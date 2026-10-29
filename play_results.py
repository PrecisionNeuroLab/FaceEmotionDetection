"""
Play a video with its FaceDx results drawn on top, in a window: the box around the patient's face and the top 4
emotions and top 3 action units of each analysed frame (about 5 per second). Nothing is written to disk.

    python play_results.py --patient S24_217 --video <file>.m2t
    python play_results.py --patient S24_217 --video <file>.m2t --start 1:20:00

Keys: space pause/play, left/right (or a/d) 10 s back/forward, down/up (or s/w) 1 min back/forward, q quit.
Frames are decoded with ffmpeg and counted from the start of the recording, so they match the CSV frame numbers.
"""

import argparse
import json
import queue
import re
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from facedx_settings import DEFAULT_CONFIG, read_settings
from make_overlay import AU_NAMES, STATUS, TOP_AUS, TOP_EMOTIONS, load_results

# OpenCV colours are (B, G, R)
WHITE, MUTED = (241, 239, 232), (169, 178, 180)
TOP_BAR, BAR, BAR_EMPTY = (117, 158, 29), (169, 178, 180), (90, 94, 95)
FACE_BOX = (165, 202, 93)
FONT = cv2.FONT_HERSHEY_SIMPLEX
KEYS = {  # waitKeyEx codes differ between window backends (GTK/Qt on Linux, Cocoa on macOS, Windows)
    'pause': {32}, 'quit': {ord('q'), 27},
    'back': {ord('a'), 65361, 2424832, 63234}, 'forward': {ord('d'), 65363, 2555904, 63235},
    'back_min': {ord('s'), 65364, 2621440, 63233}, 'forward_min': {ord('w'), 65362, 2490368, 63232},
}


def probe(path):
    # Start times (seconds) of the video stream and of the file, and the duration
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=start_time,duration:stream=start_time',
                          '-select_streams', 'v:0', '-of', 'json', path], capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    file_start = float(info['format'].get('start_time', 0))
    video_start = float(info['streams'][0].get('start_time', file_start)) if info['streams'] else file_start
    return video_start, file_start, float(info['format'].get('duration', 0))


def passthrough_option():
    # Keep every decoded frame as it is (no frames repeated or dropped): -fps_mode in ffmpeg 5.1+, -vsync before that
    test = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'nullsrc=s=16x16',
                           '-frames:v', '1', '-fps_mode', 'passthrough', '-f', 'null', '-'], capture_output=True)
    return ['-fps_mode', 'passthrough'] if test.returncode == 0 else ['-vsync', '0']


class FrameReader:
    # Decodes the video with ffmpeg from a given frame on, scaled to the display size. Each frame is numbered from its
    # timestamp (frame 0 = first video frame), the way main_Combined.py counts frames when it reads the whole video.
    # Seeking in .m2t files is not exact (decoding can start without the keyframe it needs, giving broken frames), so
    # decoding starts SEEK_MARGIN seconds early and frames are skipped until the one asked for.
    SEEK_MARGIN = 5

    def __init__(self, path, fps, size, video_start, file_start):
        self.path, self.fps, self.size = path, fps, size
        self.video_start, self.file_start = video_start, file_start
        self.proc = None
        self.target = 0
        self.passthrough = passthrough_option()

    def seek(self, frame):
        self.close()
        self.target = max(0, frame)
        # -ss counts from the start of the file; -copyts keeps the original timestamps; -noaccurate_seek passes on
        # every frame from where decoding starts, so the skipping is done here, by timestamp
        start = max(0.0, self.video_start - self.file_start + self.target / self.fps - self.SEEK_MARGIN)
        w, h = self.size
        self.proc = subprocess.Popen(
            ['ffmpeg', '-hide_banner', '-nostats', '-loglevel', 'info', '-copyts', '-noaccurate_seek',
             '-ss', f'{start:.4f}', '-i', self.path, '-an', '-sn', '-vf', f'showinfo,scale={w}:{h}', *self.passthrough,
             '-pix_fmt', 'bgr24', '-f', 'rawvideo', '-'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=w * h * 3 * 4)
        # showinfo writes one line per frame (with its timestamp) to stderr, in the same order as the frames
        self.timestamps = queue.Queue()
        threading.Thread(target=self._read_timestamps, args=(self.proc.stderr, self.timestamps), daemon=True).start()

    @staticmethod
    def _read_timestamps(stream, timestamps):
        for line in stream:
            match = re.search(rb' n: *\d+ .*pts_time: *([-\d.]+)', line)
            if match:
                timestamps.put(float(match.group(1)))
        timestamps.put(None)

    def read(self):
        # (frame number, image) of the next frame at or after the one asked for, or None at the end of the video
        w, h = self.size
        while True:
            data = self.proc.stdout.read(w * h * 3)
            if len(data) < w * h * 3:
                return None
            timestamp = self.timestamps.get(timeout=10)
            if timestamp is None:
                return None
            frame = round((timestamp - self.video_start) * self.fps)
            if frame >= self.target:
                self.target = frame + 1
                return frame, np.frombuffer(data, np.uint8).reshape(h, w, 3).copy()

    def close(self):
        if self.proc:
            self.proc.kill()
            self.proc.wait()
            self.proc = None


class Results:
    # The CSV rows as arrays, for a fast lookup of the analysed frame shown at each video frame
    def __init__(self, rows, emotions, aus):
        self.frames = rows['frame'].to_numpy()
        self.step = np.median(np.diff(self.frames)) if len(self.frames) > 1 else 1
        self.success = rows['success'].to_numpy()
        self.emotions, self.aus = emotions, aus
        self.emotion_scores = rows[emotions].to_numpy(float)
        self.au_scores = rows[aus].to_numpy(float)
        self.boxes = rows[['Facial Box X', 'Facial Box Y', 'Facial Box W', 'Facial Box H']].to_numpy(float)

    def at(self, frame):
        # Index of the last analysed frame at or before this frame, or None past the end or before the first
        i = np.searchsorted(self.frames, frame, side='right') - 1
        if i < 0 or frame - self.frames[i] >= self.step * 2:
            return None
        return i


def time_text(seconds):
    seconds = int(seconds)
    return f'{seconds // 3600}:{seconds // 60 % 60:02}:{seconds % 60:02}'


def draw(image, results, i, frame, fps, duration, scale, paused):
    h = image.shape[0]
    s = h / 720  # sizes below are for a 720-pixel-high window
    line, x0, y0 = round(24 * s), round(14 * s), round(14 * s)
    lines = []  # (text, colour, bar value or None, bar colour)

    success = results.success[i] if i is not None else None
    status = STATUS.get(int(success), 'Unknown') if success is not None else 'No results'
    lines.append((f'{status}   frame {frame:,}   {time_text(frame / fps)} / {time_text(duration)}' + ('   paused' if paused else ''), WHITE, None, None))
    if success == 1:
        x, y, w, bh = results.boxes[i] * scale
        if w > 0:
            cv2.rectangle(image, (round(x), round(y)), (round(x + w), round(y + bh)), FACE_BOX, max(2, round(2 * s)))
        lines.append(('Emotions', MUTED, None, None))
        for n, j in enumerate(np.argsort(-results.emotion_scores[i])[:TOP_EMOTIONS]):
            value = results.emotion_scores[i, j]
            lines.append((f'{value:.2f}  {results.emotions[j]}', WHITE, value, TOP_BAR if n == 0 else BAR))
        lines.append(('Action units', MUTED, None, None))
        for j in np.argsort(-results.au_scores[i])[:TOP_AUS]:
            value, name = results.au_scores[i, j], results.aus[j]
            lines.append((f'{value:.2f}  {name} {AU_NAMES[name]}', WHITE, value, BAR))
    lines.append(('space pause   left/right 10 s   up/down 1 min   q quit', MUTED, None, None))

    # Semi-transparent dark panel behind the text
    bar_w = round(90 * s)
    text_w = max(cv2.getTextSize(text, FONT, 0.55 * s, 1)[0][0] + (bar_w + round(10 * s) if value is not None else 0)
                 for text, _, value, _ in lines)
    x1, y1 = min(image.shape[1], x0 + text_w + round(20 * s)), y0 + line * len(lines) + round(10 * s)
    region = image[y0:y1, x0:x1]
    region[:] = (region * 0.45).astype(np.uint8)

    for n, (text, colour, value, bar_colour) in enumerate(lines):
        y = y0 + round(8 * s) + line * (n + 1) - round(7 * s)
        x = x0 + round(10 * s)
        if value is not None:
            top, bottom = y - round(11 * s), y
            cv2.rectangle(image, (x, top), (x + bar_w, bottom), BAR_EMPTY, -1)
            cv2.rectangle(image, (x, top), (x + round(bar_w * min(max(value, 0), 1)), bottom), bar_colour, -1)
            x += bar_w + round(10 * s)
        cv2.putText(image, text, (x, y), FONT, 0.55 * s, colour, 1, cv2.LINE_AA)


def parse_time(text):
    # 1:20:00, 20:00 or 4800 -> seconds
    seconds = 0.0
    for part in text.split(':'):
        seconds = seconds * 60 + float(part)
    return seconds


def main():
    parser = argparse.ArgumentParser(description='Play a video with its FaceDx results drawn on top (writes nothing).')
    parser.add_argument('--patient', required=True, help='Patient folder name, e.g. S24_217')
    parser.add_argument('--video', required=True, help='Video file name (e.g. <file>.m2t, looked up in RAW_ROOT and MP4_ROOT) or its full path')
    parser.add_argument('--start', default='0', help='Where to start, e.g. 1:20:00 (default: the beginning)')
    parser.add_argument('--width', type=int, default=1280, help='Window width in pixels (default: 1280, or the video width if smaller)')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='Settings file (default: local_settings.env in the repo folder)')
    args = parser.parse_args()
    settings = read_settings(args.config)

    video_path, rows, emotions, aus, fps, width, height = load_results(args.patient, args.video, settings)
    results = Results(rows, emotions, aus)
    video_start, file_start, duration = probe(video_path)
    display_w = min(args.width, width)
    display_h = round(height * display_w / width / 2) * 2
    scale = display_w / width
    print(f'Video:   {video_path} ({width}x{height}, {fps:.2f} fps, {time_text(duration)})')
    print(f'Results: {len(rows)} frames, patient found in {(rows["success"] == 1).mean():.0%}')

    reader = FrameReader(video_path, fps, (display_w, display_h), video_start, file_start)
    reader.seek(round(parse_time(args.start) * fps))
    window = 'FaceDx results'
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    paused, image, frame = False, None, reader.target
    need_frame = True  # read a frame even while paused (at the start and after a jump)
    clock_start, clock_frame = time.monotonic(), reader.target
    try:
        while True:
            if need_frame or not paused:
                got = reader.read()
                if got is not None:
                    frame, image = got
                    if need_frame:  # restart the clock from the first frame after a jump
                        clock_start, clock_frame = time.monotonic(), frame
                elif image is None:
                    sys.exit('Could not read any frames from this point of the video.')
                else:  # end of the video (or a jump past it): stay on the frame shown, paused
                    paused = True
                need_frame = False
            shown = image.copy()
            draw(shown, results, results.at(frame), frame, fps, duration, scale, paused)
            cv2.imshow(window, shown)

            # Wait until the next frame is due, so the video plays at its real speed
            delay = 30 if paused else max(1, round((clock_start + (frame + 1 - clock_frame) / fps - time.monotonic()) * 1000))
            key = cv2.waitKeyEx(delay)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1 or key in KEYS['quit']:
                break
            for name, seconds in {'back': -10, 'forward': 10, 'back_min': -60, 'forward_min': 60}.items():
                if key in KEYS[name]:
                    reader.seek(frame + round(seconds * fps))
                    need_frame = True
            if key in KEYS['pause']:
                paused = not paused
            if key in KEYS['pause']:
                clock_start, clock_frame = time.monotonic(), frame + 1
    finally:
        reader.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
