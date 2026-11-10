"""
Play a video with its FaceDx results drawn on top, in a window: the box around the patient's face and the top 4
emotions and top 3 action units of each analysed frame (about 5 per second). Only a list of frame timestamps is saved.

    python play_results.py --patient S24_217 --video <file>.m2t
    python play_results.py --patient S24_217 --video <file>.m2t --start 1:20:00
    python play_results.py --patient S24_217 --episodes                  (Happiness and Sadness episodes)
    python play_results.py --patient S24_217 --episodes Happiness

Keys: space pause/play, left/right (or a/d) 10 s back/forward, down/up (or s/w) 1 min back/forward, q quit.
With --episodes (see episodes.py): n / p next / previous episode, across all of the patient's finished videos. Each
episode plays from 3 s before it to 3 s after it, then pauses (space plays on).
Frames are numbered the way main_Combined.py numbers them: frame N is the N-th frame decoded from the start of the
video. The first time a video is played, the player lists the timestamp of every frame from the file's packets (seconds;
or by decoding the whole video if they don't fit the results), saved as frame_times.npz next to its CSVs (about 2 MB
per 2 hours), so that seeking and numbering need no fps arithmetic.
"""

import argparse
import os
import queue
import re
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from facedx_settings import DEFAULT_CONFIG, read_settings
import episodes as ep
from make_overlay import AU_NAMES, STATUS, TOP_AUS, TOP_EMOTIONS, load_results, results_name, results_path
from video_times import CACHE_NAME, fits_results, frame_times, passthrough_option, probe

# OpenCV colours are (B, G, R)
WHITE, MUTED = (241, 239, 232), (169, 178, 180)
TOP_BAR, BAR, BAR_EMPTY = (117, 158, 29), (169, 178, 180), (90, 94, 95)
FACE_BOX = (165, 202, 93)
FONT = cv2.FONT_HERSHEY_SIMPLEX
KEYS = {  # waitKeyEx codes differ between window backends (GTK/Qt on Linux, Cocoa on macOS, Windows)
    'pause': {32}, 'quit': {ord('q'), 27},
    'back': {ord('a'), 65361, 2424832, 63234}, 'forward': {ord('d'), 65363, 2555904, 63235},
    'back_min': {ord('s'), 65364, 2621440, 63233}, 'forward_min': {ord('w'), 65362, 2490368, 63232},
    'next_episode': {ord('n')}, 'previous_episode': {ord('p')},
}
EPISODE_LEAD_S = 3  # an episode jump starts this long before the episode
EPISODE_TAIL_S = 3  # and pauses this long after it ends


class FrameReader:
    # Decodes the video with ffmpeg from a given frame on, scaled to the display size. Frames are numbered by finding
    # their timestamp in the video's frame list (frame_times), so frame N is the N-th decoded frame, as in the CSVs.
    # Seeking in .m2t files is not exact (decoding can start without the keyframe it needs, giving broken frames), so
    # decoding starts SEEK_MARGIN seconds early and frames are skipped until the one asked for.
    SEEK_MARGIN = 5

    def __init__(self, path, fps, size, times, file_start, passthrough):
        self.path, self.fps, self.size = path, fps, size
        self.times, self.file_start = times, file_start
        self.order = np.argsort(times, kind='stable')
        self.sorted_times = times[self.order]
        self.tolerance = 0.25 / fps
        self.proc = None
        self.target, self.last = 0, None
        self.passthrough = passthrough

    def frame_of(self, timestamp):
        # Number of the frame with this timestamp: the one after the last frame read if it matches (so repeated
        # timestamps are counted in order), else the matching frame nearest the one asked for, or None
        nxt = None if self.last is None else self.last + 1
        if nxt is not None and nxt < len(self.times) and abs(self.times[nxt] - timestamp) <= self.tolerance:
            return nxt
        lo = np.searchsorted(self.sorted_times, timestamp - self.tolerance, side='left')
        hi = np.searchsorted(self.sorted_times, timestamp + self.tolerance, side='right')
        if lo == hi:
            return None
        candidates = self.order[lo:hi]
        return int(candidates[np.argmin(np.abs(candidates - self.target))])

    def seek(self, frame):
        self.close()
        self.target, self.last = min(max(0, frame), len(self.times)), None
        # -ss counts from the start of the file; -copyts keeps the original timestamps; -noaccurate_seek passes on
        # every frame from where decoding starts, so the skipping is done here, by frame number
        at = self.times[min(self.target, len(self.times) - 1)]
        start = max(0.0, at - self.file_start - self.SEEK_MARGIN)
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
            frame = self.frame_of(timestamp)
            if frame is None:
                continue
            self.last = frame
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


class Video:
    # One video with its results, opened for playing at a given window width
    def __init__(self, patient, name, settings, max_width):
        self.path, rows, emotions, aus, self.fps, width, height = load_results(patient, name, settings)
        self.name = results_name(patient, name, settings)  # as in the episode list
        self.results = Results(rows, emotions, aus)
        video_start, file_start, self.duration = probe(self.path)
        passthrough = passthrough_option()
        last = rows['frame'].iloc[-1] if len(rows) else -1
        fits = fits_results(rows['frame'].to_numpy())
        try:
            times = frame_times(self.path, os.path.join(results_path(patient, self.name, settings), CACHE_NAME),
                                passthrough, self.duration * self.fps, fits)
        except RuntimeError as error:
            sys.exit(str(error))
        display_w = min(max_width, width)
        display_h = round(height * display_w / width / 2) * 2
        self.scale = display_w / width
        self.reader = FrameReader(self.path, self.fps, (display_w, display_h), times, file_start, passthrough)
        print(f'Video:   {self.path} ({width}x{height}, {self.fps:.2f} fps, {time_text(self.duration)}, {len(times):,} frames)')
        print(f'Results: {len(rows)} frames, patient found in {(rows["success"] == 1).mean():.0%}')
        if len(rows) and not fits(len(times)):
            print(f'WARNING: this video has {len(times):,} frames but the results end at frame {last:,}, '
                  'so they may not line up with the picture (were they made from a different file, e.g. a converted .mp4?)')


def episode_to_jump(episodes, videos, video, frame, current, step):
    # Index of the next (step 1) or previous (step -1) episode: from the last one jumped to, or after a manual seek
    # (current None) from this position, as (video order, frame)
    if current is not None:
        k = current + step
        return k if 0 <= k < len(episodes) else None
    here = (videos.index(video.name), frame)
    keys = [(videos.index(e['video']), e['frame_start']) for e in episodes]
    if step > 0:
        return next((k for k, key in enumerate(keys) if key > here), None)
    return next((k for k in range(len(keys) - 1, -1, -1) if keys[k] < here), None)


def time_text(seconds):
    seconds = int(seconds)
    return f'{seconds // 3600}:{seconds // 60 % 60:02}:{seconds % 60:02}'


def draw(image, results, i, frame, fps, duration, scale, paused, episode_text=None):
    h = image.shape[0]
    s = h / 720  # sizes below are for a 720-pixel-high window
    line, x0, margin = round(24 * s), round(14 * s), round(14 * s)
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
    if episode_text:
        lines.append((episode_text, WHITE, None, None))
    lines.append(('space pause   left/right 10 s   up/down 1 min   ' + ('n/p next/previous episode   ' if episode_text is not None else '')
                  + 'q quit', MUTED, None, None))

    # Semi-transparent dark panel behind the text
    bar_w = round(90 * s)
    text_w = max(cv2.getTextSize(text, FONT, 0.55 * s, 1)[0][0] + (bar_w + round(10 * s) if value is not None else 0)
                 for text, _, value, _ in lines)
    # In the bottom-left corner, clear of the face (the bottom edge stays put as the number of lines changes)
    y1 = h - margin
    y0 = max(0, y1 - line * len(lines) - round(10 * s))
    x1 = min(image.shape[1], x0 + text_w + round(20 * s))
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
    parser = argparse.ArgumentParser(description='Play a video with its FaceDx results drawn on top.')
    parser.add_argument('--patient', required=True, help='Patient folder name, e.g. S24_217')
    parser.add_argument('--video', help='Video file name (e.g. <file>.m2t, looked up in RAW_ROOT and MP4_ROOT) or its full path '
                                        '(optional with --episodes: default the video of the first episode)')
    parser.add_argument('--start', default='0', help='Where to start, e.g. 1:20:00 (default: the beginning)')
    parser.add_argument('--episodes', nargs='*', choices=ep.EMOTIONS, default=None,
                        help='Find episodes of high-confidence Happiness and/or Sadness (default: both) and jump between them with n / p')
    parser.add_argument('--width', type=int, default=1280, help='Window width in pixels (default: 1280, or the video width if smaller)')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='Settings file (default: local_settings.env in the repo folder)')
    args = parser.parse_args()
    settings = read_settings(args.config)
    if args.video is None and args.episodes is None:
        parser.error('give --video, or --episodes')

    episodes, videos = [], []
    if args.episodes is not None:
        patient_dir = os.path.dirname(results_path(args.patient, 'x', settings))
        if not os.path.isdir(patient_dir):
            sys.exit(f'No results for {args.patient} in {patient_dir}')
        videos = ep.finished_videos(patient_dir)
        emotions = args.episodes or list(ep.EMOTIONS)
        episodes, thresholds = ep.find_episodes(
            patient_dir, emotions, progress=lambda n, total: print(f'\rFinding episodes: video {n}/{total}', end='', flush=True))
        print()
        for emo in emotions:
            count = sum(e['emotion'] == emo for e in episodes)
            print(f'{emo}: {count} episodes (P > {thresholds.get(emo, float("nan")):.3f}, the {ep.PERCENTILE}th percentile)')
        if not episodes and args.video is None:
            sys.exit('No episodes found.')

    stop_at = None  # frame to pause at: a few seconds after the episode jumped to
    if args.video is None:
        video = Video(args.patient, episodes[0]['video'], settings, args.width)
        current = 0
        start_frame = episodes[0]['frame_start'] - round(EPISODE_LEAD_S * video.fps)
        stop_at = episodes[0]['frame_end'] + round(EPISODE_TAIL_S * video.fps)
    else:
        video = Video(args.patient, args.video, settings, args.width)
        current = None
        start_frame = round(parse_time(args.start) * video.fps)
        if episodes and video.name not in videos:
            print(f'Note: {video.name} has no finished results folder, so n / p start from the first episode.')
            videos = [video.name] + videos
    video.reader.seek(start_frame)

    window = 'FaceDx results'
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    paused, image, frame = False, None, video.reader.target
    need_frame = True  # read a frame even while paused (at the start and after a jump)
    clock_start, clock_frame = time.monotonic(), video.reader.target
    try:
        while True:
            if need_frame or not paused:
                got = video.reader.read()
                if got is not None:
                    frame, image = got
                    if need_frame:  # restart the clock from the first frame after a jump
                        clock_start, clock_frame = time.monotonic(), frame
                elif image is None:
                    sys.exit('Could not read any frames from this point of the video.')
                else:  # end of the video (or a jump past it): stay on the frame shown, paused
                    paused = True
                need_frame = False
                if stop_at is not None and frame >= stop_at:  # end of the episode clip: pause (space plays on)
                    paused, stop_at = True, None

            episode_text = None
            if args.episodes is not None:
                inside = [k for k, e in enumerate(episodes) if e['video'] == video.name and e['frame_start'] <= frame <= e['frame_end']]
                if inside:
                    e = episodes[inside[0]]
                    episode_text = f'Episode {inside[0] + 1}/{len(episodes)}: {e["emotion"]}, {e["seconds"]:.0f} s, mean P {e["p_mean"]:.2f}'
                elif current is not None and episodes[current]['video'] == video.name:
                    e = episodes[current]
                    when = (f'starts in {(e["frame_start"] - frame) / video.fps:.0f} s' if frame < e['frame_start']
                            else 'ended' + (' (n next episode, space keep playing)' if paused else ''))
                    episode_text = f'Episode {current + 1}/{len(episodes)}: {e["emotion"]}, {when}'
                else:
                    episode_text = f'{len(episodes)} episodes'
            shown = image.copy()
            draw(shown, video.results, video.results.at(frame), frame, video.fps, video.duration, video.scale, paused, episode_text)
            cv2.imshow(window, shown)

            # Wait until the next frame is due, so the video plays at its real speed
            delay = 30 if paused else max(1, round((clock_start + (frame + 1 - clock_frame) / video.fps - time.monotonic()) * 1000))
            key = cv2.waitKeyEx(delay)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1 or key in KEYS['quit']:
                break
            for name, seconds in {'back': -10, 'forward': 10, 'back_min': -60, 'forward_min': 60}.items():
                if key in KEYS[name]:
                    video.reader.seek(frame + round(seconds * video.fps))
                    need_frame, current, stop_at = True, None, None
            for name, step in {'next_episode': 1, 'previous_episode': -1}.items():
                if key in KEYS[name] and episodes:
                    k = episode_to_jump(episodes, videos, video, frame, current, step)
                    if k is None:
                        continue  # no episode further on (or before): stay
                    if episodes[k]['video'] != video.name:
                        video.reader.close()
                        video = Video(args.patient, episodes[k]['video'], settings, args.width)
                    video.reader.seek(episodes[k]['frame_start'] - round(EPISODE_LEAD_S * video.fps))
                    need_frame, current, paused = True, k, False
                    stop_at = episodes[k]['frame_end'] + round(EPISODE_TAIL_S * video.fps)
            if key in KEYS['pause']:
                paused = not paused
                clock_start, clock_frame = time.monotonic(), frame + 1
    finally:
        video.reader.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
