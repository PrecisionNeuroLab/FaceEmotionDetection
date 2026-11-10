"""
The timestamp of every frame of a video, as stored in the file, numbered the way main_Combined.py numbers frames
(frame N = the N-th frame decoded from the start of the video), so that the FaceDx results can be put on the video's
own time line (for the player, and for lining the results up with other recordings such as iEEG).

The list is read from the video's packets (seconds for a long recording), or by decoding the whole video if the
packets don't give the frames exactly (minutes), and cached as frame_times.npz next to the video's CSVs.
write_frame_times_csv() also writes outputs_frame_times.csv there: the file's time of each analysed frame.
Needs ffmpeg and ffprobe.
"""

import json
import os
import re
import subprocess
import time

import cv2
import numpy as np
import pandas as pd

CACHE_NAME = 'frame_times.npz'
CSV_NAME = 'outputs_frame_times.csv'


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


def read_frame_times(stream, on_frame=None):
    # Timestamps of the frames reported by ffmpeg's showinfo filter on stderr, in decoding output order
    times = []
    for line in stream:
        match = re.search(rb' n: *\d+ .*pts_time: *([-\d.]+)', line)
        if match:
            times.append(float(match.group(1)))
            if on_frame:
                on_frame(len(times))
    return times


def packet_frame_times(path, passthrough):
    # Frame timestamps from the video packets, without decoding (seconds for a long recording): the packets'
    # presentation times in order, from the first frame the decoder outputs (frames before the first keyframe are
    # dropped). None if the packets can't be used this way (missing or non-increasing timestamps).
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'packet=pts_time,dts_time',
                          '-of', 'csv=p=0', path], capture_output=True, text=True)
    try:
        packets = np.array([[float(v) for v in line.split(',')[:2]] for line in out.stdout.split()])
    except ValueError:  # N/A timestamps
        return None
    if out.returncode or not len(packets) or not np.all(np.diff(packets[:, 1]) > 0):
        return None
    first = subprocess.run(['ffmpeg', '-hide_banner', '-nostdin', '-nostats', '-loglevel', 'info', '-copyts', '-i', path,
                            '-an', '-sn', '-vf', 'showinfo', *passthrough, '-frames:v', '1', '-f', 'null', '-'],
                           capture_output=True)
    first_time = read_frame_times(first.stderr.splitlines())
    if not first_time:
        return None
    times = np.sort(packets[:, 0])
    return times[times >= first_time[0] - 1e-4]


def decoded_frame_times(path, passthrough, expected_frames):
    # Frame timestamps from decoding the whole video with the same ffmpeg options as FrameReader (minutes)
    proc = subprocess.Popen(['ffmpeg', '-hide_banner', '-nostdin', '-nostats', '-loglevel', 'info', '-copyts', '-i', path,
                             '-an', '-sn', '-vf', 'showinfo', *passthrough, '-f', 'null', '-'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    started = time.monotonic()
    shown = [started]

    def progress(n):
        now = time.monotonic()
        if now - shown[0] >= 1:
            shown[0] = now
            print(f'\r  {n:,} frames, about {min(n / max(expected_frames, 1), 1):.0%}, {now - started:.0f} s', end='', flush=True)

    times = np.array(read_frame_times(proc.stderr, progress))
    proc.wait()
    print(f'\r  {len(times):,} frames in {time.monotonic() - started:.0f} s' + ' ' * 20)
    return times


def frame_times(path, cache_path, passthrough, expected_frames, fits_results):
    # Timestamp of every frame of the video: frame N of the CSVs (main_Combined.py counts decoded frames) is entry N.
    # From the packets if their count fits the results (fits_results(count)), else by decoding the whole video.
    # Cached in cache_path.
    key = (os.path.basename(path), os.path.getsize(path))
    try:
        with np.load(cache_path) as cached:
            if (str(cached['video']), int(cached['size'])) == key:
                return cached['times']
    except (OSError, KeyError, ValueError):
        pass
    print(f'Listing the frames of {key[0]} (once per video, then saved)...')
    times = packet_frame_times(path, passthrough)
    if times is None or not fits_results(len(times)):
        print('  The video packets do not give the frames exactly, so decoding the whole video (once):')
        times = decoded_frame_times(path, passthrough, expected_frames)
    if not len(times):
        raise RuntimeError(f'ffmpeg could not decode any frames of {path}')
    try:
        with open(cache_path + '.part', 'wb') as file:
            np.savez(file, times=times, video=key[0], size=key[1])
        os.replace(cache_path + '.part', cache_path)
    except OSError as error:
        print(f'  (could not save the frame list: {error})')
    return times


def fits_results(frames):
    # main_Combined.py analyses every step-th frame up to the last one it decodes, so its last analysed frame is within
    # one step of the end: a frame count that doesn't fit this means the video was decoded differently
    frames = np.asarray(frames)
    last = frames[-1] if len(frames) else -1
    step = np.median(np.diff(frames)) if len(frames) > 1 else 1
    return lambda count: 0 < count - last <= step


def write_frame_times_csv(results_dir, video_path):
    """Write results_dir/outputs_frame_times.csv: one row per analysed frame of outputs_hse.csv, with
      frame       the frame number of the other CSVs
      timestamp   their timestamp (frame / fps rounded up, an estimate)
      video_time  seconds since the first frame of the video, from the timestamps stored in the file
      pts         the timestamp stored in the file itself (seconds; the first frame's is often not 0)
    Returns the number of frames of the video, or raises RuntimeError if its frames don't fit the results."""
    rows = pd.read_csv(os.path.join(results_dir, 'outputs_hse.csv'), usecols=['frame', 'timestamp'])
    rows['frame'] = rows['frame'].round().astype(int)  # saved as floats
    _, _, duration = probe(video_path)
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30  # only for the progress shown while decoding
    capture.release()
    fits = fits_results(rows['frame'].to_numpy())
    times = frame_times(video_path, os.path.join(results_dir, CACHE_NAME), passthrough_option(), duration * fps, fits)
    if not fits(len(times)):
        raise RuntimeError(f'{os.path.basename(video_path)} has {len(times):,} frames but the results end at frame '
                           f'{rows["frame"].iloc[-1]:,}: they were not made from this file')
    pts = times[rows['frame'].to_numpy()]
    out = rows.assign(video_time=np.round(pts - times[0], 4), pts=np.round(pts, 4))
    path = os.path.join(results_dir, CSV_NAME)
    out.to_csv(path + '.part', index=False)
    os.replace(path + '.part', path)
    return len(times)
