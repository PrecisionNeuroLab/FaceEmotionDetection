"""
Episodes of sustained, high-confidence Happiness or Sadness in one patient's FaceDx results, from the face data alone.

Same rules as detect_param() in PrecisionNeuroLab/LZ-state-analysis fau/fau_emo_defs.py at the strictest sweep level
used there (98th percentile, AU signature 0.5 SD), without its brain-data checks (iEEG QC, distance from seizures):

  - Frames count if the patient's face was found and its box is >= 30 px wide. They are averaged over 4 s segments
    of real time, and a segment needs >= 12 such frames.
  - Episode: P(emotion) above its 98th percentile over all of the patient's usable segments, for >= 3 segments in a
    row (12 s), within one video.
  - Baseline: segments in runs of >= 5 (20 s) with P(Neutral) > 0.5, in the same video, centre within 10 min of the
    episode and >= 60 s from any episode of that emotion. An episode needs >= 5 baseline segments.
  - AU check: the mean z-score of the emotion's AUs (z over all usable segments) is >= 0.5 above its baseline.

Real time = CSV timestamp x 1.2045: the recordings' timestamps run about 20% fast (LZ-state-analysis fau/README.md).
"""

import os

import numpy as np
import pandas as pd

EMOTIONS = ('Happiness', 'Sadness')
SIGNATURE = {'Happiness': ['AU6', 'AU12'], 'Sadness': ['AU1', 'AU4', 'AU15']}  # prototype AUs (fau_emo_defs.py)
REAL_TIME_SCALE = 1.2045
SEGMENT_S = 4.0
MIN_FRAMES = 12
MIN_BOX = 30
PERCENTILE = 98
AU_MIN = 0.5
MIN_SEGMENTS = 3
NEUTRAL_THRESHOLD = 0.5
NEUTRAL_MIN_SEGMENTS = 5
BASELINE_WINDOW_S = 600
BASELINE_GAP_S = 60
BASELINE_MIN = 5

SIGNATURE_AUS = sorted({au for aus in SIGNATURE.values() for au in aus})


def segments(results_dir):
    # One row per 4 s segment of the video (also segments without a usable face): mean emotion probabilities and
    # signature AUs over the usable frames, face_ok, and the first and last CSV frame of the segment
    hse = pd.read_csv(os.path.join(results_dir, 'outputs_hse.csv'))
    ogau = pd.read_csv(os.path.join(results_dir, 'outputs_ogau.csv'), usecols=['frame'] + SIGNATURE_AUS)
    boxes = pd.read_csv(os.path.join(results_dir, 'outputs_bboxes.csv'), usecols=['frame', 'Facial Box W'])
    rows = hse.merge(ogau, on='frame').merge(boxes, on='frame')
    segment = np.floor(rows['timestamp'] * REAL_TIME_SCALE / SEGMENT_S).astype(int)
    usable = (rows['success'] == 1) & (rows['Facial Box W'] >= MIN_BOX)
    columns = list(hse.columns[3:]) + SIGNATURE_AUS
    grid = rows[usable].groupby(segment[usable])[columns].mean()
    grid['frames'] = rows[usable].groupby(segment[usable]).size()
    grid = grid.reindex(range(segment.max() + 1 if len(rows) else 0))
    span = rows.groupby(segment)['frame'].agg(['min', 'max']).reindex(grid.index)
    grid['frame_start'], grid['frame_end'] = span['min'], span['max']
    grid['face_ok'] = grid['frames'].fillna(0) >= MIN_FRAMES
    grid.loc[~grid['face_ok'], columns] = np.nan
    return grid


def runs_of(mask, min_len):
    # (starts, ends) of runs of True at least min_len long, ends exclusive
    d = np.diff(np.r_[0, mask.astype(int), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = ends - starts >= min_len
    return starts[keep], ends[keep]


def finished_videos(patient_dir):
    return sorted(v for v in os.listdir(patient_dir) if not v.startswith('.') and not v.endswith('.partial')
                  and os.path.exists(os.path.join(patient_dir, v, 'outputs_bboxes.csv')))


def find_episodes(patient_dir, emotions=EMOTIONS, progress=None, grids=None):
    """Episodes of every finished video in patient_dir (OUT_ROOT/outputs_Combined/<patient>), sorted by video and
    frame: dicts with video, emotion, frame_start, frame_end, seconds, mean P(emotion); plus the P thresholds.
    grids: the videos' segments() if already read."""
    videos = finished_videos(patient_dir)
    if grids is None:
        grids = {}
        for n, video in enumerate(videos, 1):
            if progress:
                progress(n, len(videos))
            grids[video] = segments(os.path.join(patient_dir, video))
    if not grids:
        return [], {}
    every = pd.concat(grids.values())
    ok = every['face_ok'].to_numpy()
    thresholds = {emo: float(np.nanquantile(every[emo].to_numpy()[ok], PERCENTILE / 100)) for emo in emotions}
    au_mean = every[SIGNATURE_AUS][ok].mean()
    au_sd = every[SIGNATURE_AUS][ok].std(ddof=0)

    episodes = []
    for video, grid in grids.items():
        ok = grid['face_ok'].to_numpy()
        t = (grid.index.to_numpy() + 0.5) * SEGMENT_S
        neutral = np.nan_to_num(grid['Neutral'].to_numpy())
        in_neutral_run = np.zeros(len(grid), bool)
        for s, e in zip(*runs_of(ok & (neutral > NEUTRAL_THRESHOLD), NEUTRAL_MIN_SEGMENTS)):
            in_neutral_run[s:e] = True
        for emo in emotions:
            p = np.nan_to_num(grid[emo].to_numpy())
            starts, ends = runs_of(ok & (p > thresholds[emo]), MIN_SEGMENTS)
            near_episode = np.zeros(len(grid), bool)
            for s, e in zip(starts, ends):
                near_episode |= (t >= t[s] - BASELINE_GAP_S) & (t <= t[e - 1] + BASELINE_GAP_S)
            au_z = ((grid[SIGNATURE[emo]] - au_mean[SIGNATURE[emo]]) / au_sd[SIGNATURE[emo]]).to_numpy()
            for s, e in zip(starts, ends):
                baseline = in_neutral_run & ~near_episode & (np.abs(t - t[s:e].mean()) <= BASELINE_WINDOW_S)
                if baseline.sum() < BASELINE_MIN:
                    continue
                if not np.nanmean(au_z[s:e]) - np.nanmean(au_z[baseline]) >= AU_MIN:
                    continue
                episodes.append({'video': video, 'emotion': emo, 'frame_start': int(grid['frame_start'].iloc[s]),
                                 'frame_end': int(grid['frame_end'].iloc[e - 1]), 'seconds': (e - s) * SEGMENT_S,
                                 'p_mean': float(p[s:e].mean())})
    episodes.sort(key=lambda ep: (videos.index(ep['video']), ep['frame_start']))
    return episodes, thresholds
