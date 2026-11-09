"""
Write an index of every video with how usable its FaceDx results are, to see at a glance which recordings are worth
looking at:

    python make_index.py                       (all patients in OUT_ROOT/outputs_Combined)
    python make_index.py --patient S24_217     (update only this patient's rows)

main_Combined.py also updates its patient's rows at the end of each run.

Writes OUT_ROOT/FaceDx_index.csv (outside outputs_Combined, so nothing that lists the results folders sees it), one
row per video: finished results, results still being processed, and raw .m2t / .wmv recordings in RAW_ROOT/<patient>/ not
processed yet. Only the CSVs are read (and the raw folder listed); nothing else is written.

Columns of a finished video:
  minutes             length of the video in real time (CSV timestamps x 1.2045 for .m2t, as in episodes.py)
  patient_found_pct   analysed frames with the patient's face (success 1)
  no_face_pct         ... with no face found (success 0)
  other_face_pct      ... with a face that isn't the patient (success -1 / -2)
  median_face_px      median width of the patient's face box
  usable_pct          4 s segments usable for episodes (>= 12 frames with the patient's face >= 30 px), as in episodes.py
  usable_minutes      the same, in minutes of real time
  happiness_episodes, sadness_episodes   episodes found by episodes.py (thresholds over all of the patient's videos)
  emotion_confidence  mean probability of the top emotion over the patient's frames (0.14 = no preference among the 7,
                      1 = certain); the model's own certainty, not a measure of accuracy
  emotion_margin      mean gap between the top two emotions' probabilities (low = often torn between two)
  face_score          mean face-detection confidence (MTCNN, 0-1) of the patient's face   } from outputs_confidence.csv:
  unchecked_pct       patient's frames with the only face in view, taken to be the patient }  only videos processed
                      without comparing it with the photos (could be a nurse or visitor)   }  since this was added
  match_distance      median VGG-Face distance of the faces compared with the photos       }
                      (lower = more alike; < 0.32 counts as the patient)                   }
  quality             good: usable_pct >= 50 and median_face_px >= 60; fair: usable_pct >= 20; poor: less
  note                the main reason for a fair or poor rating
"""

import argparse
import os

import numpy as np
import pandas as pd

import episodes as ep
from facedx_settings import DEFAULT_CONFIG, read_settings

GOOD_USABLE_PCT = 50
GOOD_FACE_PX = 60
FAIR_USABLE_PCT = 20
COLUMNS = ['patient', 'video', 'status', 'quality', 'note', 'minutes', 'patient_found_pct', 'no_face_pct',
           'other_face_pct', 'median_face_px', 'usable_pct', 'usable_minutes', 'happiness_episodes', 'sadness_episodes',
           'emotion_confidence', 'emotion_margin', 'face_score', 'unchecked_pct', 'match_distance']


def rate(row):
    # (quality, note) of a finished video
    usable, face = row['usable_pct'], row['median_face_px']
    reasons = []
    if row['patient_found_pct'] < FAIR_USABLE_PCT:
        biggest = max(('no face found', row['no_face_pct']), ('another face', row['other_face_pct']), key=lambda r: r[1])
        reasons.append(f'patient found in {row["patient_found_pct"]:.0f}% of frames (mostly {biggest[0]})')
    if not face >= GOOD_FACE_PX:
        reasons.append('no face box' if np.isnan(face) else f'face small (median {face:.0f} px)')
    if usable < GOOD_USABLE_PCT and not reasons:
        reasons.append(f'face usable in {usable:.0f}% of the video (gaps or small face at times)')
    if usable >= GOOD_USABLE_PCT and face >= GOOD_FACE_PX:
        return 'good', ''
    return ('fair' if usable >= FAIR_USABLE_PCT else 'poor'), '; '.join(reasons)


def video_row(patient, video, results_dir, grid):
    hse = pd.read_csv(os.path.join(results_dir, 'outputs_hse.csv'))
    boxes = pd.read_csv(os.path.join(results_dir, 'outputs_bboxes.csv'), usecols=['success', 'Facial Box W'])
    n = max(len(hse), 1)
    found = boxes['success'] == 1
    usable = int(grid['face_ok'].sum())
    row = {'patient': patient, 'video': video, 'status': 'finished',
           'minutes': round(hse['timestamp'].max() * ep.time_scale(video) / 60, 1) if len(hse) else 0.0,
           'patient_found_pct': round(100 * (hse['success'] == 1).sum() / n, 1),
           'no_face_pct': round(100 * (hse['success'] == 0).sum() / n, 1),
           'other_face_pct': round(100 * hse['success'].isin([-1, -2]).sum() / n, 1),
           'median_face_px': float(boxes.loc[found, 'Facial Box W'].median()) if found.any() else np.nan,
           'usable_pct': round(100 * usable / max(len(grid), 1), 1),
           'usable_minutes': round(usable * ep.SEGMENT_S / 60, 1)}
    # The emotion model's certainty on the patient's frames (its 7 probabilities add up to 1)
    scores = np.sort(hse.loc[hse['success'] == 1, hse.columns[3:]].to_numpy(float), axis=1)
    row['emotion_confidence'] = round(float(scores[:, -1].mean()), 3) if len(scores) else np.nan
    row['emotion_margin'] = round(float((scores[:, -1] - scores[:, -2]).mean()), 3) if len(scores) else np.nan
    # Face detection and patient match confidence (outputs_confidence.csv, written by main_Combined.py since it was added)
    row.update(face_score=np.nan, unchecked_pct=np.nan, match_distance=np.nan)
    confidence_path = os.path.join(results_dir, 'outputs_confidence.csv')
    if os.path.exists(confidence_path):
        conf = pd.read_csv(confidence_path)
        patient = conf[conf['success'] == 1]
        if len(patient):
            row['face_score'] = round(float(patient['face_score'].mean()), 3)
            row['unchecked_pct'] = round(100 * float((patient['patient_match'] == 'single face').mean()), 1)
            row['match_distance'] = round(float(patient['match_distance'].median()), 3) if patient['match_distance'].notna().any() else np.nan
    row['quality'], row['note'] = rate(row)
    return row


def raw_videos(settings, patient):
    # Names of the raw .m2t and .wmv recordings of the patient (all subfolders of RAW_ROOT/<patient>/)
    if not settings['RAW_ROOT']:
        return set()
    names = set()
    for _, _, files in os.walk(os.path.join(settings['RAW_ROOT'], patient)):
        names.update(f for f in files if f.lower().endswith(('.m2t', '.wmv')) and not f.startswith('.'))
    return names


def patient_rows(patient, patient_dir, settings):
    finished = ep.finished_videos(patient_dir)
    grids = {}
    for k, video in enumerate(finished, 1):
        print(f'\r{patient}: reading video {k}/{len(finished)}', end='', flush=True)
        grids[video] = ep.segments(os.path.join(patient_dir, video))
    print(f'\r{patient}: {len(finished)} finished videos' + ' ' * 20)
    episodes, _ = ep.find_episodes(patient_dir, grids=grids) if grids else ([], {})
    rows = []
    for video in finished:
        row = video_row(patient, video, os.path.join(patient_dir, video), grids[video])
        for emo in ep.EMOTIONS:
            row[f'{emo.lower()}_episodes'] = sum(e['video'] == video and e['emotion'] == emo for e in episodes)
        rows.append(row)
    processing = sorted(d[:-len('.partial')] for d in os.listdir(patient_dir) if d.endswith('.partial'))
    rows += [{'patient': patient, 'video': v, 'status': 'processing'} for v in processing]
    # Raw recordings without results (results of converted videos are named <file>.mp4)
    have = {os.path.splitext(v)[0] for v in finished + processing}
    rows += [{'patient': patient, 'video': v, 'status': 'not processed'}
             for v in sorted(raw_videos(settings, patient)) if os.path.splitext(v)[0] not in have]
    return rows


def update_index(settings, patients=None):
    """(Re)write OUT_ROOT/FaceDx_index.csv: the rows of these patients, or of all patients with results. Rows of other
    patients already in the index are always kept (also of patients whose results folder is gone).
    Also run by main_Combined.py at the end of each run, for its patient."""
    results_root = os.path.join(settings['OUT_ROOT'], 'outputs_Combined')
    out_path = os.path.join(settings['OUT_ROOT'], 'FaceDx_index.csv')
    if patients is None:
        patients = sorted(p for p in os.listdir(results_root) if not p.startswith('.') and os.path.isdir(os.path.join(results_root, p)))
    rows = []
    for patient in patients:
        patient_dir = os.path.join(results_root, patient)
        if not os.path.isdir(patient_dir):
            print(f'{patient}: no results in {patient_dir}')
            continue
        rows += patient_rows(patient, patient_dir, settings)
    redone = {row['patient'] for row in rows}

    index = pd.DataFrame(rows, columns=COLUMNS)
    if os.path.exists(out_path):
        old = pd.read_csv(out_path)
        index = pd.concat([part for part in [old[~old['patient'].isin(redone)], index] if len(part)], ignore_index=True)
        index = index.reindex(columns=COLUMNS)
    index = index.sort_values(['patient', 'video'], kind='stable')
    for col in ['happiness_episodes', 'sadness_episodes']:
        index[col] = index[col].astype('Int64')
    index.to_csv(out_path + '.part', index=False)
    os.replace(out_path + '.part', out_path)

    done = index[index['status'] == 'finished']
    counts = ', '.join(f'{(done["quality"] == q).sum()} {q}' for q in ['good', 'fair', 'poor'])
    others = ', '.join(f'{n} {s}' for s, n in index['status'].value_counts().items() if s != 'finished')
    print(f'Wrote {out_path}: {len(done)} finished videos ({counts})' + (f', {others}' if others else ''))


def main():
    parser = argparse.ArgumentParser(description='Write OUT_ROOT/FaceDx_index.csv: every video with how usable its results are.')
    parser.add_argument('--patient', nargs='+', help='Only (re)index these patients; the other rows are kept')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='Settings file (default: local_settings.env in the repo folder)')
    args = parser.parse_args()
    update_index(read_settings(args.config), args.patient)


if __name__ == '__main__':
    main()
