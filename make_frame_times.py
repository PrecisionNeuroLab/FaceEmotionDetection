"""
Add outputs_frame_times.csv to finished FaceDx results made before main_Combined.py wrote it (see video_times.py):
the time stored in the video file of each analysed frame, for lining the results up with iEEG.

    python make_frame_times.py --patient S24_217             (its finished videos without the file yet)
    python make_frame_times.py --patient S24_217 S23_199 --redo

Reads each video's timestamps from RAW_ROOT/<patient>/ or MP4_ROOT/<patient>_MP4/ (seconds per video, or minutes if
the whole video has to be decoded); the existing CSVs are not changed. Needs ffmpeg and ffprobe.
"""

import argparse
import os

import episodes as ep
from facedx_settings import DEFAULT_CONFIG, read_settings
from make_index import video_candidates, video_files
from video_times import CSV_NAME, write_frame_times_csv


def main():
    parser = argparse.ArgumentParser(description='Add outputs_frame_times.csv (frame times from the video file) to finished results.')
    parser.add_argument('--patient', nargs='+', required=True, help='Patient folder names, e.g. S24_217')
    parser.add_argument('--redo', action='store_true', help='Also rewrite the file where it already exists')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='Settings file (default: local_settings.env in the repo folder)')
    args = parser.parse_args()
    settings = read_settings(args.config)

    failed = []
    for patient in args.patient:
        patient_dir = os.path.join(settings['OUT_ROOT'], 'outputs_Combined', patient)
        if not os.path.isdir(patient_dir):
            print(f'{patient}: no results in {patient_dir}')
            continue
        videos = [v for v in ep.finished_videos(patient_dir)
                  if args.redo or not os.path.exists(os.path.join(patient_dir, v, CSV_NAME))]
        paths = video_files(settings, patient)
        print(f'{patient}: {len(videos)} videos to do')
        for k, video in enumerate(videos, 1):
            results_dir = os.path.join(patient_dir, video)
            candidates = video_candidates(video, paths)
            if not candidates:
                print(f'  {k}/{len(videos)} {video}: video file not found')
                failed.append(f'{patient}/{video}')
                continue
            errors = []
            for path in candidates:  # e.g. the .mp4 the results were made from, else the raw recording
                try:
                    frames = write_frame_times_csv(results_dir, path)
                    print(f'  {k}/{len(videos)} {video}: done ({frames:,} frames in {os.path.basename(path)})')
                    break
                except RuntimeError as error:
                    errors.append(str(error))
            else:
                print(f'  {k}/{len(videos)} {video}: not written: ' + '; '.join(errors))
                failed.append(f'{patient}/{video}')
    if failed:
        print(f'{len(failed)} videos without frame times: ' + ', '.join(failed))


if __name__ == '__main__':
    main()
