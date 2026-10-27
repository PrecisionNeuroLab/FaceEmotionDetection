"""
Convert one patient's raw .m2t recordings to .mp4 for main_Combined.py:

    python convert_to_mp4.py /path/to/raw/S24_217 --patient S24_217

Finds every .m2t under the raw folder (subfolders too) and writes <name>.mp4 into MP4_ROOT/<patient>_MP4/
(or --out-dir). Nothing is written to or deleted from the raw folder. Videos already converted are skipped;
each file is written as <name>.mp4.part and renamed when ffmpeg finishes, so an interrupted conversion is redone.
Same encoding as main_M2T_to_MP4.py: H.264, preset veryfast, CRF 18, audio copied.
"""

import argparse
import os
import subprocess
import sys

from facedx_settings import DEFAULT_CONFIG, read_settings

parser = argparse.ArgumentParser(description='Convert a patient\'s .m2t recordings to .mp4 (raw folder is only read).')
parser.add_argument('raw_dir', help='Raw folder of one patient (searched recursively for .m2t files)')
parser.add_argument('--patient', help='Patient folder name, e.g. S24_217 (output: MP4_ROOT/S24_217_MP4/)')
parser.add_argument('--out-dir', help='Output folder (instead of MP4_ROOT/<patient>_MP4/)')
parser.add_argument('--config', default=DEFAULT_CONFIG, help='Settings file (default: local_settings.env in the repo folder)')
parser.add_argument('--dry-run', action='store_true', help='List what would be converted, then stop')
args = parser.parse_args()

if not args.out_dir and not args.patient:
    sys.exit('Give --patient (output goes to MP4_ROOT/<patient>_MP4/) or --out-dir.')
raw_dir = os.path.abspath(args.raw_dir)
out_dir = os.path.abspath(args.out_dir or os.path.join(read_settings(args.config)['MP4_ROOT'], f'{args.patient}_MP4'))
if not os.path.isdir(raw_dir):
    sys.exit(f'Raw folder not found: {raw_dir}')
if out_dir == raw_dir or out_dir.startswith(raw_dir + os.sep):
    sys.exit(f'The output folder must not be inside the raw folder: {out_dir}')

m2t_files = sorted(os.path.join(path, f) for path, _, files in os.walk(raw_dir) for f in files
                   if f.lower().endswith('.m2t') and not f.startswith('.'))

# All .mp4 files go into one flat folder, so two recordings with the same name would overwrite each other
names = {}
for m2t in m2t_files:
    names.setdefault(os.path.splitext(os.path.basename(m2t))[0] + '.mp4', []).append(m2t)
clashes = {name: paths for name, paths in names.items() if len(paths) > 1}
if clashes:
    for name, paths in clashes.items():
        print(f'{name} would come from:\n  ' + '\n  '.join(paths))
    sys.exit('Some recordings share a file name; convert those folders separately with different --out-dir.')

todo = [(m2t, os.path.join(out_dir, name)) for name, (m2t,) in sorted(names.items(), key=lambda item: item[1][0])
        if not os.path.exists(os.path.join(out_dir, name))]
print(f'Raw folder: {raw_dir}\nOutput:     {out_dir}')
print(f'{len(m2t_files)} .m2t files, {len(m2t_files) - len(todo)} already converted, {len(todo)} to convert')
if args.dry_run:
    for m2t, mp4 in todo:
        print(f'  {m2t} -> {os.path.basename(mp4)}')
    sys.exit(0)

os.makedirs(out_dir, exist_ok=True)
failed = []
for n, (m2t, mp4) in enumerate(todo, 1):
    print(f'[{n}/{len(todo)}] {m2t}', flush=True)
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', m2t,
           '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-c:a', 'copy', '-f', 'mp4', mp4 + '.part']
    if subprocess.run(cmd).returncode == 0:
        os.rename(mp4 + '.part', mp4)
    else:
        failed.append(m2t)
        if os.path.exists(mp4 + '.part'):
            os.remove(mp4 + '.part')
        print(f'  ffmpeg failed on {m2t}')

print(f'Done: {len(todo) - len(failed)} converted, {len(failed)} failed')
for m2t in failed:
    print(f'  failed: {m2t}')
