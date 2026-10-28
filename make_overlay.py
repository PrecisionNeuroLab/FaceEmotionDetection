"""
Make a subtitle overlay (.ass) of the FaceDx results of one video, to watch alongside the original recording:

    python make_overlay.py --patient S24_217 --video <file>.m2t

Open the video in VLC and add the .ass file (Subtitle > Add Subtitle File...). It shows a box around the patient's
face and, about 5 times a second, the top 4 emotions and top 3 action units of that frame. No video is written:
the .ass file is up to about 20 MB of text per hour of video, saved next to the video's CSVs.
"""

import argparse
import os
import sys

import cv2
import pandas as pd

from facedx_settings import DEFAULT_CONFIG, read_settings

TOP_EMOTIONS = 4
TOP_AUS = 3
STATUS = {1: 'Patient found', 0: 'No face', -1: 'Patient not identified', -2: 'Patient not identified'}
# FACS names of the OpenGraphAU action units (the left/right AUL*/AUR* columns are left out of the overlay)
AU_NAMES = {
    'AU1': 'inner brow raiser', 'AU2': 'outer brow raiser', 'AU4': 'brow lowerer', 'AU5': 'upper lid raiser',
    'AU6': 'cheek raiser', 'AU7': 'lid tightener', 'AU9': 'nose wrinkler', 'AU10': 'upper lip raiser',
    'AU11': 'nasolabial deepener', 'AU12': 'lip corner puller', 'AU13': 'sharp lip puller', 'AU14': 'dimpler',
    'AU15': 'lip corner depressor', 'AU16': 'lower lip depressor', 'AU17': 'chin raiser', 'AU18': 'lip pucker',
    'AU19': 'tongue show', 'AU20': 'lip stretcher', 'AU22': 'lip funneler', 'AU23': 'lip tightener',
    'AU24': 'lip pressor', 'AU25': 'lips part', 'AU26': 'jaw drop', 'AU27': 'mouth stretch', 'AU32': 'lip bite',
    'AU38': 'nostril dilator', 'AU39': 'nostril compressor',
}

# ASS colours are &HBBGGRR (alpha separately, 00 = opaque)
WHITE, MUTED = '&HE8EFF1&', '&HA9B2B4&'
TOP_BAR, BAR, BAR_EMPTY = '&H759E1D&', '&HA9B2B4&', '&H5A5E5F&'
FACE_BOX = '&HA5CA5D&'


def find_video(name, patient, settings):
    # A path, or a file name looked up in RAW_ROOT/<patient>/ (all subfolders) and MP4_ROOT/<patient>_MP4/
    if os.path.isfile(name):
        return os.path.abspath(name)
    folders = [os.path.join(settings['MP4_ROOT'], f'{patient}_MP4')]
    if settings['RAW_ROOT']:
        folders.insert(0, os.path.join(settings['RAW_ROOT'], patient))
    for folder in folders:
        for path, _, files in os.walk(folder):
            if name in files:
                return os.path.join(path, name)
    sys.exit(f'Video {name} not found in: ' + ', '.join(folders) + '\nGive the full path of the video instead.')


def ass_time(seconds):
    cs = round(seconds * 100)
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f'{h}:{m:02}:{s:02}.{cs:02}'


def bar(value, width, height, colour):
    # Inline drawing: filled part + empty part, always the same total width so the numbers after it line up
    filled = round(min(max(value, 0), 1) * width)
    parts = [(filled, colour), (width - filled, BAR_EMPTY)]
    return ''.join(f'{{\\1c{c}\\p1}}m 0 0 l {w} 0 {w} {height} 0 {height}{{\\p0}}' for w, c in parts if w > 0) + f'{{\\1c{WHITE}}}'


def panel_text(row, emotions, aus, font_size):
    status = STATUS.get(int(row['success']), 'Unknown')
    if row['success'] != 1:
        return status  # scores are 0 here, so there is nothing else to show
    width, height = font_size * 4, round(font_size * 0.6)
    lines = [f'{{\\b1}}{status}{{\\b0}}  {{\\1c{MUTED}}}frame {int(row["frame"]):,}{{\\1c{WHITE}}}']
    lines.append(f'{{\\1c{MUTED}}}Emotions{{\\1c{WHITE}}}')
    top = row[emotions].astype(float).nlargest(TOP_EMOTIONS)
    for n, (name, value) in enumerate(top.items()):
        lines.append(f'{bar(value, width, height, TOP_BAR if n == 0 else BAR)} {value:.2f}  {name}')
    lines.append(f'{{\\1c{MUTED}}}Action units{{\\1c{WHITE}}}')
    for name, value in row[aus].astype(float).nlargest(TOP_AUS).items():
        lines.append(f'{bar(value, width, height, BAR)} {value:.2f}  {name} {AU_NAMES[name]}')
    return '\\N'.join(lines)


def face_box_text(row):
    if row['success'] != 1 or row['Facial Box W'] <= 0:
        return None
    x, y, w, h = (int(round(row[c])) for c in ['Facial Box X', 'Facial Box Y', 'Facial Box W', 'Facial Box H'])
    return f'{{\\pos(0,0)\\p1}}m {x} {y} l {x + w} {y} {x + w} {y + h} {x} {y + h}{{\\p0}}'


def merged_events(times, texts):
    # [(start, end, text)], with consecutive rows that show the same text joined into one event
    events = []
    for start, end, text in zip(times[:-1], times[1:], texts):
        if text is None:
            continue
        if events and events[-1][2] == text and events[-1][1] == start:
            events[-1][1] = end
        else:
            events.append([start, end, text])
    return events


parser = argparse.ArgumentParser(description='Make a subtitle overlay (.ass) of the FaceDx results of one video, for VLC.')
parser.add_argument('--patient', required=True, help='Patient folder name, e.g. S24_217')
parser.add_argument('--video', required=True, help='Video file name (e.g. <file>.m2t, looked up in RAW_ROOT and MP4_ROOT) or its full path')
parser.add_argument('--config', default=DEFAULT_CONFIG, help='Settings file (default: local_settings.env in the repo folder)')
args = parser.parse_args()
settings = read_settings(args.config)

video_path = find_video(args.video, args.patient, settings)
name = os.path.basename(video_path)
results_dir = os.path.join(settings['OUT_ROOT'], 'outputs_Combined', args.patient, name)
if not os.path.isdir(results_dir):
    sys.exit(f'No finished results for {name} in {results_dir}' +
             (' (still being processed)' if os.path.isdir(results_dir + '.partial') else ''))

capture = cv2.VideoCapture(video_path)
fps = capture.get(cv2.CAP_PROP_FPS)
width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
capture.release()
if not fps or not width:
    sys.exit(f'Could not read the frame rate and size of {video_path}')

hse = pd.read_csv(os.path.join(results_dir, 'outputs_hse.csv'))
ogau = pd.read_csv(os.path.join(results_dir, 'outputs_ogau.csv'))
bboxes = pd.read_csv(os.path.join(results_dir, 'outputs_bboxes.csv'))
emotions = list(hse.columns[3:])
aus = [c for c in ogau.columns if c in AU_NAMES]
rows = hse.merge(ogau.drop(columns=['timestamp', 'success']), on='frame').merge(
    bboxes.drop(columns=['timestamp', 'success']), on='frame').sort_values('frame')

# Times from frame numbers and the real frame rate (the CSV timestamps use the rounded-up rate, e.g. 30 for 29.97)
step = rows['frame'].diff().median() if len(rows) > 1 else 1
times = list(rows['frame'] / fps) + [(rows['frame'].iloc[-1] + step) / fps]
font_size = round(height / 32)
records = [row for _, row in rows.iterrows()]
panel = merged_events(times, [panel_text(r, emotions, aus, font_size) for r in records])
boxes = merged_events(times, [face_box_text(r) for r in records])

margin = round(font_size * 0.8)
out_path = os.path.join(results_dir, os.path.splitext(name)[0] + '.ass')
with open(out_path + '.part', 'w') as file:
    file.write('[Script Info]\nScriptType: v4.00+\n'
               f'PlayResX: {width}\nPlayResY: {height}\nScaledBorderAndShadow: yes\nWrapStyle: 2\n\n'
               '[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, '
               'Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, '
               'Alignment, MarginL, MarginR, MarginV, Encoding\n'
               # BorderStyle 3: semi-transparent box behind the text, in OutlineColour
               f'Style: Panel,Sans,{font_size},&H00E8EFF1,&H00E8EFF1,&H60000000,&H60000000,0,0,0,0,100,100,0,0,3,{round(font_size * 0.3)},0,7,{margin},{margin},{margin},1\n'
               f'Style: Box,Sans,{font_size},&HFF000000,&HFF000000,&H00{FACE_BOX[2:-1]},&HFF000000,0,0,0,0,100,100,0,0,1,{max(2, round(height / 360))},0,7,0,0,0,1\n\n'
               '[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n')
    for start, end, text in boxes:
        file.write(f'Dialogue: 0,{ass_time(start)},{ass_time(end)},Box,,0,0,0,,{text}\n')
    for start, end, text in panel:
        file.write(f'Dialogue: 1,{ass_time(start)},{ass_time(end)},Panel,,0,0,0,,{text}\n')
os.replace(out_path + '.part', out_path)

found = (rows['success'] == 1).mean()
print(f'Video:   {video_path} ({width}x{height}, {fps:.2f} fps)')
print(f'Results: {len(rows)} frames, patient found in {found:.0%}')
print(f'Overlay: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)')
print('In VLC: open the video, then Subtitle > Add Subtitle File... and pick the overlay.')
