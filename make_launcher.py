"""
Put an icon on the Linux desktop that opens the player on one patient's episodes, for someone who just double-clicks:

    python make_launcher.py --patient S24_217                       (Happiness episodes)
    python make_launcher.py --patient S24_217 --episodes Happiness Sadness

Run it with the Python that runs the player (e.g. the virtualenv's), on the machine with the videos, as the user who
will double-click it. The icon runs play_results.py from this folder with that Python, using local_settings.env here,
in a terminal window that shows the progress (and stays open on an error).
"""

import argparse
import os
import shutil
import stat
import subprocess
import sys

import episodes as ep

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def desktop_dir():
    # The desktop folder (its name depends on the language of the account)
    try:
        path = subprocess.run(['xdg-user-dir', 'DESKTOP'], capture_output=True, text=True).stdout.strip()
        if path and os.path.isdir(path):
            return path
    except OSError:
        pass
    return os.path.expanduser('~/Desktop')


def quote(arg):
    # An argument of the Exec line of a .desktop file, quoted as the Desktop Entry spec requires
    for char in '\\"`$':
        arg = arg.replace(char, '\\' + char)
    return f'"{arg}"'


def main():
    parser = argparse.ArgumentParser(description='Put an icon on the desktop that opens the player on a patient\'s episodes.')
    parser.add_argument('--patient', required=True, help='Patient folder name, e.g. S24_217')
    parser.add_argument('--episodes', nargs='+', choices=ep.EMOTIONS, default=['Happiness'],
                        help='Emotions whose episodes to show (default: Happiness)')
    parser.add_argument('--desktop', default=None, help='Folder to put the icon in (default: the desktop)')
    parser.add_argument('--config', default=None, help='Settings file for the player (default: local_settings.env in the repo folder)')
    args = parser.parse_args()

    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        sys.exit('ffmpeg not found: the player needs it. Run this from the shell where play_results.py works.')
    folder = args.desktop or desktop_dir()
    os.makedirs(folder, exist_ok=True)

    title = f'FaceDx {" + ".join(args.episodes)} {args.patient}'
    options = ['--patient', args.patient, '--episodes', *args.episodes] + (['--config', os.path.abspath(args.config)] if args.config else [])
    # Desktop sessions don't read the shell's settings, so the folder with ffmpeg is put on PATH explicitly
    path = ':'.join(dict.fromkeys([os.path.dirname(ffmpeg), os.path.dirname(sys.executable), '/usr/local/bin', '/usr/bin', '/bin']))
    command = ['env', f'PATH={path}', 'bash', os.path.join(REPO_DIR, 'launch_player.sh'), sys.executable, *options]
    entry = ('[Desktop Entry]\n'
             'Type=Application\n'
             f'Name={title}\n'
             f'Comment=Play {" and ".join(args.episodes)} episodes of {args.patient} with the FaceDx results (n / p next / previous, q quit)\n'
             f'Exec={" ".join(quote(a) for a in command)}\n'
             'Terminal=true\n'
             'Icon=video-x-generic\n')

    out = os.path.join(folder, f'facedx-{args.patient}-{"-".join(args.episodes).lower()}.desktop')
    with open(out, 'w') as file:
        file.write(entry)
    os.chmod(out, os.stat(out).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    # GNOME only launches desktop files marked as trusted (otherwise: right-click > Allow Launching)
    trusted = subprocess.run(['gio', 'set', out, 'metadata::trusted', 'true'], capture_output=True).returncode == 0 \
        if shutil.which('gio') else False
    print(f'Made {out}')
    print(f'It runs: {" ".join(command[3:])}')
    if not trusted:
        print('If double-clicking only opens it as text: right-click the icon > Allow Launching.')


if __name__ == '__main__':
    main()
