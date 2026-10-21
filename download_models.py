"""
Get the model code and weights main_Combined.py needs (run once, from any folder):

    python download_models.py

Everything goes inside the repo folder (gitignored): megraphau/ (OpenGraphAU code + checkpoints)
and model_cache/ (DeepFace's VGG-Face weights, ~580 MB). The HSEmotion models are already in HSE_models/.
"""

import os
import subprocess
import sys

from facedx_settings import REPO_DIR, read_settings

os.chdir(REPO_DIR)

# The same files as setup.py, from their original sources (setup.py's Google Drive copies no longer download):
# the Swin-B backbone from Microsoft's Swin Transformer release, OpenGraphAU stage 1 (Swin-B) from the
# OpenGraphAU authors' Google Drive (megraphau/OpenGraphAU/README.md)
SWIN_URL = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth'
SWIN_PATH = 'megraphau/checkpoints/swin_base_patch4_window7_224.pth'
OGAU_DRIVE_ID = '1nWwowmq4pQn1ACnSOOeyBy6-n0rmqTQ9'
OGAU_PATH = 'megraphau/checkpoints/OpenGprahAU-SwinB_first_stage.pth'

if not os.path.exists('megraphau'):
    print('Cloning ME-GraphAU into megraphau/')
    subprocess.run(['git', 'clone', 'https://github.com/JayRGopal/ME-GraphAU', 'megraphau'], check=True)

import gdown
import requests


def download(path, fetch):
    if os.path.exists(path):
        print(f'Already there: {path}')
        return
    print(f'Downloading {path}')
    try:
        fetch(path + '.part')
    except Exception as e:
        print(e)
    if not os.path.exists(path + '.part') or os.path.getsize(path + '.part') < 1_000_000:
        sys.exit(f'Download failed: {path}. Try again later (Google Drive limits downloads), or copy the file from another lab machine.')
    os.rename(path + '.part', path)


def fetch_url(url):
    def fetch(dest):
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(dest, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    file.write(chunk)
    return fetch


os.makedirs('megraphau/checkpoints', exist_ok=True)
download(SWIN_PATH, fetch_url(SWIN_URL))
download(OGAU_PATH, lambda dest: gdown.download(id=OGAU_DRIVE_ID, output=dest, quiet=False, use_cookies=False))

# DeepFace downloads VGG-Face on first use; do it now so the first video doesn't stall on it
os.environ.setdefault('DEEPFACE_HOME', read_settings()['DEEPFACE_HOME'])
os.makedirs(os.environ['DEEPFACE_HOME'], exist_ok=True)
from deepface import DeepFace
DeepFace.build_model('VGG-Face')
print(f'DeepFace VGG-Face weights in {os.environ["DEEPFACE_HOME"]}/.deepface/weights/')
print('Done.')
