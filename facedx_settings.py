import os

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(REPO_DIR, 'local_settings.env')

# Defaults here, overridden by the settings file (KEY=VALUE lines), then by command-line options
DEFAULT_SETTINGS = {
    'MP4_ROOT': '/home/klab/NAS/Analysis/MP4', # videos in MP4_ROOT/<patient>_MP4/, patient photos in MP4_ROOT/<patient>_MP4/targetFace/
    'OUT_ROOT': '/home/klab/NAS/Analysis', # outputs in OUT_ROOT/outputs_Combined/<patient>/ and OUT_ROOT/outputs_Combined_PatData/<patient>/
    'DEEPFACE_HOME': os.path.join(REPO_DIR, 'model_cache'), # DeepFace keeps its VGG-Face weights in DEEPFACE_HOME/.deepface/
}


def read_settings(path=DEFAULT_CONFIG):
    settings = dict(DEFAULT_SETTINGS)
    if not os.path.exists(path):
        return settings
    with open(path) as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                print(f'WARNING: ignoring line {line_num} of {path} (expected KEY=VALUE): {line}')
                continue
            key, value = [part.strip() for part in line.split('=', 1)]
            if key not in DEFAULT_SETTINGS:
                print(f'WARNING: ignoring unknown setting {key} in {path}')
                continue
            settings[key] = os.path.expanduser(value.strip('"\''))
    return settings
