<a id="readme-top"></a>



[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Lab Website][lab-website-shield]][lab-website-url]
[![Twitter][twitter-shield]][twitter-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/JayRGopal/FaceEmotionDetection">
    <img src="readme_assets/FaceDx_Logo.png" alt="Logo" width="385" height="270">
  </a>

  <h1 align="center">FaceDx</h1>

  <p align="center">
    <h3>A Markerless Computer Vision Approach For Continuous Quantification of Internal States and Affective Behaviors in Clinical Settings</h3>
    <a href="https://github.com/JayRGopal/FaceEmotionDetection/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/JayRGopal/FaceEmotionDetection/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#project-overview">Project Overview</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contacts">Contacts</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- PROJECT OVERVIEW -->
## Project Overview

![Product Demo](readme_assets/product_demo.png)

FaceDx is a fully integrated computer vision workflow to analyze internal states, affect, pain, and short-term behaviors in the clinical setting. Importantly, our approach is markerless and does not require any fine-tuning, meaning FaceDx can automatically process any videos in which the given subject's face is visible. We bring together pre-trained, open source models that output facial action units (AUs) and emotions on a frame-by-frame basis. 

FaceDx has been shown to accurately decode self-reported long-term mood scores, as well as short-term behaviors such as smiles, frowns, or neutral expressions. Importantly, we are one of the first to conduct this validation in a clinical setting without the need for significant human intervention and analysis.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

We use a combination of the most widely employed deep learning and computer vision libraries for our custom clinical monitoring pipeline.
* **PyTorch**: MTCNN, OpenGraphAU
* **TensorFlow**: HSEmotion, DeepFace & Partial Verify,
* **OpenCV**: Video Processing, Intermediate Image Saving, Visualizer

<br />
<div align="center">
  <a href="https://github.com/JayRGopal/FaceEmotionDetection">
    <img src="readme_assets/pytorch.png" alt="Logo" width="451" height="112">
  </a>
  <br />
  <a href="https://github.com/JayRGopal/FaceEmotionDetection">
    <img src="readme_assets/tensorflow.png" alt="Logo" width="422" height="270">
  </a>

  <a href="https://github.com/JayRGopal/FaceEmotionDetection">
    <img src="readme_assets/opencv.png" alt="Logo" width="262" height="347">
  </a>

</div>



<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This covers the face pipeline, `main_Combined.py`: emotions (HSEmotion) and action units (OpenGraphAU) for every
video of one patient, with face verification so only the patient's face is scored.

### Prerequisites

* Linux with an NVIDIA GPU (a CPU works for testing: a few minutes per minute of video)
* Python 3.11 and `ffmpeg` (`sudo apt install ffmpeg`)
* About 3 GB of disk for the model files

### Installation

From the folder you want the repo in:

```bash
git clone https://github.com/PrecisionNeuroLab/FaceEmotionDetection.git
cd FaceEmotionDetection
python3.11 -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir -r requirements_face.txt
python download_models.py
```

`requirements_face.txt` has the versions of the lab's environment (`environments/cvmonitor.yaml`: PyTorch 2.0.1,
TensorFlow 2.13). `conda env create -f environments/cvmonitor.yaml` gives the full lab environment instead
(then `pip install "gdown>=5.2"` for the download script).

`download_models.py` clones ME-GraphAU into `megraphau/` and downloads the OpenGraphAU and Swin-B weights into
`megraphau/checkpoints/` and DeepFace's VGG-Face weights into `model_cache/`, all inside the repo folder (gitignored).

Check that PyTorch and TensorFlow see the GPU:

```bash
python -c "import torch, tensorflow as tf; print(torch.cuda.is_available(), tf.config.list_physical_devices('GPU'))"
```

If TensorFlow finds no GPU, run with `--force-hse-cpu` (only HSEmotion, the small emotion model, then runs on the CPU).

**Settings (once):** copy `local_settings.env.example` to `local_settings.env` (gitignored) and keep only the lines
you change. You don't need to edit it after updates.

```
MP4_ROOT=/home/klab/NAS/Analysis/MP4
OUT_ROOT=/home/klab/NAS/Analysis
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Expected layout: one folder of videos per patient, with photos of the patient's face in `targetFace/`:

```
<MP4_ROOT>/<patient>_MP4/*.mp4
<MP4_ROOT>/<patient>_MP4/targetFace/*.jpg     # a few clear photos of the patient's face
```

**1. Convert the raw recordings** (`.m2t`, searched in all subfolders) to `.mp4`. Nothing in the raw folder is
changed; finished files are skipped on a rerun.

```bash
python convert_to_mp4.py /path/to/raw/S24_217 --patient S24_217 --dry-run
python convert_to_mp4.py /path/to/raw/S24_217 --patient S24_217
```

To convert just one video (e.g. for a test), with the same settings:

```bash
ffmpeg -i /path/to/raw/S24_217/HospitalRoom/NKT/EEG2100/<session>/<file>.m2t -c:v libx264 -preset veryfast -crf 18 -c:a copy <MP4_ROOT>/S24_217_MP4/<file>.mp4
```

**2. Add the patient photos** to `<MP4_ROOT>/S24_217_MP4/targetFace/`.

**3. Check the folders and models, then test on one video:**

```bash
python main_Combined.py --patient S24_217 --dry-run
python main_Combined.py --patient S24_217 --limit 1
```

**4. Run all videos in the background** (keeps going if the connection drops):

```bash
nohup python main_Combined.py --patient S24_217 > facedx_S24_217.log 2>&1 &
tail -f facedx_S24_217.log
```

If it stops, run the same command again: finished videos are skipped, and a video that was interrupted is redone
from the start (results are written to `<video>.partial/` and renamed when the video is finished).
Other options: `--device cuda:1`, `--video-dir`, `--face-dir`, `--config` (`python main_Combined.py --help`).

**Outputs**, one folder per video in `<OUT_ROOT>/outputs_Combined/<patient>/<video>.mp4/`, one row per analysed
frame (about 5 per second):

| File | Contents |
|---|---|
| `outputs_hse.csv` | probabilities of Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise |
| `outputs_ogau.csv` | probabilities of 41 facial action units (AU1 ... AU39, plus left/right AUs) |
| `outputs_bboxes.csv` | position and size of the patient's face |

Each row has `frame`, `timestamp` (seconds from the start of the video) and `success`: `1` patient's face found,
`0` no face, `-1` several faces and none verified as the patient. Scores are 0 when `success` is not 1.
About 1% of the face crops matched to the patient are saved to `<OUT_ROOT>/outputs_Combined_PatData/<patient>/` for
checking the verification; they are identifiable images, so keep them on the lab machine.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

TODO

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACTS -->
## Contacts

Yuhao "Danny" Huang, MD - [@YuhaoHuangMD](https://twitter.com/YuhaoHuangMD) - Yhhuang@Stanford.edu

Jay Gopal - [@JayRGopal](https://twitter.com/JayRGopal) - Jay_Gopal@Brown.edu & JGopal@Stanford.edu

Corey Keller, MD, PhD - [@DrCoreyKeller](https://twitter.com/DrCoreyKeller) - CKeller1@Stanford.edu


Project Link: [https://github.com/JayRGopal/FaceEmotionDetection](https://github.com/JayRGopal/FaceEmotionDetection)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments


* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Font Awesome](https://fontawesome.com)
* [README Template](https://github.com/othneildrew/Best-README-Template)
* [OpenGraphAU](https://github.com/lingjivoo/OpenGraphAU)
* [HSEmotion](https://github.com/av-savchenko/hsemotion)
* [DeepFace](https://github.com/serengil/deepface)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/JayRGopal/FaceEmotionDetection
[contributors-url]: https://github.com/JayRGopal/FaceEmotionDetection/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/JayRGopal/FaceEmotionDetection
[forks-url]: https://github.com/JayRGopal/FaceEmotionDetection/network/members
[stars-shield]: https://img.shields.io/github/stars/JayRGopal/FaceEmotionDetection
[stars-url]: https://github.com/JayRGopal/FaceEmotionDetection/stargazers
[issues-shield]: https://img.shields.io/github/issues/JayRGopal/FaceEmotionDetection
[issues-url]: https://github.com/JayRGopal/FaceEmotionDetection/issues
[lab-website-shield]: https://img.shields.io/website?url=https%3A%2F%2Fprecisionneuro.stanford.edu%2F
[lab-website-url]: https://precisionneuro.stanford.edu/
[linkedin-shield]: https://img.shields.io/badge/LinkedIn-JayRGopal-blue
[linkedin-url]: https://linkedin.com/in/jay-gopal/
[twitter-shield]: https://img.shields.io/twitter/follow/JayRGopal
[twitter-url]: https://twitter.com/JayRGopal
