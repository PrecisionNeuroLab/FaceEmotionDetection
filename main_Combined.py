import os
import sys
import shutil
import time
import datetime
import argparse
from facedx_settings import REPO_DIR, DEFAULT_CONFIG, read_settings

# Model files (HSE_models/, megraphau/checkpoints/) are found relative to the repo folder
os.chdir(REPO_DIR)


def create_parser():
    parser = argparse.ArgumentParser(description='FaceDx: emotions (HSEmotion) and action units (OpenGraphAU) for every video of one patient.')
    parser.add_argument('--patient', type=str, required=True,
                        help='Patient folder name, e.g. S23_174 (videos in MP4_ROOT/S23_174_MP4/, or RAW_ROOT/S23_174/ with --m2t)')
    parser.add_argument('--m2t', action='store_true',
                        help='Read the raw .m2t recordings in RAW_ROOT/<patient>/ (all subfolders) instead of converted .mp4 files')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use, e.g. cuda:0, cuda:1, cpu (default: cuda:0 if a GPU is found, else cpu)')
    parser.add_argument('--force-hse-cpu', action='store_true',
                        help='Run HSEmotion (TensorFlow) on the CPU')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG,
                        help='Settings file (default: local_settings.env in the repo folder)')
    parser.add_argument('--video-dir', type=str, default=None,
                        help='Folder with the .mp4 videos (default: MP4_ROOT/<patient>_MP4), or with --m2t the raw folder (default: RAW_ROOT/<patient>)')
    parser.add_argument('--face-dir', type=str, default=None,
                        help='Folder with .jpg photos of the patient (default: MP4_ROOT/<patient>_MP4/targetFace, or <video-dir>/targetFace)')
    parser.add_argument('--save-face-samples', action='store_true',
                        help='Save about 1%% of the face crops matched to the patient as .jpg, to check the verification (default: none saved)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Process at most this many videos (for a test run)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the folders and the videos left to process, then stop (loads no models)')
    return parser


def list_videos(video_dir, m2t=False):
    # {file name: path}. .mp4 files directly in video_dir, or .m2t files anywhere under it (raw recordings are in session subfolders)
    if not m2t:
        return {vid: os.path.join(video_dir, vid) for vid in os.listdir(video_dir) if not vid.startswith('.')
                and vid.endswith('.mp4') and os.path.isfile(os.path.join(video_dir, vid))}
    paths = {}
    for path, _, files in os.walk(video_dir):
        for vid in files:
            if vid.lower().endswith('.m2t') and not vid.startswith('.'):
                paths.setdefault(vid, []).append(os.path.join(path, vid))
    # Outputs go into one folder per file name, so two recordings with the same name would mix
    clashes = {vid: found for vid, found in paths.items() if len(found) > 1}
    if clashes:
        for vid, found in clashes.items():
            print(f'{vid} found in:\n  ' + '\n  '.join(found))
        sys.exit('Some recordings share a file name; run those session folders separately with --video-dir.')
    return {vid: found[0] for vid, found in paths.items()}


def list_unprocessed(videos, output_dir):
    # A video is done once its output folder exists: the folder is renamed into place only after the whole video is processed
    return [vid for vid in sorted(videos, key=videos.get) if not os.path.exists(os.path.join(output_dir, vid))]


parser = create_parser()
args = parser.parse_args()
settings = read_settings(args.config)

# Main parameters
PAT_NOW = args.patient
MP4_DIRECTORY = os.path.join(settings['MP4_ROOT'], f'{PAT_NOW}_MP4')
if args.m2t and not args.video_dir and not settings['RAW_ROOT']:
    sys.exit(f'Set RAW_ROOT (folder with one raw folder per patient) in {args.config}, or pass --video-dir.')
VIDEO_DIRECTORY = os.path.abspath(args.video_dir or (os.path.join(settings['RAW_ROOT'], PAT_NOW) if args.m2t else MP4_DIRECTORY))
# Photos are never looked for inside the raw folder
SUBJECT_FACE_IMAGE_FOLDER = os.path.abspath(args.face_dir or os.path.join(
    args.video_dir if args.video_dir and not args.m2t else MP4_DIRECTORY, 'targetFace'))
OUTPUT_DIRECTORY = os.path.abspath(os.path.join(settings['OUT_ROOT'], 'outputs_Combined', PAT_NOW))
OUTPUT_DIRECTORY_PARTIAL_VERIFY = os.path.abspath(os.path.join(settings['OUT_ROOT'], 'outputs_Combined_PatData', PAT_NOW))
os.environ.setdefault('DEEPFACE_HOME', settings['DEEPFACE_HOME'])

print(f'Settings file:  {args.config}' + ('' if os.path.exists(args.config) else ' (not found, using defaults)'))
print(f'Videos:         {VIDEO_DIRECTORY}')
print(f'Patient photos: {SUBJECT_FACE_IMAGE_FOLDER}')
print(f'Outputs:        {OUTPUT_DIRECTORY}')
print(f'Face samples:   ' + (OUTPUT_DIRECTORY_PARTIAL_VERIFY if args.save_face_samples else 'not saved (--save-face-samples to save them)'))
if not os.path.isdir(VIDEO_DIRECTORY):
    sys.exit(f'Video folder not found: {VIDEO_DIRECTORY}\nSet {"RAW_ROOT" if args.m2t else "MP4_ROOT"} in {args.config}, or pass --video-dir.')
patient_photos = [f for f in os.listdir(SUBJECT_FACE_IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg'))] \
    if os.path.isdir(SUBJECT_FACE_IMAGE_FOLDER) else []
video_paths = list_videos(VIDEO_DIRECTORY, m2t=args.m2t)
unprocessed_videos = list_unprocessed(video_paths, OUTPUT_DIRECTORY)[:args.limit]

if args.dry_run:
    print(f'Patient photos found: {len(patient_photos)}' + ('' if patient_photos else '  <-- add .jpg photos of the patient before running'))
    for path in ['HSE_models/affectnet_emotions/mobilenet_7.h5', 'megraphau/checkpoints/OpenGprahAU-SwinB_first_stage.pth',
                 'megraphau/checkpoints/swin_base_patch4_window7_224.pth']:
        print(f'Model file {path}: ' + ('ok' if os.path.exists(path) else 'MISSING (run: python download_models.py)'))
    print(f'Videos to process: {len(unprocessed_videos)}' + (f' (--limit {args.limit})' if args.limit else ''))
    for vid in unprocessed_videos:
        print(f'  {os.path.relpath(video_paths[vid], VIDEO_DIRECTORY)}')
    sys.exit(0)

if not patient_photos:
    sys.exit(f'No .jpg photos of the patient in {SUBJECT_FACE_IMAGE_FOLDER}: they are needed to tell the patient apart from other people.')

from utils import *
from utilsHSE import *
from utilsVerify import *
import cv2
import torch
from facenet_pytorch import MTCNN
import gc

device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
FORCE_HSE_CPU = args.force_hse_cpu
print(f'Device: {device}' + (' (HSEmotion on CPU)' if FORCE_HSE_CPU else ''))

if 'cuda' in device:
  use_cuda = True
else:
  use_cuda = False

if use_cuda:
  torch.cuda.empty_cache()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


"""

Full Pipeline - HSE and OpenGraphAU
Detection via MTCNN
Verification using DeepFace (Model: VGG-Face)

Success column:
1: Successful!
0: Failure: Zero faces detected
-1: Failure: 2+ faces detected, and verification failed (across all images in loop!)
-2: Failure: 1 face detected, and verification failed (across all images in loop). Only possible with Verify_Every_Frame = True
"""

# Choose which pipelines to run
Run_HSE = True
Run_OpenGraphAU = True
Do_Verification = True 
Partial_Verify = True # Verify, then find nearest face within distance max (below)
Verify_Every_Frame = False # Verify all frames, even if only 1 person is detected
VERIFY_THRESHOLD = 0.32 # Maximum distance threshold (below this, faces are deemed "verified")
Face_Detector = 'MTCNN' # Options: ['MTCNN', 'RetinaFace']

# Additional parameters
BATCH_SIZE = 2000
HSE_MODEL_TYPE = 'mobilenet_7.h5'
OPENGRAPHAU_MODEL_TYPE = 'OpenGraphAU'
OPENGRAPHAU_MODEL_BACKBONE = 'swin_transformer_base'
OPENGRAPHAU_MODEL_PATH = os.path.abspath('megraphau/checkpoints/OpenGprahAU-SwinB_first_stage.pth')
INPUT_SIZE = (224, 224)
FPS_EXTRACTING = 5 # we'll extract this many fps from the video for analysis
DISTANCE_MAX_PARTIAL_VERIFY = 30
SAVE_PROB_PARTIAL_VERIFY = 0.01 if args.save_face_samples else 0 # share of patient face crops saved as .jpg

# Make output dirs if they don't exist
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
if args.save_face_samples and not os.path.exists(OUTPUT_DIRECTORY_PARTIAL_VERIFY):
    os.makedirs(OUTPUT_DIRECTORY_PARTIAL_VERIFY)

# Function that gets us the output folder for each input video
SAVE_PATH_FOLDER = lambda video_name: os.path.join(OUTPUT_DIRECTORY, f'{video_name}')
SAVE_PATH_FOLDER_PARTIAL_VERIFY = lambda video_name: os.path.join(OUTPUT_DIRECTORY_PARTIAL_VERIFY, f'{video_name}')

# For timing estimation
num_vids = len(unprocessed_videos)
start_time = time.time()

TIMING_VERBOSE = True # yes/no do we print times for sub-processes within videos?

# Load models
mtcnn = MTCNN(keep_all=True, post_process=False, min_face_size=40, device=device)

if Run_HSE:
  model_hse = get_emotion_predictor(HSE_MODEL_TYPE, device=device, FORCE_HSE_CPU=FORCE_HSE_CPU)

if Run_OpenGraphAU:
  model_ogau = load_network(model_type=OPENGRAPHAU_MODEL_TYPE, backbone=OPENGRAPHAU_MODEL_BACKBONE, path=OPENGRAPHAU_MODEL_PATH, device=device)

if Do_Verification:
  assert has_jpg_or_jpeg_files(SUBJECT_FACE_IMAGE_FOLDER), "No jpg or jpeg files found in SUBJECT_FACE_IMAGE_FOLDER. Can't do facial verification!"


# Loop through all videos
for i in unprocessed_videos:
  video_path = video_paths[i]
  print(f'===== Video {unprocessed_videos.index(i) + 1}/{num_vids}: {i}')

  frame_now = 0 # this is what we save in outputs file and print

  running = True
  
  fps = get_fps(path=video_path, extracting_fps=FPS_EXTRACTING) # FPS at which we're extracting

  # If video is corrupted, skip it
  if fps == 0:
    running = False
    print('-'*15)
    print(f'MAJOR WARNING! SKIPPING CORRUPTED VIDEO: {i}')
    print('-'*15)

  # Save paths/folders
  # Results are written to <video>.partial/ and renamed to <video>/ when the video is finished,
  # so a video interrupted by a crash is redone from the start on the next run
  final_folder_now = SAVE_PATH_FOLDER(i)
  save_folder_now = final_folder_now + '.partial'
  save_folder_partial_verify_now = SAVE_PATH_FOLDER_PARTIAL_VERIFY(i)
  if os.path.exists(save_folder_now):
    print(f'Removing unfinished results from an earlier run: {save_folder_now}')
    shutil.rmtree(save_folder_now)
  os.mkdir(save_folder_now)
  if Do_Verification and Partial_Verify and args.save_face_samples:
    os.makedirs(save_folder_partial_verify_now, exist_ok=True)
  save_path_hse = os.path.join(save_folder_now, f'outputs_hse.csv')
  save_path_ogau = os.path.join(save_folder_now, f'outputs_ogau.csv') 
  save_path_bboxes = os.path.join(save_folder_now, f'outputs_bboxes.csv') 

  if TIMING_VERBOSE: 
    time1 = time.time()

  # Garbage collector
  gc.collect()

  # Extract video frames
  capture = cv2.VideoCapture(video_path)
  ims = []
  real_frame_numbers = []
  real_fps = math.ceil(capture.get(cv2.CAP_PROP_FPS)) # real FPS of the video
  frame_division = real_fps // FPS_EXTRACTING # Helps us only analyze 5 fps (or close to it)
  frameNr = 0 # Track frame number
  while running:
      # Extract frames continuously
      success, frame = capture.read()
      if success:
          if frameNr % frame_division == 0:
              # We are only saving SOME frames (e.g. extracting 5 fps)
              frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
              ims.append(frame)
              real_frame_numbers.append(frameNr)
          if (frameNr % BATCH_SIZE == 0) and (frameNr > 0) and len(real_frame_numbers) > 0:
              # Let's do analysis, save results, and reset ims!
              ims = np.array(ims)
              print(f"Extracted Ims, Frames {frame_now} to {frameNr} in {i}") 
              if TIMING_VERBOSE:
                time2 = time.time()
                print('Time: ', time2 - time1)

              # Face detection
              if Do_Verification:
                if Face_Detector == 'MTCNN':
                  faces, is_null, all_bboxes = extract_faces_with_verify(ims, INPUT_SIZE, SUBJECT_FACE_IMAGE_FOLDER, partialVerify=Partial_Verify, \
                                                                         mtcnn=mtcnn, verifyAll=Verify_Every_Frame, verify_threshold=VERIFY_THRESHOLD, \
                                                             distance_max=DISTANCE_MAX_PARTIAL_VERIFY, save_folder_path=save_folder_partial_verify_now, \
                                                              real_frame_numbers=real_frame_numbers, saveProb=SAVE_PROB_PARTIAL_VERIFY)
                elif Face_Detector == 'RetinaFace':
                  faces, is_null = detect_extract_faces(ims, INPUT_SIZE)
              else:
                if Face_Detector == 'MTCNN':
                  faces, is_null, all_bboxes = extract_faces_mtcnn(ims, INPUT_SIZE, mtcnn=mtcnn, real_frame_numbers=real_frame_numbers)
                elif Face_Detector == 'RetinaFace':
                  faces, is_null = detect_extract_faces(ims, INPUT_SIZE)
              print(f"Detected Faces")
              if TIMING_VERBOSE:
                time3 = time.time()
                print('Time: ', time3 - time2) 

              # Get predictions of relevant network
              if Run_HSE:
                faces_for_hse = convert_to_gpu_tensor(faces, device=device, FORCE_HSE_CPU=FORCE_HSE_CPU)
                hse_scores_real = hse_preds(faces_for_hse, model_hse, model_type=HSE_MODEL_TYPE, device=device, FORCE_HSE_CPU=FORCE_HSE_CPU)
                hse_scores_real[is_null > 0] = 0 # clear the predictions from frames w/o faces!
                del faces_for_hse
                print("Got Network Predictions: HSE")
              
              if use_cuda:
                torch.cuda.empty_cache()

              if Run_OpenGraphAU:
                image_evaluator = image_eval()
                faces_ogau = mtcnn_to_torch(faces)
                faces_ogau = image_evaluator(faces_ogau)
                faces_ogau = faces_ogau.to(device)
                ogau_predictions = get_model_preds(faces_ogau, model_ogau, model_type=OPENGRAPHAU_MODEL_TYPE, device=device)
                ogau_predictions[is_null > 0] = 0 # clear the predictions from frames w/o faces!
                print("Got Network Predictions: OGAU")

              if use_cuda:
                torch.cuda.empty_cache()
              
              if TIMING_VERBOSE:
                time4 = time.time()
                print('Time: ', time4 - time3)

              # Save outputs to a CSV
              frames = np.array(real_frame_numbers).reshape(-1, 1)
              real_frame_numbers = []

              if Run_HSE:
                csv_save_HSE(labels=hse_scores_real, is_null=is_null, frames=frames, save_path=save_path_hse, fps=real_fps)
                del hse_scores_real
                print(f"Saved HSE CSV to {save_path_hse}!")
              
              if Run_OpenGraphAU:
                csv_save(labels=ogau_predictions, is_null=is_null, frames=frames, save_path=save_path_ogau, fps=real_fps)
                print(f"Saved OpenGraphAU CSV to {save_path_ogau}!")

              if 'all_bboxes' in globals():
                csv_save_bboxes(labels=all_bboxes[['Facial Box X', 'Facial Box Y', 'Facial Box W', 'Facial Box H']].values,  is_null=is_null, frames=frames, save_path=save_path_bboxes, fps=real_fps)
                print(f"Saved Facial Bboxes CSV to {save_path_bboxes}!")
              
              frame_now = frameNr

              # Reset ims for the next batch!
              del ims
              del frames
              del faces
              ims = []

              # Reset timing
              if TIMING_VERBOSE: 
                time1 = time.time()
              
              # Garbage collector
              gc.collect()
      else:
          # We're out of frames!
          running = False

          # Let's do analysis, save results, and reset ims!
          ims = np.array(ims)
          print(f"Extracted Ims, Frames {frame_now} to {frameNr} in {i}") 
          if TIMING_VERBOSE:
            time2 = time.time()
            print('Time: ', time2 - time1)

          # Face detection
          if Do_Verification:
            if Face_Detector == 'MTCNN':
              faces, is_null, all_bboxes = extract_faces_with_verify(ims, INPUT_SIZE, SUBJECT_FACE_IMAGE_FOLDER, partialVerify=Partial_Verify, \
                                                                     mtcnn=mtcnn, verifyAll=Verify_Every_Frame, verify_threshold=VERIFY_THRESHOLD, \
                                                         distance_max=DISTANCE_MAX_PARTIAL_VERIFY, save_folder_path=save_folder_partial_verify_now, \
                                                          real_frame_numbers=real_frame_numbers, saveProb=SAVE_PROB_PARTIAL_VERIFY)
            elif Face_Detector == 'RetinaFace':
              faces, is_null = detect_extract_faces(ims, INPUT_SIZE)
          else:
            if Face_Detector == 'MTCNN':
              faces, is_null, all_bboxes = extract_faces_mtcnn(ims, INPUT_SIZE, mtcnn=mtcnn, real_frame_numbers=real_frame_numbers)
            elif Face_Detector == 'RetinaFace':
              faces, is_null = detect_extract_faces(ims, INPUT_SIZE)
          print(f"Detected Faces")
          if TIMING_VERBOSE:
            time3 = time.time()
            print('Time: ', time3 - time2) 
          
          # Get predictions of relevant network
          if Run_HSE:
            faces_for_hse = convert_to_gpu_tensor(faces, device=device, FORCE_HSE_CPU=FORCE_HSE_CPU)
            hse_scores_real = hse_preds(faces_for_hse, model_hse, model_type=HSE_MODEL_TYPE, device=device, FORCE_HSE_CPU=FORCE_HSE_CPU)    
            hse_scores_real[is_null > 0] = 0 # clear the predictions from frames w/o faces!
            del faces_for_hse
            print("Got Network Predictions: HSE")

          if use_cuda:
            torch.cuda.empty_cache()
          
          if Run_OpenGraphAU:
            image_evaluator = image_eval()
            faces_ogau = mtcnn_to_torch(faces)
            faces_ogau = image_evaluator(faces_ogau)
            faces_ogau = faces_ogau.to(device)
            ogau_predictions = get_model_preds(faces_ogau, model_ogau, model_type=OPENGRAPHAU_MODEL_TYPE, device=device)
            ogau_predictions[is_null > 0] = 0 # clear the predictions from frames w/o faces!
            print("Got Network Predictions: OGAU")

          if use_cuda:
            torch.cuda.empty_cache()

          if TIMING_VERBOSE:
            time4 = time.time()
            print('Time: ', time4 - time3)

          # Save outputs to a CSV
          frames = np.array(real_frame_numbers).reshape(-1, 1)
          real_frame_numbers = []

          if Run_HSE:
            csv_save_HSE(labels=hse_scores_real, is_null=is_null, frames=frames, save_path=save_path_hse, fps=real_fps)
            del hse_scores_real
            print(f"Saved HSE CSV to {save_path_hse}!")

          if Run_OpenGraphAU:
            csv_save(labels=ogau_predictions, is_null=is_null, frames=frames, save_path=save_path_ogau, fps=real_fps)
            print(f"Saved OpenGraphAU CSV to {save_path_ogau}!")

          if 'all_bboxes' in globals():
            csv_save_bboxes(labels=all_bboxes[['Facial Box X', 'Facial Box Y', 'Facial Box W', 'Facial Box H']].values,  is_null=is_null, frames=frames, save_path=save_path_bboxes, fps=real_fps)
            print(f"Saved Facial Bboxes CSV to {save_path_bboxes}!")

          frame_now = frameNr

          # Reset ims to save space
          del ims
          del frames
          del faces
          ims = []

          # Reset timing
          if TIMING_VERBOSE: 
            time1 = time.time()
          
          # Garbage collector
          gc.collect()
          
      frameNr = frameNr + 1
  capture.release()
  os.rename(save_folder_now, final_folder_now)
  print(f'Finished {i}: {final_folder_now}')

  # Time estimation
  elapsed_time = time.time() - start_time
  iterations_left = num_vids - unprocessed_videos.index(i) - 1
  time_per_iteration = elapsed_time / (unprocessed_videos.index(i) + 1)
  time_left = time_per_iteration * iterations_left
  time_left_formatted = str(datetime.timedelta(seconds=int(time_left)))
  
  # print an update on the progress
  print("Approximately ", time_left_formatted, " left to complete analyzing all videos")

 

