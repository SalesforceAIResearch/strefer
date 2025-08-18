import argparse
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

from helper import load_video_frames_with_timestamps, save_frames, create_image_grid
from helper import build_and_bfs as get_frame_index_ordering
from matcher import HungarianMatcher
import shutil
import joblib


TMP_DIR = Path('./output/tmp')
referring_tracklets = joblib.load(os.path.join(TMP_DIR, 'referring_tracklets.pkl'))

# process the box detection results
video_segments = referring_tracklets['masklets']
OBJECTS = referring_tracklets['referrings']
confidences = referring_tracklets['object_box_gdinoscores']

print(f"class_names: {OBJECTS}")
print(f"confidences: {confidences}")

ID_TO_OBJECTS = {i: '{}_{}'.format(obj, round(confidences[i-1], 2))  for i, obj in enumerate(OBJECTS, start=1)}
OBJECTS_TO_ID = {ID_TO_OBJECTS[k].split('_')[0]:k for k in ID_TO_OBJECTS}

# convert mask FROM rle format
from pycocotools import mask as maskUtils

for frame_idx in video_segments:
    for object_idx in video_segments[frame_idx]:

        masks = [maskUtils.decode(mask_rle) for mask_rle in video_segments[frame_idx][object_idx]]
        masks = np.stack(masks, axis=0).astype(bool)
        # masks = video_segments[frame_idx][object_idx] 
        # (num_mask, h, w) -> num_mask == 1
        
        video_segments[frame_idx][object_idx] = masks
        

n_frames = len(video_segments)
video_dir = os.path.join(TMP_DIR, 'video')
TRACK_DIR = Path(os.path.join(TMP_DIR, 'referring_track'))
TRACK_DIR.mkdir(parents=True, exist_ok=False)
pil_images = []
for frame_idx in range(n_frames):
    segments = video_segments[frame_idx]
    img = cv2.imread(os.path.join(video_dir, f"{frame_idx}.jpg"))
    
    object_ids = list(segments.keys())
    masks = list(segments.values())
    masks = np.concatenate(masks, axis=0)
    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
        mask=masks, # (n, h, w)
        class_id=np.array(object_ids, dtype=np.int32),
    )
    box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
    mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

    annotated_image_path = os.path.join(TRACK_DIR, f"{frame_idx}.jpg")
    cv2.imwrite(annotated_image_path, annotated_frame)
    pil_images.append(Image.open(annotated_image_path))


# --- Convert the annotated frames to video and image grids
create_video_from_images(
    TRACK_DIR, os.path.join(TRACK_DIR, "referring_expression_masklets_results.mp4"), frame_rate=1)
create_image_grid(pil_images, num_columns=8, figsize=(80, 20), 
                  grid_image_save_path=os.path.join(TRACK_DIR, "referring_expression_masklets_results.png"))
 