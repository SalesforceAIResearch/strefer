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


n_frames = 32

START_FRAME = 'heuristic'
# START_FRAME = 'first'
# START_FRAME = 'end'

expressions = joblib.load('../tmp.pkl')
referrings = expressions['referrings']
generalized_nouns = expressions['generalized_nouns']
video_path = expressions['video_path']

TMP_DIR = Path('./output/tmp')
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
TMP_DIR.mkdir(parents=True, exist_ok=False)

unique_generalized_nouns = list(set(generalized_nouns))
if 'person' in unique_generalized_nouns:  # make 'person' the first
    unique_generalized_nouns.remove('person')
    unique_generalized_nouns = ['person'] + unique_generalized_nouns

# VERY important: text queries need to be lowercased + end with a dot
text = '.'.join(unique_generalized_nouns) + '. '

start_time = "00:00:00.000"
end_time = None
    
video_frames, video_arr, frame_timestamps = load_video_frames_with_timestamps(
    video_path, n_frames=n_frames, start_time=start_time, end_time=end_time)
save_frames(video_frames, list(range(len(video_frames))), os.path.join(TMP_DIR, 'video'))

################## Grounded-SAM-2 Settings ##################
DUMP_JSON_RESULTS = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
# GROUNDING_MODEL = "IDEA-Research/grounding-dino-base"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# environment settings
# use bfloat16
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# build grounding dino from huggingface
model_id = GROUNDING_MODEL
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)
#############################################################

"""
Step 1: Locate the video frame with the most entities
"""
def post_process_detection(results, score_threshold=0.5):
    scores = results[0]['scores']
    boxes = results[0]['boxes']
    text_labels = results[0]['text_labels']
    labels = results[0]['labels']

    # --- remove empty and low-score  detections 
    new_labels = []
    for i in range(len(labels)):
        if labels[i] != '' and scores[i] >= score_threshold:
            new_labels.append(labels[i])
    
    num_detected_objects = len(new_labels)

    new_text_labels = []
    new_scores = torch.zeros((num_detected_objects))
    new_boxes = torch.zeros((num_detected_objects, 4))
    new_boxid = 0
    for i in range(len(labels)):
        if labels[i] != '' and scores[i] >= score_threshold:
            new_scores[new_boxid] = scores[i]
            new_boxes[new_boxid] = boxes[i]
            new_text_labels.append(text_labels[i])
            new_boxid += 1

    new_results = {
        'scores': new_scores,
        'boxes': new_boxes,
        'text_labels': new_text_labels,
        'labels': new_labels
    }
    new_results = [new_results]
    return new_results
    
    
print("generalized_nouns:", generalized_nouns)


if START_FRAME == 'first':
    frame_indices = list(range(n_frames))  # loop over frames of a video from start to end
elif START_FRAME == 'end':
    frame_indices = list(range(n_frames))
    frame_indices = frame_indices[::-1]
else:
    frame_indices = get_frame_index_ordering(n_frames) # loop over frames of a video with ordering heuristics


frame_index_for_tracking = frame_indices[-1]

max_detected_record = {
    'max_num_detected_objs': -1,
    'max_detected_objs_f_id': 0,
    'max_detected_objs_results': None
}
for f_id in frame_indices:
    image = video_frames[f_id]

    # --- prompt Grounding DINO image predictor to get the box for each frame
    inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.25,  # higher the box_threshold, fewer detection results
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    
    # print(f_id, results[0]["labels"], 'before post process')
    results = post_process_detection(results)
    print(f_id, results[0]["labels"], 'after post process')
        
    num_detected_objs = len(results[0]["labels"])

    if num_detected_objs > max_detected_record['max_num_detected_objs']:
        max_detected_record['max_num_detected_objs'] = num_detected_objs
        max_detected_record['max_detected_objs_f_id'] = f_id
        max_detected_record['max_detected_objs_results'] = results

    if num_detected_objs >= len(generalized_nouns):
        frame_index_for_tracking = f_id
        
        break


# --- visualize the box on the image
if (frame_index_for_tracking == frame_indices[-1]) and (max_detected_record['max_detected_objs_f_id'] != frame_index_for_tracking):
    frame_index_for_tracking = max_detected_record['max_detected_objs_f_id']
    results = max_detected_record['max_detected_objs_results']

confidences = results[0]["scores"].cpu().numpy().tolist()
class_names = results[0]["labels"]
class_ids = np.array(list(range(len(class_names))))

labels = [
    f"{idx}: {class_name} {confidence:.2f}"
    for idx, (class_name, confidence) in enumerate(zip(class_names, confidences))
]

img = video_arr[frame_index_for_tracking]
detections = sv.Detections(
    xyxy=results[0]["boxes"].cpu().numpy(),  # (n, 4)
    class_id=class_ids
)

box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
cv2.imwrite(os.path.join(TMP_DIR, f"frame{frame_index_for_tracking}-detected_nouns.jpg"), annotated_frame)
        

"""
Step 2: Obtain mask tracklets
"""
# process the box detection results
input_boxes = results[0]["boxes"].cpu().numpy()
OBJECTS = results[0]["labels"]
assert len(OBJECTS) > 0
# print(OBJECTS)

confidences = results[0]["scores"].cpu().numpy().tolist()
class_names = results[0]["labels"]

print(f"input_boxes: {input_boxes}")
print(f"confidences: {confidences}")
print(f"class_names: {class_names}")

ID_TO_OBJECTS = {i: '{}_{}'.format(obj, round(confidences[i-1], 2))  for i, obj in enumerate(OBJECTS, start=1)}

print(f"start backward and forward tracking at frame {frame_index_for_tracking}")

video_forward = list()
f_ordering_forward = list(range(frame_index_for_tracking, n_frames))
for f_id in f_ordering_forward:
    video_forward.append(video_frames[f_id])
save_frames(video_forward, list(range(len(video_forward))), os.path.join(TMP_DIR, 'video_forward'))

video_backward = list()
f_ordering_backward = list(range(0,frame_index_for_tracking+1))
f_ordering_backward = f_ordering_backward[::-1] # reverse
for f_id in f_ordering_backward:
    video_backward.append(video_frames[f_id])
save_frames(video_backward, list(range(len(video_forward))), os.path.join(TMP_DIR, 'video_backward'))

two_direction_tracking_results = {
    'forward': dict(),
    'backward': dict()
}

boxes_matcher = HungarianMatcher()  # matching boxes on the overlapping frame (frame_index_for_tracking)

def matching_makes_a_difference(matching):
    for k in matching:
        if matching[k] != k:
            return True
    else:
        return False
        
for video_dir in [os.path.join(TMP_DIR, 'video_forward'), os.path.join(TMP_DIR, 'video_backward')]:
    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    # ['0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg', '10.jpg', '11.jpg', '12.jpg', '13.jpg', '14.jpg', '15.jpg', '16.jpg']
    
    # init video predictor state
    inference_state = video_predictor.init_state(video_path=video_dir)
    ann_frame_idx = 0  # the frame index we interact with
    
    # --- register each object's positive points to video predictor with seperate add_new_points call
    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            box=box,
        )
    
    # --- propagate the video predictor to get the segmentation results for each frame
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    # --- visualize the segment results across the video and save them
    if 'backward' in video_dir:
        TRACK_DIR = Path(os.path.join(TMP_DIR, 'track_backward'))
        TRACK_DIR.mkdir(parents=True, exist_ok=True)

        overlap_frame_masks = list(video_segments[0].values())
        overlap_frame_masks = np.concatenate(overlap_frame_masks, axis=0)
        overlap_frame_boxes = sv.mask_to_xyxy(overlap_frame_masks)

        _, _, matching = boxes_matcher(
            overlap_frame_boxes,
            two_direction_tracking_results['forward']['overlap_frame_boxes']
            )
        print(f"matching: {matching}")
        if matching_makes_a_difference(matching):
            print("matching makes a difference")
        
        two_direction_tracking_results['backward']['overlap_frame_boxes'] = overlap_frame_boxes
        two_direction_tracking_results['backward']['video_segments'] = video_segments

        matching_postprocessed = dict()
        for k in matching:
            matching_postprocessed[k+1] = matching[k]+1  # object id starts from 1
        two_direction_tracking_results['matching'] = matching_postprocessed
        
    else:
        TRACK_DIR = Path(os.path.join(TMP_DIR, 'track_forward'))
        TRACK_DIR.mkdir(parents=True, exist_ok=True)

        overlap_frame_masks = list(video_segments[0].values())
        overlap_frame_masks = np.concatenate(overlap_frame_masks, axis=0)
        overlap_frame_boxes = sv.mask_to_xyxy(overlap_frame_masks)

        two_direction_tracking_results['forward']['overlap_frame_boxes'] = overlap_frame_boxes
        two_direction_tracking_results['forward']['video_segments'] = video_segments
        
    pil_images = []
    for frame_idx, segments in video_segments.items():
        img = cv2.imread(os.path.join(video_dir, frame_names[frame_idx]))
        
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
        TRACK_DIR, os.path.join(TRACK_DIR, "sam2_gdino_tracklets_results.mp4"))
    create_image_grid(pil_images, num_columns=8, figsize=(80, 20), 
                      grid_image_save_path=os.path.join(TRACK_DIR, "sam2_gdino_tracklets_results.png"))
    
    
# --- merge the forward and backward tracking results
video_segments_backward = two_direction_tracking_results['backward']['video_segments']
video_segments_forward = two_direction_tracking_results['forward']['video_segments']
matching = two_direction_tracking_results['matching']

video_segments = dict()
for i in range(len(f_ordering_backward)):
    # print(f_ordering_backward[i], i)
    video_segments[f_ordering_backward[i]] = video_segments_backward[i]
    
for i in range(len(video_segments_forward)):
    # print(f_ordering_forward[i], i)
    apply_matching = dict()
    for backward_obj_idx in matching:
        try:
            apply_matching[backward_obj_idx] = video_segments_forward[i][matching[backward_obj_idx]]
        except KeyError:
            import pdb;pdb.set_trace()
    video_segments[f_ordering_forward[i]] = apply_matching  # video_segments_forward[i]


def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle
    

video_dir = os.path.join(TMP_DIR, 'video')
TRACK_DIR = Path(os.path.join(TMP_DIR, 'track'))
TRACK_DIR.mkdir(parents=True, exist_ok=False)
pil_images = []
for frame_idx in range(n_frames):
    segments = video_segments[frame_idx]
    img = cv2.imread(os.path.join(video_dir, f"{frame_idx}.jpg"))
    
    object_ids = list(segments.keys())
    masks = list(segments.values())
    masks = np.concatenate(masks, axis=0)
    
    # # convert mask into rle format
    # mask_rles = [single_mask_to_rle(mask) for mask in masks]
    
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
    TRACK_DIR, os.path.join(TRACK_DIR, "noun_phrase_masklets_results.mp4"))
create_image_grid(pil_images, num_columns=8, figsize=(80, 20), 
                  grid_image_save_path=os.path.join(TRACK_DIR, "noun_phrase_masklets_results.png"))

# convert mask INTO rle format
for frame_idx in video_segments:
    for object_idx in video_segments[frame_idx]:

        # masks = video_segments[frame_idx][object_idx] 
        # (num_mask, h, w) -> num_mask == 1
        
        mask_rles = [single_mask_to_rle(mask) for mask in video_segments[frame_idx][object_idx]]
        video_segments[frame_idx][object_idx] = mask_rles
        
    
noun_tracklets = {
    'masklets': video_segments,  # keyed by frame_idx, values are keyed by object_idx
    'object_box_names': OBJECTS,
    'object_box_gdinoscores': confidences,
    'object_box_on_the_start_tracking_frame': input_boxes,
    'start_tracking_frame_idx': frame_index_for_tracking,
    'referrings': referrings,
    'generalized_nouns': generalized_nouns
}
joblib.dump(noun_tracklets, os.path.join(TMP_DIR, 'noun_tracklets.pkl'))
