import torch
import os
import re
from typing import Any, Dict, List, Union
import numpy as np
import joblib
from pathlib import Path
from PIL import Image

from rexseek.builder import load_rexseek_model
from rexseek.utils.inference_utils import (
    modify_processor_resolution,
    prepare_input_for_rexseek,
)

ERR_CODE = -1


def inference_rexseek(
    image: Image.Image,
    image_processor: Any,
    tokenizer: Any,
    rexseek_model: Any,
    crop_size_raw: tuple[int, int],
    candidate_boxes: torch.Tensor,
    question: str,
) -> str:
    """Process an image with RexSeek model to answer questions about detected objects.

    Args:
        image: Input PIL image
        image_processor: Processor for image preprocessing
        tokenizer: Tokenizer for text processing
        rexseek_model: The RexSeek model instance
        crop_size_raw: Tuple of (height, width) for image cropping
        template: Dictionary containing prompt templates for model input
        candidate_boxes: Tensor of detected object bounding boxes
        question: Question to ask about the detected objects

    Returns:
        str: Model's answer to the question about detected objects
    """
    # start inference for rexseek
    template = dict(
        SYSTEM=("<|im_start|>system\n{system}<|im_end|>\n"),
        INSTRUCTION=("<|im_start|>user\n{input}<|im_end|>\n" "<|im_start|>assistant\n"),
        SUFFIX="<|im_end|>",
        SUFFIX_AS_EOS=True,
        SEP="\n",
        STOP_WORDS=["<|im_end|>", "<|endoftext|>"],
    )
    # print(f"question: {question}")
    
    data_dict = prepare_input_for_rexseek(
        image,
        image_processor,
        tokenizer,
        candidate_boxes,
        question,
        crop_size_raw,
        template,
    )
    
    input_ids = data_dict["input_ids"]
    pixel_values = data_dict["pixel_values"]
    pixel_values_aux = data_dict["pixel_values_aux"]
    gt_boxes = data_dict["gt_boxes"]

    input_ids = input_ids.to(device="cuda", non_blocking=True)
    pixel_values = pixel_values.to(
        device="cuda", dtype=torch.float16, non_blocking=True
    )
    pixel_values_aux = pixel_values_aux.to(
        device="cuda", dtype=torch.float16, non_blocking=True
    )

    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        output_ids = rexseek_model.generate(
            input_ids,
            pixel_values=pixel_values,
            pixel_values_aux=pixel_values_aux,
            gt_boxes=gt_boxes.to(dtype=torch.float16, device="cuda"),
            do_sample=False,
            max_new_tokens=512,
            temperature=0.0,
            top_p=None,
        )
    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
    return answer


def convert_all_cate_prediction_to_ans(prediction: str) -> Dict[str, List[str]]:
    # Define the pattern to extract ground truth labels and object tags
    pattern = r"<ground>(.*?)<\/ground><objects>(.*?)<\/objects>"

    # Find all matches in the prediction string
    matches = re.findall(pattern, prediction)

    # Initialize the dictionary to store the result
    ans = {}
    existed_obj_index = []
    for ground, objects in matches:
        ground = ground.strip()
        # Extract the object tags, e.g., <obj0>, <obj1>, <obj2>
        object_tags = re.findall(r"<obj\d+>", objects)
        filtered_object_tags = []
        for object_tag in object_tags:
            if object_tag in existed_obj_index:
                continue
            filtered_object_tags.append(object_tag)
            existed_obj_index.append(object_tag)

        if len(filtered_object_tags) == 0:
            continue
        # Add the ground truth label and associated object tags to the dictionary
        if ground not in ans:
            ans[ground] = object_tags

    return ans


def partition_input(noun_tracklets):
    referrings = noun_tracklets['referrings']
    generalized_nouns = noun_tracklets['generalized_nouns']
    object_box_names = noun_tracklets['object_box_names']

    map_r2b= dict()
    for i in range(len(generalized_nouns)):
        map_r2b[i] = set()
        for j in range(len(object_box_names)):
            if generalized_nouns[i] == object_box_names[j]:
                map_r2b[i].add(str(j))

    map_r2b_new = dict()
    for k in map_r2b:
        if len(map_r2b[k]) > 0:  # omit if the referring has no box found
            map_r2b_new[str(k)] = '_'.join(sorted(list(map_r2b[k])))
    map_r2b = map_r2b_new

    map_b2r = dict()
    for k in map_r2b:
        if map_r2b[k] not in map_b2r:
            map_b2r[map_r2b[k]] = set()
        map_b2r[map_r2b[k]].add(k)

    map_b2r_new = dict()
    for k in map_b2r:
        map_b2r_new[k] =  '_'.join(sorted(list(map_b2r[k])))
    map_b2r = map_b2r_new
    
    partitions = []
    for box_indices_str in map_b2r:
        referring_indices_str = map_b2r[box_indices_str]
        box_indices_int = {int(idx) for idx in box_indices_str.split('_')}
        box_indices_int_lst = list(box_indices_int)
        referring_indices_int = {int(idx) for idx in referring_indices_str.split('_')}

        box_names = noun_tracklets['object_box_names']
        new_box_names = [box_names[idx] for idx in box_indices_int]

        scores = noun_tracklets['object_box_gdinoscores']
        new_scores = [scores[idx] for idx in box_indices_int]

        new_object_box_on_the_start_tracking_frame =  np.array(
            [noun_tracklets['object_box_on_the_start_tracking_frame'][idx] for idx in box_indices_int])
        
        video_segments = noun_tracklets['masklets']
        new_video_segments = dict()
        for frame_idx in video_segments:
            new_video_segments[frame_idx] = dict()

            segments = video_segments[frame_idx]
            for segments_obj_idx in segments:
                if segments_obj_idx - 1 in box_indices_int:
                    new_segments_obj_idx = box_indices_int_lst.index(segments_obj_idx - 1) + 1
                    new_video_segments[frame_idx][new_segments_obj_idx] = video_segments[frame_idx][segments_obj_idx]

        new_referrings = [referrings[idx] for idx in referring_indices_int]
        new_generalized_nouns = [generalized_nouns[idx] for idx in referring_indices_int]
        
        
        referring_tracklets = {
            'masklets': new_video_segments,  # keyed by frame_idx, values are keyed by object_idx
            'object_box_names': new_box_names,
            'object_box_gdinoscores': new_scores,
            'object_box_on_the_start_tracking_frame': new_object_box_on_the_start_tracking_frame,
            'start_tracking_frame_idx': noun_tracklets['start_tracking_frame_idx'],
            'referrings': new_referrings,
            'generalized_nouns': new_generalized_nouns
        }
        partitions.append(referring_tracklets)
        
    return partitions

def merge_partition_results(partition_results, noun_tracklets):
    num_partion_handled_prev = 0
    all_referrings = []
    all_generalized_nouns = []
    all_object_box_names = []
    all_scores = []
    all_object_box_on_the_start_tracking_frame = []
    all_video_segments = dict()
    for referring_tracklets_partition in partition_results:
        if referring_tracklets_partition != ERR_CODE:
            all_referrings += referring_tracklets_partition['referrings']
            all_generalized_nouns += referring_tracklets_partition['generalized_nouns']
            all_object_box_names += referring_tracklets_partition['object_box_names']
            all_scores += referring_tracklets_partition['object_box_gdinoscores']
            all_object_box_on_the_start_tracking_frame.append(referring_tracklets_partition['object_box_on_the_start_tracking_frame'])

            video_segments = referring_tracklets_partition['masklets']
            for frame_idx in video_segments:
                if frame_idx not in all_video_segments:
                    all_video_segments[frame_idx] = dict()
    
                segments = video_segments[frame_idx]
                for segments_obj_idx in segments:
                    new_segments_obj_idx = num_partion_handled_prev + segments_obj_idx
                    all_video_segments[frame_idx][new_segments_obj_idx] = video_segments[frame_idx][segments_obj_idx]

            num_partion_handled_prev += len(referring_tracklets_partition['referrings'])
            
    if len(all_object_box_on_the_start_tracking_frame) > 0:
        all_object_box_on_the_start_tracking_frame = np.vstack(all_object_box_on_the_start_tracking_frame)
    
    if len(all_referrings) > 0:
        
        referring_tracklets = {
            'masklets': all_video_segments,  # keyed by frame_idx, values are keyed by object_idx
            'object_box_names': all_object_box_names,
            'object_box_gdinoscores': all_scores,
            'object_box_on_the_start_tracking_frame': all_object_box_on_the_start_tracking_frame,
            'start_tracking_frame_idx': noun_tracklets['start_tracking_frame_idx'],
            'referrings': all_referrings,
            'generalized_nouns': all_generalized_nouns
        }
        return referring_tracklets
    else:
        return ERR_CODE

def referring_matching(TMP_DIR, noun_tracklets):
    
    referrings = noun_tracklets['referrings']
    generalized_nouns = noun_tracklets['generalized_nouns']
    
    object_box_on_the_start_tracking_frame = noun_tracklets["object_box_on_the_start_tracking_frame"]
    start_tracking_frame_idx = noun_tracklets["start_tracking_frame_idx"]
    
    if (referrings == generalized_nouns) or ((len(referrings) == 1) and (len(object_box_on_the_start_tracking_frame) == 1)):
        return noun_tracklets
    
    # load image
    input_image_path = os.path.join(TMP_DIR, 'video', f"{start_tracking_frame_idx}.jpg")
    image = Image.open(input_image_path)

    candidate_boxes = torch.Tensor(object_box_on_the_start_tracking_frame)
    
    # load rexseek model
    model_path = "IDEA-Research/RexSeek-3B"
    tokenizer, rexseek_model, image_processor, context_len = load_rexseek_model(
        model_path
    )
    image_processor, crop_size_raw = modify_processor_resolution(image_processor)


    referring2boxid = dict()
    for referring in referrings:
        # inference rexseek
        question = f"Please detect {referring} in this image. Answer the question with object indexes."
        answer = inference_rexseek(
            image,
            image_processor,
            tokenizer,
            rexseek_model,
            crop_size_raw,
            candidate_boxes,
            question,
        )
        # print(f"RexSeek answer: {answer}")
        
        ans = convert_all_cate_prediction_to_ans(answer)
                
        pred_box_index = []
        for k, v in ans.items():
            for box in v:
                obj_idx = int(box[4:-1])
                if obj_idx < len(candidate_boxes):
                    pred_box_index.append(obj_idx)

        referring2boxid[referring] = pred_box_index

    print(referring2boxid)

    #####
    print('post processing...')
    # post processing to ensure no overlapping assignment
    valid = True
    boxid2referring = dict()
    for referring in referring2boxid:
        if len(referring2boxid[referring]) == 1:
            if referring2boxid[referring][0] in boxid2referring:
                valid = False
            boxid2referring[referring2boxid[referring][0]] = referring
    print(boxid2referring)
    for referring in referring2boxid:
        if len(referring2boxid[referring]) > 1:
            for boxid in referring2boxid[referring]:
                if boxid in boxid2referring:
                    continue
                boxid2referring[boxid] = referring
                break
    print(boxid2referring)

    referring2boxid = {boxid2referring[boxid]:boxid for boxid in boxid2referring}
    
    if len(referring2boxid) < 1:
        valid = False
    
    referrings_new = []
    generalized_nouns_new = []
    for i in range(len(referrings)):
        if referrings[i] in referring2boxid:
            referrings_new.append(referrings[i])
            generalized_nouns_new.append(generalized_nouns[i])
    referrings = referrings_new
    generalized_nouns = generalized_nouns_new
    
    if valid:
        print('processing referring matching succesfully')

        referring2boxid = {boxid2referring[boxid]:boxid for boxid in boxid2referring}
        
        print(boxid2referring) 
        print(referring2boxid) 
        
        scores = noun_tracklets['object_box_gdinoscores']
        new_scores = [scores[referring2boxid[referring]] for referring in referrings]

        new_object_box_on_the_start_tracking_frame =  np.array(
            [noun_tracklets['object_box_on_the_start_tracking_frame'][referring2boxid[referring]] for referring in referrings])
        
        new_object_box_name = [noun_tracklets['object_box_names'][referring2boxid[referring]] for referring in referrings]

        video_segments = noun_tracklets['masklets']
        new_video_segments = dict()
        for frame_idx in video_segments:
            new_video_segments[frame_idx] = dict()

            segments = video_segments[frame_idx]
            for segments_obj_idx in segments:
                if segments_obj_idx - 1 in boxid2referring:
                    new_segments_obj_idx = referrings.index(boxid2referring[segments_obj_idx-1]) + 1
                    new_video_segments[frame_idx][new_segments_obj_idx] = video_segments[frame_idx][segments_obj_idx]

        referring_tracklets = {
            'masklets': new_video_segments,  # keyed by frame_idx, values are keyed by object_idx
            'object_box_names': new_object_box_name,
            'object_box_gdinoscores': new_scores,
            'object_box_on_the_start_tracking_frame': new_object_box_on_the_start_tracking_frame,
            'start_tracking_frame_idx': noun_tracklets['start_tracking_frame_idx'],
            'referrings': referrings,
            'generalized_nouns': generalized_nouns
        }
        return referring_tracklets
    
    else:
        print('processing referring matching failed! <--- NOTE')

        print(boxid2referring) 
        print(referring2boxid) 

        return ERR_CODE

    
if __name__ == "__main__":
    TMP_DIR = '{}/Grounded-SAM-2/output/tmp'.format(Path(__file__).resolve().parent.parent)
    noun_tracklets = joblib.load(os.path.join(TMP_DIR, 'noun_tracklets.pkl'))

    # referrings = noun_tracklets['referrings']
    # generalized_nouns = noun_tracklets['generalized_nouns']
    
    ######## partition
    partitions = partition_input(noun_tracklets)

    ######## processing
    partition_results = []
    for noun_tracklets_partition in partitions:
        partition_results.append(
            referring_matching(TMP_DIR, noun_tracklets_partition)
        )
        
    ######## merging
    referring_tracklets = merge_partition_results(partition_results, noun_tracklets)
    
    if referring_tracklets != ERR_CODE:
        joblib.dump(referring_tracklets, os.path.join(TMP_DIR, 'referring_tracklets.pkl'))
