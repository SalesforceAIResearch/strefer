import math
import os
import sys
import argparse
import json
from collections import OrderedDict
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
import torch
import torchvision
import transformers
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import importlib
from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import clips_at_random_timestamps

import open_flamingo
import open_flamingo.src.kosmos
from open_flamingo import create_model_and_transforms
from open_flamingo.train.any_res_data_utils import process_images
import open_flamingo.inference.utils as utils

importlib.reload(open_flamingo.src.kosmos)
sys.path.append('..')
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "error"

backend_video = 'opencv'
# backend_video = 'torchcodec'

import supervision as sv
from supervision.draw.color import ColorPalette
CUSTOM_COLOR_MAP = [
"#e6194b",
"#3cb44b",
"#ffe119",
"#0082c8",
"#f58231",
"#911eb4",
"#46f0f0",
"#f032e6",
"#d2f53c",
"#fabebe",
"#008080",
"#e6beff",
"#aa6e28",
"#fffac8",
"#800000",
"#aaffc3",
]

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--ckpt_pth', type=str, help='Checkpoint path')
    parser.add_argument('--cache_dir', help='Directory to cache', required=True)
    parser.add_argument('--output_dir', help='Directory to output', required=True)
    parser.add_argument('--data_root', help='Directory to data root', required=True)
    
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    
    parser.add_argument('--num_frames', default=4, type=int, help='Number of frames of a video')  
    parser.add_argument("--save_video_frames", action="store_true", help="Saves the frames used during evalution.")
    parser.add_argument("--show_masks", action="store_true", help="Saves the frames used during evalution.")
    
    parser.add_argument("--temporal_encoder", type=str, default='gttm', help="Temporal encoder used in the model")
    parser.add_argument("--num_timestamp_tokens", type=int, default=0)

    parser.add_argument(
        '--mask_referring',
        default=False, action='store_true'
    )
    
    parser.add_argument("--prompt_type", type=str, default="general", choices=['general', 'mcq', 'videomme_sub', 'videomme_nosub', 'mcq_caption'], help="Choose a specific type depending upon the dataset used.")
    return parser.parse_args()

def time_to_seconds(timestamp):
    """Convert timestamp (HH:MM:SS.sss) to seconds."""
    h, m, s = map(float, timestamp.split(":"))
    total_seconds = h * 3600 + m * 60 + s
    return total_seconds

def seconds_to_timetoken(cur_seconds, duration_seconds, num_timetokens):
    return round(num_timetokens * (cur_seconds/duration_seconds))

def timetoken_to_seconds(timetoken, duration_seconds, num_timetokens):
    return round(duration_seconds * (timetoken/num_timetokens))

def time_to_frame(timestamp, fps):
    """Convert timestamp (HH:MM:SS.sss) to frame index."""
    h, m, s = map(float, timestamp.split(":"))
    total_seconds = h * 3600 + m * 60 + s
    return int(total_seconds * fps)

def load_video_opencv(video_path, num_frames, start_time="00:00:00.000", end_time=None):
    # Open video file
    cv2_vr = cv2.VideoCapture(video_path)
    if not cv2_vr.isOpened():
        print("Error: Could not open video file.", video_path)
        return -1, None, None

    # Get video properties
    fps = cv2_vr.get(cv2.CAP_PROP_FPS)
    total_frames = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print("Error: No frames found in video.", video_path)
        return -1, None, None

    # Convert start and end times to frame indices
    start_frame = time_to_frame(start_time, fps)
    end_frame = time_to_frame(end_time, fps) if end_time else total_frames - 1

    # Ensure valid frame range
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame, min(end_frame, total_frames - 1))

    # Compute duration of the extracted segment in seconds
    duration_seconds = (end_frame - start_frame) / fps

    # Generate frame indices to sample within the range
    frame_id_list = np.linspace(start_frame, end_frame, num_frames, dtype=int)

    video_data = []
    frame_timestamps = []

    for i, frame_idx in enumerate(frame_id_list):
        # cv2_vr.set(1, frame_idx)
        cv2_vr.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cv2_vr.read()
        if not ret:
            # print(f"Warning: Frame at index {frame_idx} could not be read.", video_path)
            if i > 0:
                video_data.append(video_data[-1])  # Replicate last frame if read fails
                frame_timestamps.append(frame_timestamps[-1])
            continue
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # video_data.append(torch.from_numpy(frame).permute(2, 0, 1))
        video_data.append(frame)

        # Compute timestamp for the frame in seconds
        frame_time = frame_idx / fps
        frame_timestamps.append(frame_time)

    cv2_vr.release()
    video_data = [Image.fromarray(f).convert('RGB') for f in video_data]

    if len(video_data) < num_frames:
        print("Error: Certain frames were not read", video_path)
        return -1, None, None
    
    return video_data, frame_timestamps, duration_seconds

def load_video_torchcodec(video_path, num_frames, start_time="00:00:00.000", end_time=None, min_duration=0):

    decoder = VideoDecoder(video_path)

    # Convert the start and end times to frame seconds and indices
    start_seconds = time_to_seconds(start_time)
    end_seconds = time_to_seconds(end_time) if end_time else decoder.metadata.duration_seconds

    duration_seconds = end_seconds - start_seconds
    if duration_seconds < min_duration:
        return -1, None, None
    
    frame_interval = (end_seconds - start_seconds) / num_frames


    clip = clips_at_random_timestamps(
        decoder,
        num_clips=1,
        num_frames_per_clip=num_frames,
        seconds_between_frames=frame_interval,
        sampling_range_start=start_seconds,
        sampling_range_end=start_seconds+frame_interval,
    )
    
    video = clip.data[0]
    frame_time_in_seconds = clip.pts_seconds[0]
    video_cue_duration_in_seconds = duration_seconds

    video = [torchvision.transforms.functional.to_pil_image(f) for f in video]
    return video, frame_time_in_seconds, video_cue_duration_in_seconds


def load_video(video_path, num_frames, start_time="00:00:00.000", end_time=None):
    if backend_video == 'opencv':
        return load_video_opencv(video_path, num_frames, start_time, end_time)
    elif backend_video == 'torchcodec':
        return load_video_torchcodec(video_path, num_frames, start_time, end_time)
    else:
        print(f"Error: Video loading using {backend_video} is not supported.")
        os._exit(0)

def get_model_output(model, sample_set, file_processor, cfg, tokenizer, args):
    print("sample_set: {}".format(sample_set))

    if ('clip_timestamps' in sample_set) and (
        not ('<start_time>' in sample_set['question']) or '<end_time>' in sample_set['question']):
        
        images, frame_time_in_seconds, video_cue_duration_in_seconds = load_video(
            sample_set['video_path'], 
            args.num_frames,
            start_time=sample_set['clip_timestamps'][0], 
            end_time=sample_set['clip_timestamps'][1],
        )

    else:
        images, frame_time_in_seconds, video_cue_duration_in_seconds = load_video(
            sample_set['video_path'], 
            args.num_frames
        )

    image_size = [img.size for img in images]
    image_size = [image_size]
    vision_x = [file_processor([img]) for img in images]
    vision_x = torch.stack(vision_x)
    vision_x = vision_x.unsqueeze(0)
    
    if args.save_video_frames:
        for f_id, image in enumerate(images):
            image.save(os.path.join(args.output_dir, '{}.png'.format(f_id)))

    if '<start_time>' in sample_set['question']:
        start_second = time_to_seconds(sample_set['start_time'])
        assert start_second < video_cue_duration_in_seconds
        start_time = seconds_to_timetoken(start_second, video_cue_duration_in_seconds, args.num_timestamp_tokens-1)
        sample_set['question'] = sample_set['question'].replace('<start_time>', '<{}>'.format(start_time))
        
        start_second_convertedback = timetoken_to_seconds(start_time, video_cue_duration_in_seconds, args.num_timestamp_tokens-1)
        print('start_second_convertedback: {}'.format(start_second_convertedback))
        
    if '<end_time>' in sample_set['question']:
        end_second = time_to_seconds(sample_set['end_time'])
        assert end_second < video_cue_duration_in_seconds
        end_time = seconds_to_timetoken(end_second, video_cue_duration_in_seconds, args.num_timestamp_tokens-1)
        sample_set['question'] = sample_set['question'].replace('<end_time>', '<{}>'.format(end_time))
        
        end_second_convertedback = timetoken_to_seconds(end_time, video_cue_duration_in_seconds, args.num_timestamp_tokens-1)
        print('end_second_convertedback: {}'.format(end_second_convertedback))
        
    
    # Get the prompt and prompt tokens
    question = sample_set['question']
    subtitles = sample_set.get('subtitles', None) 
    prompt, lang_x = utils.get_prompt(question, tokenizer, cfg, args.prompt_type, subtitles)
    print("prompt: ",prompt)
    # print("lang_x: ",lang_x)

    kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=1024, top_p=None, num_beams=1)
    generated_text = model.generate(
        vision_x = vision_x.to(torch.device(args.device)),
        lang_x = lang_x['input_ids'].to(torch.device(args.device)),
        image_size = image_size,
        attention_mask = lang_x['attention_mask'].to(torch.device(args.device)),
        **kwargs_default)

    # Decode the tokens
    output = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    if 'Phi-3' in cfg.lm_path: output = output.split('<|end|>')[0]
    
    print('output: ',output)
    return output


def create_image_grid(pil_images, num_columns=8, figsize=(20, 5), 
                      grid_image_save_path='grid_image.png'):
    num_rows = (len(pil_images) + num_columns - 1) // num_columns

    img_width, img_height = pil_images[0].size
    grid_width = num_columns * img_width
    grid_height = num_rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    for idx, image in enumerate(pil_images):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        position = (col_idx * img_width, row_idx * img_height)
        grid_image.paste(image, position)

    plt.figure(figsize=figsize)
    plt.imshow(grid_image)
    plt.axis('off')
    # plt.show()
    plt.savefig(grid_image_save_path, bbox_inches='tight', pad_inches=0)
    return grid_image
            

def visualize_mask_on_frame(frame_path, masks, color_idx=1):
    img = cv2.imread(frame_path)
    
    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
        mask=masks, # (n, h, w) <class 'numpy.ndarray'>
        class_id=np.array([color_idx], dtype=np.int32), # it can be any interger here
    )
    
    box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

    cv2.imwrite(frame_path, annotated_frame)
    print('{} saved!'.format(frame_path))
    return

    
def get_model_output_with_mask_input(model, sample_set, file_processor, cfg, tokenizer, args):
    from open_flamingo.train.data_utils import annToMask, process_video, load_video_frames_with_timestamps

    mask_single_frame = False
            
    if not mask_single_frame:
                
        ann_indices = []
        frame_nums = 1
        all_frames = set()
        
        for ann in sample_set['annotation']: # for each object mask annotation
            all_frames.update(list(ann.keys()))
        all_frames = list(all_frames)
        frame_nums = len(all_frames)
    
        all_frames = [int(f) for f in all_frames]
        all_frames = sorted(all_frames)
        all_frames = [str(f) for f in all_frames]
    
        for ann in sample_set['annotation']:
            frame_list = list(ann.keys())
        
            frame_list = [int(f) for f in frame_list]
            frame_list = sorted(frame_list)
            frame_list = [str(f) for f in frame_list]
        
            indices = []
            for frame in frame_list:
                indices.append(all_frames.index(frame))
            ann_indices.append(indices)
         
        all_frames = [int(f) for f in all_frames]

    else:  # select a random single frame for mask referring
        ann_indices = []
        frame_nums = 1
        all_frames = set()

        selected_frame_per_obj = []
        
        for ann in sample_set['annotation']: # for each object mask annotation
            if len(all_frames) == 0:
                selected_frame = random.sample(list(ann.keys()),1)[0]
            else:
                overlapped_frames = all_frames & set(ann.keys())
                if not overlapped_frames:
                    selected_frame = random.sample(list(ann.keys()),1)[0]
                else:
                    selected_frame = random.sample(list(overlapped_frames),1)[0]
            all_frames.add(selected_frame)
            selected_frame_per_obj.append(selected_frame)
                    
        all_frames = list(all_frames)
        frame_nums = len(all_frames)

        all_frames = [int(f) for f in all_frames]
        all_frames = sorted(all_frames)
        all_frames = [str(f) for f in all_frames]

        obj_idx = 0
        for ann in sample_set['annotation']:
            indices = [all_frames.index(selected_frame_per_obj[obj_idx])]
            ann_indices.append(indices)

            obj_idx += 1
         
        all_frames = [int(f) for f in all_frames]
    

    if 'videorefer' in sample_set:
        video_frames, mask_frames, height, width = process_video(
            sample_set['path'], num_frames=args.num_frames, frame_idx=all_frames, 
            MAX_FRAMES=args.num_frames)
        
    else:
        video_frames, _, _, video_cue_duration_in_seconds = load_video_frames_with_timestamps(
            sample_set['path'], n_frames=args.num_frames, start_time="00:00:00.000", end_time=None)
        height, width = np.array(video_frames[0]).shape[:2]
        
        if args.num_frames == frame_nums:
            mask_frames = video_frames
        else:
            mask_frames = [video_frames[frame_idx] for frame_idx in all_frames]


    images = video_frames
    
    if args.save_video_frames:
        for f_id, image in enumerate(images):
            image.save(os.path.join(args.output_dir, '{}.png'.format(f_id)))

    if args.show_masks:        
        pil_images = []
        for f_id in range(len(all_frames)):
            image = mask_frames[f_id]
            ann_fidx = str(all_frames[f_id])

            os.makedirs(os.path.join(args.output_dir, 'mask_frames'), exist_ok=True)
            f_path = os.path.join(args.output_dir, 'mask_frames', '{}.jpg'.format(f_id))
            if os.path.exists(f_path):
                os.remove(f_path)
            image.save(f_path)

            # get the mask
            for obj_idx in range(len(sample_set['annotation'])):   # loop over obj
                ann = sample_set['annotation'][obj_idx]
                if ann_fidx in ann:
                    mask = annToMask(ann[ann_fidx]['segmentation'], None, None)
                    mask = np.expand_dims(np.array(mask), axis=0).astype(bool)
            
                    visualize_mask_on_frame(f_path, mask, color_idx=obj_idx)
                    
            pil_images.append(Image.open(f_path))
            
    
        mask_grid_path = os.path.join(args.output_dir, "mask_vis_results.png")
        if os.path.exists(mask_grid_path):
            os.remove(mask_grid_path)
        create_image_grid(pil_images, num_columns=8, figsize=(80, 20), 
                          grid_image_save_path=mask_grid_path)


    image_size = [img.size for img in images]
    image_size = [image_size]
    vision_x = [file_processor([img]) for img in images]
    vision_x = torch.stack(vision_x)
    vision_x = vision_x.unsqueeze(0)

    mask_frames = [file_processor([img]) for img in mask_frames]
    mask_frames = torch.stack(mask_frames)
    mask_frames = mask_frames.unsqueeze(0)


    masks = []
    if 'height' in sample_set:
        h = sample_set['height']
        w = sample_set['width']
    else:
        h = None
        w = None

    if not mask_single_frame:
        for ann in sample_set['annotation']:
            frame_list = list(ann.keys())

            # sort frame ids
            frame_list = [int(f) for f in frame_list]
            frame_list = sorted(frame_list)
            frame_list = [str(f) for f in frame_list]
            
            for ann_idx in frame_list:
                
                try:
                    mask = annToMask(ann[ann_idx]['segmentation'], h, w)
                except:
                    mask = np.zeros((height, width))
                masks.append(mask)
    else:
        obj_idx = 0
        for ann in sample_set['annotation']:
            try:
                mask = annToMask(
                    ann[selected_frame_per_obj[obj_idx]]['segmentation'], h, w)
            except:
                mask = np.zeros((height, width))
            masks.append(mask)

            obj_idx += 1

        
    mask_frame_count = 0
    for obj_idx in range(len(ann_indices)):
        mask_frame_count += len(ann_indices[obj_idx])
    assert len(masks) == mask_frame_count

    
    masks = [torch.Tensor(np.array(masks)).to(torch.device(args.device))]
    ann_indices = [ann_indices]
    frame_nums = [frame_nums]

    question = sample_set["conversations"][0]['value']
    # question = "<image>\nPlease give a detailed description of the highlighted object <region> in the video."

    if '<start_time>' in question:
        start_second = time_to_seconds(sample_set['start_time'])
        assert start_second < video_cue_duration_in_seconds
        start_time = seconds_to_timetoken(start_second, video_cue_duration_in_seconds, args.num_timestamp_tokens-1)
        question = question.replace('<start_time>', '<{}>'.format(start_time))
        
        start_second_convertedback = timetoken_to_seconds(start_time, video_cue_duration_in_seconds, args.num_timestamp_tokens-1)
        print('start_second_convertedback: {}'.format(start_second_convertedback))
        
    if '<end_time>' in question:
        end_second = time_to_seconds(sample_set['end_time'])
        assert int(end_second) < video_cue_duration_in_seconds
        end_time = seconds_to_timetoken(end_second, video_cue_duration_in_seconds, args.num_timestamp_tokens-1)
        question = question.replace('<end_time>', '<{}>'.format(end_time))
        
        end_second_convertedback = timetoken_to_seconds(end_time, video_cue_duration_in_seconds, args.num_timestamp_tokens-1)
        print('end_second_convertedback: {}'.format(end_second_convertedback))
    

    # Get the prompt and prompt tokens
    question = question.replace('<image>\n', '')
    question = question.replace('\n<image>', '')
    question = question.replace('<image>', '')
    
    subtitles = sample_set.get('subtitles', None) 
    prompt, lang_x = utils.get_prompt(question, tokenizer, cfg, args.prompt_type, subtitles)

    # print("video path: ", sample_set['path'])
    # print('%'*50)
    print("prompt: ",prompt)
    # print('%'*50)
    # # print("lang_x: ",lang_x)

    
    kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=1024, top_p=None, num_beams=1)
    generated_text = model.generate(
        vision_x = vision_x.to(torch.device(args.device)),
        lang_x = lang_x['input_ids'].to(torch.device(args.device)),
        image_size = image_size,
        attention_mask = lang_x['attention_mask'].to(torch.device(args.device)),
        mask_frames=mask_frames.to(torch.device(args.device)),
        frame_nums=frame_nums,
        masks=masks,
        ann_indices=ann_indices,
        **kwargs_default)

    # Decode the tokens
    output = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    if 'Phi-3' in cfg.lm_path: output = output.split('<|end|>')[0]

    # print('%'*50)
    # print('Pred: ',output)
    
    GT_answer = sample_set["conversations"][1]['value']
    # print('%'*50)
    # print('GT answer: ',GT_answer)

    return output


def run_inference(args):
    if os.path.exists(args.output_dir):
        import shutil
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the model
    # Set model conf
    cfg, additional_kwargs = utils.set_model_conf(args)
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path=cfg.vision_encoder_path,
        clip_vision_encoder_pretrained=cfg.vision_encoder_pretrained,
        lang_model_path=cfg.lm_path,
        tokenizer_path=cfg.lm_path,
        model_family=cfg.model_family,
        **additional_kwargs 
    )
    # Load the model with checkpoint
    ckpt = torch.load(cfg.ckpt_pth)["model_state_dict"]
    new_state_dict = OrderedDict()
    print("{} loaded!".format(cfg.ckpt_pth))

    for key, value in ckpt.items():
        new_key = key.replace('.gttm.', '.temporal_encoder.')
        new_key = new_key.replace('.ttm.', '.temporal_encoder.')
        new_key = new_key.replace('.tts.', '.temporal_encoder.')
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict, strict=True)
    torch.cuda.empty_cache()
    model = model.eval().cuda()

    image_proc = partial(process_images, image_processor=image_processor, model_cfg=cfg)


    with open('../assets/inference_sample.json', 'r') as f:
        sample_set = json.load(f)
    sample_set['path'] = os.path.join(args.data_root, sample_set['path'])

    # Model inference
    if 'annotation' in sample_set:
        output = get_model_output_with_mask_input(model = model,
                            sample_set = sample_set,
                            file_processor = image_proc,
                            cfg = cfg,
                            tokenizer = tokenizer,
                            args = args)
    else:
        output = get_model_output(model = model,
                                sample_set = sample_set,
                                file_processor = image_proc,
                                cfg = cfg,
                                tokenizer = tokenizer,
                                args = args)

    prediction = {
            'question': sample_set["conversations"][0]['value'],
            'GT_answer': sample_set["conversations"][1]['value'],
            'pred': output
        }
    

    print('\n'*3+'%'*50)
    print("Video path: ", sample_set['path'])
    print("Question: ", prediction['question'])
    print("Prediction: ", prediction['pred'])
    print("GT_answer: ", prediction['GT_answer'])
    print('%'*50)
    
    
    return
    
    


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    run_inference(args)
    
    
