import ast
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from multiprocessing import Value

import braceexpand
import webdataset as wds
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from pycocotools import mask as maskUtils
import numpy as np

import cv2
from PIL import Image
from datetime import timedelta


def save_data_dict_as_npz(data_dict, file_path="saved_data_dict.npz"):
    # Save to .npz
    np.savez_compressed(file_path, **data_dict)
    return
    

def load_data_dict_from_npz(file_path="saved_data_dict.npz"):
    # Load .npz
    with np.load(file_path, allow_pickle=True) as data:
        data_dict = {key: data[key] for key in data.files}
    return data_dict


def annToMask(mask_ann, h=None, w=None):
    if isinstance(mask_ann, list):
        rles = maskUtils.frPyObjects(mask_ann, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, h, w)
    else:
        # print("rle TRUE")
        # rle
        rle = mask_ann
    
    mask = maskUtils.decode(rle)
    return mask


def time_to_frame(time_str, fps):
    if time_str is None:
        return 0
    h, m, s = time_str.split(":")
    seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return int(seconds * fps)


def frame_to_timestamp(frame_idx, fps):
    seconds = frame_idx / fps
    return str(timedelta(seconds=seconds))[:-3]  # Trim microseconds to milliseconds


def load_video_frames_with_timestamps(video_path, n_frames=16, start_time="00:00:00.000", end_time=None):
    """
    Load frames from a video using OpenCV and return timestamps.
    
    Args:
        video_path (str): Path to the video file or directory of frames.
        n_frames (int): Number of frames to sample.
        start_time (str): Start timestamp in 'HH:MM:SS.sss' format.
        end_time (str): End timestamp in 'HH:MM:SS.sss' format (optional).
    
    Returns:
        Tuple: (List of PIL.Image frames, 
        Array of video frames (frames, height, width, channels),
        List of timestamp strings)
    """
    # Open video file
    cv2_vr = cv2.VideoCapture(video_path)
    if not cv2_vr.isOpened():
        print("Error: Could not open video file.", video_path)
        os._exit(0)

    fps = cv2_vr.get(cv2.CAP_PROP_FPS)
    total_frames = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = time_to_frame(start_time, fps)
    end_frame = time_to_frame(end_time, fps) if end_time else total_frames - 1

    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame, min(end_frame, total_frames - 1))

    # Compute duration of the extracted segment in seconds
    duration_seconds = (end_frame - start_frame) / fps

    frame_id_list = np.linspace(start_frame, end_frame, n_frames, dtype=int)

    video_frames = []
    video_arr = []
    timestamps = []
    for i, frame_idx in enumerate(frame_id_list):
        cv2_vr.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cv2_vr.read()
        if not ret:
            if i > 0:
                video_frames.append(video_frames[-1])  # Replicate last frame if read fails
                video_arr.append(video_arr[-1])
                timestamps.append(timestamps[-1])
            continue
        video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        video_arr.append(frame)
        timestamps.append(frame_to_timestamp(frame_idx, fps))

    cv2_vr.release()
    
    video_arr = np.array(video_arr)
    video_frames = [Image.fromarray(image) for image in np.array(video_frames)]

    return video_frames, video_arr, timestamps, duration_seconds
    

from decord import VideoReader, cpu
import imageio
import torch


def frame_sample(duration, mode='uniform', num_frames=None, fps=None):
    NUM_FRAMES_PER_SECOND = 1
    if mode == 'uniform':
        assert num_frames is not None, "Number of frames must be provided for uniform sampling."
        # Calculate the size of each segment from which a frame will be extracted
        seg_size = float(duration - 1) / num_frames

        frame_ids = []
        for i in range(num_frames):
            # Calculate the start and end indices of each segment
            start = seg_size * i
            end   = seg_size * (i + 1)
            # Append the middle index of the segment to the list
            frame_ids.append((start + end) / 2)

        return np.round(np.array(frame_ids) + 1e-6).astype(int)
        # return np.linspace(0, duration-1, num_frames, dtype=int)
    elif mode == 'fps':
        assert fps is not None, "FPS must be provided for FPS sampling."
        segment_len = min(fps // NUM_FRAMES_PER_SECOND, duration)
        return np.arange(segment_len // 2, duration, segment_len, dtype=int)
    else:
        raise ImportError(f'Unsupported frame sampling mode: {mode}')


def process_video(video_path, s=None, e=None, num_frames=16, frame_idx=None, MAX_FRAMES=32):
    if isinstance(video_path, str):
        if s is not None and e is not None:
            s = s if s >= 0. else 0.
            e = e if e >= 0. else 0.
            if s > e:
                s, e = e, s
            elif s == e:
                e = s + 1

        # Loading Video
        if os.path.isdir(video_path):                
            frame_files = sorted(os.listdir(video_path))

            fps = 3
            num_frames_of_video = len(frame_files)
        elif video_path.endswith('.gif'):
            gif_reader = imageio.get_reader(video_path)

            fps = 25
            num_frames_of_video = len(gif_reader)
        else:
            vreader = VideoReader(video_path, ctx=cpu(0), num_threads=1)

            fps = vreader.get_avg_fps()
            num_frames_of_video = len(vreader)

        # Determine frame range & Calculate frame indices
        f_start = 0                       if s is None else max(int(s * fps) - 1, 0)
        f_end   = num_frames_of_video - 1 if e is None else min(int(e * fps) - 1, num_frames_of_video - 1)
        frame_indices = list(range(f_start, f_end + 1))

        duration = len(frame_indices)
        
        # Sampling frame indices 
        if num_frames is None:
            sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='fps', fps=fps)]
        else:
            sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='uniform', num_frames=num_frames)]

        # Acquire frame data
        if os.path.isdir(video_path): 
            video_data = [Image.open(os.path.join(video_path, frame_files[f_idx])) for f_idx in sampled_frame_indices]
            frame_data = []
            if frame_idx is not None:
                for idx in frame_idx:
                    frame = Image.open(os.path.join(video_path, frame_files[idx])).convert('RGB')
                    frame_data.append(np.array(frame))
            else:
                frame_data = None
        elif video_path.endswith('.gif'):
            video_data = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)) for idx, frame in enumerate(gif_reader) if idx in sampled_frame_indices]
            if frame_idx is not None:
                frame_data = [frame for index, frame in enumerate(gif_reader) if index in frame_idx]
            else:
                frame_data = None
        else:
            try:
                video_data = [Image.fromarray(frame) for frame in vreader.get_batch(sampled_frame_indices).asnumpy()]
            except:
                video_data = [Image.fromarray(frame) for frame in vreader.get_batch(sampled_frame_indices).numpy()]
            if frame_idx is not None:
                try:
                    frame_data = vreader.get_batch(frame_idx).asnumpy()
                except:
                    frame_data = vreader.get_batch(frame_idx).numpy()
            else:
                frame_data = None

    else:
        raise ValueError(f"Unsupported video path type: {type(video_path)}")

    while num_frames is not None and len(video_data) < num_frames:
        video_data.append(Image.fromarray(np.zeros((*video_data[-1].size, 3), dtype=np.uint8)))

    # MAX_FRAMES filter
    video_data = video_data[:MAX_FRAMES]

    height, width = np.array(video_data[0]).shape[:2]

    video = [f for f in video_data]
    if frame_data is not None:
        frame_data = [Image.fromarray(f.numpy() if isinstance(f, torch.Tensor) else f) for f in frame_data]
    return video, frame_data, height, width


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    """
    DataInfo is a dataclass that holds information about a dataset.
    """

    name: str
    dataloader: DataLoader
    batch_size: int
    loss_multiplier: int
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_dataset_size(shards):
    """
    Get the number of samples in a dataset and the number of shards in a dataset
    based on the shards list.
    Returns None for the number of samples if is undefined.
    One can define the number of samples using a sizes.json file in the same directory
    or a __len__ file in the same directory.
    """
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards[0])
    sizes_filename = os.path.join(dir_path, "sizes.json")
    len_filename = os.path.join(dir_path, "__len__")
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, "r"))
        total_size = sum(
            [
                int(sizes[os.path.basename(shard)])
                if os.path.basename(shard) in sizes
                else 0
                for shard in shards_list
            ]
        )
    elif os.path.exists(len_filename):
        total_size = ast.literal_eval(open(len_filename, "r").read())
    else:
        total_size = None  # num samples undefined
    num_shards = len(shards_list)
    return total_size, num_shards


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    if "images in sample" not in repr(exn):
        logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def group_by_keys_nothrow(
    data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None
):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        if (
            current_sample is None
            or prefix != current_sample["__key__"]
            or suffix in current_sample
        ):
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


class detshuffle2(wds.PipelineStage):
    def __init__(
        self,
        bufsize=1000,
        initial=100,
        seed=0,
        epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            seed = pytorch_worker_seed(epoch)
        else:
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.
        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = wds.shardlists.expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            self.epoch += 1
            epoch = self.epoch

        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            yield dict(url=self.rng.choice(self.urls))
