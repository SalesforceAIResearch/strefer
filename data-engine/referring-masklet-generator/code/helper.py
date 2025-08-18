import os
import numpy as np
import pickle
import cv2
from PIL import Image
from datetime import timedelta


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
    if os.path.isdir(video_path):  # If input is a folder of frames
        video_data = []
        for filename in sorted(os.listdir(video_path)):
            img_path = os.path.join(video_path, filename)
            if not os.path.isdir(img_path):
                img = Image.open(img_path)
                video_data.append(img.convert('RGB'))
        return np.array(video_data), []

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

    return video_frames, video_arr, timestamps


def save_frames(frame_lst, framename_lst, frame_output_dir):
    import shutil
    if os.path.exists(frame_output_dir):
        shutil.rmtree(frame_output_dir)
    os.makedirs(frame_output_dir, exist_ok=True)

    for f_id, image in enumerate(frame_lst):
        # print(f_id)
        image.save(os.path.join(frame_output_dir, '{}.jpg'.format(f_id)))
    return


import matplotlib.pyplot as plt

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
            

from collections import deque

def build_balanced_tree_indices(start, end):
    """
    Recursively build a balanced binary tree of indices from 'start' to 'end'.
    Returns the root node of the tree, or None if start > end.
    Each node is represented as a dictionary with 'index', 'left', and 'right'.
    """
    if start > end:
        return None

    mid = (start + end) // 2
    return {
        'index': mid,
        'left': build_balanced_tree_indices(start, mid - 1),
        'right': build_balanced_tree_indices(mid + 1, end)
    }

def bfs_tree(root):
    """
    Perform a BFS on the tree whose root is 'root'.
    'root' is a dictionary node with 'index', 'left', and 'right'.
    Returns a list of node indices in BFS order.
    """
    if root is None:
        return []

    queue = deque([root])
    bfs_order = []

    while queue:
        node = queue.popleft()
        bfs_order.append(node['index'])

        if node['left'] is not None:
            queue.append(node['left'])
        if node['right'] is not None:
            queue.append(node['right'])

    return bfs_order

def build_and_bfs(n):
    """
    Given an integer n (the length of your list),
    1. Build a balanced binary tree of indices [0..n-1].
    2. Perform a BFS traversal on that tree.
    3. Return the BFS order of the node indices.
    """
    root = build_balanced_tree_indices(0, n - 1)
    return bfs_tree(root)