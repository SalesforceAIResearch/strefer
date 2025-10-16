#!/bin/bash

# Initialize conda
__conda_setup="$('/root/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/root/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/root/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup


# Activate conda environment
source /path/to/miniconda3/bin/activate
conda activate blip-3-strefer
export PYTHONPATH=/path/to/blip-3-strefer
cd /path/to/blip-3-strefer

# Set common variables
cache_dir='/path/to/.cache'
data_root='/path/to/data/root'

########### model specific settings
ckpt_path="/path/to/blip_3_strefer_checkpoint_0.pt"
num_frames=32
temporal_encoder="gttm"
num_timestamp_tokens=32


########### inference general settings
output_dir="./inference_results"

CUDA_VISIBLE_DEVICES=0 python3 open_flamingo/inference/run_inference_video_qa_with_mask.py \
  --ckpt_pth ${ckpt_path} \
  --cache_dir ${cache_dir} \
  --save_video_frames \
  --show_masks \
  --output_dir ${output_dir} \
  --data_root ${data_root} \
  --num_frames ${num_frames} \
  --temporal_encoder ${temporal_encoder} \
  --num_timestamp_tokens ${num_timestamp_tokens} \
  --mask_referring \
  --prompt_type general
