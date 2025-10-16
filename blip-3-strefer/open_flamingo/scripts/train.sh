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
export TRANSFORMERS_CACHE='/path/to/.cache/huggingface/hub'
export HF_HOME='/path/to/.cache/huggingface/hub'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pretrained_ckpt="/path/to/blip3_checkpoint_0.pt"

# Replace this line with your own wandb api key.
wandb login --relogin --host=XXXXXXX --relogin; # Use your wandb login info

##########################################################################################
exp_prefix_name=1010
exp_postfix_name="training_with_one_gpu"
data_path='data_configs/final_recipe.yaml'
exp_name="/path/to/outputs/blip3video_training/${exp_prefix_name}-${exp_postfix_name}"
data_root='/path/to/data/root'
num_frames=32
min_video_seconds=0
num_timestamp_tokens=32
temporal_encoder=gttm
batch_size=2
checkpoint_steps=5000
wandb_project="wandb_project_name"
master_port=9650

if [[ -e $exp_name ]]; then
    rm -rf "$exp_name"
fi

mkdir "$exp_name"

python open_flamingo/src/data_partition.py --data_root ${data_root} --data_path ${data_path}


export PYTHONPATH="$PYTHONPATH:open_flamingo"

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 --master_port ${master_port} open_flamingo/train/instruction_finetune_vid.py \
    --lm_path microsoft/Phi-3-mini-4k-instruct \
    --tokenizer_path microsoft/Phi-3-mini-4k-instruct \
    --conv_template_name phi_3 \
    --vision_encoder_path google/siglip-so400m-patch14-384 \
    --vision_encoder_pretrained google \
    --model_family kosmos \
    --num_vision_tokens 128 \
    --pretrained ${pretrained_ckpt} \
    --data_path ${data_path} \
    --dataset_resampled \
    --image_aspect_ratio pad \
    --batch_size ${batch_size} \
    --checkpoint_steps ${checkpoint_steps} \
    --fsdp \
    --gradient_checkpointing \
    --fsdp_sharding_strategy hybrid \
    --workers 16 \
    --num_epochs 1 \
    --warmup_steps 500 \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --lr_scheduler cosine \
    --precision amp_bf16 \
    --num_frames ${num_frames} \
    --min_video_seconds ${min_video_seconds} \
    --num_timestamp_tokens ${num_timestamp_tokens} \
    --temporal_encoder ${temporal_encoder} \
    --mask_referring \
    --wandb_project ${wandb_project} \
    --report_to_wandb \
    --run_name ${exp_name} 2>&1 | tee ${exp_name}/terminal_output.log;
