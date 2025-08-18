#!/bin/bash

# Define base path variables
MINICONDA_PATH="/path/to/miniconda3"
STREFER_PATH="/path/to/strefer"
CACHE_PATH="/path/to/cache"

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
source "$MINICONDA_PATH/bin/activate"
conda activate grounded-sam-2

# Set environment variables
export PYTHONPATH="$STREFER_PATH/data-engine/referring-masklet-generator/Grounded-SAM-2"
export TRANSFORMERS_CACHE="$CACHE_PATH/huggingface/hub"

# Change directory and run the script
cd "$PYTHONPATH"
time python "$PYTHONPATH/3.vis_tracklet.py"
