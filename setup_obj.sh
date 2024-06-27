#!/bin/bash

source /opt/conda/bin/activate llmgame

export GSA_PATH=/root/Grounded-Segment-Anything/

DEFAULT_N=1  # Replace with the actual default value you want
DEFAULT_GPU_ID=0 

N=${1:-$DEFAULT_N}
GPU_ID=${2:-$DEFAULT_GPU_ID}

python main_vis_vec.py -n "$N" --gpu_id "$GPU_ID"