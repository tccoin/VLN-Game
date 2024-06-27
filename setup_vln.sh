#!/bin/bash

source /opt/conda/bin/activate llmgame

export GSA_PATH=/root/Grounded-Segment-Anything/

DEFAULT_N=1  # Replace with the actual default value you want
DEFAULT_GPU_ID=0 

N=${1:-$DEFAULT_N}
GPU_ID=${2:-$DEFAULT_GPU_ID}

# Run the Python script with the parameters
python main_vln_vec.py -n "$N" --gpu_id "$GPU_ID"