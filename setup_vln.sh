#!/bin/bash

source /opt/conda/bin/activate llmgame

export GSA_PATH=/root/Grounded-Segment-Anything/

# Default values for the parameters
DEFAULT_VLN_MODE="clip"  # Default mode as a string
DEFAULT_GPU_ID=0  # Default GPU ID

# Assign arguments with defaults
VLN_MODE="${1:-$DEFAULT_VLN_MODE}"
GPU_ID="${2:-$DEFAULT_GPU_ID}"

# Run the Python script with the parameters
python main_vln.py --vln_mode "$VLN_MODE" --gpu_id "$GPU_ID"