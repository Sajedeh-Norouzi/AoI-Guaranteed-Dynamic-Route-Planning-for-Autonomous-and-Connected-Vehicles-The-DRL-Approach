#!/bin/bash

# Create the logs directory if it doesn't exist
mkdir -p logs

# GPU ID to use (example: GPU 1)
gpu_id=4

# Run the command with nohup, setting CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$gpu_id nohup python3 main.py > logs/output.log &

