#!/bin/bash

# This script runs the MAE training for a catchment foundation model.
#
# --- Key Parameters ---
# --input_dir:        Path to your TIFF image data.
# --output_dir:       Directory where models and logs will be saved.
# --vit_model_name:   Size of the Vision Transformer.
#                     Options: vit_small_patch16_224, vit_base_patch16_224, vit_large_patch16_224
# --max_epochs:       Total number of training epochs.
# --batch_size:       Number of images per batch.
# --num_workers:      Number of CPU cores for data loading.
# --continue_training: Uncomment this flag to resume from the latest checkpoint.


apptainer exec --nv --bind /scratch:/scratch catchment_gpu_venv.sif \
python main.py train \
    --input_dir "sample" \
    --output_dir "catchment_run_1" \
    --vit_model_name "vit_base_patch16_224" \
    --max_epochs 50 \
    --batch_size 32 \
    --num_workers 16 \
    # --continue_training