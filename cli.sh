#!/bin/bash

apptainer exec --nv --bind /scratch:/scratch catchment_gpu_venv.sif \
python main.py train \
    --input_dir "sample" \
    --output_dir "catchment_run_1" \
    --max_epochs 50 \
    --batch_size 16 \
    --num_workers 8 \
    # --continue_training