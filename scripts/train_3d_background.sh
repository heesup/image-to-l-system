#!/usr/bin/env bash
# Background training script for the 3D plant diffusion model on Ubuntu GPU.
# Usage: ./scripts/train_3d_background.sh > train_3d_background.log 2>&1 &

export PYTHONUNBUFFERED=1

PYTHON_BIN="/data/heesup/miniconda3/envs/py310/bin/python"
DATA_ROOT="/data/heesup/codes/Digital-Crops/projects/syntheticdata_generation/build/output"

$PYTHON_BIN -m diffusion_based.training.train_diffusion_3d \
  --epochs 200 \
  --batch_size 2 \
  --max_nodes 2048 \
  --lr 3e-4 \
  --helios_data_root "$DATA_ROOT" \
  --num_samples 100 \
  --pretrain_existence_epochs 10 \
  --save_path diffusion_based/checkpoints/diffusion_model_3d_200ep.pt \
  --best_save_path diffusion_based/checkpoints/best_3d_model_200ep.pt

