#!/usr/bin/env bash
# Background training script for the 3D plant diffusion model on Ubuntu GPU.
# Usage: ./scripts/train_3d_background.sh > train_3d_background.log 2>&1 &

export PYTHONUNBUFFERED=1
# Use the HPC-wide CUDA-enabled PyTorch environment on this node.
PYTHON_BIN="/cvmfs/hpc.ucdavis.edu/sw/conda/environments/pytorch-2.9.1/bin/python"
DATA_ROOT="Digital-Crops/projects/syntheticdata_generation/build/output"

export DISPLAY=:1.0

$PYTHON_BIN -m diffusion_based.training.train_diffusion_3d \
  --data-dir "$DATA_ROOT" \
  --epochs 200 \
  --batch-size 2 \
  --max_nodes 2048 \
  --lr 3e-4 \
  --num_samples 100 \
  --pretrain_existence_epochs 10 \
  --save_path diffusion_based/checkpoints/diffusion_model_3d_200ep.pt \
  --best_save_path diffusion_based/checkpoints/best_3d_model_200ep.pt

