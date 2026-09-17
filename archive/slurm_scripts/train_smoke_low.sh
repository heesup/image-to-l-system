#!/bin/bash
#SBATCH --job-name=fm_smoke_low
#SBATCH --output=slurm_scripts/logs/fm_smoke_low_%j.log
#SBATCH --error=slurm_scripts/logs/fm_smoke_low_%j.log
#SBATCH --account=publicgrp
#SBATCH --partition=low
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -e
REPO_ROOT="/home/lion397/codes/image-to-l-system"
cd "$REPO_ROOT"
PYTHON_BIN="/home/lion397/.conda/envs/digital-crops/bin/python"
TORCHRUN_BIN="/home/lion397/.conda/envs/digital-crops/bin/torchrun"

$TORCHRUN_BIN --standalone --nproc_per_node=2 \
    plant_recon/training/train_hierarchical_flow_matching.py \
    --render_fraction 0.167 \
    --render_grad_start_epoch 4 \
    --epochs 6 \
    --save_every 5 \
    --max_train_samples 600 \
    --batch_size 8 \
    --slots_per_phytomer 10 \
    --flow_granularity phytomer \
    --output_dir plant_recon/checkpoints/fm_smoke_low \
    --freeze_backbone \
    2>&1
