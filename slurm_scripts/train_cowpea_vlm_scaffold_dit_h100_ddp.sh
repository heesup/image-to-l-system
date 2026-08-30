#!/bin/bash
# =============================================================================
# SLURM 2x H100 DDP Multi-GPU Launcher for VLM-Scaffold-DiT (Coarse-to-Fine)
# =============================================================================
# Allocates 2x NVIDIA H100 GPUs on node gpu-10-58 with torchrun DDP & W&B logging.
#
# Usage:
#   sbatch slurm_scripts/train_cowpea_vlm_scaffold_dit_h100_ddp.sh
# =============================================================================

#SBATCH --job-name=vlm_mmdit_ddp
#SBATCH --account=publicgrp
#SBATCH --partition=low
#SBATCH --constraint="gpu:h100" # "gpu:h100|gpu:a100" 
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_scripts/logs/train_vlm_scaffold_h100_%j.log
#SBATCH --error=slurm_scripts/logs/train_vlm_scaffold_h100_%j.log

set -e

REPO_ROOT="/home/lion397/codes/image-to-l-system"
PYTHON_BIN="/home/lion397/.conda/envs/digital-crops/bin/python"
TORCHRUN_BIN="/home/lion397/.conda/envs/digital-crops/bin/torchrun"

mkdir -p "${REPO_ROOT}/slurm_scripts/logs"
mkdir -p "${REPO_ROOT}/diffusion_based/checkpoints/fm"

cd ${REPO_ROOT}

echo "============================================================"
echo "🚀 SLURM Job ID:    ${SLURM_JOB_ID}"
echo "Node Allocated:     ${SLURM_NODELIST}"
echo "Partition:          ${SLURM_JOB_PARTITION}"
echo "Start Time:         $(date)"
echo "GPUs Available:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "============================================================"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST="8.0;9.0+PTX"
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0

# Run 2x A100/H100 DDP via torchrun with 4-Channel (RGB 3ch + Depth 1ch) Render Loss
# + 17D->XML->Helios round-trip validation column in eval debug images
${TORCHRUN_BIN} \
    --nproc_per_node=2 \
    --master_port=29505 \
    diffusion_based/training/train_cowpea_vlm_scaffold_dit_ddp.py \
    --epochs 60 \
    --batch-size 32 \
    --grad-accum-steps 2 \
    --lr 2.5e-4 \
    --eval-every 2 \
    --max-slots 4096 \
    --embed-dim 768 \
    --decoder-layers 12 \
    --num-heads 12 \
    --cond-drop-prob 0.10 \
    --guidance-scale 2.0 \
    --render-loss-weight 0.15 \
    --helios-roundtrip \
    --noise-sigma 0.05 \
    --resume \
    --cache-dir dataset/helios_data/cowpea_shard \
    --data-root dataset/helios_data/cowpea \
    --save-dir diffusion_based/checkpoints/fm \
    --save-name cowpea_vlm_scaffold_dit_h100_ddp.pt \
    --num-workers 8 \
    --use-wandb \
    --wandb-project cowpea-vlm-scaffold-dit \
    --wandb-group vlm-mmdit-b128-rgbd-renderloss

echo "============================================================"
echo "🏁 Training Finished Successfully at $(date)"
echo "============================================================"
