#!/bin/bash
# =============================================================================
# SLURM 2x H100 DDP Multi-GPU Training Launcher for Cowpea 232M DiT-Large
# =============================================================================
# Allocates 2x NVIDIA H100 GPUs on node gpu-10-58 with torchrun DDP & W&B logging.
#
# Usage:
#   sbatch slurm_scripts/train_cowpea_dit_h100_ddp.sh
# =============================================================================

#SBATCH --job-name=dit_h100_ddp
#SBATCH --account=publicgrp
#SBATCH --partition=low
#SBATCH --nodelist=gpu-10-58
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_scripts/logs/train_h100_ddp_%j.log
#SBATCH --error=slurm_scripts/logs/train_h100_ddp_%j.log

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
export TORCH_CUDA_ARCH_LIST="9.0+PTX"
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0

# Run 2x H100 DDP via torchrun
${TORCHRUN_BIN} \
    --nproc_per_node=2 \
    --master_port=29500 \
    diffusion_based/training/train_cowpea_dit_100k_ddp.py \
    --epochs 60 \
    --batch-size 32 \
    --grad-accum-steps 2 \
    --lr 4e-4 \
    --eval-every 2 \
    --cache-dir dataset/helios_data/cowpea_shard \
    --data-root dataset/helios_data/cowpea \
    --save-dir diffusion_based/checkpoints/fm \
    --save-name cowpea_dit_large_2xh100_ddp.pt \
    --use-wandb

EXIT_CODE=$?
echo "============================================================"
echo "Training finished with exit code ${EXIT_CODE} at $(date)"
echo "============================================================"
exit ${EXIT_CODE}
