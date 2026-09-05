#!/bin/bash
# =============================================================================
# SLURM Launcher: Part Flow Matching training (26D curvature-encoding)
# Lean resources: 9.3M-param model -> 2 CPU / 16G RAM / 1 GPU schedules fast.
# =============================================================================
#SBATCH --job-name=fm_curv
#SBATCH --partition=gpu-6000_ada-h
#SBATCH --account=geminigrp
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_scripts/logs/fm_curv_%j.log
#SBATCH --error=slurm_scripts/logs/fm_curv_%j.log

set -e
REPO_ROOT="/home/lion397/codes/image-to-l-system"
PYTHON_BIN="/home/lion397/.conda/envs/digital-crops/bin/python"
TORCHRUN_BIN="/home/lion397/.conda/envs/digital-crops/bin/torchrun"
mkdir -p "${REPO_ROOT}/slurm_scripts/logs" "${REPO_ROOT}/diffusion_based/checkpoints/fm"
cd ${REPO_ROOT}

# Number of GPU ranks = GPUs actually allocated (respects --gres)
NPROC=${SLURM_GPUS_ON_NODE:-$(nvidia-smi --list-gpus | wc -l)}
# Global batch stays constant: per-rank batch = GLOBAL / ranks
BATCH_SIZE=$(( 256 / NPROC ))
if [[ ${BATCH_SIZE} -lt 1 ]]; then BATCH_SIZE=1; fi
MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 400 ))

if [ -f ~/.bashrc ]; then source ~/.bashrc; fi
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export TORCH_CUDA_ARCH_LIST="8.0;9.0+PTX"

echo "Node: ${SLURM_NODELIST} | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"

echo "torchrun ranks: ${NPROC} | per-rank batch: ${BATCH_SIZE} | master port: ${MASTER_PORT}"
${TORCHRUN_BIN} \
    --nproc_per_node=${NPROC} \
    --master_port=${MASTER_PORT} \
    diffusion_based/training/train_part_flow_matching.py \
    --data_root dataset/helios_data \
    --max_nodes 512 \
    --image_size 128 \
    --embed_dim 256 \
    --encoder_layers 6 \
    --decoder_layers 4 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${FM_EPOCHS:-50} \
    --lr 2e-4 \
    --num_workers 6 \
    --cache_dir dataset/cache/cowpea_curv26 \
    --prior_type scaffold \
    --vis_every 1 --vis_samples 3 \
    --use-wandb --wandb-project part-flow-matching --wandb-group fm-curv26 \
    --checkpoint_dir diffusion_based/checkpoints/fm_curv

echo "Training finished at $(date)"
