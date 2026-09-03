#!/bin/bash
# =============================================================================
# SLURM DDP Multi-GPU Launcher for VLM-Scaffold-DiT (Coarse-to-Fine)
# =============================================================================
# Allocates N GPUs via SLURM (--gres) and spawns one DDP rank per allocated GPU
# on the node SLURM assigns. No hardcoded GPU model / count / node / port.
#
# Usage:
#   sbatch slurm_scripts/train_cowpea_vlm_scaffold_dit_ddp.sh
# =============================================================================

#SBATCH --job-name=vlm_mmdit_ddp
#SBATCH --account=publicgrp
#SBATCH --partition=low
#SBATCH --constraint="gpu:h100" # "gpu:h100|gpu:a100"
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_scripts/logs/train_vlm_scaffold_%j.log
#SBATCH --error=slurm_scripts/logs/train_vlm_scaffold_%j.log

set -e

REPO_ROOT="/home/lion397/codes/image-to-l-system"
PYTHON_BIN="/home/lion397/.conda/envs/digital-crops/bin/python"
TORCHRUN_BIN="/home/lion397/.conda/envs/digital-crops/bin/torchrun"

mkdir -p "${REPO_ROOT}/slurm_scripts/logs"
mkdir -p "${REPO_ROOT}/diffusion_based/checkpoints/fm"

cd ${REPO_ROOT}

# Number of GPU ranks = GPUs actually allocated by SLURM (respects --gres,
# partition limits, and fairshare rewrites) — falls back to 1 outside SLURM.
NPROC=${SLURM_GPUS_ON_NODE:-$(nvidia-smi --list-gpus | wc -l)}

# Per-device batch scales with available VRAM: measured 22 GB / rank at batch 8
# (~3 GB fixed + ~2.3 GB/sample on H100 NVL 96 GB for this model), so batch 32
# fits ≈ 77 GB with ~18 GB headroom (no expandable_segments on this platform).
# Keep the EFFECTIVE global batch = 128 (batch × ranks × grad-accum) so LR 2.5e-4
# stays valid: 4 GPUs × 32 × 1 = 128 (grad-accum drops 2 → 1).
BATCH_SIZE=$(( 128 / NPROC ))
if [[ ${BATCH_SIZE} -lt 1 ]]; then BATCH_SIZE=1; fi
ACCUM=1
if [[ ${NPROC} -le 1 ]]; then ACCUM=2; fi

echo "============================================================"
echo "🚀 SLURM Job ID:    ${SLURM_JOB_ID}"
echo "Node Allocated:     ${SLURM_NODELIST}"
echo "Partition:          ${SLURM_JOB_PARTITION}"
echo "Allocated GPUs:     ${NPROC}x $(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | tr '\n' ' ')"
echo "Ranks (nproc):      ${NPROC} | per-rank batch: ${BATCH_SIZE} | global batch: $(( BATCH_SIZE * NPROC * ACCUM ))"
echo "Start Time:         $(date)"
echo "============================================================"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST="8.0;9.0+PTX"
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0

# DDP via torchrun: one rank per allocated GPU (SLURM CUDA_VISIBLE_DEVICES
# already restricts visibility, so no fixed device mapping). Free port chosen
# from SLURM's allocated range to avoid collisions between concurrent jobs.
# 4-Channel (RGB 3ch + Depth 1ch) Render Loss + 17D->XML->Helios round-trip
# validation column in eval debug images.
# DDP via torchrun: one rank per allocated GPU (SLURM CUDA_VISIBLE_DEVICES
# already restricts visibility, so no fixed device mapping). Port derived from
# the SLURM job ID (unique per job → no collisions between concurrent jobs).
# 4-Channel (RGB 3ch + Depth 1ch) Render Loss + 17D->XML->Helios round-trip
# validation column in eval debug images.
MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 400 ))
echo "torchrun master port: ${MASTER_PORT}"
${TORCHRUN_BIN} \
    --nproc_per_node=${NPROC} \
    --master_port=${MASTER_PORT} \
    diffusion_based/training/train_cowpea_vlm_scaffold_dit_ddp.py \
    --epochs 60 \
    --batch-size ${BATCH_SIZE} \
    --grad-accum-steps ${ACCUM} \
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
    --cache-dir dataset/helios_data/cowpea_shard \
    --data-root dataset/helios_data/cowpea \
    --save-dir diffusion_based/checkpoints/fm \
    --save-name cowpea_vlm_scaffold_dit_h100_ddp.pt \
    --num-workers 3 \
    --use-wandb \
    --wandb-project cowpea-vlm-scaffold-dit \
    --wandb-group vlm-mmdit-b128-rgbd-renderloss

echo "============================================================"
echo "🏁 Training Finished Successfully at $(date)"
echo "============================================================"
