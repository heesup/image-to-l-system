#!/bin/bash
#SBATCH --job-name=phytomer_vae
#SBATCH --output=slurm_scripts/logs/phytomer_vae_%j.log
#SBATCH --error=slurm_scripts/logs/phytomer_vae_%j.log
#SBATCH --account=geminigrp
#SBATCH --partition=gpu-6000_ada-h
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# Trains PhytomerVAE (phytomer-level latent) on canonical 10-slot packets (v8:
# hybrid coarse+residual latent, default 128D = 48 coarse + 10 slots x 8
# rotation-residual -- see diffusion_based/models/phytomer_vae.py docstring).
# Single GPU is sufficient (tiny MLP VAE, ~1M params).
# Override via env: LATENT_DIM=128 RESIDUAL_DIM=8 EPOCHS=60 MAX_FILES=4000
# After training, PKT_CACHE_DIR=dataset/cache/cowpea_curv26_pkt precomputes the
# per-sample packet targets with the best checkpoint (v8 hybrid-latent format --
# PKT_VERSION bumped in generate_cache.py so stale 64D-latent files are rebuilt).
set -e

REPO_ROOT="/home/lion397/codes/image-to-l-system"
PYTHON_BIN="/home/lion397/.conda/envs/digital-crops/bin/python"

mkdir -p "${REPO_ROOT}/slurm_scripts/logs"
mkdir -p "${REPO_ROOT}/diffusion_based/checkpoints/phytomer_vae"
cd ${REPO_ROOT}

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================================================================================"
echo "Training PhytomerVAE-${LATENT_DIM:-128}D (residual ${RESIDUAL_DIM:-8}/slot) on phytomer packets"
echo "Job ID: $SLURM_JOB_ID | Host: $(hostname)"
echo "Date: $(date)"
echo "================================================================================"

${PYTHON_BIN} diffusion_based/training/train_phytomer_vae.py \
    --latent-dim "${LATENT_DIM:-128}" \
    --residual-dim "${RESIDUAL_DIM:-8}" \
    --hidden-dim 256 \
    --epochs "${EPOCHS:-60}" \
    --batch-size 4096 \
    --lr 1e-3 \
    --beta-kl 1e-3 \
    --max-files "${MAX_FILES:-4000}" \
    --seed "${SEED:-0}" \
    --packet-cache "/tmp/opencode/phytomer_packets_${MAX_FILES:-4000}_v2.pt" \
    --checkpoint-dir "${CHECKPOINT_DIR:-diffusion_based/checkpoints/phytomer_vae_v8}" \
    --pkt-cache-dir "${PKT_CACHE_DIR:-}" \
    --pkt-workers "${PKT_WORKERS:-28}" \
    --device cuda:0

echo "PhytomerVAE Training Completed at $(date)"
