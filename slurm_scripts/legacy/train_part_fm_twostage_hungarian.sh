#!/bin/bash
# =============================================================================
# SLURM Launcher: Two-Stage Part Flow Matching with Hungarian Bipartite Matcher
# 13D continuous geometry velocity + cascaded discrete organ-type classification.
# Lean resources: 9.55M-param model -> 2 GPU DDP, global batch 256.
# =============================================================================
#SBATCH --job-name=fm_twostage
#SBATCH --account=geminigrp
#SBATCH --partition=gpu-6000_ada-h
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_scripts/logs/fm_scaleup_%j.log
#SBATCH --error=slurm_scripts/logs/fm_scaleup_%j.log

set -e
REPO_ROOT="/home/lion397/codes/image-to-l-system"
PYTHON_BIN="/home/lion397/.conda/envs/digital-crops/bin/python"
TORCHRUN_BIN="/home/lion397/.conda/envs/digital-crops/bin/torchrun"
mkdir -p "${REPO_ROOT}/slurm_scripts/logs" "${REPO_ROOT}/diffusion_based/checkpoints/fm_scaleup_180m"
cd ${REPO_ROOT}

NPROC=${SLURM_GPUS_ON_NODE:-$(nvidia-smi --list-gpus | wc -l)}
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)

# Dynamic VRAM-Adaptive Per-Rank Batch Size for 180M Model (embed_dim=768, max_slots=4096)
if [[ -n "${FORCE_BATCH_SIZE}" ]]; then
    BATCH_SIZE=${FORCE_BATCH_SIZE}
elif [[ ${VRAM_MB} -ge 70000 ]]; then
    # 80GB H100 / A100-80GB
    BATCH_SIZE=32
elif [[ ${VRAM_MB} -ge 40000 ]]; then
    # 48GB RTX 6000 Ada / RTX A6000: safe peak ~33GB for 180M model
    BATCH_SIZE=16
elif [[ ${VRAM_MB} -ge 20000 ]]; then
    # 24GB/32GB (V100 32GB, RTX 3090/4090)
    BATCH_SIZE=8
else
    BATCH_SIZE=4
fi
GLOBAL_BATCH=$(( BATCH_SIZE * NPROC ))
MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 400 ))

if [ -f ~/.bashrc ]; then source ~/.bashrc; fi
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export TORCH_CUDA_ARCH_LIST="8.0;9.0+PTX"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "Node: ${SLURM_NODELIST} | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"

echo "torchrun ranks: ${NPROC} | per-rank batch: ${BATCH_SIZE} (global: ${GLOBAL_BATCH}) | master port: ${MASTER_PORT}"
${TORCHRUN_BIN} \
    --nproc_per_node=${NPROC} \
    --master_port=${MASTER_PORT} \
    diffusion_based/training/train_part_flow_matching.py \
    --data_root dataset/helios_data \
    --max_nodes 4096 \
    --node_dim 13 \
    --matcher hungarian \
    --image_size 256 \
    --embed_dim 768 \
    --encoder_layers 16 \
    --decoder_layers 12 \
    --num_heads 12 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${FM_EPOCHS:-500} \
    --lr 2e-4 \
    --num_workers 6 \
    --cache_dir dataset/cache/cowpea_curv26 \
    --prior_type scaffold \
    --vis_every 25 --vis_samples 3 \
    --use-wandb --wandb-project part-flow-matching --wandb-group fm-scaleup-180m \
    --checkpoint_dir diffusion_based/checkpoints/fm_scaleup_180m

echo "Training finished at $(date)"
