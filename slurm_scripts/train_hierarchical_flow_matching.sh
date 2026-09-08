#!/bin/bash
#SBATCH --job-name=hierarchical_fm
#SBATCH --output=slurm_scripts/logs/hierarchical_fm_%j.log
#SBATCH --error=slurm_scripts/logs/hierarchical_fm_%j.log
#SBATCH --account=geminigrp
#SBATCH --partition=gpu-6000_ada-h
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -e

REPO_ROOT="/home/lion397/codes/image-to-l-system"
PYTHON_BIN="/home/lion397/.conda/envs/digital-crops/bin/python"
TORCHRUN_BIN="/home/lion397/.conda/envs/digital-crops/bin/torchrun"

mkdir -p "${REPO_ROOT}/slurm_scripts/logs"
mkdir -p "${REPO_ROOT}/diffusion_based/checkpoints/hierarchical_latent_fm"
cd ${REPO_ROOT}

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GPU count detection
NPROC=${SLURM_GPUS_ON_NODE:-$(nvidia-smi --list-gpus | wc -l)}
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)

# Python Runtime Profiling: Automatically probes model parameters and autograd activations
# to achieve safe GPU VRAM utilization on any GPU architecture (H100, A100, RTX 6000 Ada).
# Can be manually overridden via FORCE_BATCH_SIZE=<int>.
BATCH_ARG=${FORCE_BATCH_SIZE:-48}
TARGET_RATIO=0.88

echo "================================================================================"
echo "Starting Hierarchical Matryoshka Botanical Flow Matching Training"
echo "Job ID: $SLURM_JOB_ID | Host: $(hostname) | GPUs allocated: $NPROC"
echo "Per-GPU VRAM: ${VRAM_MB} MiB | Batch Mode: ${BATCH_ARG} (Target VRAM: ${TARGET_RATIO})"
echo "Active Anchor Capacity: K=512 | Fine Slots per Anchor: M=8 (Max 4,096 slots)"
echo "Date: $(date)"
echo "================================================================================"

# Pick a dynamic port to avoid collision
MASTER_PORT=$(shuf -i 29500-29999 -n 1)

# Checkpoint Resume configuration
EXTRA_ARGS=""
if [ -n "${INIT_CHECKPOINT}" ] && [ -f "${INIT_CHECKPOINT}" ]; then
    EXTRA_ARGS="--init_checkpoint ${INIT_CHECKPOINT}"
    if [ "${RESUME:-1}" = "1" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --resume"
    fi
    echo "Resuming from checkpoint: ${INIT_CHECKPOINT}"
fi

${TORCHRUN_BIN} --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
    diffusion_based/training/train_hierarchical_flow_matching.py \
    ${EXTRA_ARGS} \
    --data_dir dataset/helios_data/cowpea \
    --cache_dir dataset/cache/cowpea_curv26 \
    --output_dir diffusion_based/checkpoints/hierarchical_latent_fm \
    --epochs 500 \
    --batch_size "${BATCH_ARG}" \
    --target_vram_ratio "${TARGET_RATIO}" \
    --lr 2e-4 \
    --node_dim 16 \
    --organ_vae_checkpoint diffusion_based/checkpoints/organ_vae/organ_latent_vae_best.pt \
    --max_anchors 512 \
    --slots_per_anchor 8 \
    --embed_dim 384 \
    --vit_layers 8 \
    --vit_heads 8 \
    --coarse_layers 4 \
    --fine_layers 6 \
    --depth_weight 0.5 \
    --color_weight 0.2 \
    --silhouette_weight 2.0 \
    --render_sub_batch 4 \
    --render_ratio 0.25 \
    --save_every 25 \
    --eval_every 1 \
    --wandb_project part-flow-matching \
    --wandb_run_name "hierarchical-3stage-cascaded-cowpea"

echo "Training Completed at $(date)"
