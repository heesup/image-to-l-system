#!/bin/bash
#SBATCH --job-name=fm_smoke_test
#SBATCH --output=slurm_scripts/logs/fm_smoke_%j.log
#SBATCH --error=slurm_scripts/logs/fm_smoke_test_%j.log
#SBATCH --account=publicgrp
#SBATCH --partition=low
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1:00:00

# Smoke test for the Job 38237555 deadlock fixes (commit 4b70266).
# Goal: prove grad_norm stays finite across the render-gate activation (epoch 4+)
# AND across an in-loop eval (eval_min_interval_minutes=0 -> eval fires every epoch
# from epoch 1, exercising the model.train() restore path immediately).
# PASS criteria: Epochs 1..6 complete with 0 [Recovery] lines and eval panels logged.

set -e

REPO_ROOT="/home/lion397/codes/image-to-l-system"
PYTHON_BIN="/home/lion397/.conda/envs/digital-crops/bin/python"
TORCHRUN_BIN="/home/lion397/.conda/envs/digital-crops/bin/torchrun"

REPO_ROOT="/home/lion397/codes/image-to-l-system"
PYTHON_BIN="/home/lion397/.conda/envs/digital-crops/bin/python"
TORCHRUN_BIN="/home/lion397/.conda/envs/digital-crops/bin/torchrun"

BATCH_ARG=${FORCE_BATCH_SIZE:-8}
TARGET_RATIO=0.60
BACKBONE=${BACKBONE:-dinov2_vits14}
OUTPUT_DIR=${OUTPUT_DIR:-diffusion_based/checkpoints/fm_smoke_test}
EPOCHS=6
SAVE_EVERY=100
FREEZE_BACKBONE=${FREEZE_BACKBONE:-1}
FREEZE_ARGS=""
if [ "${FREEZE_BACKBONE}" = "1" ]; then
    FREEZE_ARGS="--freeze_backbone"
fi

mkdir -p "${REPO_ROOT}/slurm_scripts/logs"
cd ${REPO_ROOT}
mkdir -p "${OUTPUT_DIR}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NPROC=${SLURM_GPUS_ON_NODE:-$(nvidia-smi --list-gpus | wc -l)}
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)

echo "================================================================================"
echo "Starting SMOKE TEST: deadlock-fix verification (commit 4b70266)"
echo "Job ID: $SLURM_JOB_ID | Host: $(hostname) | GPUs allocated: $NPROC"
echo "Per-GPU VRAM: ${VRAM_MB} MiB | Batch: ${BATCH_ARG} (ratio ${TARGET_RATIO})"
echo "Epochs: ${EPOCHS} | Render gate at ep 4 | Eval EVERY epoch (train() restore exercised)"
echo "Date: $(date)"
echo "================================================================================"

MASTER_PORT=$(shuf -i 29500-29999 -n 1)

${TORCHRUN_BIN} --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
    diffusion_based/training/train_hierarchical_flow_matching.py \
    --data_dir dataset/helios_data/cowpea \
    --cache_dir dataset/cache/cowpea_curv26 \
    --output_dir "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_ARG}" \
    --target_vram_ratio "${TARGET_RATIO}" \
    --lr 3e-4 \
    --backbone_lr_ratio 0.3 \
    --phy_count_weight 2.0 \
    --dap_weight 0.05 \
    --init_phytomer_count 50.0 \
    --node_dim 16 \
    --organ_vae_checkpoint diffusion_based/checkpoints/organ_vae/organ_latent_vae_best.pt \
    --flow_granularity phytomer \
    --backbone "${BACKBONE}" \
    --matcher_type greedy \
    ${FREEZE_ARGS} \
    --phytomer_latent_dim 128 \
    --phytomer_residual_dim 8 \
    --phytomer_vae_checkpoint diffusion_based/checkpoints/phytomer_vae_v8/phytomer_vae_128d_best.pt \
    --pkt_cache_dir dataset/cache/cowpea_curv26_pkt \
    --max_phytomers 512 \
    --slots_per_phytomer 10 \
    --embed_dim 384 \
    --vit_layers 8 \
    --vit_heads 8 \
    --coarse_layers 4 \
    --fine_layers 6 \
    --depth_weight 0.5 \
    --color_weight 0.0 \
    --silhouette_weight 1.0 \
    --render_fraction 0.167 \
    --render_grad_start_epoch 4 \
    --scale_weight 1.0 \
    --save_every "${SAVE_EVERY}" \
    --eval_every 1 \
    --eval_min_interval_minutes 0 \
    --eval_samples_per_bucket 1 \
    --dap_buckets 8 \
    --max_train_samples 512 \
    --wandb_project part-flow-matching \
    --wandb_run_name "fm-smoke-deadlock-fix-$(date +%m%d_%H%M)"

echo "SMOKE TEST COMPLETE: $(date)"