#!/bin/bash
#SBATCH --job-name=hierarchical_fm
#SBATCH --output=slurm_scripts/logs/hierarchical_fm_%j.log
#SBATCH --error=slurm_scripts/logs/hierarchical_fm_%j.log
#SBATCH --account=geminigrp
#SBATCH --partition=gpu-6000_ada-h
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00

# Defaults below are the CURRENT recipe (v9 lineage, 2026-09-14): PhytomerVAE
# v9_tl_rw4_20k + terminal-last packet cache _pkt_v9, lr 1e-4, batch 48/GPU,
# render loss from epoch 11 on 1/6 of the batch (3% was a throughput compromise
# from when one rendered plant cost 0.65 s; batched, 8 plants cost ~0.1 s), a
# checkpoint every 5 epochs and
# an eval every epoch (30-min floor). Every knob is an env override, e.g.
#   mkdir -p slurm_scripts/logs/$(date +%Y%m%d) && sbatch --output=slurm_scripts/logs/$(date +%Y%m%d)/hierarchical_fm_%j.log \
#          slurm_scripts/train_hierarchical_flow_matching.sh                       # plain: this recipe, 2 GPUs, geminigrp; log in today's folder
#   sbatch --partition=low --account=publicgrp --gres=gpu:a100:4 --time=7-00:00:00 \
#          --requeue --export=ALL,AUTO_RESUME=1 slurm_scripts/train_hierarchical_flow_matching.sh
#   INIT_CHECKPOINT=<...>/hierarchical_fm_epoch_015.pt RESUME=1 sbatch ...           # continue a lineage elsewhere
#   TRAIN_VAE=1 sbatch slurm_scripts/train_hierarchical_flow_matching.sh             # train a fresh VAE first, then FM with it
# The v8 recipe (bottom-to-top packets) needs PHYTOMER_VAE_CHECKPOINT=.../phytomer_vae_v8/...,
# PKT_CACHE_DIR=dataset/cache/cowpea_curv26_pkt and PHYTOMER_TERMINAL_LAST=0 together; never mix.

set -e

REPO_ROOT="/home/lion397/codes/image-to-l-system"
PYTHON_BIN="/home/lion397/.conda/envs/digital-crops/bin/python"
TORCHRUN_BIN="/home/lion397/.conda/envs/digital-crops/bin/torchrun"

# Python Runtime Profiling: Automatically probes model parameters and autograd activations
# to achieve safe GPU VRAM utilization on any GPU architecture (H100, A100, RTX 6000 Ada).
# Batch size: FORCE_BATCH_SIZE=<int> per GPU (default 48, the setting the v9 lineage was validated with);
# FORCE_BATCH_SIZE=auto = runtime VRAM probe (target_vram_ratio of total VRAM).
BATCH_ARG=${FORCE_BATCH_SIZE:-48}
TARGET_RATIO=0.88

# Image backbone for scaling A/B (see diffusion_based/models/dinov2_ray_encoder.py):
#   dinov2_vits14 (control) | dinov2_vitb14 | dinov2_vitl14 |
#   dinov3_vits16 | dinov3_vitb16 | dinov3_vitl16 | dinov3_vitl16_sat
BACKBONE=${BACKBONE:-dinov2_vits14}
OUTPUT_DIR=${OUTPUT_DIR:-diffusion_based/checkpoints/hierarchical_fm_v9}
EPOCHS=${EPOCHS:-500}
SAVE_EVERY=${SAVE_EVERY:-5}
FREEZE_BACKBONE=${FREEZE_BACKBONE:-1}
FREEZE_ARGS=""
if [ "${FREEZE_BACKBONE}" = "1" ]; then
    FREEZE_ARGS="--freeze_backbone"
fi

DETECT_ANOMALY_ARGS=""
if [ "${DETECT_ANOMALY:-0}" = "1" ]; then
    DETECT_ANOMALY_ARGS="--detect_anomaly"
fi
# STAGE3_GEOMETRY=1: Stage 3 generates the child's position (relative to its fixed
# parent), roll and scale in the flow state with the latent (design doc §2.1;
# the 2026-09-14 GT-substitution ablation put node position first among the
# per-node errors). A latent-only checkpoint widens on load (latent block kept).
STAGE3_ARGS=""
if [ "${STAGE3_GEOMETRY:-0}" = "1" ]; then
    STAGE3_ARGS="--stage3_geometry --stage3_geom_weight ${STAGE3_GEOM_WEIGHT:-4.0}"
fi
# STAGE3_GT_NODES=1: teacher forcing -- Stage 3 conditioned on the GT node geometry
# of matched nodes (upper bound of the latent path; a diagnostic arm, not a recipe).
# STAGE3_GT_NODES_P / STAGE3_GT_NODES_JITTER_CM: scheduled teacher forcing (GT node with
# probability P per matched node, Gaussian jitter on the GT position); defaults = pure.
if [ "${STAGE3_GT_NODES:-0}" = "1" ]; then
    STAGE3_ARGS="${STAGE3_ARGS} --stage3_gt_nodes --stage3_gt_nodes_p ${STAGE3_GT_NODES_P:-1.0} --stage3_gt_nodes_jitter_cm ${STAGE3_GT_NODES_JITTER_CM:-0.0}"
fi
# RENDER_TO_LATENT=1: the render loss also trains Stage 3's shape latent (default: flow loss only).
if [ "${RENDER_TO_LATENT:-0}" = "1" ]; then
    STAGE3_ARGS="${STAGE3_ARGS} --render_to_latent"
fi
# LATENT_NORM=1: flow-match the standardized VAE latent (unit variance per dim).
if [ "${LATENT_NORM:-0}" = "1" ]; then
    STAGE3_ARGS="${STAGE3_ARGS} --latent_norm"
fi
# EVAL_SET_FILE: score every arm / subset on the same plants (matched by prefix, force-included in subsets).
# HOLDOUT_PER_BUCKET: stratified held-out plants removed from training and reported as [Holdout] each eval.
if [ -n "${EVAL_SET_FILE:-}" ]; then
    STAGE3_ARGS="${STAGE3_ARGS} --eval_set_file ${EVAL_SET_FILE}"
fi
if [ "${HOLDOUT_PER_BUCKET:-0}" != "0" ]; then
    STAGE3_ARGS="${STAGE3_ARGS} --holdout_samples_per_bucket ${HOLDOUT_PER_BUCKET}"
fi
# NODE_TOKEN_WINDOW (default 1): WxW mean-pooled node-local token; T0_FRAC (default 0): fraction of each batch at t=0.
if [ "${NODE_TOKEN_WINDOW:-1}" != "1" ]; then
    STAGE3_ARGS="${STAGE3_ARGS} --node_token_window ${NODE_TOKEN_WINDOW}"
fi
if [ "${T0_FRAC:-0}" != "0" ]; then
    STAGE3_ARGS="${STAGE3_ARGS} --t0_frac ${T0_FRAC}"
fi
# EMA_DECAY=<0.999>: keep an EMA of the weights and save <checkpoint>_ema.pt for evaluation (0/unset = off).
if [ "${EMA_DECAY:-0}" != "0" ]; then
    STAGE3_ARGS="${STAGE3_ARGS} --ema_decay ${EMA_DECAY}"
fi
# RENDER_INPUT_CAMERA=1: render the training loss in the cached input's camera frame (GT plant bbox centre) instead of the origin window.
if [ "${RENDER_INPUT_CAMERA:-0}" = "1" ]; then
    STAGE3_ARGS="${STAGE3_ARGS} --render_input_camera 1"
fi
# MULTIZOOM=1: all four cache zoom levels as image tokens (design doc §2.7).
if [ "${MULTIZOOM:-0}" = "1" ]; then
    STAGE3_ARGS="${STAGE3_ARGS} --multizoom"
fi
# COVERAGE_WEIGHT / EXIST_COUNT_WEIGHT: Stage 2 node coverage (GT centre -> nearest active node) and
# soft active-count losses (design doc §2.7); 0 = off.
if [ "${COVERAGE_WEIGHT:-0}" != "0" ]; then
    STAGE3_ARGS="${STAGE3_ARGS} --coverage_weight ${COVERAGE_WEIGHT}"
fi
if [ "${EXIST_COUNT_WEIGHT:-0}" != "0" ]; then
    STAGE3_ARGS="${STAGE3_ARGS} --exist_count_weight ${EXIST_COUNT_WEIGHT}"
fi

mkdir -p "${REPO_ROOT}/slurm_scripts/logs"
cd ${REPO_ROOT}
mkdir -p "${OUTPUT_DIR}"

# Per-run artifact folder: the self-consistency panels live here beside this
# run's own log, so runs stay comparable instead of overwriting each other at
# fixed filenames under docs/results/assets. SLURM will not create a directory
# for --output, so the log is written where it always was and symlinked in
# rather than moved (it is appended to for the life of the job).
RUN_TAG="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
# Logs are organized by start date (Heesup, 2026-09-15): this run's panels go under logs/<YYYYMMDD>/run_<tag>/.
# The SBATCH --output path above is static; submit with --output=slurm_scripts/logs/$(date +%Y%m%d)/hierarchical_fm_%j.log
# (directory created here) or let tools/organize_logs.py file top-level logs into their date folder later.
LOG_DAY="$(date +%Y%m%d)"
mkdir -p "${REPO_ROOT}/slurm_scripts/logs/${LOG_DAY}"
RUN_DIR="${REPO_ROOT}/slurm_scripts/logs/${LOG_DAY}/run_${RUN_TAG}"
mkdir -p "${RUN_DIR}"
if [ -n "${SLURM_JOB_ID}" ]; then
    # relative link: valid whether the job log sits at the top level or in this date folder
    if [ -f "${REPO_ROOT}/slurm_scripts/logs/${LOG_DAY}/hierarchical_fm_${SLURM_JOB_ID}.log" ]; then
        ln -sfn "../hierarchical_fm_${SLURM_JOB_ID}.log" "${RUN_DIR}/run.log"
    else
        ln -sfn "../../hierarchical_fm_${SLURM_JOB_ID}.log" "${RUN_DIR}/run.log"
    fi
fi
echo "Run artifacts: ${RUN_DIR} (self-consistency panels + run.log symlink)"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Packet leaflet order must match the VAE and the cache: "1" (terminal leaflet
# last) for v9_tl_* VAEs and _pkt_v9; "0" (bottom-to-top) for v8 and _pkt.
export PHYTOMER_TERMINAL_LAST=${PHYTOMER_TERMINAL_LAST:-1}

# GPU count detection
NPROC=${SLURM_GPUS_ON_NODE:-$(nvidia-smi --list-gpus | wc -l)}

# TRAIN_VAE=1: train the PhytomerVAE first, in this same allocation, then run
# FM with it. FM encodes the Stage 3 target latents on the fly from the cached
# packets (2026-09-14), so a new VAE needs no packet-cache rebuild; only the
# packet order must agree (PHYTOMER_TERMINAL_LAST, exported above). Default 0:
# the VAE is a shared frozen component -- retraining it per FM run
# re-randomises the latent space and makes runs incomparable, so retrain it
# when the packet format changes, not per run. The recipe below is the one
# v9_tl_rw4_20k was trained with (rot weight 4, 20,000 files, 120 epochs).
if [ "${TRAIN_VAE:-0}" = "1" ]; then
    VAE_DIR=${VAE_CHECKPOINT_DIR:-diffusion_based/checkpoints/phytomer_vae_$(date +%Y%m%d_%H%M)}
    mkdir -p "${VAE_DIR}"
    echo ">>> [VAE] training PhytomerVAE-${VAE_LATENT_DIM:-128}D into ${VAE_DIR} (files ${VAE_MAX_FILES:-20000}, epochs ${VAE_EPOCHS:-120}, rot weight ${VAE_ROT_WEIGHT:-4}, terminal-last ${PHYTOMER_TERMINAL_LAST})"
    ${PYTHON_BIN} diffusion_based/training/train_phytomer_vae.py \
        --cache-dir "${CACHE_DIR:-dataset/cache/cowpea_curv26}" \
        --latent-dim "${VAE_LATENT_DIM:-128}" \
        --residual-dim "${VAE_RESIDUAL_DIM:-8}" \
        --hidden-dim 256 \
        --epochs "${VAE_EPOCHS:-120}" \
        --batch-size 4096 \
        --lr 1e-3 \
        --beta-kl 1e-3 \
        --rot-weight "${VAE_ROT_WEIGHT:-4}" \
        --max-files "${VAE_MAX_FILES:-20000}" \
        --seed "${SEED:-0}" \
        --packet-cache "${VAE_DIR}/packets_${VAE_MAX_FILES:-20000}files.pt" \
        --checkpoint-dir "${VAE_DIR}" \
        --device cuda:0
    PHYTOMER_VAE_CHECKPOINT="${VAE_DIR}/phytomer_vae_${VAE_LATENT_DIM:-128}d_best.pt"
    echo ">>> [VAE] done at $(date): ${PHYTOMER_VAE_CHECKPOINT}"
fi
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)

echo "================================================================================"
echo "Starting Hierarchical Matryoshka Botanical Flow Matching Training"
echo "Job ID: $SLURM_JOB_ID | Host: $(hostname) | GPUs allocated: $NPROC"
echo "Per-GPU VRAM: ${VRAM_MB} MiB | Batch Mode: ${BATCH_ARG} (Target VRAM: ${TARGET_RATIO})"
echo "Phytomer Capacity: 512 phytomers x ${SLOTS_PER_PHYTOMER:-10} slots/phytomer"
echo "Render gate: fast warmup bypass (epochs 1-3) -> active at epoch ${RENDER_GRAD_START_EPOCH:-11}"
echo "Eval cadence: every ${EVAL_EVERY:-1} epochs OR every ${EVAL_MIN_INTERVAL_MINUTES:-30} min (time fallback)"
echo "Render fraction: ${RENDER_FRACTION:-0.167} of batch per step (batch-relative; 2-scale pyramid 1x/2x — profiling 2026-09-09: render was 70% of step time)${RENDER_FRACTION_FINAL:+ -> ramping to ${RENDER_FRACTION_FINAL} over ${RENDER_FRACTION_RAMP_EPOCHS:-0} epochs}"
echo "Flow granularity: ${FLOW_GRANULARITY:-phytomer} (hybrid decoupled: 128D VAE latent flow + Stage 2 3D scaffold)"
echo "Phytomer VAE: ${PHYTOMER_VAE_CHECKPOINT:-diffusion_based/checkpoints/phytomer_vae_v9_tl_rw4_20k/phytomer_vae_128d_best.pt}"
echo "Pkt cache dir: ${PKT_CACHE_DIR:-dataset/cache/cowpea_curv26_pkt_v9} (missing samples fall back to on-the-fly) | PHYTOMER_TERMINAL_LAST=${PHYTOMER_TERMINAL_LAST}"
echo "LR: ${LR:-1e-4} | Save every ${SAVE_EVERY} epochs | Stage 3 geometry: ${STAGE3_GEOMETRY:-0} | GT nodes: ${STAGE3_GT_NODES:-0} (p ${STAGE3_GT_NODES_P:-1.0}, jitter ${STAGE3_GT_NODES_JITTER_CM:-0.0} cm) | render->latent: ${RENDER_TO_LATENT:-0} | latent norm: ${LATENT_NORM:-0} | coverage w: ${COVERAGE_WEIGHT:-0} | multizoom: ${MULTIZOOM:-0} | token window: ${NODE_TOKEN_WINDOW:-1} | t0 frac: ${T0_FRAC:-0} | count w: ${EXIST_COUNT_WEIGHT:-0} | Git: $(git rev-parse --short HEAD 2>/dev/null)"
echo "EMA decay: ${EMA_DECAY:-0} (0 = off)"
echo "Render camera: ${RENDER_INPUT_CAMERA:-0} (1 = cached input's GT-bbox-centred frame, 0 = origin window)"
echo "Backbone: ${BACKBONE}${FREEZE_ARGS:+ (frozen)} | Output: ${OUTPUT_DIR} | Epochs: ${EPOCHS}"
echo "Train subset: ${MAX_TRAIN_SAMPLES:-0} (0 = full dataset) | eval set file: ${EVAL_SET_FILE:-none} | holdout/bucket: ${HOLDOUT_PER_BUCKET:-0}"
echo "Date: $(date)"
echo "================================================================================"

# Pick a dynamic port to avoid collision
MASTER_PORT=$(shuf -i 29500-29999 -n 1)

# Checkpoint Resume configuration
EXTRA_ARGS=""
if [ -n "${SEED}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --seed ${SEED}"
    echo "Seeded run: ${SEED} (init + shuffling reproducible; needed to iterate on intermittent failures)"
fi
# AUTO_RESUME=1: on a (re)start pick up the newest checkpoint in OUTPUT_DIR, so a
# preempted/requeued job on a preemptible partition (e.g. `low`) continues
# instead of restarting; falls back to INIT_CHECKPOINT (or scratch) when none.
if [ "${AUTO_RESUME:-0}" = "1" ]; then
    # newest RAW checkpoint (the <checkpoint>_ema.pt files written with EMA_DECAY carry no optimizer state)
    LATEST_CKPT=$(ls -t "${OUTPUT_DIR}"/hierarchical_fm_epoch_*.pt 2>/dev/null | grep -v '_ema\.pt$' | head -1)
    if [ -n "${LATEST_CKPT}" ]; then
        INIT_CHECKPOINT="${LATEST_CKPT}"
        RESUME=1
        echo "AUTO_RESUME: continuing from ${INIT_CHECKPOINT}"
    fi
fi
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
    --cache_dir "${CACHE_DIR:-dataset/cache/cowpea_curv26}" \
    --output_dir "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_ARG}" \
    --target_vram_ratio "${TARGET_RATIO}" \
    --num_workers "${NUM_WORKERS:-8}" \
    --lr "${LR:-1e-4}" \
    --backbone_lr_ratio "${BACKBONE_LR_RATIO:-0.3}" \
    --phy_count_weight "${PHY_COUNT_WEIGHT:-2.0}" \
    --dap_weight "${DAP_WEIGHT:-0.05}" \
    --init_phytomer_count "${INIT_PHYTOMER_COUNT:-50.0}" \
    --node_dim 16 \
    --organ_vae_checkpoint diffusion_based/checkpoints/organ_vae/organ_latent_vae_best.pt \
    --flow_granularity "${FLOW_GRANULARITY:-phytomer}" \
    --backbone "${BACKBONE}" \
    --matcher_type "${MATCHER_TYPE:-greedy}" \
    ${FREEZE_ARGS} \
    ${DETECT_ANOMALY_ARGS} \
    ${STAGE3_ARGS} \
    --phytomer_latent_dim "${PHYTOMER_LATENT_DIM:-128}" \
    --phytomer_residual_dim "${PHYTOMER_RESIDUAL_DIM:-8}" \
    --phytomer_vae_checkpoint "${PHYTOMER_VAE_CHECKPOINT:-diffusion_based/checkpoints/phytomer_vae_v9_tl_rw4_20k/phytomer_vae_128d_best.pt}" \
    --pkt_cache_dir "${PKT_CACHE_DIR:-dataset/cache/cowpea_curv26_pkt_v9}" \
    --max_phytomers 512 \
    --slots_per_phytomer "${SLOTS_PER_PHYTOMER:-10}" \
    --embed_dim 384 \
    --vit_layers 8 \
    --vit_heads 8 \
    --coarse_layers 4 \
    --fine_layers 6 \
    --depth_weight 0.5 \
    --color_weight 0.0 \
    --silhouette_weight 1.0 \
    --render_fraction "${RENDER_FRACTION:-0.167}" \
    --render_grad_start_epoch "${RENDER_GRAD_START_EPOCH:-11}" \
    ${RENDER_FRACTION_FINAL:+--render_fraction_final "${RENDER_FRACTION_FINAL}" --render_fraction_ramp_epochs "${RENDER_FRACTION_RAMP_EPOCHS:-20}"} \
    --scale_weight "${SCALE_WEIGHT:-1.0}" \
    --save_every "${SAVE_EVERY}" \
    --eval_every "${EVAL_EVERY:-1}" \
    --eval_min_interval_minutes "${EVAL_MIN_INTERVAL_MINUTES:-30}" \
    --eval_samples_per_bucket "${EVAL_SAMPLES_PER_BUCKET:-2}" \
    --dap_buckets "${DAP_BUCKETS:-8}" \
    --max_train_samples "${MAX_TRAIN_SAMPLES:-0}" \
    --figure_dir "${RUN_DIR}" \
    --wandb_project part-flow-matching \
    --wandb_run_name "${WANDB_RUN_NAME:-hierarchical-fm-v9}"

echo "Training Completed at $(date)"
