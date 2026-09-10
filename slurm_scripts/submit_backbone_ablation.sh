#!/bin/bash
# =============================================================================
# Backbone Scaling A/B Dispatcher
# =============================================================================
# Submits one training run per image backbone with an isolated output dir and
# wandb run name, so checkpoints/panels never collide with the main run.
# Runs are chained sequentially (--dependency=afterany) by default to avoid
# contending for the same GPUs.
#
# Cached (runs out of the box):
#   dinov2_vits14     22M   (control, current)
#   dinov2_vitb14     87M
#   dinov2_vitl14    300M
#   dinov3_vitl16_sat 300M  (DINOv3 ViT-L/16, SAT-493M satellite pretraining)
# License-gated DINOv3 web weights (set DINOV3_WEIGHTS=/path/or/url first):
#   dinov3_vits16 | dinov3_vitb16 | dinov3_vitl16
#
# Usage:
#   ./slurm_scripts/submit_backbone_ablation.sh --dry-run
#   ./slurm_scripts/submit_backbone_ablation.sh --submit
#   ARMS="dinov2_vits14 dinov2_vitb14" EPOCHS=30 ./slurm_scripts/submit_backbone_ablation.sh --submit
#   FREEZE_BACKBONE=1 ./slurm_scripts/submit_backbone_ablation.sh --submit   # linear-probe arms
# =============================================================================

set -e

REPO_ROOT="/home/lion397/codes/image-to-l-system"
cd ${REPO_ROOT}

ARMS="${ARMS:-dinov2_vits14 dinov2_vitb14 dinov3_vitl16_sat}"
EPOCHS="${EPOCHS:-50}"
EVAL_EVERY="${EVAL_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-10}"
FLOW_GRANULARITY="${FLOW_GRANULARITY:-phytomer}"
SUBMIT=0
SEQUENTIAL=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit) SUBMIT=1; shift ;;
        --dry-run) SUBMIT=0; shift ;;
        --parallel) SEQUENTIAL=0; shift ;;
        --arms) ARMS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "Backbone Scaling A/B"
echo "============================================================"
echo "Arms:        ${ARMS}"
echo "Epochs:      ${EPOCHS} (eval/save every ${EVAL_EVERY}/${SAVE_EVERY})"
echo "Granularity: ${FLOW_GRANULARITY}"
echo "Sequential:  ${SEQUENTIAL} (chained via --dependency=afterany)"
echo "Freeze:      ${FREEZE_BACKBONE:-0}"
echo "Submit:      ${SUBMIT}"
echo "============================================================"

PREV_JOB=""
for ARM in ${ARMS}; do
    OUT="diffusion_based/checkpoints/ablation_${ARM}"
    NAME="ablation-${ARM}-${FLOW_GRANULARITY}-e${EPOCHS}"
    DEP=""
    if [[ "${SEQUENTIAL}" == "1" && -n "${PREV_JOB}" ]]; then
        DEP="--dependency=afterany:${PREV_JOB}"
    fi

    if [[ "${SUBMIT}" == "1" ]]; then
        echo ""
        echo ">>> Submitting ${ARM} -> ${OUT}"
        JOB_ID=$(env FLOW_GRANULARITY="${FLOW_GRANULARITY}" \
                     BACKBONE="${ARM}" \
                     OUTPUT_DIR="${OUT}" \
                     WANDB_RUN_NAME="${NAME}" \
                     EPOCHS="${EPOCHS}" \
                     EVAL_EVERY="${EVAL_EVERY}" \
                     SAVE_EVERY="${SAVE_EVERY}" \
                     bash slurm_scripts/submit_train_best_gpu.sh ${DEP} 2>&1 | tee /dev/stderr | grep -oP 'Submitted batch job \K\d+' | tail -1)
        if [[ -z "${JOB_ID}" ]]; then
            echo "ERROR: failed to submit ${ARM}"
            exit 1
        fi
        echo "    Job ${JOB_ID} | wandb: ${NAME}"
        PREV_JOB="${JOB_ID}"
    else
        echo "[dry-run] BACKBONE=${ARM} OUTPUT_DIR=${OUT} WANDB_RUN_NAME=${NAME} EPOCHS=${EPOCHS} ${DEP}"
    fi
done

echo ""
echo "Done. Monitor: squeue -u $USER"
echo "Checkpoints: diffusion_based/checkpoints/ablation_<arm>/"
