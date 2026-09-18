#!/bin/bash
#SBATCH --job-name=real_plant_detector
#SBATCH --output=outputs/logs/real_plant_detector_%j.log
#SBATCH --error=outputs/logs/real_plant_detector_%j.log
#SBATCH --account=geminigrp
#SBATCH --partition=gpu-6000_ada-h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

# Fine-tunes a lightweight YOLO11n-seg detector on the Roboflow t4_plant_weed_seg export
# (159 real nadir tunnel-cart images, classes: plant/weed) to localize individual cowpea
# plants for use_cases/real_world/dataset/real_field_dataset.py. Small dataset -> short job.
#   sbatch scripts/train_real_plant_detector.sh
# Weights + curves land under outputs/logs/real_plant_detector/ (see --name below);
# move/rename per run if training more than once.

set -e
# Repo root. Under sbatch the script runs from a SPOOLED COPY
# (/var/spool/slurmd/job<ID>/slurm_script), so "$(dirname "${BASH_SOURCE[0]}")/.." resolves to
# /var/spool/slurmd and not to the repo. That made every SLURM submission die on the first mkdir
# with "Permission denied" -- silently, since the failure looked like a filesystem problem rather
# than a path bug (2026-09-17; introduced by the d78796f restructure, which replaced a hardcoded
# absolute path with the BASH_SOURCE form that only works for a local `bash scripts/...` run).
# SLURM_SUBMIT_DIR is the directory sbatch was invoked from. Try it first, fall back to
# BASH_SOURCE for local runs, and in both cases walk up to the repo marker so a submission made
# from a subdirectory still resolves. Fail loudly rather than guess.
_repo_root_from() {
    local d="${1:-}"
    [ -n "$d" ] || return 1
    d="$(cd "$d" 2>/dev/null && pwd)" || return 1
    while [ -n "$d" ] && [ "$d" != "/" ]; do
        if [ -d "$d/plant_recon" ] && [ -d "$d/scripts" ]; then
            printf '%s\n' "$d"
            return 0
        fi
        d="$(dirname "$d")"
    done
    return 1
}
REPO_ROOT="$(_repo_root_from "${SLURM_SUBMIT_DIR:-}")" \
    || REPO_ROOT="$(_repo_root_from "$(dirname "${BASH_SOURCE[0]}")/..")" \
    || { echo "ERROR: cannot locate the repo root. Tried SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-}'" \
              "and '$(dirname "${BASH_SOURCE[0]}")/..'. Submit with 'sbatch -D /path/to/image-to-l-system ...'" \
              "or run the script from inside the repo." >&2; exit 1; }
PYTHON_BIN="/home/lion397/.conda/envs/digital-crops/bin/python"
cd "$REPO_ROOT"

"$PYTHON_BIN" use_cases/real_world/detector/train_yolo_detector.py \
    --data "${DATA_YAML:-use_cases/real_world/data/roboflow_t4_plant_weed_seg/1/data.yaml}" \
    --weights "${WEIGHTS:-outputs/weights/yolo11n-seg.pt}" \
    --epochs "${EPOCHS:-150}" \
    --imgsz "${IMGSZ:-1280}" \
    --batch "${BATCH:-8}" \
    --project "outputs/logs" \
    --name "real_plant_detector_${SLURM_JOB_ID:-local}"
