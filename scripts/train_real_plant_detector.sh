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
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
