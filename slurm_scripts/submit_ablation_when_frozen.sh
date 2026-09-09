#!/bin/bash
# =============================================================================
# 100k Dataset Freeze Watcher -> Fraction-Ablation Ladder (Runs A & B)
# =============================================================================
# Waits until the Helios dataset generation finishes (all helios_pipe_* jobs done
# and cache >= --target-samples), then cancels the interim training job (whose
# sample list was frozen on a partial dataset) and submits the two fraction
# ablation runs IN PARALLEL on two 4-GPU nodes:
#
#   Run A (parity):   batch 48, render_fraction 1/6   (~8 days / 500 epochs)
#   Run B (dense):    batch 32, render_fraction 1/2   (~13 days / 500 epochs)
#
# Decision gate (day ~4-6): compare val/dice_loss, val/cos_color_loss,
# val/silhouette_iou at matched wall-clock. If Run B clearly beats Run A,
# launch Config C (batch 16, render_fraction 1.0, ~25 days) for the densest
# self-consistency signal.
#
# Usage:
#   nohup bash slurm_scripts/submit_ablation_when_frozen.sh > slurm_scripts/logs/ablation_watcher.log 2>&1 &
#   bash slurm_scripts/submit_ablation_when_frozen.sh --dry-run
#   bash slurm_scripts/submit_ablation_when_frozen.sh --force          # skip the freeze wait
#   bash slurm_scripts/submit_ablation_when_frozen.sh --cancel-interim 38146809
# =============================================================================

set -u

REPO_ROOT="/home/lion397/codes/image-to-l-system"
LOGS_DIR="${REPO_ROOT}/slurm_scripts/logs"
TRAIN_SCRIPT="${REPO_ROOT}/slurm_scripts/train_hierarchical_flow_matching.sh"
CACHE_DIR="${REPO_ROOT}/dataset/cache/cowpea_curv26"

TARGET_SAMPLES=100000
TARGET_TOLERANCE=0.99       # accept >= 99% of target (a few XMLs may fail permanently)
CHECK_INTERVAL_SEC=600      # poll every 10 minutes
DRY_RUN=0
FORCE=0
CANCEL_INTERIM=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --force) FORCE=1 ;;
        --cancel-interim) CANCEL_INTERIM="$2"; shift ;;
        --target-samples) TARGET_SAMPLES="$2"; shift ;;
        --check-interval) CHECK_INTERVAL_SEC="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p "${LOGS_DIR}"
cd "${REPO_ROOT}"

echo "================================================================================"
echo "100k Freeze Watcher & Ablation Ladder Orchestrator"
echo "Target: ${TARGET_SAMPLES} cached .pt samples (accept >= $(python3 -c "print(int(${TARGET_SAMPLES} * ${TARGET_TOLERANCE}))"))"
echo "Poll interval: ${CHECK_INTERVAL_SEC}s | Dry-run: ${DRY_RUN} | Force: ${FORCE}"
echo "Start: $(date)"
echo "================================================================================"

# ---------------------------------------------------------------------------
# Step 0: optionally cancel the interim (partially-frozen-dataset) training job
# ---------------------------------------------------------------------------
if [[ -n "${CANCEL_INTERIM}" ]]; then
    if squeue -u "$USER" -j "${CANCEL_INTERIM}" -h 2>/dev/null | grep -q .; then
        echo "Cancelling interim training job ${CANCEL_INTERIM} (frozen on partial dataset)..."
        if [[ "${DRY_RUN}" -eq 0 ]]; then
            scancel "${CANCEL_INTERIM}"
            sleep 15
        fi
    else
        echo "Interim job ${CANCEL_INTERIM} not running — nothing to cancel."
    fi
fi

# ---------------------------------------------------------------------------
# Step 1: wait for dataset freeze
# ---------------------------------------------------------------------------
freeze_reached() {
    # find(1) instead of ls glob: arg-list overflows at ~70k+ files
    local cache_count
    cache_count=$(find "${CACHE_DIR}" -name "*.pt" 2>/dev/null | wc -l)
    local pipes_running
    pipes_running=$(squeue -u "$USER" -h 2>/dev/null | grep -c "helios_pipe" || true)
    local threshold
    threshold=$(python3 -c "print(int(${TARGET_SAMPLES} * ${TARGET_TOLERANCE}))")
    local no_pipes_drained=0
    if [[ "${pipes_running}" -eq 0 ]] && [[ "${cache_count}" -gt $((threshold - 5000)) ]]; then
        no_pipes_drained=1
    fi
    if [[ "${cache_count}" -ge "${threshold}" ]] || [[ "${no_pipes_drained}" -eq 1 ]]; then
        echo "FREEZE: cache=${cache_count} (>= ${threshold}) | helios_pipe jobs running=${pipes_running}"
        return 0
    fi
    echo "waiting: cache=${cache_count}/${TARGET_SAMPLES} | helios_pipe jobs running=${pipes_running}"
    return 1
}

while true; do
    if freeze_reached; then
        break
    fi
    if [[ "${FORCE}" -eq 1 ]]; then
        echo "--force given: skipping freeze wait."
        break
    fi
    sleep "${CHECK_INTERVAL_SEC}"
done

# ---------------------------------------------------------------------------
# Step 2: sanity-check the frozen dataset (DAP balance of the final cache)
# ---------------------------------------------------------------------------
echo "Analyzing frozen dataset DAP balance..."
python3 - << 'EOF'
import glob, os, re
from collections import defaultdict
by_bucket = defaultdict(int)
for f in glob.glob("dataset/cache/cowpea_curv26/*.pt"):
    m = re.search(r"dap(\d+)", os.path.basename(f))
    d = int(m.group(1)) if m else 0
    by_bucket[(d - 1) // 10] += 1
total = sum(by_bucket.values())
print(f"  total cache: {total}")
for b in sorted(by_bucket):
    frac = by_bucket[b] / max(total, 1) * 100
    flag = " <-- SKEWED" if frac > 20 else ""
    print(f"  DAP {b*10+1:3d}-{b*10+10:3d}: {by_bucket[b]:6d} ({frac:4.1f}%){flag}")
if total < 90000:
    print("  WARNING: dataset below 90k — ablation runs will train on the partial set.")
EOF

# ---------------------------------------------------------------------------
# Step 3: submit Run A (parity) and Run B (dense) on separate nodes
# ---------------------------------------------------------------------------
SUBMIT_RUN() {
    local run_name="$1"
    local batch="$2"
    local fraction="$3"
    local log_path="${LOGS_DIR}/ablation_${run_name}_%j.log"
    local export_str="ALL,FORCE_BATCH_SIZE=${batch},RENDER_FRACTION=${fraction},WANDB_RUN_NAME=ablation_${run_name}_b${batch}_f${fraction}"

    echo "--------------------------------------------------------------"
    echo "Submitting Run ${run_name}: batch=${batch}, render_fraction=${fraction}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "  [DRY-RUN] sbatch --job-name=ablation_${run_name} --output=${log_path} --export=${export_str} ${TRAIN_SCRIPT}"
        echo "dryrun"
        return 0
    fi
    sbatch --job-name="ablation_${run_name}" \
        --output="${log_path}" \
        --error="${log_path}" \
        --export="${export_str}" \
        "${TRAIN_SCRIPT}" | grep -oP 'Submitted batch job \K\d+'
}

# Run A: parity (batch 48, f=1/6) — primary partition (RTX 6000 Ada 4x)
JID_A=$(SUBMIT_RUN "A_parity_f1_6" 48 0.167)

# Run B: dense (batch 32, f=1/2) — try H100/low partition first if free, else same partition.
# The train script picks whatever partition it is submitted to; submitting B to the
# 'low' partition (H100/A100) avoids competing with A for Ada nodes.
PARTITION_B=$(squeue -u "$USER" -h -o "%P" 2>/dev/null | head -1)
if scontrol show partition low 2>/dev/null | grep -q "State=UP"; then
    RUN_B_PARTITION="low"
else
    RUN_B_PARTITION="gpu-6000_ada-h"
fi
echo "Run B target partition: ${RUN_B_PARTITION}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    JID_B="dryrun"
    echo "  [DRY-RUN] Run B would go to partition ${RUN_B_PARTITION}"
else
    JID_B=$(sbatch --job-name="ablation_B_dense_f1_2" \
        --partition="${RUN_B_PARTITION}" \
        --account="${SLURM_ACCOUNT:-geminigrp}" \
        --output="${LOGS_DIR}/ablation_B_dense_f1_2_%j.log" \
        --error="${LOGS_DIR}/ablation_B_dense_f1_2_%j.log" \
        --export="ALL,FORCE_BATCH_SIZE=32,RENDER_FRACTION=0.5,WANDB_RUN_NAME=ablation_B_dense_b32_f0_5" \
        "${TRAIN_SCRIPT}" | grep -oP 'Submitted batch job \K\d+')
    echo "  Submitted Run B (partition ${RUN_B_PARTITION}) as job ${JID_B}"
fi

echo "================================================================================"
echo "Ablation Ladder Launched at $(date)"
echo "  Run A (parity, B48 f=1/6):  job ${JID_A:-n/a}   (~8 days / 500 ep)"
echo "  Run B (dense,  B32 f=1/2):  job ${JID_B:-n/a}   (~13 days / 500 ep)"
echo "Decision gate: compare val/dice_loss, val/cos_color_loss, val/silhouette_iou at"
echo "matched wall-clock (day ~4-6). If B clearly wins, launch Config C"
echo "(batch 16, render_fraction 1.0, ~25 days) for full-batch self-consistency."
echo "================================================================================"