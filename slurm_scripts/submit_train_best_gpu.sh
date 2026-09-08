#!/bin/bash
# =============================================================================
# Smart SLURM Dispatcher: Selects the Best Available GPU Partition & Submits
# =============================================================================
# Evaluates cluster resource availability in real time:
#   1. publicgrp / low with H100 (if idle)
#   2. publicgrp / low with A100 (if idle)
#   3. geminigrp / gpu-6000_ada-h with RTX 6000 Ada (primary workhorse)
#
# Supports native SLURM dependencies (e.g. --dependency=afterok:job1:job2...)
# The inner script train_part_fm_twostage_hungarian.sh will automatically detect
# the allocated GPU VRAM and scale batch size to fully saturate GPU memory!
# =============================================================================

REPO_ROOT="/home/lion397/codes/image-to-l-system"
TRAIN_SCRIPT="${REPO_ROOT}/slurm_scripts/train_hierarchical_flow_matching.sh"

echo "============================================================"
echo "SLURM Intelligent GPU Dispatcher"
echo "============================================================"

# Default candidate
TARGET_PARTITION="gpu-6000_ada-h"
TARGET_ACCOUNT="geminigrp"
TARGET_GRES="gpu:4"
CHOSEN_REASON="Default high-throughput RTX 6000 Ada node"

# 1. Check if H100 on low partition has free GPUs
H100_ALLOC=$(scontrol show node gpu-10-58 2>/dev/null | grep "AllocTRES=" | grep -o "gres/gpu=[0-9]*" | cut -d= -f2 || echo 4)
if [[ "${H100_ALLOC:-4}" -le 2 ]]; then
    TARGET_PARTITION="low"
    TARGET_ACCOUNT="publicgrp"
    TARGET_GRES="gpu:h100:2"
    CHOSEN_REASON="H100 80GB node (gpu-10-58) has ${H100_ALLOC}/4 allocated; using remaining 2 GPUs"
fi

# 2. Check if A100 on low partition has free GPUs (gpu-3-38, gpu-5-46, gpu-4-56)
if [[ "${TARGET_PARTITION}" != "low" ]]; then
    for A100_NODE in gpu-3-38 gpu-5-46; do
        A100_ALLOC=$(scontrol show node ${A100_NODE} 2>/dev/null | grep "AllocTRES=" | grep -o "gres/gpu=[0-9]*" | cut -d= -f2 || echo 4)
        if [[ "${A100_ALLOC:-4}" -eq 0 ]]; then
            TARGET_PARTITION="low"
            TARGET_ACCOUNT="publicgrp"
            TARGET_GRES="gpu:a100:4"
            CHOSEN_REASON="A100 80GB node (${A100_NODE}) is completely idle (0/4 allocated)"
            break
        fi
    done
fi

echo "Selected Allocation:"
echo "  Partition: ${TARGET_PARTITION}"
echo "  Account:   ${TARGET_ACCOUNT}"
echo "  Gres:      ${TARGET_GRES}"
echo "  Reason:    ${CHOSEN_REASON}"
echo "============================================================"

# Pass any extra arguments (e.g. --dependency=afterok:...) directly to sbatch
SUBMIT_CMD=(
    sbatch
    --partition="${TARGET_PARTITION}"
    --account="${TARGET_ACCOUNT}"
    --gres="${TARGET_GRES}"
    "$@"
    "${TRAIN_SCRIPT}"
)

echo "Executing: ${SUBMIT_CMD[*]}"
"${SUBMIT_CMD[@]}"
