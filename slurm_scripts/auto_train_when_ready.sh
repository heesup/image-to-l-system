#!/bin/bash
# =============================================================================
# Automated Dataset Completion & 100% Top-View Differentiable Training Launcher
# =============================================================================
# Monitors 40 parallel Helios synthesis jobs until all 10,000 samples are built.
# Automatically transitions GPU resources from prototype job 38142469 to the
# production 10,000-sample training run with 100% full-batch Top-View Dice rendering.
# =============================================================================

REPO_ROOT="/home/lion397/codes/image-to-l-system"
DATASET_DIR="${REPO_ROOT}/dataset/helios_data/cowpea"
CACHE_DIR="${REPO_ROOT}/dataset/cache/cowpea_curv26"
LOGS_DIR="${REPO_ROOT}/slurm_scripts/logs"
OLD_TRAIN_JOB_ID="38142469"
TARGET_XML_COUNT=10000

echo "================================================================================"
echo "Helios Dataset Watcher & Auto-Train Orchestrator"
echo "Target XML Goal: ${TARGET_XML_COUNT} samples"
echo "Monitoring 40 parallel synthesis jobs (38142663 - 38142702)..."
echo "Start Time: $(date)"
echo "================================================================================"

while true; do
    XML_COUNT=$(ls -1 "${DATASET_DIR}"/*.xml 2>/dev/null | wc -l)
    CACHE_COUNT=$(ls -1 "${CACHE_DIR}"/*.pt 2>/dev/null | wc -l)
    
    # Check running SLURM generator jobs
    GEN_JOBS_RUNNING=$(squeue -u $USER 2>/dev/null | grep -c "helios_pipe" || echo 0)
    
    echo "[$(date +'%T')] Dataset Status: XMLs=${XML_COUNT}/${TARGET_XML_COUNT} | Cached=${CACHE_COUNT} | Active Workers=${GEN_JOBS_RUNNING}"
    
    if [[ "${XML_COUNT}" -ge "${TARGET_XML_COUNT}" ]] || [[ "${GEN_JOBS_RUNNING}" -eq 0 && "${XML_COUNT}" -gt 9500 ]]; then
        echo "================================================================================"
        echo "✅ Dataset Generation Complete! Total XMLs: ${XML_COUNT}"
        echo "================================================================================"
        break
    fi
    
    sleep 30
done

# Step 2: Gracefully stop the 1k prototype training run if still active
if squeue -u $USER -j "${OLD_TRAIN_JOB_ID}" -h 2>/dev/null | grep -q "${OLD_TRAIN_JOB_ID}"; then
    echo "Stopping 1k prototype training job ${OLD_TRAIN_JOB_ID} to release 2x H100 GPUs..."
    scancel "${OLD_TRAIN_JOB_ID}"
    sleep 10
fi

# Step 3: Launch production training job with 10k samples & 100% Top-View Dice rendering
echo "Submitting 10,000-sample production training job with full-batch Top-View differentiable loss..."
cd "${REPO_ROOT}"
bash "${REPO_ROOT}/slurm_scripts/submit_train_best_gpu.sh"

echo "================================================================================"
echo "Pipeline Transition Finished at $(date)"
echo "================================================================================"
