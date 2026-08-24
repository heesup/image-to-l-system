#!/bin/bash
# =============================================================================
# Unified End-to-End SLURM Pipeline: Helios XML Synthesis + GPU 26D Sharding
# =============================================================================
# Accelerates dataset generation across a fleet of SLURM 'low' partition GPU nodes.
# Performs both:
#   [Phase 1] Helios C++ Simulation & XML Synthesis (DAP 1~100 × 100 Seeds)
#             -> dataset/helios_data/<species>/
#   [Phase 2] GPU Multi-Arch Rendering & 26D Tensor Sharding (100K samples)
#             -> dataset/helios_data/<species>_shard/
#
# Usage:
#   ./slurm_scripts/generate_helios_dataset_jobs.sh --dry-run
#   ./slurm_scripts/generate_helios_dataset_jobs.sh --submit
#   ./slurm_scripts/generate_helios_dataset_jobs.sh --plant-types cowpea --seeds 100 --total-samples 100000 --submit
#   ./slurm_scripts/generate_helios_dataset_jobs.sh --skip-xml --submit      # Run only GPU sharding
#   ./slurm_scripts/generate_helios_dataset_jobs.sh --skip-shards --submit   # Run only C++ XML generation
# =============================================================================

set -e

REPO_ROOT="/home/lion397/codes/image-to-l-system"
LOGS_DIR="${REPO_ROOT}/slurm_scripts/logs"
DATASET_DIR="${REPO_ROOT}/dataset/helios_data"
PYTHON_BIN="/home/lion397/.conda/envs/digital-crops/bin/python"

# Default configuration
PLANT_TYPES="cowpea"
GENOTYPES="random"
NUM_JOBS=40
DAP_MIN=1
DAP_MAX=100
SEEDS=100
TOTAL_SAMPLES=100000
SHARD_SIZE=100
IMAGE_SIZE=512
MAX_SLOTS=4096
WORKERS_PER_NODE=4
PARTITION="low"
ACCOUNT="publicgrp"
TIME_LIMIT="08:00:00"
CPUS_PER_JOB=8
MEM_PER_JOB="32G"
EXCLUDE_NODES=""

RUN_XML=true
RUN_SHARDS=true
SUBMIT=false
DRY_RUN=false

# Command line parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        --submit)
            SUBMIT=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --exclude)
            EXCLUDE_NODES="$2"
            shift 2
            ;;
        --plant-types|--species)
            PLANT_TYPES="$2"
            shift 2
            ;;
        --genotypes)
            GENOTYPES="$2"
            shift 2
            ;;
        --num-jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --dap-min)
            DAP_MIN="$2"
            shift 2
            ;;
        --dap-max)
            DAP_MAX="$2"
            shift 2
            ;;
        --total-samples)
            TOTAL_SAMPLES="$2"
            shift 2
            ;;
        --shard-size)
            SHARD_SIZE="$2"
            shift 2
            ;;
        --image-size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --max-slots)
            MAX_SLOTS="$2"
            shift 2
            ;;
        --skip-xml)
            RUN_XML=false
            shift
            ;;
        --skip-shards)
            RUN_SHARDS=false
            shift
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--submit] [--dry-run] [--plant-types P] [--seeds S] [--total-samples N] [--max-slots M] [--skip-xml] [--skip-shards]"
            exit 1
            ;;
    esac
done

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_LOG_DIR="${LOGS_DIR}/unified_${PLANT_TYPES}_${TIMESTAMP}"
SHARDS_DIR="${DATASET_DIR}/${PLANT_TYPES}_shard"

mkdir -p "${BATCH_LOG_DIR}"
mkdir -p "${DATASET_DIR}/${PLANT_TYPES}"
mkdir -p "${SHARDS_DIR}"

TOTAL_DAPS=$((DAP_MAX - DAP_MIN + 1))
DAPS_PER_JOB=$(( (TOTAL_DAPS + NUM_JOBS - 1) / NUM_JOBS ))
SAMPLES_PER_WORKER=$(( TOTAL_SAMPLES / NUM_JOBS ))

echo "============================================================"
echo "Unified Helios Pipeline: XML Synthesis + GPU 26D Sharding"
echo "============================================================"
echo "Plant Types:       ${PLANT_TYPES}"
echo "Genotypes:         ${GENOTYPES}"
echo "Total DAPs:        ${DAP_MIN} to ${DAP_MAX} (${TOTAL_DAPS} DAPs × ${SEEDS} seeds)"
echo "Parallel Jobs:     ${NUM_JOBS}"
echo "DAPs per Job:      ~${DAPS_PER_JOB}"
echo "Total Shard Goal:  ${TOTAL_SAMPLES} samples (${SAMPLES_PER_WORKER} / worker)"
echo "Max Organ Slots:   ${MAX_SLOTS}"
echo "Raw XML Output:    ${DATASET_DIR}/${PLANT_TYPES}"
echo "Shard Output:      ${SHARDS_DIR}"
echo "Partition:         ${PARTITION} (gres=gpu:1)"
echo "Account:           ${ACCOUNT}"
echo "Batch Log Dir:     ${BATCH_LOG_DIR}"
echo "Run Phases:        XML=${RUN_XML}, Shards=${RUN_SHARDS}"
echo "============================================================"
echo ""

JOB_FILES=()

for ((job_idx=0; job_idx<NUM_JOBS; job_idx++)); do
    JOB_DAP_START=$(( DAP_MIN + (job_idx * TOTAL_DAPS) / NUM_JOBS ))
    JOB_DAP_END=$(( DAP_MIN + ((job_idx + 1) * TOTAL_DAPS) / NUM_JOBS - 1 ))
    if [[ $JOB_DAP_END -gt $DAP_MAX ]]; then
        JOB_DAP_END=$DAP_MAX
    fi
    if [[ $JOB_DAP_START -gt $DAP_MAX ]]; then
        JOB_DAP_START=$DAP_MAX
    fi
    if [[ $JOB_DAP_END -lt $JOB_DAP_START ]]; then
        JOB_DAP_END=$JOB_DAP_START
    fi

    JOB_NAME="helios_pipe_${job_idx}_dap${JOB_DAP_START}-${JOB_DAP_END}"
    JOB_SCRIPT="${BATCH_LOG_DIR}/job_${job_idx}.sh"
    JOB_LOG="${BATCH_LOG_DIR}/${JOB_NAME}_%j.log"

    EXCLUDE_DIRECTIVE=""
    if [[ -n "$EXCLUDE_NODES" ]]; then
        EXCLUDE_DIRECTIVE="#SBATCH --exclude=${EXCLUDE_NODES}"
    fi

    cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_JOB}
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM_PER_JOB}
#SBATCH --time=${TIME_LIMIT}
${EXCLUDE_DIRECTIVE}
#SBATCH --output=${JOB_LOG}
#SBATCH --error=${JOB_LOG}

echo "============================================================"
echo "SLURM Job: \${SLURM_JOB_NAME} (ID: \${SLURM_JOB_ID})"
echo "Worker:    ${job_idx} / $((NUM_JOBS - 1))"
echo "Node:      \${SLURM_NODELIST}"
echo "Partition: \${SLURM_JOB_PARTITION}"
echo "Start:     \$(date)"
echo "Plant:     ${PLANT_TYPES} | DAP Range: ${JOB_DAP_START}-${JOB_DAP_END} (${SEEDS} seeds)"
echo "Target:    ${SAMPLES_PER_WORKER} shard samples (Slots: ${MAX_SLOTS})"
echo "============================================================"

if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

cd ${REPO_ROOT}

# 1. GPU Health Check & Fault-Tolerant Auto-Resubmit
GPU_HEALTH_OK=true
if ! command -v nvidia-smi &> /dev/null; then
    GPU_HEALTH_OK=false
elif ! nvidia-smi &> /dev/null || nvidia-smi 2>&1 | grep -qiE "Failed to get device handle|Unknown Error|No devices were found|GPU is lost"; then
    GPU_HEALTH_OK=false
elif ! ${PYTHON_BIN} -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() > 0; torch.zeros(1).cuda()" &> /dev/null; then
    GPU_HEALTH_OK=false
fi

if [[ "\$GPU_HEALTH_OK" == false ]]; then
    echo ""
    echo "============================================================"
    echo "⚠️ [FAULTY GPU DETECTED] Allocated GPU on node \${SLURM_NODELIST} is unresponsive or broken!"

    SENTINEL_FILE="${BATCH_LOG_DIR}/.tarpit_${job_idx}"
    echo "\${SLURM_JOB_ID}" > "\${SENTINEL_FILE}"

    # Loop: keep re-submitting until a replacement verifiably completes all shards
    ATTEMPT=0
    while true; do
        ATTEMPT=\$((ATTEMPT + 1))
        echo "1. [Attempt \${ATTEMPT}] Re-submitting replacement job for ${JOB_NAME}..."
        NEW_JOB_ID=\$(sbatch "${JOB_SCRIPT}" | grep -oP 'Submitted batch job \K\d+')
        if [[ -z "\${NEW_JOB_ID}" ]]; then
            echo "   ERROR: sbatch failed to return a job ID. Retrying in 120s..."
            sleep 120
            continue
        fi

        # Derive the log path SLURM will write for the replacement job
        NEW_LOG="${BATCH_LOG_DIR}/${JOB_NAME}_\${NEW_JOB_ID}.log"
        echo "   Replacement job submitted: \${NEW_JOB_ID}"
        echo "   Watching log: \${NEW_LOG}"
        echo "2. Holding faulty GPU slot — polling until replacement verifiably succeeds..."
        echo "============================================================"

        # Wait for the replacement job to leave the SLURM queue
        while squeue -j "\${NEW_JOB_ID}" -h &> /dev/null; do
            sleep 60
        done

        # Job left the queue — check WHY by inspecting its log
        if [[ -f "\${NEW_LOG}" ]] && grep -q "successfully completed all phases" "\${NEW_LOG}"; then
            echo "[Tarpit] ✅ Replacement job \${NEW_JOB_ID} completed successfully. Releasing faulty GPU slot."
            rm -f "\${SENTINEL_FILE}"
            exit 1
        else
            # Determine failure reason from log
            REASON="unknown"
            if [[ -f "\${NEW_LOG}" ]]; then
                if grep -qi "CANCELLED\|preempt" "\${NEW_LOG}"; then
                    REASON="preempted/cancelled"
                elif grep -qi "OOM\|out of memory\|Killed" "\${NEW_LOG}"; then
                    REASON="OOM"
                elif grep -qi "FAULTY GPU DETECTED" "\${NEW_LOG}"; then
                    REASON="landed on another faulty GPU"
                elif grep -qi "Error\|Traceback" "\${NEW_LOG}"; then
                    REASON="runtime error"
                fi
            else
                REASON="log file not found (job may not have started)"
            fi
            echo "[Tarpit] ⚠️  Replacement job \${NEW_JOB_ID} did NOT complete successfully (reason: \${REASON})."
            echo "         Re-submitting a new replacement in 60s..."
            sleep 60
        fi
    done
fi



echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:\${PYTHONPATH}"

# -----------------------------------------------------------------------------
# Phase 1: Helios C++ Plant Simulation & XML Synthesis
# -----------------------------------------------------------------------------
if [[ "${RUN_XML}" == true ]]; then
    echo ""
    echo ">>> [Phase 1/2] Synthesizing Helios C++ Plant Structures (DAP ${JOB_DAP_START}-${JOB_DAP_END}, ${SEEDS} seeds)..."
    ${PYTHON_BIN} scripts/generate_helios_dataset.py \\
        --plant-types "${PLANT_TYPES}" \\
        --genotypes "${GENOTYPES}" \\
        --dap-min ${JOB_DAP_START} \\
        --dap-max ${JOB_DAP_END} \\
        --seeds ${SEEDS} \\
        --workers ${WORKERS_PER_NODE} \\
        --output-dir "${DATASET_DIR}"
    
    XML_STATUS=\$?
    if [[ \$XML_STATUS -ne 0 ]]; then
        echo "Error: Phase 1 XML generation exited with code \${XML_STATUS}"
        exit \$XML_STATUS
    fi
fi

# -----------------------------------------------------------------------------
# Phase 2: Python GPU Differentiable 26D Tensor Sharding
# -----------------------------------------------------------------------------
if [[ "${RUN_SHARDS}" == true ]]; then
    echo ""
    echo ">>> [Phase 2/2] Generating 26D Flow Matching Tensor Shards (Worker ${job_idx}/${NUM_JOBS}, ${SAMPLES_PER_WORKER} samples)..."
    ${PYTHON_BIN} diffusion_based/dataset/generate_tensor_shards.py \\
        --species "${PLANT_TYPES}" \\
        --data-root "${DATASET_DIR}" \\
        --output-dir "${SHARDS_DIR}" \\
        --total-samples ${TOTAL_SAMPLES} \\
        --num-workers ${NUM_JOBS} \\
        --worker-id ${job_idx} \\
        --shard-size ${SHARD_SIZE} \\
        --image-size ${IMAGE_SIZE} \\
        --max-slots ${MAX_SLOTS} \\
        --max-templates 30 \\
        --device cuda

    SHARD_STATUS=\$?
    if [[ \$SHARD_STATUS -ne 0 ]]; then
        echo "Error: Phase 2 Tensor Sharding exited with code \${SHARD_STATUS}"
        exit \$SHARD_STATUS
    fi
fi

echo ""
echo "============================================================"
echo "Worker ${job_idx} successfully completed all phases at \$(date)"
echo "============================================================"
exit 0
EOF

    chmod +x "$JOB_SCRIPT"
    JOB_FILES+=("$JOB_SCRIPT")
    echo "Generated job ${job_idx}: DAP ${JOB_DAP_START}-${JOB_DAP_END} -> ${JOB_SCRIPT}"
done

echo ""
echo "Created ${#JOB_FILES[@]} unified SLURM job scripts."
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo "DRY RUN complete. To submit, run with --submit"
    exit 0
fi

if [[ "$SUBMIT" == true ]]; then
    echo "Submitting jobs to SLURM '${PARTITION}' partition..."
    submitted_count=0
    failed_count=0

    for job_file in "${JOB_FILES[@]}"; do
        output=$(sbatch "$job_file" 2>&1 || true)
        if echo "$output" | grep -q "Submitted batch job"; then
            job_id=$(echo "$output" | grep -oP 'Submitted batch job \K\d+')
            echo "  ✓ Submitted: $(basename "$job_file") (Job ID: ${job_id})"
            submitted_count=$((submitted_count + 1))
        else
            echo "  ✗ Failed to submit $(basename "$job_file"): $output"
            failed_count=$((failed_count + 1))
        fi
    done

    echo ""
    echo "============================================================"
    echo "Submission Summary: ${submitted_count} submitted, ${failed_count} failed"
    echo "Monitor:   squeue -u $USER"
    echo "Logs:      ${BATCH_LOG_DIR}"
    echo "Shards:    ${SHARDS_DIR}"
    echo "============================================================"
else
    echo "Run with --submit to submit all ${#JOB_FILES[@]} jobs to SLURM."
fi
