#!/bin/bash
# =============================================================================
# Generate and Submit SLURM Array Jobs for Cowpea 100K Dataset Synthesis
# =============================================================================
# GPU 렌더링을 사용해 Cowpea 100K 데이터셋을 클러스터에서 분산 생성합니다.
# low 파티션의 GPU 노드를 gres=gpu:1 로 요청하여 NVIDIA GPU 렌더링을 보장합니다.
#
# Usage:
#   ./slurm_scripts/generate_cowpea_100k_jobs.sh --dry-run
#   ./slurm_scripts/generate_cowpea_100k_jobs.sh --submit
#   ./slurm_scripts/generate_cowpea_100k_jobs.sh --num-workers 40 --total-samples 100000 --submit
#   ./slurm_scripts/generate_cowpea_100k_jobs.sh --partition gpu-6000_ada-h --account geminigrp --submit
# =============================================================================

set -e

REPO_ROOT="/home/lion397/codes/image-to-l-system"
LOGS_DIR="${REPO_ROOT}/slurm_scripts/logs"
PYTHON_BIN="/home/lion397/.conda/envs/digital-crops/bin/python"

# Default configuration
TOTAL_SAMPLES=100000
NUM_WORKERS=40
OUTPUT_DIR="${REPO_ROOT}/dataset/cache_cowpea_100k"
SHARD_SIZE=100
IMAGE_SIZE=128
MAX_SLOTS=512
PARTITION="low"
ACCOUNT="publicgrp"
TIME_LIMIT="02:00:00"
CPUS_PER_JOB=4
MEM_PER_JOB="16G"

# Command line parsing
SUBMIT=false
DRY_RUN=false

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
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --total-samples)
            TOTAL_SAMPLES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
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
            echo "Usage: $0 [--submit] [--dry-run] [--num-workers N] [--total-samples N] [--partition P] [--account A]"
            exit 1
            ;;
    esac
done

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_LOG_DIR="${LOGS_DIR}/cowpea_100k_${TIMESTAMP}"
mkdir -p "${BATCH_LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

SAMPLES_PER_WORKER=$(( TOTAL_SAMPLES / NUM_WORKERS ))

echo "============================================================"
echo "SLURM Cowpea 100K Dataset GPU Array Generation"
echo "============================================================"
echo "Total Samples:     ${TOTAL_SAMPLES}"
echo "Workers (Array):   ${NUM_WORKERS}"
echo "Samples/Worker:    ${SAMPLES_PER_WORKER}"
echo "Output Dir:        ${OUTPUT_DIR}"
echo "Partition:         ${PARTITION}  (gres=gpu:1)"
echo "Account:           ${ACCOUNT}"
echo "Batch Log Dir:     ${BATCH_LOG_DIR}"
echo "============================================================"
echo ""

# Generate a single array job script (cleaner than N individual scripts)
ARRAY_SCRIPT="${BATCH_LOG_DIR}/cowpea_100k_array.sh"
ARRAY_LOG="${BATCH_LOG_DIR}/worker_%A_%a.log"

cat > "${ARRAY_SCRIPT}" << EOF
#!/bin/bash
#SBATCH --job-name=cowpea_100k
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --array=0-$((NUM_WORKERS - 1))
#SBATCH --cpus-per-task=${CPUS_PER_JOB}
#SBATCH --mem=${MEM_PER_JOB}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${ARRAY_LOG}
#SBATCH --error=${ARRAY_LOG}

echo "============================================================"
echo "SLURM Job: \${SLURM_JOB_NAME} (Array ID: \${SLURM_JOB_ID})"
echo "Worker:    \${SLURM_ARRAY_TASK_ID} / $((NUM_WORKERS - 1))"
echo "Node:      \${SLURM_NODELIST}"
echo "Partition: \${SLURM_JOB_PARTITION}"
echo "Start:     \$(date)"
echo "============================================================"

if command -v nvidia-smi &> /dev/null; then
    echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:\${PYTHONPATH}"
cd ${REPO_ROOT}

${PYTHON_BIN} diffusion_based/dataset/generate_cowpea_100k.py \\
    --total-samples ${TOTAL_SAMPLES} \\
    --num-workers ${NUM_WORKERS} \\
    --worker-id \${SLURM_ARRAY_TASK_ID} \\
    --output-dir ${OUTPUT_DIR} \\
    --shard-size ${SHARD_SIZE} \\
    --image-size ${IMAGE_SIZE} \\
    --max-slots ${MAX_SLOTS} \\
    --device cuda

EXIT_CODE=\$?
echo "Worker \${SLURM_ARRAY_TASK_ID} finished with exit code \${EXIT_CODE} at \$(date)"
exit \${EXIT_CODE}
EOF

chmod +x "${ARRAY_SCRIPT}"
echo "Generated array job script -> ${ARRAY_SCRIPT}"
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo "DRY RUN complete. To submit, run with --submit"
    echo "Submit command: sbatch ${ARRAY_SCRIPT}"
    exit 0
fi

if [[ "$SUBMIT" == true ]]; then
    echo "Submitting ${NUM_WORKERS}-worker GPU array job to SLURM '${PARTITION}' partition..."
    output=$(sbatch "${ARRAY_SCRIPT}" 2>&1 || true)
    if echo "$output" | grep -q "Submitted batch job"; then
        job_id=$(echo "$output" | grep -oP 'Submitted batch job \K\d+')
        echo "  ✓ Submitted array job (Job ID: ${job_id}, Workers: ${NUM_WORKERS})"
        echo ""
        echo "============================================================"
        echo "Monitor: squeue -u \$USER"
        echo "Logs:    ${BATCH_LOG_DIR}/worker_${job_id}_<worker_id>.log"
        echo "Cancel:  scancel ${job_id}"
        echo "============================================================"
    else
        echo "  ✗ Failed to submit: ${output}"
        exit 1
    fi
else
    echo "Run with --submit to launch all ${NUM_WORKERS} GPU workers."
    echo "Submit command: sbatch ${ARRAY_SCRIPT}"
fi
