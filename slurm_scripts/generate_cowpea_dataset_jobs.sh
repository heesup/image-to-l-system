#!/bin/bash
# =============================================================================
# Generate and Submit SLURM Jobs for Cowpea Dataset Generation
# =============================================================================
# Accelerates dataset generation across a fleet of SLURM 'low' partition farm nodes.
# Divides 1000 samples (DAP 1..100 × 10 seeds) across parallel GPU worker nodes.
#
# Usage:
#   ./slurm_scripts/generate_cowpea_dataset_jobs.sh --submit
#   ./slurm_scripts/generate_cowpea_dataset_jobs.sh --dry-run
#   ./slurm_scripts/generate_cowpea_dataset_jobs.sh --num-jobs 20 --submit
# =============================================================================

set -e

REPO_ROOT="/home/lion397/codes/image-to-l-system"
LOGS_DIR="${REPO_ROOT}/slurm_scripts/logs"
DATASET_DIR="${REPO_ROOT}/dataset/helios_data"
PYTHON_BIN="/home/lion397/.conda/envs/digital-crops/bin/python"

# Default configuration (10x scaled: 100 DAPs × 100 seeds = 10,000 samples)
NUM_JOBS=20
DAP_MIN=1
DAP_MAX=100
SEEDS=100
WORKERS_PER_NODE=4
PARTITION="low"
ACCOUNT="publicgrp"
TIME_LIMIT="08:00:00"
CPUS_PER_JOB=8
MEM_PER_JOB="32G"

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
        --num-jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
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
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--submit] [--dry-run] [--num-jobs N] [--seeds S] [--partition P] [--account A]"
            exit 1
            ;;
    esac
done

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_LOG_DIR="${LOGS_DIR}/batch_${TIMESTAMP}"
mkdir -p "${BATCH_LOG_DIR}"
mkdir -p "${DATASET_DIR}"

TOTAL_DAPS=$((DAP_MAX - DAP_MIN + 1))
DAPS_PER_JOB=$(( (TOTAL_DAPS + NUM_JOBS - 1) / NUM_JOBS ))

echo "=============================================="
echo "SLURM Cowpea Dataset Batch Generation"
echo "=============================================="
echo "Total DAPs: ${DAP_MIN} to ${DAP_MAX} (${TOTAL_DAPS} DAPs × ${SEEDS} seeds = $((TOTAL_DAPS * SEEDS)) samples)"
echo "Parallel SLURM Jobs: ${NUM_JOBS}"
echo "DAPs per Job: ~${DAPS_PER_JOB}"
echo "Partition: ${PARTITION}"
echo "Account: ${ACCOUNT}"
echo "Batch Log Dir: ${BATCH_LOG_DIR}"
echo "Dataset Dir: ${DATASET_DIR}"
echo "=============================================="
echo ""

JOB_FILES=()

for ((job_idx=0; job_idx<NUM_JOBS; job_idx++)); do
    JOB_DAP_START=$(( DAP_MIN + job_idx * DAPS_PER_JOB ))
    JOB_DAP_END=$(( JOB_DAP_START + DAPS_PER_JOB - 1 ))
    if [[ $JOB_DAP_END -gt $DAP_MAX ]]; then
        JOB_DAP_END=$DAP_MAX
    fi

    if [[ $JOB_DAP_START -gt $DAP_MAX ]]; then
        break
    fi

    JOB_NAME="cowpea_ds_${job_idx}_dap${JOB_DAP_START}-${JOB_DAP_END}"
    JOB_SCRIPT="${BATCH_LOG_DIR}/job_${job_idx}_dap${JOB_DAP_START}-${JOB_DAP_END}.sh"
    JOB_OUT="${BATCH_LOG_DIR}/${JOB_NAME}_%j.out"
    JOB_ERR="${BATCH_LOG_DIR}/${JOB_NAME}_%j.err"

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
#SBATCH --output=${JOB_OUT}
#SBATCH --error=${JOB_ERR}

echo "=============================================="
echo "SLURM Job: \${SLURM_JOB_NAME} (ID: \${SLURM_JOB_ID})"
echo "Node: \${SLURM_NODELIST}"
echo "Partition: \${SLURM_JOB_PARTITION}"
echo "Start: \$(date)"
echo "DAP Range: ${JOB_DAP_START} to ${JOB_DAP_END} (${SEEDS} seeds)"
echo "=============================================="

if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

cd ${REPO_ROOT}

if command -v nvidia-smi &> /dev/null; then
    echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
fi

echo "Starting generation..."
${PYTHON_BIN} scripts/generate_cowpea_dataset.py \\
    --dap-min ${JOB_DAP_START} \\
    --dap-max ${JOB_DAP_END} \\
    --seeds ${SEEDS} \\
    --workers ${WORKERS_PER_NODE} \\
    --output-dir ${DATASET_DIR}

EXIT_CODE=\$?
echo "Job finished with exit code \${EXIT_CODE} at \$(date)"
exit \${EXIT_CODE}
EOF

    chmod +x "$JOB_SCRIPT"
    JOB_FILES+=("$JOB_SCRIPT")
    echo "Generated job ${job_idx}: DAP ${JOB_DAP_START}-${JOB_DAP_END} ($(( (JOB_DAP_END - JOB_DAP_START + 1) * SEEDS )) samples) -> ${JOB_SCRIPT}"
done

echo ""
echo "Created ${#JOB_FILES[@]} SLURM job scripts."
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
    echo "=============================================="
    echo "Submission Summary: ${submitted_count} submitted, ${failed_count} failed"
    echo "Monitor with: squeue -u $USER"
    echo "Log files: ${BATCH_LOG_DIR}"
    echo "=============================================="
else
    echo "Run with --submit to submit all ${#JOB_FILES[@]} jobs to SLURM."
fi
