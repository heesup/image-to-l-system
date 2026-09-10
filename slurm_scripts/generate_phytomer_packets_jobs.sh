#!/bin/bash
# =============================================================================
# Multi-node phytomer packet targets — XML-DIRECT backfill
# generate_cache.py --mode pkt reads XML directly (bypasses the image cache),
# builds packets with EXACT XML phytomer membership, batch-encodes VAE latents
# (CPU or GPU) -> dataset/cache/<crop>_curv26_pkt/
# Use this to (re)build packet targets for an existing image cache without
# re-rendering. New datasets get pkt embedded by the unified pipeline instead.
# =============================================================================
# Usage:
#   ./slurm_scripts/generate_phytomer_packets_jobs.sh --dry-run
#   ./slurm_scripts/generate_phytomer_packets_jobs.sh --submit
#   ./slurm_scripts/generate_phytomer_packets_jobs.sh --num-jobs 40 --submit   # GPU nodes (default)
#   ./slurm_scripts/generate_phytomer_packets_jobs.sh --gres "" --submit        # CPU-only fallback
# =============================================================================

set -e

REPO_ROOT="/home/lion397/codes/image-to-l-system"
LOGS_DIR="${REPO_ROOT}/slurm_scripts/logs"
PYTHON_BIN="/home/lion397/.conda/envs/digital-crops/bin/python"

DATA_DIR="dataset/helios_data/cowpea"
OUT_DIR="dataset/cache/cowpea_curv26_pkt"
VAE_CKPT="diffusion_based/checkpoints/phytomer_vae_xml/phytomer_vae_64d_best.pt"
TOOL="diffusion_based/dataset/generate_cache.py"

NUM_JOBS=40
PARTITION="low"
ACCOUNT="publicgrp"
CPUS_PER_JOB=8
GRES_PER_JOB="gpu:1"   # default GPU nodes (VAE encode is ~10x faster); set "" for CPU-only
MEM_PER_JOB="32G"
TIME_LIMIT="06:00:00"
WORKERS_PER_JOB=8
DEVICE="cuda:0"
SUBMIT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-jobs) NUM_JOBS="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --vae-checkpoint) VAE_CKPT="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --gres) GRES_PER_JOB="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --workers) WORKERS_PER_JOB="$2"; shift 2 ;;
        --submit) SUBMIT=1; shift ;;
        --dry-run) SUBMIT=0; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$GRES_PER_JOB" ]]; then
    DEVICE="cpu"
fi

mkdir -p "${LOGS_DIR}"
cd ${REPO_ROOT}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_LOG_DIR="${LOGS_DIR}/pkt_xml_${TIMESTAMP}"
mkdir -p "${BATCH_LOG_DIR}"

# Shard the XML file list across jobs.
MASTER_LIST="${BATCH_LOG_DIR}/xml_master_list.txt"
find ${DATA_DIR} -maxdepth 1 -name "*_plant_*.xml" | sort > "${MASTER_LIST}"
TOTAL=$(wc -l < "${MASTER_LIST}")
PER_JOB=$(( (TOTAL + NUM_JOBS - 1) / NUM_JOBS ))

echo "============================================================"
echo "Multi-node XML-direct pkt precompute"
echo "============================================================"
echo "XML Dir:      ${DATA_DIR} (${TOTAL} files)"
echo "Output:       ${OUT_DIR}"
echo "VAE:          ${VAE_CKPT}"
echo "Parallel Jobs: ${NUM_JOBS} (${PER_JOB} XML / job)"
echo "Partition:    ${PARTITION} (gres: ${GRES_PER_JOB:-none}, ${CPUS_PER_JOB} cpus/job)"
echo "Device:       ${DEVICE} (workers: ${WORKERS_PER_JOB})"
echo "Batch Log:    ${BATCH_LOG_DIR}"
echo "Submit:       ${SUBMIT}"
echo "============================================================"

for ((job_idx=0; job_idx<NUM_JOBS; job_idx++)); do
    START=$(( job_idx * PER_JOB ))
    END=$(( START + PER_JOB ))
    if [[ $START -ge $TOTAL ]]; then break; fi
    if [[ $END -gt $TOTAL ]]; then END=$TOTAL; fi

    JOB_NAME="pkt_xml_${job_idx}"
    JOB_SCRIPT="${BATCH_LOG_DIR}/job_${job_idx}.sh"
    JOB_LOG="${BATCH_LOG_DIR}/${JOB_NAME}_%j.log"
    LIST_FILE="${BATCH_LOG_DIR}/xml_list_${job_idx}.txt"
    sed -n "$((START + 1)),${END}p" "${MASTER_LIST}" > "${LIST_FILE}"

    GRES_DIRECTIVE=""
    if [[ -n "$GRES_PER_JOB" ]]; then
        GRES_DIRECTIVE="#SBATCH --gres=${GRES_PER_JOB}"
    fi

    cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_JOB}
#SBATCH --mem=${MEM_PER_JOB}
#SBATCH --time=${TIME_LIMIT}
${GRES_DIRECTIVE}
#SBATCH --output=${JOB_LOG}
#SBATCH --error=${JOB_LOG}

cd ${REPO_ROOT}
export PYTHONUNBUFFERED=1
export PYTHONPATH=.
echo "Job ${job_idx}: ${START}..${END} (${LIST_FILE})"
${PYTHON_BIN} ${TOOL} \
    --mode pkt \
    --data-root ${DATA_DIR} \
    --output-dir ${OUT_DIR} \
    --vae-checkpoint ${VAE_CKPT} \
    --workers ${WORKERS_PER_JOB} \
    --batch-size 128 \
    --device ${DEVICE} \
    --file-list ${LIST_FILE}
EOF

    chmod +x "$JOB_SCRIPT"
    if [[ $SUBMIT -eq 1 ]]; then
        echo "Submitting ${JOB_SCRIPT} ..."
        sbatch "$JOB_SCRIPT"
    else
        echo "[dry-run] ${JOB_SCRIPT} (${START}..${END})"
    fi
done

echo "Done. Job scripts in ${BATCH_LOG_DIR}/job_*.sh"