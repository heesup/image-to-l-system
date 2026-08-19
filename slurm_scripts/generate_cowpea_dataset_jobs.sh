#!/bin/bash
# =============================================================================
# [DEPRECATED] SLURM Jobs Generator for Plant Dataset Generation
# =============================================================================
# This script has been superseded by `slurm_scripts/generate_helios_dataset_jobs.sh`.
#
# `generate_helios_dataset_jobs.sh` supports:
#   - Multi-species (cowpea, bean, sorghum, soybean, maize)
#   - Genotypes & Phenotypic variations
#   - Subfolder separation
#
# Forwarding execution to generate_helios_dataset_jobs.sh...
# =============================================================================

echo "[DEPRECATED] slurm_scripts/generate_cowpea_dataset_jobs.sh is deprecated." >&2
echo "Forwarding to slurm_scripts/generate_helios_dataset_jobs.sh..." >&2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/generate_helios_dataset_jobs.sh" "$@"
