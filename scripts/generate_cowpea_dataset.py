#!/usr/bin/env python3
"""
[DEPRECATED] This script has been superseded by `scripts/generate_helios_dataset.py`.

`scripts/generate_helios_dataset.py` provides:
  - Multi-species support (cowpea, bean, sorghum, soybean, maize)
  - Genotype archetype presets (bush, spreading, vine, dwarf, tall)
  - Subfolder organization per species (e.g. dataset/helios_data/cowpea/)
  - Dynamic shoot & phytomer phenotype parameter distributions

This file is maintained as a backward-compatible wrapper that forwards to `scripts/generate_helios_dataset.py`.
"""

import sys
import os
import warnings

warnings.warn(
    "scripts/generate_cowpea_dataset.py is DEPRECATED. "
    "Please use scripts/generate_helios_dataset.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

print("[DEPRECATED] scripts/generate_cowpea_dataset.py is deprecated. Forwarding to scripts/generate_helios_dataset.py...\n", file=sys.stderr)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HELIOS_SCRIPT = os.path.join(REPO_ROOT, "scripts", "generate_helios_dataset.py")

if __name__ == "__main__":
    import subprocess
    cmd = [sys.executable, HELIOS_SCRIPT, "--plant-types", "cowpea"] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))