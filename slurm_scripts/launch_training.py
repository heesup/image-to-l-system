import subprocess
import sys

cmd = [
    "sbatch",
    "--partition=low",
    "--account=publicgrp",
    "--gres=gpu:h100:2",
    "/home/lion397/codes/image-to-l-system/slurm_scripts/train_hierarchical_flow_matching.sh"
]
print(f"Submitting: {' '.join(cmd)}")
res = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
sys.exit(res.returncode)
