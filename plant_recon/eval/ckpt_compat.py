"""Remap paths stored inside checkpoints written before the 2026-09-16 repository restructure.

A checkpoint carries the `args` of the run that produced it, and the eval scripts read the VAE paths out
of them. Runs from before the restructure recorded `diffusion_based/checkpoints/...`, which no longer
exists (the package is `plant_recon/`, checkpoints live in `outputs/checkpoints/`, logs in
`outputs/logs/`), so loading any older checkpoint -- including the best lineage, v10_cam ep160 -- fails
with FileNotFoundError. Newer checkpoints already carry the new paths and pass through untouched.
"""
import os
from typing import Any, Dict

_REMAP = (
    ("diffusion_based/checkpoints/", "outputs/checkpoints/"),
    ("slurm_scripts/logs/", "outputs/logs/"),
    ("diffusion_based/", "plant_recon/"),
    ("real_world/", "use_cases/real_world/"),
)


def fix_path(p: str) -> str:
    """Rewrite one stored path to its post-restructure location, if the original is gone."""
    if not isinstance(p, str) or not p or os.path.exists(p):
        return p
    for old, new in _REMAP:
        if p.startswith(old):
            cand = new + p[len(old):]
            if os.path.exists(cand):
                return cand
    return p


def fix_ckpt_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Remap every path-shaped value in a checkpoint's stored args, in place."""
    for k, v in list(args.items()):
        if isinstance(v, str) and ("/" in v):
            args[k] = fix_path(v)
    return args
