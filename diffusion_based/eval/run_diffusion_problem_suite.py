"""
Unified Problem Suite Entry Point (Diffusion & Backpropagation Solvers).

This script delegates to the unified problem suite in `diffusion_based/eval/run_backprop_problem_suite.py`,
providing full support for:
  - Fresh Diffusion training with Parameter MSE + Image + VGG Perceptual Loss + EMA + CFG Dropout.
  - Classifier-Free Guidance (CFG s=2.0) DDIM Reverse Sampling.
  - 3 Problem Tiers: Easy (non-relevant start / target recovery), Medium (seed expansion), Hard (random topology).
  - Pre-trained checkpoint evaluation via `--checkpoint <path>`.
  - Batch holdout evaluation via `--val_pattern <globs>`.
"""

import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.eval.run_backprop_problem_suite import (
    main,
    solve_problem_diffusion,
    optimize_backprop,
    train_diffusion_fresh,
    plot_problem,
    render_target,
    make_non_relevant_source_plant,
    make_seed_plant,
    make_random_topology,
)

if __name__ == "__main__":
    main()
