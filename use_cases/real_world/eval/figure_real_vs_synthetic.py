#!/usr/bin/env python
"""Qualitative real-vs-predicted comparison grid (white-background academic style, matching
tools/compare_same_plants_figure.py's rcParams/layout): rows = real plants, one RGB + one Depth
column per stage (real crop, Approach 1 cold, Approach 2 before/after), plus one Mask column for
the real crop — the detector's own segmentation mask, i.e. the actual Dice-loss foreground
target run_approach2_refine.py optimizes against (see docs/results/20260915_real_image_first_test.md
§4.1: this mask nearly fills the frame at 4x/8x zoom, a contributor to the "canvas inflation"
failure diagnosed there). No IoU column — there is no ground truth for a real photo; the DAP
shown is Stage 1's own self-predicted estimate, flagged as such since the DAP head was trained
only on Helios-simulated appearances. All depth panels share one fixed color scale
(real_world/eval/viz_utils.DEPTH_VMAX_M) so a degenerate flat/inflated prediction is visually
obvious as a near-uniform patch, not auto-scaled away.

Reads the outputs of run_approach1_cold.py and/or run_approach2_refine.py (matched by the
shared `<prefix>` naming; each stage has a `<prefix>_<suffix>.png` / `_depth.png`, and the real
crop additionally an optional `<prefix>_input_mask.png`), `results.json` for pred_dap.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from real_world.eval.viz_utils import DEPTH_CMAP, DEPTH_VMAX_M


def _load(path):
    return Image.open(path) if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach1_dir", default="", help="run_approach1_cold.py --out directory")
    ap.add_argument("--approach2_dir", default="", help="run_approach2_refine.py --out directory")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Real cowpea field images vs. trained-pipeline reconstruction")
    a = ap.parse_args()
    if not a.approach1_dir and not a.approach2_dir:
        raise SystemExit("pass --approach1_dir and/or --approach2_dir")

    src_dir = a.approach2_dir or a.approach1_dir
    rows_meta = json.load(open(os.path.join(src_dir, "results.json")))["rows"][:a.limit]

    # (stage_key, title, source_dir, file_suffix, panel_kinds) — panel_kinds is the ordered list
    # of "<suffix>[_<kind>].png" files to show for that stage: "rgb" -> "<suffix>.png",
    # anything else -> "<suffix>_<kind>.png". The real crop gets an extra "mask" panel; the
    # predicted stages don't have an independent mask (nothing renders one), only rgb + depth.
    stages = [("input", "Real crop (rover, nadir)", src_dir, "input", ["rgb", "depth", "mask"])]
    if a.approach1_dir:
        stages.append(("a1", "Approach 1: cold generation", a.approach1_dir, "render", ["rgb", "depth"]))
    if a.approach2_dir:
        stages.append(("a2_before", "Approach 2: before refinement", a.approach2_dir, "before", ["rgb", "depth"]))
        stages.append(("a2_after", "Approach 2: after refinement", a.approach2_dir, "after", ["rgb", "depth"]))

    # flat column list: (stage_key, title, source_dir, file_suffix, kind)
    cols = [(key, title, sdir, suffix, kind) for key, title, sdir, suffix, kinds in stages for kind in kinds]
    n_cols = len(cols)
    letters = [chr(ord('a') + k) for k in range(n_cols)]

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                          "font.size": 9, "axes.edgecolor": "#444444", "axes.linewidth": 0.6, "text.color": "#111111"})
    fig, axes = plt.subplots(len(rows_meta), n_cols, figsize=(2.3 * n_cols, 2.5 * len(rows_meta)), facecolor="white")
    axes = np.atleast_2d(axes)

    for r, meta in enumerate(rows_meta):
        prefix = meta["prefix"]
        for c, (key, title, sdir, suffix, kind) in enumerate(cols):
            ax = axes[r, c]; ax.set_xticks([]); ax.set_yticks([])
            fname = f"{prefix}_{suffix}.png" if kind == "rgb" else f"{prefix}_{suffix}_{kind}.png"
            im = _load(os.path.join(sdir, fname))
            if im is not None:
                ax.imshow(im)
            else:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", color="#666666")
            if key == "input" and kind == "rgb":
                ax.set_ylabel(f"pred. DAP {meta.get('pred_dap', float('nan')):.0f}\n(Stage 1 estimate)", fontsize=8.5)
            if key == "a2_after" and kind == "rgb" and "nodes_moved_cm" in meta:
                ax.set_xlabel(f"nodes moved {meta['nodes_moved_cm']:.1f} cm", fontsize=8.5)
            if r == 0:
                ax.set_title(f"({letters[c]}) {title}\n{kind}", fontsize=9)

    fig.suptitle(a.title + f"\n(no ground truth for real photos — qualitative comparison only, {len(rows_meta)} plants)",
                 fontsize=10, y=0.998)
    plt.tight_layout(rect=(0, 0, 0.93, 0.96))

    # one shared colorbar for every depth panel (fixed scale, see viz_utils.DEPTH_VMAX_M)
    cbar_ax = fig.add_axes((0.945, 0.15, 0.012, 0.7))
    sm = cm.ScalarMappable(cmap=DEPTH_CMAP, norm=matplotlib.colors.Normalize(vmin=0, vmax=DEPTH_VMAX_M))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("canopy height (m)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)

    fig.savefig(a.out, dpi=300, bbox_inches="tight", facecolor="white")
    print("saved", a.out)


if __name__ == "__main__":
    main()
