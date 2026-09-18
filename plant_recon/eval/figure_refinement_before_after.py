"""Compose the refinement renders saved by eval_test_time_refinement.py --save_renders into one panel:
ground truth, the network's cold sample, and the refined result, one row per plant.

White-background journal style (single sans family, thin grey frames, 300 dpi, sentence-case titles),
sized to a double column so it can go straight into the paper draft.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--renders", required=True, help="folder written by --save_renders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_plants", type=int, default=6)
    ap.add_argument("--title", default="Test-time refinement against the input canopy height map")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.5,
                         "figure.facecolor": "white", "axes.facecolor": "white"})

    metas = sorted(glob.glob(os.path.join(a.renders, "*_meta.json")))
    rows = []
    for m in metas:
        d = json.load(open(m))
        idx = d["index"]
        trio = [os.path.join(a.renders, f"{idx}_{k}.png") for k in ("gt", "before", "after")]
        if all(os.path.exists(t) for t in trio):
            rows.append((d, trio))
    if not rows:
        raise SystemExit(f"no complete gt/before/after trios under {a.renders}")
    rows.sort(key=lambda r: r[0]["dap"])
    if len(rows) > a.max_plants:                       # keep a DAP-spread subset
        sel = np.linspace(0, len(rows) - 1, a.max_plants).round().astype(int)
        rows = [rows[i] for i in sel]

    fig, axes = plt.subplots(len(rows), 3, figsize=(5.2, 1.85 * len(rows)))
    axes = np.atleast_2d(axes)
    titles = ("Ground truth", "Network sample", "After refinement")
    for r, (d, trio) in enumerate(rows):
        for c, (ax, path, t) in enumerate(zip(axes[r], trio, titles)):
            ax.imshow(np.asarray(Image.open(path).convert("RGB")), interpolation="nearest")
            if r == 0:
                ax.set_title(t, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.5); s.set_color("0.45")
            if c == 0:
                ax.set_ylabel(f"DAP {d['dap']}", fontsize=8)
            if c == 2:
                ax.text(0.5, -0.07, f"IoU {d['iou_before'] * 100:.1f} → {d['iou_after'] * 100:.1f}%",
                        transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#0b0b0b")
    fig.suptitle(a.title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"{len(rows)} plants -> {a.out}")


if __name__ == "__main__":
    main()
