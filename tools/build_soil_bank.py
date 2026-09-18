"""Extract real bare-soil patches from the AgML GEMINI cowpea frames, for compositing under the
synthetic flat renders (docs/handovers/20260916-sim-to-real-assessment §2.1: the training RGB is a
flat-shaded plant on a uniform tan ground, and the measured appearance gap costs ~10 points of raw
strict P).

A patch is kept only where the frame carries no plant/weed box and no rover rig, so the bank is bare
soil at the real camera's own resolution and lighting. Patches are stored at full frame resolution
together with the frame's pixels-per-metre, so a consumer can cut a window of a given SIZE IN METRES
and resize it to the render's pixel grid -- the soil texture then sits at the same physical scale as
the plant it goes under, at any zoom level.

Scale: the rover frame is 2592 px wide and the rig margins (real_plant_crop_utils.DEFAULT_ROVER_MARGINS)
leave 75% of it, which run_multiplant_scene.py maps to --plot_width_m (1.3 m by default), so
px/m = 2592 * 0.75 / 1.3 ~ 1495. That is ~7x the cache's zoom-1x grid (1.2 m over 256 px = 213 px/m).
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from use_cases.real_world.dataset.real_plant_crop_utils import DEFAULT_ROVER_MARGINS  # noqa: E402


def boxes_from_label(path, W, H):
    """YOLO label file -> list of (x1, y1, x2, y2) in pixels of the FULL frame."""
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, w, h = (float(v) for v in p[1:5])
        out.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
    return out


def free_patches(W, H, boxes, margins, patch_px, stride_px, pad_px):
    """Axis-aligned patch_px squares inside the rig margins that miss every box by pad_px."""
    l, r, t, b = margins
    x0, x1 = int(W * l), int(W * (1 - r))
    y0, y1 = int(H * t), int(H * (1 - b))
    grown = [(bx0 - pad_px, by0 - pad_px, bx1 + pad_px, by1 + pad_px) for bx0, by0, bx1, by1 in boxes]
    out = []
    for y in range(y0, y1 - patch_px + 1, stride_px):
        for x in range(x0, x1 - patch_px + 1, stride_px):
            px1, py1 = x + patch_px, y + patch_px
            if any(not (px1 <= gx0 or x >= gx1 or py1 <= gy0 or y >= gy1) for gx0, gy0, gx1, gy1 in grown):
                continue
            out.append((x, y))
    return out


def greenness(rgb):
    """Excess-green fraction: rejects a patch that still holds leaves the boxes missed."""
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    return float(((2 * g - r - b) > 20).mean())


def warmth(rgb):
    """Fraction of pixels that look like soil (warm: R above B). The tunnel-cart rig is grey-blue metal,
    pink-white plastic and bright LED bars, none of which is warm, so a patch holding any rig structure
    scores well below a clean one -- measured on the first bank: rig patches 0.859-0.866, clean soil up to
    1.000. The rig also reaches further into the frame than the detection-time margins assume (every
    contaminated patch began at x = 285 px, just inside the 11% left margin), so this filter runs on top of
    the wider margins below rather than instead of them."""
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    return float(((r > b + 8) & (r >= g - 5)).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(REPO / "use_cases/real_world/data/agml_gemini_plant_detection_2022"))
    ap.add_argument("--splits", default="train,valid")
    ap.add_argument("--out", default=str(REPO / "dataset/soil_bank_agml"))
    ap.add_argument("--patch_px", type=int, default=768, help="patch side in FRAME pixels (~0.51 m at 1495 px/m)")
    ap.add_argument("--stride_px", type=int, default=512)
    ap.add_argument("--pad_px", type=int, default=64, help="keep this far from every plant/weed box")
    ap.add_argument("--min_warmth", type=float, default=0.98,
                    help="reject a patch whose warm-pixel fraction is below this (rig structure; see warmth())")
    ap.add_argument("--margins", default="0.20,0.22,0.02,0.02",
                    help="left,right,top,bottom margins stripped before cutting; wider than the detection-time "
                         "rover margins because the rig reaches further in than those assume")
    ap.add_argument("--max_green", type=float, default=0.02, help="reject a patch with more than this excess-green fraction")
    ap.add_argument("--per_frame", type=int, default=2)
    ap.add_argument("--max_patches", type=int, default=400)
    ap.add_argument("--plot_width_m", type=float, default=1.3)
    a = ap.parse_args()

    margins = tuple(float(v) for v in a.margins.split(","))
    out = Path(a.out); (out / "patches").mkdir(parents=True, exist_ok=True)
    kept, rejected_green, rejected_rig, frames = 0, 0, 0, 0
    px_per_m = None
    rng = np.random.default_rng(0)
    for split in a.splits.split(","):
        for img_path in sorted(glob.glob(os.path.join(a.data_dir, split, "images", "*.jpg"))):
            if kept >= a.max_patches:
                break
            im = Image.open(img_path).convert("RGB")
            W, H = im.size
            if px_per_m is None:
                # The frame's physical scale is set by the DETECTION-time convention (the rover margins
                # real_plant_crop_utils uses, mapped to --plot_width_m), not by the wider margins this
                # script cuts patches inside; widening the cut must not rescale the texture.
                px_per_m = W * (1 - DEFAULT_ROVER_MARGINS[0] - DEFAULT_ROVER_MARGINS[1]) / a.plot_width_m
            lab = os.path.join(a.data_dir, split, "labels", Path(img_path).stem + ".txt")
            cand = free_patches(W, H, boxes_from_label(lab, W, H), margins, a.patch_px, a.stride_px, a.pad_px)
            if not cand:
                continue
            frames += 1
            arr = np.asarray(im)
            rng.shuffle(cand)
            taken = 0
            for (x, y) in cand:
                if taken >= a.per_frame or kept >= a.max_patches:
                    break
                patch = arr[y:y + a.patch_px, x:x + a.patch_px]
                if greenness(patch) > a.max_green:
                    rejected_green += 1
                    continue
                if warmth(patch) < a.min_warmth:
                    rejected_rig += 1
                    continue
                Image.fromarray(patch).save(out / "patches" / f"{Path(img_path).stem}_{x}_{y}.jpg", quality=95)
                kept += 1; taken += 1
    meta = {"px_per_m": px_per_m, "patch_px": a.patch_px, "patch_m": a.patch_px / px_per_m if px_per_m else None,
            "n_patches": kept, "n_frames_used": frames, "rejected_green": rejected_green, "rejected_rig": rejected_rig,
            "min_warmth": a.min_warmth, "margins_used": list(margins),
            "source": a.data_dir, "splits": a.splits, "plot_width_m": a.plot_width_m, "margins": list(margins)}
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
