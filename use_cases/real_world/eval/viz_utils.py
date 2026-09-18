"""Shared depth-channel colorization for the real_world eval scripts and comparison figure.
One fixed color scale across all panels (real input, Approach 1, Approach 2 before/after) so
depth maps are visually comparable — in particular, a degenerate flat/inflated prediction (the
known "canvas inflation" failure mode, see docs/experiments/20260915-real-image-first-test/20260915-real-image-first-test.md)
shows up as a near-uniform color patch rather than being auto-scaled to look textured.
"""
import numpy as np

# Canopy-height convention already used elsewhere in the project (ground=0, canopy up to
# ~0.5 m; docs/todo/phase3_advanced_depth_and_deformable_vision.md, depth_anything_calib.py's
# assumed_max_canopy_m default) — reused here as the fixed colorbar range.
DEPTH_VMAX_M = 0.5
DEPTH_CMAP = "viridis"  # muted, print-safe, perceptually uniform (pinned figure-style convention)


def colorize_depth(depth: np.ndarray, vmax: float = DEPTH_VMAX_M, cmap: str = DEPTH_CMAP) -> np.ndarray:
    """depth: (H, W) float, meters, 0 = ground/no data. Returns (H, W, 3) uint8."""
    import matplotlib
    d = np.nan_to_num(np.asarray(depth, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.clip(d / vmax, 0.0, 1.0)
    rgba = matplotlib.colormaps[cmap](norm)
    return (rgba[..., :3] * 255).astype(np.uint8)


# Muted plant-green on a near-white ground, distinct from the depth colormap so a reader never
# confuses the two single-channel panels at a glance.
MASK_FG_COLOR = (46, 125, 50)
MASK_BG_COLOR = (245, 245, 240)


def colorize_mask(mask: np.ndarray, fg=MASK_FG_COLOR, bg=MASK_BG_COLOR) -> np.ndarray:
    """mask: (H, W) float/bool in [0, 1] — the detector segmentation mask used as the Dice-loss
    foreground target (use_cases/real_world/dataset/real_plant_crop_utils.py::build_mask_pyramid).
    Returns (H, W, 3) uint8, foreground tinted green on a light background."""
    m = np.clip(np.nan_to_num(np.asarray(mask, dtype=np.float32), nan=0.0), 0.0, 1.0)[..., None]
    fg_a = np.asarray(fg, dtype=np.float32)
    bg_a = np.asarray(bg, dtype=np.float32)
    out = bg_a * (1 - m) + fg_a * m
    return out.astype(np.uint8)
