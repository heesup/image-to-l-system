"""Pseudo-CHM depth for real RGB-only crops, via Depth Anything V2.

The model's 16-channel input and the differentiable-renderer optimization target
(plant_recon/eval/eval_test_time_refinement.py) both expect a canopy-height-model (CHM)
channel: 0 at the ground, increasing with canopy height, in METERS, consistent with the
synthetic Helios camera convention (ground=0, canopy up to ~0.5 m). Real images have no depth
sensor, so this module estimates it monocularly and converts it to that convention using the
one confirmed real-camera parameter: the rover's nadir camera sits 1.5 m above the ground
(real_world/eval/run_approach2_refine.py and friends pass this as camera_height_m).

Calibration has no automatic ground-truth anchor in a real photo (see the plan's open risks) —
this is a documented heuristic, not a trusted metric signal. Treat its output as a soft prior;
every optimization script that consumes it keeps a --no_depth_loss fallback to silhouette-only
supervision for exactly this reason.
"""
from typing import Optional

import numpy as np
import torch
from PIL import Image

_METRIC_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
_RELATIVE_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

_pipe_cache = {}


def _get_pipeline(model_id: str, device: Optional[str] = None):
    if model_id not in _pipe_cache:
        from transformers import pipeline
        _pipe_cache[model_id] = pipeline(
            task="depth-estimation", model=model_id,
            device=0 if (device is None and torch.cuda.is_available()) else device)
    return _pipe_cache[model_id]


def estimate_depth_map(img: Image.Image, metric: bool = True) -> np.ndarray:
    """Returns a (H, W) float32 array. If metric=True: camera-to-surface distance in meters
    (Depth Anything's Metric-Indoor checkpoint). If metric=False: an unscaled relative map
    where LARGER values mean CLOSER to the camera (Depth Anything's default relative convention)."""
    model_id = _METRIC_MODEL_ID if metric else _RELATIVE_MODEL_ID
    pipe = _get_pipeline(model_id)
    out = pipe(img)
    depth = np.array(out["predicted_depth"] if "predicted_depth" not in out or not hasattr(out["predicted_depth"], "cpu")
                      else out["predicted_depth"].squeeze().cpu().numpy())
    if depth.shape[:2] != (img.height, img.width):
        depth = np.array(Image.fromarray(depth).resize((img.width, img.height), Image.BILINEAR))
    return depth.astype(np.float32)


def calibrate_to_chm(depth_map: np.ndarray, camera_height_m: float = 1.5, metric: bool = True,
                      ground_percentile: float = 5.0, canopy_percentile: float = 99.0,
                      assumed_max_canopy_m: float = 0.5) -> np.ndarray:
    """Converts a raw depth map to canopy-height-meters, ground≈0, matching the synthetic CHM
    convention. Two paths:
      metric=True: height = camera_height_m - depth_from_camera (direct, since the metric
        checkpoint already reports meters). Clamped to [0, camera_height_m].
      metric=False: no physical units exist, so an affine fit anchors the LOW-percentile tail
        of the map (Depth Anything's relative convention: larger value = closer to the camera,
        so the low tail is the farthest pixels, i.e. assumed soil) to height 0, and the
        HIGH-percentile tail (closest pixels, assumed canopy top) to `assumed_max_canopy_m` —
        a coarse heuristic, not a measurement; empirically (2026-09-15, a t4_plant_weed_seg
        sample) the RELATIVE checkpoint resolves individual leaf structure clearly while the
        Metric-Indoor checkpoint (trained on room-scale Hypersim) keys off the rail structure
        and reads the canopy as flat — prefer metric=False for this macro top-down crop scene.
    Either way the result is clamped to [0, camera_height_m] and NaN/inf-safe.
    """
    d = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
    if metric:
        height = camera_height_m - d
    else:
        d_ground = np.percentile(d, ground_percentile)   # low tail: farthest == ground
        d_canopy = np.percentile(d, canopy_percentile)   # high tail: closest == tallest canopy
        span = max(d_canopy - d_ground, 1e-6)
        height = (d - d_ground) / span * assumed_max_canopy_m
    return np.clip(height, 0.0, camera_height_m).astype(np.float32)


def pseudo_chm_for_crop(img_rgb: Image.Image, camera_height_m: float = 1.5, metric: bool = False) -> np.ndarray:
    """One-shot: RGB crop -> calibrated canopy-height map (H, W) meters. metric=False (the
    relative checkpoint) is the empirically-better default for this scene type; see
    calibrate_to_chm's docstring."""
    raw = estimate_depth_map(img_rgb, metric=metric)
    return calibrate_to_chm(raw, camera_height_m=camera_height_m, metric=metric)
