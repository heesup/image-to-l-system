"""Per-plant crop pipeline for real rover/tunnel-cart images: strip the visible rig structure,
detect individual cowpea plants, and build a 16-channel (RGB+pseudo-CHM, 4-zoom pyramid) tensor
matching plant_recon/dataset/generate_cache.py's exact convention so the trained model can
consume a real crop the same way it consumes a synthetic cache sample.

Rover-margin defaults come from visually inspecting real_world/data/roboflow_t4_plant_weed_seg
samples (2026-09-15): the images are nadir shots from inside a tunnel-cart rig (GEMINI "MAGIC"
imaging cart) with metal rails + LED light bars occupying the left/right edges. Roboflow's own
plant/weed polygon annotations stay within roughly x in [0.11, 0.92] of the frame (1st-99th
percentile over both classes), consistent with ~10-12% margins on each side; top/bottom show no
rig structure in the inspected samples.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# (left, right, top, bottom) as fractions of image width/height, stripped before detection.
DEFAULT_ROVER_MARGINS = (0.11, 0.14, 0.0, 0.0)

PYRAMID_ZOOMS = (1.0, 2.0, 4.0, 8.0)
REFERENCE_WINDOW_SIZE = 1.2  # matches generate_cache.py's zoom-1x window == 1.2x the plant's own extent


def crop_rover_margins(img: Image.Image, margins: Tuple[float, float, float, float] = DEFAULT_ROVER_MARGINS) -> Image.Image:
    left, right, top, bottom = margins
    w, h = img.size
    box = (int(w * left), int(h * top), int(w * (1 - right)), int(h * (1 - bottom)))
    return img.crop(box)


@dataclass
class PlantDetection:
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 in the (margin-cropped) image
    center: Tuple[float, float]
    conf: float
    cls: int
    mask: Optional[np.ndarray] = None  # (H, W) bool, same resolution as the (margin-cropped) image


def detect_plants(img: Image.Image, weights_path: str, conf: float = 0.25, plant_class_id: int = 0) -> List[PlantDetection]:
    """Runs the fine-tuned YOLO11n-seg detector (real_world/detector/train_yolo_detector.py
    output) and keeps only the 'plant' class (id 0 in data.yaml: [plant, weed]) — weeds are
    detected too but excluded from the crops fed to the reconstruction pipeline. Also carries
    each detection's segmentation mask (the model is trained as -seg): Approach 2
    (real_world/eval/run_approach2_refine.py) uses it as a cleaner silhouette target than
    thresholding the noisy pseudo-depth channel."""
    from ultralytics import YOLO
    model = YOLO(weights_path)
    res = model.predict(img, conf=conf, verbose=False)[0]
    out = []
    if res.boxes is None:
        return out
    masks = None
    if res.masks is not None:
        masks = torch.nn.functional.interpolate(res.masks.data.unsqueeze(1).float(), size=(img.height, img.width),
                                                  mode="bilinear", align_corners=False).squeeze(1) > 0.5
    for i, (box, c, cls) in enumerate(zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist())):
        if int(cls) != plant_class_id:
            continue
        x1, y1, x2, y2 = box
        mask = masks[i].cpu().numpy() if masks is not None else bbox_to_mask((x1, y1, x2, y2), img.width, img.height)
        out.append(PlantDetection(bbox=(x1, y1, x2, y2), center=((x1 + x2) / 2, (y1 + y2) / 2), conf=c, cls=int(cls), mask=mask))
    return out


def bbox_to_mask(bbox: Tuple[float, float, float, float], width: int, height: int) -> np.ndarray:
    """(H, W) bool rectangle mask from a box, for detectors trained without segmentation (2026-09-16,
    the AgML gemini_plant_detection_2022 source: bounding boxes only, no polygons). Coarser than a
    real per-plant segmentation mask but still a much tighter silhouette prior than the alternative
    (thresholding the noisy pseudo-depth channel, which build_mask_pyramid falls back to when this
    is also absent) -- a rectangle at least bounds how far a refined organ can plausibly spread
    before the Dice term penalises it, which a depth threshold over a blurry Depth-Anything blob
    does not."""
    m = np.zeros((height, width), dtype=bool)
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    m[max(y1, 0):min(y2, height), max(x1, 0):min(x2, width)] = True
    return m


def _crop_square_resize(arr: np.ndarray, center: Tuple[float, float], window_px: float, out_size: int) -> np.ndarray:
    """Crops a `window_px`-wide square centred at `center` from `arr` (H, W[, C]), zero-padding
    past the image border, then resizes to (out_size, out_size). Uses torch bilinear interpolate
    (not PIL) so both uint8-range RGB and arbitrary-scale float depth resize the same way."""
    h, w = arr.shape[:2]
    cx, cy = center
    half = window_px / 2.0
    x1, y1, x2, y2 = int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))
    pad_l, pad_t = max(0, -x1), max(0, -y1)
    pad_r, pad_b = max(0, x2 - w), max(0, y2 - h)
    sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    crop = arr[sy1:sy2, sx1:sx2]
    if crop.ndim == 2:
        crop = crop[..., None]
    crop = np.pad(crop, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="constant")
    t = torch.from_numpy(crop.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
    t = torch.nn.functional.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return t.squeeze(0).permute(1, 2, 0).numpy()  # (out_size, out_size, C)


def build_pyramid_16ch(img_rgb: Image.Image, det: PlantDetection, depth_m: np.ndarray,
                        image_size: int = 128, zooms: Tuple[float, ...] = PYRAMID_ZOOMS,
                        reference_window_size: float = REFERENCE_WINDOW_SIZE) -> torch.Tensor:
    """Builds a (4*len(zooms), image_size, image_size) tensor: per zoom level, RGB normalized to
    [-1,1] (matching generate_cache.py: clamp[0,1] then (x-0.5)/0.5) concatenated with the
    pseudo-CHM depth channel (clamp(min=0), meters). `depth_m` must already be aligned to
    `img_rgb` (same crop, same resolution) — see depth_anything_calib.pseudo_chm_for_crop.
    """
    rgb = np.asarray(img_rgb.convert("RGB"), dtype=np.float32) / 255.0
    x1, y1, x2, y2 = det.bbox
    bbox_size = max(x2 - x1, y2 - y1)
    base_window = bbox_size * reference_window_size
    channel_imgs = []
    for z in zooms:
        window = base_window / z
        rgb_crop = _crop_square_resize(rgb, det.center, window, image_size)  # (S,S,3) in [0,1]
        depth_crop = _crop_square_resize(depth_m[..., None], det.center, window, image_size)  # (S,S,1)
        rgb_t = torch.from_numpy(rgb_crop).permute(2, 0, 1).clamp(0.0, 1.0)
        rgb_norm = (rgb_t - 0.5) / 0.5
        depth_t = torch.from_numpy(depth_crop).permute(2, 0, 1).clamp(min=0.0)
        channel_imgs.append(torch.cat([rgb_norm, depth_t], dim=0))
    return torch.cat(channel_imgs, dim=0).float()  # (4*len(zooms), S, S)


def build_mask_pyramid(det: PlantDetection, image_size: int = 128, zooms: Tuple[float, ...] = PYRAMID_ZOOMS,
                        reference_window_size: float = REFERENCE_WINDOW_SIZE) -> Optional[torch.Tensor]:
    """Same per-zoom crop/resize as build_pyramid_16ch, applied to the detector's own
    segmentation mask instead of RGB/depth — a cleaner real silhouette target for Approach 2's
    Dice loss than thresholding the pseudo-depth channel. Returns None if `det.mask` is absent
    (a plain-box, non-seg detector) — callers fall back to depth-threshold silhouette."""
    if det.mask is None:
        return None
    x1, y1, x2, y2 = det.bbox
    base_window = max(x2 - x1, y2 - y1) * reference_window_size
    mask_f = det.mask.astype(np.float32)
    out = []
    for z in zooms:
        crop = _crop_square_resize(mask_f[..., None], det.center, base_window / z, image_size)
        out.append(torch.from_numpy(crop).squeeze(-1).clamp(0.0, 1.0))
    return torch.stack(out, dim=0).float()  # (len(zooms), S, S)
