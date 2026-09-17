"""Real-photo analogue of plant_recon/dataset/part_array_dataset.py::PartArrayDataset, for
the inference-only path: one item per DETECTED plant crop, not per XML file, and with no GT
(no `nodes`/`existence_mask` ground truth exists for a real photo). Does not subclass
PartArrayDataset — none of its XML/cache machinery applies here.

Returns the subset of keys real_world/eval/run_approach1_cold.py and run_approach2_refine.py
actually read at inference: "image" (16,128,128, matching generate_cache.py's convention),
"dap" (a placeholder — Stage 1 self-predicts DAP from the image, this key is never read as
model input), "prefix", "jpeg" (bookkeeping for saved outputs).
"""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from real_world.dataset.depth_anything_calib import pseudo_chm_for_crop
from real_world.dataset.real_plant_crop_utils import (DEFAULT_ROVER_MARGINS, PlantDetection,
                                                        build_mask_pyramid, build_pyramid_16ch,
                                                        crop_rover_margins, detect_plants)


class RealFieldPlantDataset(Dataset):
    def __init__(self, image_paths: List[str], detector_weights: str,
                 margins: Tuple[float, float, float, float] = DEFAULT_ROVER_MARGINS,
                 conf: float = 0.25, camera_height_m: float = 1.5, image_size: int = 128,
                 depth_downsample: int = 3):
        """
        image_paths: source field images (pre rover-margin-crop).
        depth_downsample: Depth-Anything runs on a `1/depth_downsample`-scaled copy of the
            margin-cropped image for speed, then the CHM is upsampled back — the model's own
            depth loss/conditioning is already very low-resolution (128px), so this costs
            negligible accuracy.
        """
        self.detector_weights = detector_weights
        self.margins = margins
        self.conf = conf
        self.camera_height_m = camera_height_m
        self.image_size = image_size
        self.depth_downsample = depth_downsample
        self.items: List[Tuple[str, Image.Image, PlantDetection]] = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            cropped = crop_rover_margins(img, margins)
            dets = detect_plants(cropped, detector_weights, conf=conf)
            for det in dets:
                self.items.append((p, cropped, det))
        if not self.items:
            raise RuntimeError(f"No plants detected across {len(image_paths)} images (conf>={conf}).")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, cropped, det = self.items[i]
        small = cropped.resize((max(1, cropped.width // self.depth_downsample),
                                 max(1, cropped.height // self.depth_downsample)))
        chm_small = pseudo_chm_for_crop(small, camera_height_m=self.camera_height_m, metric=False)
        # nearest-neighbor-safe float upsample (avoid PIL's dtype restrictions on float arrays)
        chm_t = torch.from_numpy(chm_small).float()[None, None]
        chm_full = torch.nn.functional.interpolate(chm_t, size=cropped.size[::-1], mode="bilinear",
                                                     align_corners=False)[0, 0].numpy()
        image = build_pyramid_16ch(cropped, det, chm_full, image_size=self.image_size)
        mask_pyr = build_mask_pyramid(det, image_size=self.image_size)  # (4, S, S) or None
        prefix = f"{Path(path).stem}_plant{i}"
        return {
            "image": image,
            "dap": torch.tensor(30.0, dtype=torch.float32),  # placeholder; DAP is self-predicted at inference
            "prefix": prefix,
            "jpeg": path,
            "bbox": det.bbox,
            "det_conf": det.conf,
            "mask_pyramid": mask_pyr,
        }
