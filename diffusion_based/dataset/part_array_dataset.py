"""
Dataset for paired (rendered image, 14D part-centric PlantOrganArray tensor) samples.

The 14D part tensor layout (per organ):
    [OrganType(0), Base(1..3), Rot6D(4..9), Scale(10..12), Existence(13)]

Normalization (fixed, hand-tuned to unit-ish scale for flow matching):
    - organ type (col 0):  / 9.0  -> [0, 1]  (categorical, rounded at inference)
    - base (cols 1..3):    * 100.0 -> ~[-1, 1] (world coords are ~cm scale)
    - rot6d (cols 4..9):   unchanged (already [-1, 1])
    - scale (cols 10..12): unchanged (already [0, 1])
    - existence (col 13):  unchanged (already [0, 1])
"""

import os
import glob
from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    P14_COL_ORGAN_TYPE,
    P14_COL_BASE_X,
    P14_COL_BASE_Z,
    P14_COL_EXISTENCE,
    NUM_FEATURES_14D,
)

# Fixed normalization constants (see module docstring)
ORGAN_TYPE_SCALE = 9.0
BASE_SCALE = 100.0


class PartArrayDataset(Dataset):
    """Loads Helios XML -> 14D part tensor + rendered image."""

    def __init__(
        self,
        data_root: str,
        max_nodes: int = 2048,
        image_size: int = 128,
        device: torch.device = None,
        use_gt_renderer_image: bool = True,
        exclude_globs: List[str] = None,
        include_globs: List[str] = None,
        cache_dir: str = None,
    ):
        self.data_root = os.path.abspath(data_root)
        self.max_nodes = max_nodes
        self.image_size = image_size
        self.use_gt_renderer_image = use_gt_renderer_image
        self.node_dim = NUM_FEATURES_14D
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self._cached_renderer = None
        self._image_cache = {}
        self._tensor_cache = {}

        self._transform_tensor = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        import fnmatch as _fnmatch
        xml_paths = sorted(glob.glob(os.path.join(self.data_root, "*_plant_*.xml")))
        if include_globs:
            xml_paths = [p for p in xml_paths if any(_fnmatch.fnmatch(os.path.basename(p), pat) for pat in include_globs)]
        elif exclude_globs:
            xml_paths = [p for p in xml_paths if not any(_fnmatch.fnmatch(os.path.basename(p), pat) for pat in exclude_globs)]
        self.samples = [self._resolve_pair(p) for p in xml_paths]

    def _resolve_pair(self, xml_path: str) -> Dict[str, str]:
        prefix = os.path.basename(xml_path).split("_plant_")[0]
        xml_dir = os.path.dirname(os.path.abspath(xml_path))
        jpeg_path = ""
        for suffix in ("_vis.jpeg", "_rad.jpeg"):
            candidate = os.path.join(xml_dir, f"{prefix}{suffix}")
            if os.path.exists(candidate):
                jpeg_path = candidate
                break
        return {"xml": xml_path, "jpeg": jpeg_path, "prefix": prefix}

    def normalize(self, p14: torch.Tensor) -> torch.Tensor:
        """Normalize a (N, 14) part tensor to unit-ish scale."""
        out = p14.clone()
        out[:, P14_COL_ORGAN_TYPE] = out[:, P14_COL_ORGAN_TYPE] / ORGAN_TYPE_SCALE
        out[:, P14_COL_BASE_X:P14_COL_BASE_Z + 1] = out[:, P14_COL_BASE_X:P14_COL_BASE_Z + 1] * BASE_SCALE
        return out

    def denormalize(self, p14: torch.Tensor) -> torch.Tensor:
        """Undo normalization."""
        out = p14.clone()
        out[:, P14_COL_ORGAN_TYPE] = out[:, P14_COL_ORGAN_TYPE] * ORGAN_TYPE_SCALE
        out[:, P14_COL_BASE_X:P14_COL_BASE_Z + 1] = out[:, P14_COL_BASE_X:P14_COL_BASE_Z + 1] / BASE_SCALE
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        cache_key = sample["xml"]

        # Load (or compute) the 14D part tensor ONCE, then derive both the
        # rendered image and the padded/normalized training tensor from it.
        p14 = None
        if self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, f"{sample['prefix']}.pt")
            if os.path.exists(cache_path):
                p14 = torch.load(cache_path, map_location="cpu")
        if p14 is None:
            gt_array = PlantOrganArray.from_xml_file_typed(sample["xml"])
            p14 = gt_array.to_part_tensor_14d(device=torch.device("cpu"))

        # Image
        if cache_key in self._image_cache:
            image_tensor = self._image_cache[cache_key]
        else:
            rgb = None
            if self.cache_dir is not None:
                img_path = os.path.join(self.cache_dir, f"{sample['prefix']}_img.pt")
                if os.path.exists(img_path):
                    rgb = torch.load(img_path, map_location="cpu")
            if rgb is None:
                if self._cached_renderer is None:
                    from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
                    self._cached_renderer = HeliosPyTorchRenderer(image_size=self.image_size).to(self.device)
                gt_array = PlantOrganArray.from_xml_file_typed(sample["xml"])
                with torch.no_grad():
                    rgb = self._cached_renderer.render_part_tensor_14d(
                        p14.to(self.device), template_organ_array=gt_array, camera_height=1.0,
                        elevation_deg=90.0, device=self.device, focus_plant=True,
                        use_kinematics_tree=False, differentiable=False,
                    )
                rgb = rgb.cpu()
            image_tensor = self._transform_tensor(rgb).cpu()
            self._image_cache[cache_key] = image_tensor

        # 14D part tensor (padded + normalized)
        if cache_key in self._tensor_cache:
            nodes, existence_mask, num_nodes = self._tensor_cache[cache_key]
        else:
            N = min(p14.shape[0], self.max_nodes)
            nodes = torch.zeros((self.max_nodes, self.node_dim), dtype=torch.float32)
            nodes[:N] = p14[:N]
            nodes = self.normalize(nodes)

            existence_mask = torch.zeros(self.max_nodes, dtype=torch.float32)
            existence_mask[:N] = 1.0

            num_nodes = torch.tensor(N, dtype=torch.long)
            self._tensor_cache[cache_key] = (nodes, existence_mask, num_nodes)

        return {
            "image": image_tensor,
            "nodes": nodes,
            "existence_mask": existence_mask,
            "num_nodes": num_nodes,
            "xml_path": sample["xml"],
            "prefix": sample["prefix"],
        }
