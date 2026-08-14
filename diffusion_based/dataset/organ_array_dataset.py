"""
Dataset for paired (rendered image, normalized 94D PlantOrganArray tensor) samples.
"""

import os
import glob
from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from diffusion_based.models.plant_organ_array import PlantOrganArray


class OrganArrayDataset(Dataset):
    """
    Loads Helios `_vis.jpeg` images and matching `*_plant_*.xml` files,
    returning image + normalized 94D organ array tensors.

    Channel layout:
      - columns 0..92: continuous parameters, standardized by dataset mean/std
      - column 93: existence probability, kept in [0, 1]

    Args:
        data_root: directory containing *.xml and *_vis.jpeg files
        max_nodes: pad/truncate organ arrays to this length
        image_size: resize input image to (image_size, image_size)
        single_xml_path: if provided, only load this one XML (for overfit tests)
    """

    def __init__(
        self,
        data_root: str,
        max_nodes: int = 64,
        image_size: int = 256,
        single_xml_path: str = None,
        device: torch.device = None,
    ):
        self.data_root = os.path.abspath(data_root)
        self.max_nodes = max_nodes
        self.image_size = image_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self._transform_tensor = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        if single_xml_path is not None:
            self.samples = [self._resolve_pair(single_xml_path)]
        else:
            xml_paths = sorted(glob.glob(os.path.join(self.data_root, "*_plant_*.xml")))
            self.samples = [self._resolve_pair(p) for p in xml_paths]

        # Compute dataset-wide min/max bounds from ALL available XML files,
        # so a single-sample overfit test still uses stable global statistics.
        self.min_vals, self.max_vals = self._compute_global_min_max()

    def _resolve_pair(self, xml_path: str) -> Dict[str, str]:
        prefix = os.path.basename(xml_path).split("_plant_")[0]
        xml_dir = os.path.dirname(os.path.abspath(xml_path))
        jpeg_path = os.path.join(xml_dir, f"{prefix}_vis.jpeg")
        if not os.path.exists(jpeg_path):
            # Look for any *_vis.jpeg in the same directory as fallback
            candidates = sorted(glob.glob(os.path.join(xml_dir, "*_vis.jpeg")))
            if candidates:
                jpeg_path = candidates[0]
        return {"xml": xml_path, "jpeg": jpeg_path, "prefix": prefix}

    def _compute_global_min_max(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-channel min/max over every available *_plant_*.xml file."""
        xml_paths = sorted(glob.glob(os.path.join(self.data_root, "*_plant_*.xml")))
        mins = torch.full((93,), float("inf"), dtype=torch.float32)
        maxs = torch.full((93,), float("-inf"), dtype=torch.float32)

        for xml_path in xml_paths:
            try:
                organ_array = PlantOrganArray.from_xml_file(xml_path)
                tensor = organ_array.tensor[:, :93]  # exclude existence column
                if tensor.shape[0] == 0:
                    continue
                local_min = tensor.min(dim=0).values
                local_max = tensor.max(dim=0).values
                mins = torch.minimum(mins, local_min)
                maxs = torch.maximum(maxs, local_max)
            except Exception:
                continue

        # Avoid zero range
        range_vals = torch.where((maxs - mins) < 1e-6, torch.ones_like(maxs), maxs - mins)
        return mins, range_vals

    def normalize(self, nodes: torch.Tensor) -> torch.Tensor:
        """Min-max normalize continuous channels (0..92) to [0, 1]. Existence channel (93) stays in [0,1]."""
        device = nodes.device
        min_vals = self.min_vals.to(device)
        range_vals = self.max_vals.to(device)
        out = nodes.clone()
        out[:, :93] = (out[:, :93] - min_vals) / range_vals
        out[:, :93] = torch.clamp(out[:, :93], 0.0, 1.0)
        out[:, 93] = torch.clamp(out[:, 93], 0.0, 1.0)
        return out

    def denormalize(self, nodes: torch.Tensor) -> torch.Tensor:
        """Undo min-max normalization for continuous channels (0..92)."""
        device = nodes.device
        min_vals = self.min_vals.to(device)
        range_vals = self.max_vals.to(device)
        out = nodes.clone()
        out[:, :93] = out[:, :93] * range_vals + min_vals
        out[:, 93] = torch.clamp(out[:, 93], 0.0, 1.0)
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        if os.path.exists(sample["jpeg"]):
            image = Image.open(sample["jpeg"]).convert("RGB")
            image_tensor = self.transform(image)
        else:
            # Fallback: render the XML directly through the PyTorch renderer.
            # This requires a CUDA device because nvdiffrast does not support CPU.
            from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
            organ_array = PlantOrganArray.from_xml_file(sample["xml"])
            renderer = HeliosPyTorchRenderer(image_size=self.image_size)
            rgb = renderer.render_organ_array(
                organ_array,
                azimuth_deg=0.0,
                elevation_deg=90.0,
                camera_height=1.0,
                background="black",
                device=self.device,
                differentiable=False,
                focus_plant=True,
                existence_threshold=0.5,
            )
            image_tensor = self._transform_tensor(rgb)

        organ_array = PlantOrganArray.from_xml_file(sample["xml"])
        raw_tensor = organ_array.tensor  # (N, 94)
        N = min(raw_tensor.shape[0], self.max_nodes)

        nodes = torch.zeros((self.max_nodes, 94), dtype=torch.float32)
        nodes[:N] = raw_tensor[:N]

        # Set existence for padded slots to 0
        nodes[N:, 93] = 0.0

        existence_mask = torch.zeros(self.max_nodes, dtype=torch.float32)
        existence_mask[:N] = 1.0

        nodes = self.normalize(nodes)

        num_nodes = torch.tensor(N, dtype=torch.long)

        return {
            "image": image_tensor,
            "nodes": nodes,
            "existence_mask": existence_mask,
            "num_nodes": num_nodes,
            "xml_path": sample["xml"],
            "jpeg_path": sample["jpeg"],
            "prefix": sample["prefix"],
        }
