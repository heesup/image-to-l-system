"""
Dataset for paired (rendered image, normalized 40D typed PlantOrganArray tensor) samples.

Provides:
  - 40D typed organ-array tensors padded/truncated to max_nodes
  - per-organ-type column relevance mask (row_relevance) for masked MSE
  - dataset-wide robust (percentile-clipped) min/max normalization stats
  - optional image-space augmentation (photometric jitter + gaussian noise +
    blur + random erasing); never geometric flips (breaks 3D chirality)
"""

import os
import glob
from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    ORGAN_COLUMN_MASK,
)


class OrganArrayDataset(Dataset):
    """
    Loads Helios `_vis.jpeg` images and matching `*_plant_*.xml` files,
    returning image + normalized 40D typed organ array tensors.

    Channel layout:
      - columns 0..38: continuous/categorical parameters, min-max normalized
      - column 39: existence probability, kept in [0, 1]
      - column 11: organ_type (categorical, treated as integer)

    Args:
        data_root: directory containing *.xml and *_vis.jpeg files
        max_nodes: pad/truncate organ arrays to this length
        image_size: resize input image to (image_size, image_size)
        single_xml_path: if provided, only load this one XML (for overfit tests)
        device: torch device for the on-the-fly render fallback
        use_typed_layout: if True (default), parse XML into the (N, 40) typed
            organ-row layout; if False, keep the legacy (N, 94) layout.
        augment: if True, apply stochastic image-space augmentation.
        percentile: clip normalization stats to [percentile, 100-percentile].
        exclude_globs: list of basename glob patterns to EXCLUDE (for
            train/val splitting, e.g. ['*seed02*']).
        include_globs: list of basename glob patterns to INCLUDE (keeps only
            matching samples; mutually exclusive with exclude_globs).
    """

    def __init__(
        self,
        data_root: str,
        max_nodes: int = 2048,
        image_size: int = 256,
        single_xml_path: str = None,
        device: torch.device = None,
        use_typed_layout: bool = True,
        use_gt_renderer_image: bool = False,
        augment: bool = False,
        percentile: float = 1.0,
        exclude_globs: List[str] = None,
        include_globs: List[str] = None,
    ):
        self.data_root = os.path.abspath(data_root)
        self.max_nodes = max_nodes
        self.image_size = image_size
        self.use_typed_layout = use_typed_layout
        self.use_gt_renderer_image = use_gt_renderer_image
        self.augment = augment
        self.percentile = percentile
        self.node_dim = 40 if use_typed_layout else 94
        self.existence_col = 39 if use_typed_layout else 93
        self.categorical_col = 11 if use_typed_layout else None
        self.continuous_cols = [c for c in range(self.node_dim - 1) if c != self.categorical_col]
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cached_renderer = None
        self._image_cache = {}
        self._tensor_cache = {}

        base_transform = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
        if self.augment:
            # Photometric / noise augmentation only. No geometric flips: a
            # horizontal flip mirrors the 3D plant and would not match the
            # organ-array tensor when re-rendered.
            aug_transform = [
                transforms.RandomApply([transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)], p=0.8),
                transforms.RandomApply([transforms.GaussianBlur(
                    kernel_size=(3, 3), sigma=(0.1, 1.5))], p=0.3),
                transforms.RandomPosterize(bits=4, p=0.15),
            ]
            self.transform = transforms.Compose(aug_transform + base_transform)
            self._transform_tensor = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.transform = transforms.Compose(base_transform)
            self._transform_tensor = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        import fnmatch as _fnmatch

        if single_xml_path is not None:
            self.samples = [self._resolve_pair(single_xml_path)]
        else:
            xml_paths = sorted(glob.glob(os.path.join(self.data_root, "*_plant_*.xml")))
            if include_globs:
                xml_paths = [
                    p for p in xml_paths
                    if any(_fnmatch.fnmatch(os.path.basename(p), pat) for pat in include_globs)
                ]
            elif exclude_globs:
                xml_paths = [
                    p for p in xml_paths
                    if not any(_fnmatch.fnmatch(os.path.basename(p), pat) for pat in exclude_globs)
                ]
            self.samples = [self._resolve_pair(p) for p in xml_paths]

        # Compute dataset-wide min/max bounds from ALL available XML files,
        # so a single-sample overfit test still uses stable global statistics.
        self.min_vals, self.max_vals = self._compute_global_min_max()

    def _resolve_pair(self, xml_path: str) -> Dict[str, str]:
        prefix = os.path.basename(xml_path).split("_plant_")[0]
        xml_dir = os.path.dirname(os.path.abspath(xml_path))
        jpeg_path = ""
        for suffix in ("_vis.jpeg", "_rad.jpeg"):
            candidate = os.path.join(xml_dir, f"{prefix}{suffix}")
            if os.path.exists(candidate):
                jpeg_path = candidate
                break
        if not jpeg_path:
            # Look for any *_vis.jpeg / *_rad.jpeg in the same directory as fallback
            candidates = sorted(glob.glob(os.path.join(xml_dir, "*_vis.jpeg")) +
                                glob.glob(os.path.join(xml_dir, "*_rad.jpeg")))
            if candidates:
                jpeg_path = candidates[0]
        return {"xml": xml_path, "jpeg": jpeg_path, "prefix": prefix}

    def _compute_global_min_max(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-channel min/max over every available *_plant_*.xml file.

        The categorical organ_type column (11) and existence column are
        excluded from the continuous min/max statistics. Percentile clipping
        makes the stats robust to outliers (e.g. tiny radii, 0-360 angles).
        """
        xml_paths = sorted(glob.glob(os.path.join(self.data_root, "*_plant_*.xml")))
        n_cont = len(self.continuous_cols)

        all_values: List[torch.Tensor] = []
        for xml_path in xml_paths:
            try:
                if self.use_typed_layout:
                    organ_array = PlantOrganArray.from_xml_file_typed(xml_path)
                else:
                    organ_array = PlantOrganArray.from_xml_file(xml_path)
                tensor = organ_array.tensor[:, self.continuous_cols]  # continuous channels only
                if tensor.shape[0] == 0:
                    continue
                all_values.append(tensor)
            except Exception:
                continue

        if len(all_values) == 0:
            mins = torch.zeros((n_cont,), dtype=torch.float32)
            return mins, torch.ones((n_cont,), dtype=torch.float32)

        stacked = torch.cat(all_values, dim=0)  # (total_rows, n_cont)
        lo = self.percentile
        hi = 100.0 - self.percentile
        mins = torch.quantile(stacked, lo / 100.0, dim=0)
        maxs = torch.quantile(stacked, hi / 100.0, dim=0)

        # Avoid zero range
        range_vals = torch.where((maxs - mins) < 1e-6, torch.ones_like(maxs), maxs - mins)
        return mins, range_vals

    def normalize(self, nodes: torch.Tensor) -> torch.Tensor:
        """Min-max normalize continuous channels to [0, 1]. Existence channel stays in [0,1].
        The categorical organ_type column (11) is left as an integer class index."""
        device = nodes.device
        min_vals = self.min_vals.to(device)
        range_vals = self.max_vals.to(device)
        norm_cont = torch.clamp((nodes[:, self.continuous_cols] - min_vals) / range_vals, 0.0, 1.0)
        norm_exist = torch.clamp(nodes[:, self.existence_col:self.existence_col + 1], 0.0, 1.0)

        col_list = []
        cont_idx = 0
        for c in range(nodes.shape[1]):
            if c == self.existence_col:
                col_list.append(norm_exist)
            elif c in self.continuous_cols:
                col_list.append(norm_cont[:, cont_idx:cont_idx + 1])
                cont_idx += 1
            else:
                col_list.append(nodes[:, c:c + 1])
        return torch.cat(col_list, dim=1)

    def denormalize(self, nodes: torch.Tensor) -> torch.Tensor:
        """Undo min-max normalization for continuous channels."""
        device = nodes.device
        min_vals = self.min_vals.to(device)
        range_vals = self.max_vals.to(device)
        denorm_cont = nodes[:, self.continuous_cols] * range_vals + min_vals
        denorm_exist = torch.clamp(nodes[:, self.existence_col:self.existence_col + 1], 0.0, 1.0)

        col_list = []
        cont_idx = 0
        for c in range(nodes.shape[1]):
            if c == self.existence_col:
                col_list.append(denorm_exist)
            elif c in self.continuous_cols:
                col_list.append(denorm_cont[:, cont_idx:cont_idx + 1])
                cont_idx += 1
            else:
                col_list.append(nodes[:, c:c + 1])
        return torch.cat(col_list, dim=1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        if self.use_gt_renderer_image or not os.path.exists(sample["jpeg"]):
            cache_key = sample["xml"]
            if cache_key in self._image_cache:
                image_tensor = self._image_cache[cache_key]
            else:
                # Render the ground-truth XML directly through the differentiable PyTorch renderer
                if self._cached_renderer is None:
                    from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
                    self._cached_renderer = HeliosPyTorchRenderer(image_size=self.image_size).to(self.device)
                
                if self.use_typed_layout:
                    gt_array = PlantOrganArray.from_xml_file_typed(sample["xml"])
                else:
                    gt_array = PlantOrganArray.from_xml_file(sample["xml"])
                
                with torch.no_grad():
                    rgb = self._cached_renderer.render_organ_array(
                        gt_array,
                        azimuth_deg=0.0,
                        elevation_deg=90.0,
                        camera_height=1.0,
                        background="black",
                        device=self.device,
                        differentiable=False,
                        focus_plant=True,
                        existence_threshold=0.1,
                    )
                image_tensor = self._transform_tensor(rgb).cpu()
                self._image_cache[cache_key] = image_tensor
        else:
            image = Image.open(sample["jpeg"]).convert("RGB")
            image_tensor = self.transform(image)

        if cache_key in self._tensor_cache:
            nodes, existence_mask, row_relevance, num_nodes = self._tensor_cache[cache_key]
        else:
            if self.use_typed_layout:
                organ_array = PlantOrganArray.from_xml_file_typed(sample["xml"])
            else:
                organ_array = PlantOrganArray.from_xml_file(sample["xml"])
            raw_tensor = organ_array.tensor  # (N, node_dim)
            N = min(raw_tensor.shape[0], self.max_nodes)

            nodes = torch.zeros((self.max_nodes, self.node_dim), dtype=torch.float32)
            nodes[:N] = raw_tensor[:N]

            # Set existence for padded slots to 0
            nodes[N:, self.existence_col] = 0.0

            existence_mask = torch.zeros(self.max_nodes, dtype=torch.float32)
            existence_mask[:N] = 1.0

            nodes = self.normalize(nodes)

            # Per-node column relevance mask for the continuous MSE: which columns
            # carry real signal for each node's organ_type.
            organ_types = nodes[:, self.categorical_col].long().clamp(0, ORGAN_COLUMN_MASK.shape[0] - 1)
            row_relevance = ORGAN_COLUMN_MASK[organ_types].clone()  # (max_nodes, node_dim)
            row_relevance[N:] = False  # padded rows contribute nothing

            num_nodes = torch.tensor(N, dtype=torch.long)
            self._tensor_cache[cache_key] = (nodes, existence_mask, row_relevance, num_nodes)

        return {
            "image": image_tensor,
            "nodes": nodes,
            "existence_mask": existence_mask,
            "row_relevance": row_relevance,
            "num_nodes": num_nodes,
            "xml_path": sample["xml"],
            "jpeg_path": sample["jpeg"],
            "prefix": sample["prefix"],
        }
