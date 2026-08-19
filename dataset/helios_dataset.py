"""
HeliosPlantDataset: PyTorch Dataset for paired plant images and 16D Part-Centric Tensors.

Supports both flat directories and multi-species subfolder trees:
    <data_root>/<species>/<prefix>_0000_rad.jpeg
    <data_root>/<species>/<prefix>_0000_plant_0000.xml
    <data_root>/<species>/<prefix>_0000_params.json
    <data_root>/<species>/<prefix>_0000_masks.json
    <data_root>/<species>/<prefix>_0000_camera.json

Each sample produces a 16D part-centric tensor using `PlantOrganArray`:
    [Existence(0), OrganType(1), Base(2..4), Rot6D(5..10), Scale(11..13),
     Curvature(14), PhyllotacticAngle(15)]
"""

import os
import glob
import json
import fnmatch
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    P_COL_EXISTENCE,
    P_COL_ORGAN_TYPE,
    P_COL_BASE_X,
    P_COL_BASE_Y,
    P_COL_BASE_Z,
    P_COL_ROT_0,
    P_COL_ROT_5,
    P_COL_SCALE_X,
    P_COL_SCALE_Y,
    P_COL_SCALE_Z,
    P_COL_CURVATURE,
    P_COL_PHYLLOTACTIC_ANGLE,
    NUM_FEATURES,
    ORGAN_ROOT_META,
    ORGAN_SHOOT_META,
    ORGAN_INTERNODE,
    ORGAN_PETIOLE,
    ORGAN_LEAF,
    ORGAN_BUD,
    ORGAN_PEDUNCLE,
    ORGAN_FLOWER,
    ORGAN_FRUIT,
    ORGAN_FLOWER_CLOSED,
    ORGAN_BUD_ABORTED,
)

# Normalization constants (matching flow-matching pipeline conventions)
ORGAN_TYPE_SCALE = 10.0
BASE_SCALE = 100.0          # Convert meters -> cm range
CURVATURE_SCALE = 100.0     # Degrees
PHYLLOTACTIC_SCALE = 180.0  # Degrees

ORGAN_CATEGORIES = [
    ORGAN_ROOT_META, ORGAN_SHOOT_META, ORGAN_INTERNODE, ORGAN_PETIOLE, ORGAN_LEAF,
    ORGAN_BUD, ORGAN_PEDUNCLE, ORGAN_FLOWER, ORGAN_FRUIT, ORGAN_FLOWER_CLOSED,
    ORGAN_BUD_ABORTED,
]
CATEGORY_TO_IDX = {ot: i for i, ot in enumerate(ORGAN_CATEGORIES)}
EMPTY_IDX = len(ORGAN_CATEGORIES)          # 11: index of "empty" slot
NUM_ORGAN_CATEGORIES = len(ORGAN_CATEGORIES) + 1  # 12 classes (11 real + 1 empty)

# Flow Matching representation layout
FM_OT_END = NUM_ORGAN_CATEGORIES
FM_BASE_START = FM_OT_END
FM_BASE_END = FM_OT_END + 3
FM_ROT_START = FM_BASE_END
FM_ROT_END = FM_BASE_END + 6
FM_SCALE_START = FM_ROT_END
FM_SCALE_END = FM_ROT_END + 3
FM_CURV_IDX = FM_SCALE_END
FM_PHYLLO_IDX = FM_SCALE_END + 1
FM_NODE_DIM = FM_SCALE_END + 2   # 12 + 3 + 6 + 3 + 1 + 1 = 26 dimensions


class HeliosPlantDataset(Dataset):
    """Modern 16D Part-Centric PyTorch Dataset for Helios synthetic plant data.

    Recursively scans `data_root` for paired JPEG and XML files across all species
    subfolders or within a single crop directory.
    """

    def __init__(
        self,
        data_root: str,
        max_parts: int = 512,
        image_size: int = 256,
        species: Optional[Union[str, List[str]]] = None,
        transform: Optional[transforms.Compose] = None,
        normalize_parts: bool = True,
        return_fm_format: bool = False,
        include_globs: Optional[List[str]] = None,
        exclude_globs: Optional[List[str]] = None,
    ):
        """
        Args:
            data_root: Root path containing dataset (e.g. `dataset/helios_data` or `dataset/helios_data/cowpea`).
            max_parts: Maximum number of organs/parts per plant (padded with zero existence).
            image_size: Image resize dimension (H=W=image_size).
            species: Filter to specific species (e.g. 'cowpea', ['cowpea', 'bean']), or None for all.
            transform: Optional custom torchvision transform for RGB images.
            normalize_parts: Whether continuous columns of 16D part tensor are normalized.
            return_fm_format: If True, returns flow-matching encoded tensor (26D) alongside canonical (16D).
            include_globs: Filename glob patterns to include.
            exclude_globs: Filename glob patterns to exclude.
        """
        self.data_root = os.path.abspath(data_root)
        self.max_parts = max_parts
        self.image_size = image_size
        self.normalize_parts = normalize_parts
        self.return_fm_format = return_fm_format
        self.node_dim = NUM_FEATURES  # 16

        if species is None:
            self.species_filter = None
        elif isinstance(species, str):
            self.species_filter = [s.strip().lower() for s in species.split(",") if s.strip()]
        else:
            self.species_filter = [s.strip().lower() for s in species if s.strip()]

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        self.samples = self._scan_dataset(include_globs, exclude_globs)

    def _scan_dataset(
        self,
        include_globs: Optional[List[str]],
        exclude_globs: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """Recursively scan for paired XML, JPEG, and JSON metadata."""
        xml_pattern = os.path.join(self.data_root, "**", "*_plant_*.xml")
        xml_paths = sorted(glob.glob(xml_pattern, recursive=True))

        if not xml_paths:
            # Fallback for flat structure or non-standard naming
            xml_paths = sorted(glob.glob(os.path.join(self.data_root, "*.xml")))

        samples = []
        for xml_path in xml_paths:
            basename = os.path.basename(xml_path)

            if include_globs and not any(fnmatch.fnmatch(basename, pat) for pat in include_globs):
                continue
            if exclude_globs and any(fnmatch.fnmatch(basename, pat) for pat in exclude_globs):
                continue

            xml_dir = os.path.dirname(os.path.abspath(xml_path))
            prefix = basename.split("_plant_")[0] if "_plant_" in basename else basename.rsplit(".", 1)[0]

            # Infer species from subfolder name or filename prefix
            parent_folder = os.path.basename(xml_dir).lower()
            inferred_species = parent_folder if parent_folder in ("cowpea", "bean", "sorghum", "soybean", "maize") else prefix.split("_")[0].lower()

            if self.species_filter and inferred_species not in self.species_filter:
                continue

            # Look for best available image: _rad.jpeg (primary), _vis.jpeg, or .jpeg
            jpeg_path = ""
            for suffix in ("_rad.jpeg", "_vis.jpeg", ".jpeg", ".jpg", ".png"):
                candidate = os.path.join(xml_dir, f"{prefix}{suffix}")
                if os.path.exists(candidate):
                    jpeg_path = candidate
                    break

            params_path = os.path.join(xml_dir, f"{prefix}_params.json")
            if not os.path.exists(params_path):
                params_path = None

            masks_path = os.path.join(xml_dir, f"{prefix}_masks.json")
            if not os.path.exists(masks_path):
                masks_path = None

            camera_path = os.path.join(xml_dir, f"{prefix}_camera.json")
            if not os.path.exists(camera_path):
                camera_path = None

            samples.append({
                "xml": xml_path,
                "jpeg": jpeg_path,
                "params": params_path,
                "masks": masks_path,
                "camera": camera_path,
                "prefix": prefix,
                "species": inferred_species,
                "dir": xml_dir,
            })

        return samples

    def _load_params(self, params_path: Optional[str]) -> Dict[str, Any]:
        defaults = {
            "dap": 10,
            "camera_height": 1.0,
            "distance_from_center": 0.01,
            "azimuth_angle": 0.0,
            "elevation_angle": 45.0,
            "plant_type": "cowpea",
            "genotype": "random",
        }
        if params_path is None or not os.path.exists(params_path):
            return defaults
        try:
            with open(params_path, "r") as f:
                params = json.load(f)
            positioning = params.get("camera", {}).get("positioning", {})
            metadata = params.get("metadata", {})
            defaults["dap"] = metadata.get("dap", defaults["dap"])
            defaults["plant_type"] = metadata.get("plant_type", defaults["plant_type"])
            defaults["genotype"] = metadata.get("genotype", defaults["genotype"])
            defaults["camera_height"] = positioning.get("camera_height", defaults["camera_height"])
            defaults["distance_from_center"] = positioning.get("distance_from_center", defaults["distance_from_center"])
            defaults["azimuth_angle"] = positioning.get("azimuth_angle", defaults["azimuth_angle"])
        except Exception:
            pass
        return defaults

    @staticmethod
    def normalize_part_tensor(part: torch.Tensor) -> torch.Tensor:
        """Normalize canonical (N, 16) part tensor to unit-ish scale for neural training."""
        out = part.clone()
        out[:, P_COL_ORGAN_TYPE] = out[:, P_COL_ORGAN_TYPE] / ORGAN_TYPE_SCALE
        out[:, P_COL_BASE_X:P_COL_BASE_Z + 1] = out[:, P_COL_BASE_X:P_COL_BASE_Z + 1] * BASE_SCALE
        out[:, P_COL_CURVATURE] = out[:, P_COL_CURVATURE] / CURVATURE_SCALE
        out[:, P_COL_PHYLLOTACTIC_ANGLE] = out[:, P_COL_PHYLLOTACTIC_ANGLE] / PHYLLOTACTIC_SCALE
        return out

    @staticmethod
    def denormalize_part_tensor(part: torch.Tensor) -> torch.Tensor:
        """Undo normalization back to physical metric units."""
        out = part.clone()
        out[:, P_COL_ORGAN_TYPE] = out[:, P_COL_ORGAN_TYPE] * ORGAN_TYPE_SCALE
        out[:, P_COL_BASE_X:P_COL_BASE_Z + 1] = out[:, P_COL_BASE_X:P_COL_BASE_Z + 1] / BASE_SCALE
        out[:, P_COL_CURVATURE] = out[:, P_COL_CURVATURE] * CURVATURE_SCALE
        out[:, P_COL_PHYLLOTACTIC_ANGLE] = out[:, P_COL_PHYLLOTACTIC_ANGLE] * PHYLLOTACTIC_SCALE
        return out

    @staticmethod
    def encode_fm(part: torch.Tensor) -> torch.Tensor:
        """Convert canonical (N, 16) part tensor to (N, 26) flow-matching layout with one-hot organ categories."""
        N = part.shape[0]
        out = torch.zeros((N, FM_NODE_DIM), dtype=part.dtype, device=part.device)
        ot = part[:, P_COL_ORGAN_TYPE].long()
        exist = part[:, P_COL_EXISTENCE]

        for i, cat in enumerate(ORGAN_CATEGORIES):
            mask = (ot == cat) & (exist > 0.5)
            out[mask, i] = 1.0
        out[exist <= 0.5, EMPTY_IDX] = 1.0

        out[:, FM_BASE_START:FM_BASE_END] = part[:, P_COL_BASE_X:P_COL_BASE_Z + 1] * BASE_SCALE
        out[:, FM_ROT_START:FM_ROT_END] = part[:, P_COL_ROT_0:P_COL_ROT_5 + 1]
        out[:, FM_SCALE_START:FM_SCALE_END] = part[:, P_COL_SCALE_X:P_COL_SCALE_Z + 1]
        out[:, FM_CURV_IDX] = part[:, P_COL_CURVATURE] / CURVATURE_SCALE
        out[:, FM_PHYLLO_IDX] = part[:, P_COL_PHYLLOTACTIC_ANGLE] / PHYLLOTACTIC_SCALE
        return out

    @staticmethod
    def decode_fm(fm: torch.Tensor) -> torch.Tensor:
        """Convert (N, 26) flow-matching tensor back to canonical (N, 16) part tensor."""
        N = fm.shape[0]
        out = torch.zeros((N, NUM_FEATURES), dtype=fm.dtype, device=fm.device)
        p_empty = fm[:, EMPTY_IDX]
        exist = (p_empty <= 0.5).float()
        out[:, P_COL_EXISTENCE] = exist

        ot_logits = fm[:, :len(ORGAN_CATEGORIES)]
        cat_indices = torch.argmax(ot_logits, dim=-1)
        cats_tensor = torch.tensor(ORGAN_CATEGORIES, dtype=torch.float32, device=fm.device)
        out[:, P_COL_ORGAN_TYPE] = cats_tensor[cat_indices] * exist

        out[:, P_COL_BASE_X:P_COL_BASE_Z + 1] = (fm[:, FM_BASE_START:FM_BASE_END] / BASE_SCALE) * exist.unsqueeze(-1)
        out[:, P_COL_ROT_0:P_COL_ROT_5 + 1] = fm[:, FM_ROT_START:FM_ROT_END]
        out[:, P_COL_SCALE_X:P_COL_SCALE_Z + 1] = fm[:, FM_SCALE_START:FM_SCALE_END] * exist.unsqueeze(-1)
        out[:, P_COL_CURVATURE] = (fm[:, FM_CURV_IDX] * CURVATURE_SCALE) * exist
        out[:, P_COL_PHYLLOTACTIC_ANGLE] = (fm[:, FM_PHYLLO_IDX] * PHYLLOTACTIC_SCALE) * exist
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # 1. Load image (RGB)
        if sample["jpeg"] and os.path.exists(sample["jpeg"]):
            try:
                image = Image.open(sample["jpeg"]).convert("RGB")
            except Exception:
                image = Image.new("RGB", (self.image_size, self.image_size), "black")
        else:
            image = Image.new("RGB", (self.image_size, self.image_size), "black")

        image_tensor = self.transform(image)

        # 2. Parse 16D Part-Centric Tensor via PlantOrganArray
        poa = PlantOrganArray.from_xml_file(sample["xml"])
        raw_part_tensor = poa.to_part_tensor()  # (N_active, 16)
        num_active = min(raw_part_tensor.shape[0], self.max_parts)

        # Build padded (max_parts, 16) tensor
        padded_parts = torch.zeros((self.max_parts, NUM_FEATURES), dtype=torch.float32)
        padded_parts[:num_active] = raw_part_tensor[:num_active]

        # Normalized copy
        if self.normalize_parts:
            norm_parts = self.normalize_part_tensor(padded_parts)
        else:
            norm_parts = padded_parts

        # 3. Parameters & Conditions
        params = self._load_params(sample["params"])
        dap = float(params.get("dap", 10))
        dap_norm = np.clip(dap / 100.0, 0.0, 1.0)

        # Camera pose representation
        azimuth_deg = float(params.get("azimuth_angle", 0.0))
        cam_height = float(params.get("camera_height", 1.0))
        dist = float(params.get("distance_from_center", 0.01))
        camera_pose = torch.tensor([
            (azimuth_deg % 360.0) / 360.0,
            np.clip(cam_height / 5.0, 0.0, 1.0),
            np.clip(dist / 5.0, 0.0, 1.0),
        ], dtype=torch.float32)

        result: Dict[str, Any] = {
            "image": image_tensor,
            "parts": norm_parts,                          # (max_parts, 16) normalized
            "raw_parts": padded_parts,                    # (max_parts, 16) physical metric
            "existence_mask": padded_parts[:, P_COL_EXISTENCE],  # (max_parts,)
            "organ_types": padded_parts[:, P_COL_ORGAN_TYPE].long(),  # (max_parts,)
            "num_parts": torch.tensor(num_active, dtype=torch.long),
            "dap": torch.tensor([dap_norm], dtype=torch.float32),
            "dap_raw": torch.tensor([dap], dtype=torch.float32),
            "camera_pose": camera_pose,
            "species": sample["species"],
            "prefix": sample["prefix"],
            "xml_path": sample["xml"],
            "jpeg_path": sample["jpeg"],
        }

        # Optional flow-matching encoding
        if self.return_fm_format:
            result["fm_tensor"] = self.encode_fm(padded_parts)  # (max_parts, 26)

        return result


# Convenience alias for seamless interoperability
HeliosPartDataset = HeliosPlantDataset


def create_helios_dataloader(
    data_root: str,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    max_parts: int = 512,
    image_size: int = 256,
    species: Optional[Union[str, List[str]]] = None,
    return_fm_format: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    """Helper to instantiate a PyTorch DataLoader for the 16D Helios plant dataset."""
    dataset = HeliosPlantDataset(
        data_root=data_root,
        max_parts=max_parts,
        image_size=image_size,
        species=species,
        return_fm_format=return_fm_format,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )
