"""HeliosPlantDataset: loads paired JPEG + XML + params.json plant samples."""

import os
import glob
import json
from typing import Dict, Any, List, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.helios_xml_parser import parse_helios_xml


class HeliosPlantDataset(Dataset):
    """PyTorch Dataset for Helios synthetic plant image + XML structure pairs.

    Scans `data_root` for files matching:
        <prefix>_vis.jpeg  -> RGB rendered image
        <prefix>_plant_*.xml -> one or more plant XML structures
        <prefix>_params.json -> generation parameters incl. DAP & camera pose

    Each __getitem__ returns:
        {
            "image": (3, H, W) normalized tensor,
            "raw_image": PIL.Image,
            "nodes": (max_nodes, 15),
            "adj_matrix": (max_nodes, max_nodes),
            "parent_indices": (max_nodes,),
            "existence_mask": (max_nodes,),
            "organ_types": (max_nodes,),
            "shoot_ids": (max_nodes,),
            "phytomer_indices": (max_nodes,),
            "camera_pose": (2,),    # [azimuth_norm, elevation_norm]
            "dap": (1,),            # [dap_norm]
            "num_nodes": int,
            "xyz_min": (3,),
            "xyz_scale": (3,),
            "xml_path": str,
        }
    """

    def __init__(self, data_root: str, max_nodes: int = 2048, image_size: int = 256,
                 normalize_coords: bool = True):
        self.data_root = os.path.abspath(data_root)
        self.max_nodes = max_nodes
        self.image_size = image_size
        self.normalize_coords = normalize_coords

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        self.samples: List[Dict[str, str]] = self._scan_samples()

    def _scan_samples(self) -> List[Dict[str, str]]:
        """Collect matched (jpeg, xml, params) triplets."""
        jpeg_paths = sorted(glob.glob(os.path.join(self.data_root, "*_vis.jpeg")))
        samples = []
        for jpeg_path in jpeg_paths:
            basename = os.path.basename(jpeg_path)
            prefix = basename.replace("_vis.jpeg", "")
            # Find XML(s) for this prefix
            xml_pattern = os.path.join(self.data_root, f"{prefix}_plant_*.xml")
            xml_paths = sorted(glob.glob(xml_pattern))
            params_path = os.path.join(self.data_root, f"{prefix}_params.json")
            if not os.path.exists(params_path):
                params_path = None
            for xml_path in xml_paths:
                samples.append({
                    "jpeg": jpeg_path,
                    "xml": xml_path,
                    "params": params_path,
                    "prefix": prefix,
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_params(self, params_path: Optional[str]) -> Dict[str, Any]:
        defaults = {
            "dap": 10,
            "camera_height": 1.0,
            "distance_from_center": 0.01,
            "azimuth_angle": 0.0,
        }
        if params_path is None or not os.path.exists(params_path):
            return defaults
        with open(params_path, "r") as f:
            params = json.load(f)
        positioning = params.get("camera", {}).get("positioning", {})
        metadata = params.get("metadata", {})
        defaults["dap"] = metadata.get("dap", defaults["dap"])
        defaults["camera_height"] = positioning.get("camera_height", defaults["camera_height"])
        defaults["distance_from_center"] = positioning.get("distance_from_center", defaults["distance_from_center"])
        defaults["azimuth_angle"] = positioning.get("azimuth_angle", defaults["azimuth_angle"])
        return defaults

    @staticmethod
    def _camera_pose_to_tensor(camera_height: float, distance: float, azimuth_deg: float) -> torch.Tensor:
        # Normalize azimuth to [0, 1]
        azimuth_norm = (azimuth_deg % 360.0) / 360.0
        # Approximate elevation angle from camera height and distance
        if distance > 1e-6:
            elevation_deg = np.degrees(np.arctan2(camera_height, distance))
        else:
            elevation_deg = 90.0
        elevation_norm = np.clip(elevation_deg / 90.0, 0.0, 1.0)
        return torch.tensor([azimuth_norm, elevation_norm], dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        raw_image = Image.open(sample["jpeg"]).convert("RGB")
        image_tensor = self.transform(raw_image)

        params = self._load_params(sample["params"])

        xml_data = parse_helios_xml(
            sample["xml"],
            max_nodes=self.max_nodes,
            normalize=self.normalize_coords,
        )

        # Override DAP if params.json has a more reliable value
        dap = params.get("dap", xml_data["dap"])
        dap_norm = np.clip(dap / 90.0, 0.0, 1.0)

        camera_pose = self._camera_pose_to_tensor(
            params["camera_height"],
            params["distance_from_center"],
            params["azimuth_angle"],
        )

        return {
            "image": image_tensor,
            "raw_image": raw_image,
            "nodes": torch.tensor(xml_data["nodes"], dtype=torch.float32),
            "adj_matrix": torch.tensor(xml_data["adj_matrix"], dtype=torch.float32),
            "parent_indices": torch.tensor(xml_data["parent_indices"], dtype=torch.long),
            "existence_mask": torch.tensor(xml_data["existence_mask"], dtype=torch.float32),
            "organ_types": torch.tensor(xml_data["organ_types"], dtype=torch.long),
            "shoot_ids": torch.tensor(xml_data["shoot_ids"], dtype=torch.long),
            "phytomer_indices": torch.tensor(xml_data["phytomer_indices"], dtype=torch.long),
            "camera_pose": camera_pose,
            "dap": torch.tensor([dap_norm], dtype=torch.float32),
            "num_nodes": torch.tensor(xml_data["num_nodes"], dtype=torch.long),
            "xyz_min": torch.tensor(xml_data["xyz_min"], dtype=torch.float32),
            "xyz_scale": torch.tensor(xml_data["xyz_scale"], dtype=torch.float32),
            "xml_path": sample["xml"],
            "jpeg_path": sample["jpeg"],
        }
