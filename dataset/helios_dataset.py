"""HeliosPlantDataset: loads paired JPEG + XML + params.json plant samples.

Supports both the original 15D node representation (dataset.helios_xml_parser)
and the full 25D organ-node representation (diffusion_based.models.helios_xml_parser).
"""

import os
import glob
import json
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.helios_xml_parser import parse_helios_xml

# 25D parser is only imported when requested so the legacy 15D path stays dependency-free.
try:
    from diffusion_based.models.helios_xml_parser import HeliosXMLParser
except Exception:  # pragma: no cover
    HeliosXMLParser = None


class HeliosPlantDataset(Dataset):
    """PyTorch Dataset for Helios synthetic plant image + XML structure pairs.

    Scans `data_root` for files matching:
        <prefix>_vis.jpeg  -> RGB rendered image
        <prefix>_plant_*.xml -> one or more plant XML structures
        <prefix>_params.json -> generation parameters incl. DAP & camera pose

    Each __getitem__ returns (node_dim = 15 or 25 depending on ``node_dim``):
        {
            "image": (3, H, W) normalized tensor,
            "nodes": (max_nodes, node_dim),
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

    When ``node_dim == 25`` the richer organ-node representation from
    ``diffusion_based.models.helios_xml_parser`` is used. It stores a full
    3x3 local-to-world rotation matrix for leaves, an expanded 6-class one-hot
    organ type vector, and explicit parent indices for every node.
    """

    def __init__(self, data_root: str, max_nodes: int = 2048, image_size: int = 256,
                 normalize_coords: bool = True, node_dim: int = 15):
        if node_dim not in (15, 25):
            raise ValueError(f"HeliosPlantDataset supports node_dim=15 or 25, got {node_dim}")
        if node_dim == 25 and HeliosXMLParser is None:
            raise RuntimeError("25D dataset mode requires diffusion_based.models.helios_xml_parser.HeliosXMLParser")
        self.data_root = os.path.abspath(data_root)
        self.max_nodes = max_nodes
        self.image_size = image_size
        self.normalize_coords = normalize_coords
        self.node_dim = node_dim

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        self.samples: List[Dict[str, str]] = self._scan_samples()

    def _scan_samples(self) -> List[Dict[str, str]]:
        """Collect matched (jpeg, xml, params) triplets."""
        xml_paths = sorted(glob.glob(os.path.join(self.data_root, "*_plant_*.xml")))
        if not xml_paths:
            xml_paths = sorted(glob.glob(os.path.join(self.data_root, "*.xml")))

        samples = []
        for xml_path in xml_paths:
            basename = os.path.basename(xml_path)
            # Extract prefix (e.g., cowpea_dap005_seed00_0000_plant_0000.xml -> cowpea_dap005_seed00_0000)
            prefix = basename.split("_plant_")[0] if "_plant_" in basename else basename.rsplit(".", 1)[0]
            jpeg_path = os.path.join(self.data_root, f"{prefix}_vis.jpeg")
            if not os.path.exists(jpeg_path):
                # Check for alternative jpeg extension or prepare for auto-render
                alt_jpeg = os.path.join(self.data_root, f"{prefix}.jpeg")
                if os.path.exists(alt_jpeg):
                    jpeg_path = alt_jpeg

            params_path = os.path.join(self.data_root, f"{prefix}_params.json")
            if not os.path.exists(params_path):
                params_path = None

            samples.append({
                "jpeg": jpeg_path,
                "xml": xml_path,
                "params": params_path,
                "prefix": prefix,
            })
        return samples

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
        azimuth_norm = (azimuth_deg % 360.0) / 360.0
        if distance > 1e-6:
            elevation_deg = np.degrees(np.arctan2(camera_height, distance))
        else:
            elevation_deg = 90.0
        elevation_norm = np.clip(elevation_deg / 90.0, 0.0, 1.0)
        return torch.tensor([azimuth_norm, elevation_norm], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize_25d_nodes(self, node_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize 25D node array positions/scale/radius to [0,1].

        Returns the normalized array along with metric ``xyz_min`` and
        ``xyz_scale`` so downstream rendering can reconstruct metric units.
        """
        positions = node_arr[:, :3].copy()
        xyz_min = positions.min(axis=0)
        xyz_max = positions.max(axis=0)
        xyz_scale = xyz_max - xyz_min
        xyz_scale = np.where(xyz_scale < 1e-6, 1.0, xyz_scale)

        if self.normalize_coords:
            node_arr[:, :3] = (positions - xyz_min) / xyz_scale
            node_arr[:, 3] = np.clip(node_arr[:, 3] / 1.0, 0.0, 1.0)
            node_arr[:, 4] = np.clip(node_arr[:, 4] / 0.1, 0.0, 1.0)

        return node_arr, xyz_min.astype(np.float32), xyz_scale.astype(np.float32)

    def _build_25d_adjacency(
        self,
        parent_indices: np.ndarray,
        existence_mask: np.ndarray,
        num_active: int,
    ) -> np.ndarray:
        """Build an undirected adjacency matrix from parent indices."""
        max_nodes = self.max_nodes
        adj_matrix = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        for i in range(num_active):
            j = parent_indices[i]
            if i != j and j < num_active and existence_mask[i] > 0.5 and existence_mask[j] > 0.5:
                adj_matrix[j, i] = 1.0
                adj_matrix[i, j] = 1.0
        return adj_matrix

    def _render_python_projection(self, xml_path: str, py_proj_path: str):
        """Render a 2D projection PNG of the 3D plant graph using Python for visual comparison."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from dataset.helios_xml_parser import parse_helios_xml

        xml_data = parse_helios_xml(xml_path, max_nodes=self.max_nodes, normalize=False)
        nodes = xml_data["nodes"]
        existence = xml_data["existence_mask"]
        organ_types = xml_data["organ_types"]
        parents = xml_data["parent_indices"]
        num_nodes = xml_data["num_nodes"]

        colors = {0: "#8B4513", 1: "#ADFF2F", 2: "#228B22", 3: "#FFD700"}
        fig, ax = plt.subplots(figsize=(4, 4), dpi=self.image_size // 4)
        ax.set_facecolor("black")

        for i in range(num_nodes):
            if existence[i] < 0.5:
                continue
            parent = parents[i]
            p_xyz = nodes[parent, 0:3]
            c_xyz = nodes[i, 0:3]
            organ = int(organ_types[i])
            c = colors.get(organ, "#228B22")
            ax.plot([p_xyz[0], c_xyz[0]], [p_xyz[2], c_xyz[2]], color=c, linewidth=1.2)

        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(py_proj_path, facecolor="black", edgecolor="none", pad_inches=0)
        plt.close(fig)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # 1. Render & save Python 2D projection render (_py_proj.png) for side-by-side comparison
        py_proj_path = os.path.join(self.data_root, f"{sample['prefix']}_py_proj.png")
        if not os.path.exists(py_proj_path):
            try:
                self._render_python_projection(sample["xml"], py_proj_path)
            except Exception:
                pass

        # 2. Auto-render fallback _vis.jpeg if native Helios image is not present
        if not os.path.exists(sample["jpeg"]):
            try:
                self._render_python_projection(sample["xml"], sample["jpeg"])
            except Exception:
                blank = Image.new("RGB", (self.image_size, self.image_size), "black")
                blank.save(sample["jpeg"])

        image = Image.open(sample["jpeg"]).convert("RGB")
        image_tensor = self.transform(image)

        params = self._load_params(sample["params"])

    def _parse_25d_xml(self, xml_path: str) -> Dict[str, Any]:
        """Parse a Helios XML into the 25D organ-node layout.

        Positions/length/radius are normalized to [0,1] using the per-plant
        bounding box so the diffusion model can train on a consistent scale.
        The original metric bounds are returned in ``xyz_min`` / ``xyz_scale``
        so the renderer can denormalize when needed.
        """
        parser = HeliosXMLParser(xml_path)
        parser.parse()
        nodes_25 = parser.get_all_organ_nodes()

        # If the plant exceeds the node budget, truncate (parser emits nodes in
        # a meaningful order). A smarter budget policy can be added later.
        if len(nodes_25) > self.max_nodes:
            print(f"Warning: Plant has {len(nodes_25)} nodes, pruning to max_nodes={self.max_nodes}.")
            nodes_25 = nodes_25[:self.max_nodes]

        N = len(nodes_25)
        max_nodes = self.max_nodes

        # Stack raw 25D vectors and normalize
        node_arr = np.stack([n.to_vec() for n in nodes_25], axis=0).astype(np.float32)
        node_arr, xyz_min, xyz_scale = self._normalize_25d_nodes(node_arr)

        # Build output arrays padded to max_nodes
        nodes_out = np.zeros((max_nodes, 25), dtype=np.float32)
        nodes_out[:N] = node_arr

        existence_mask = np.zeros(max_nodes, dtype=np.float32)
        existence_mask[:N] = 1.0

        parent_indices = np.zeros(max_nodes, dtype=np.int64)
        parent_indices[:N] = np.array([n.parent_idx for n in nodes_25], dtype=np.int64)
        # Clamp self/root parents to 0 to keep the parent CE logic simple
        parent_indices = np.where(parent_indices >= 0, parent_indices, 0)
        parent_indices[N:] = 0

        adj_matrix = self._build_25d_adjacency(parent_indices, existence_mask, N)

        organ_types = np.argmax(nodes_out[:, 14:20], axis=1)
        shoot_ids = nodes_out[:, 20].astype(np.int64)
        phytomer_indices = nodes_out[:, 21].astype(np.int64)

        # Infer DAP from the XML plant_age tag
        dap = parser.plant_age if parser.plant_age is not None else 10

        return {
            "nodes": nodes_out,
            "adj_matrix": adj_matrix,
            "parent_indices": parent_indices,
            "existence_mask": existence_mask,
            "organ_types": organ_types,
            "shoot_ids": shoot_ids,
            "phytomer_indices": phytomer_indices,
            "dap": dap,
            "num_nodes": N,
            "xyz_min": xyz_min,
            "xyz_scale": xyz_scale,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # 1. Render & save Python 2D projection render (_py_proj.png) for side-by-side comparison
        py_proj_path = os.path.join(self.data_root, f"{sample['prefix']}_py_proj.png")
        if not os.path.exists(py_proj_path):
            try:
                self._render_python_projection(sample["xml"], py_proj_path)
            except Exception:
                pass

        # 2. Auto-render fallback _vis.jpeg if native Helios image is not present
        if not os.path.exists(sample["jpeg"]):
            try:
                self._render_python_projection(sample["xml"], sample["jpeg"])
            except Exception:
                blank = Image.new("RGB", (self.image_size, self.image_size), "black")
                blank.save(sample["jpeg"])

        image = Image.open(sample["jpeg"]).convert("RGB")
        image_tensor = self.transform(image)

        params = self._load_params(sample["params"])

        # Load XML into the requested node representation
        if self.node_dim == 25:
            xml_data = self._parse_25d_xml(sample["xml"])
        else:
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
