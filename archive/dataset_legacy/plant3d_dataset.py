import math
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from torchvision import transforms
from typing import Dict, Any, List, Tuple


class Plant3DDataset(Dataset):
    """Dataset generating 3D Botanical Plants (Stems + 3D Leaves) with 2D Perspective Projection Images.

    Node feature layout (15D):
        0-2: x, y, z (base position)
        3:   length / scale
        4:   radius / thickness
        5-7: pitch, yaw, roll (degrees normalized to [0,1])
        8-11: organ_type one-hot [internode, petiole, leaf, floral_bud]
        12:  shoot_id
        13:  phytomer_idx
        14:  existence
    """

    def __init__(self, num_samples: int = 100, max_nodes: int = 2048, image_size: int = 256,
                 fixed_seed: bool = True):
        self.num_samples = num_samples
        self.max_nodes = max_nodes
        self.image_size = image_size
        self.fixed_seed = fixed_seed

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.samples: List[Dict[str, Any]] = []
        self._generate_synthetic_3d_plants()

    def _angle_to_norm(self, deg: float) -> float:
        """Map [-180, 180] degrees to [0, 1]."""
        return (deg + 180.0) / 360.0

    def _make_15d_node(self, x: float, y: float, z: float,
                       length: float, radius: float,
                       pitch_deg: float, yaw_deg: float, roll_deg: float,
                       organ_type: int, shoot_id: int, phytomer_idx: int) -> np.ndarray:
        """Create a single 15D node vector."""
        one_hot = np.zeros(4, dtype=np.float32)
        one_hot[organ_type] = 1.0
        return np.array([
            x, y, z,
            length,
            radius,
            self._angle_to_norm(pitch_deg),
            self._angle_to_norm(yaw_deg),
            self._angle_to_norm(roll_deg),
            one_hot[0], one_hot[1], one_hot[2], one_hot[3],
            float(shoot_id),
            float(phytomer_idx),
            1.0,  # existence
        ], dtype=np.float32)

    def _generate_synthetic_3d_plants(self):
        if self.fixed_seed:
            np.random.seed(42)

        for idx in range(self.num_samples):
            nodes = np.zeros((self.max_nodes, 15), dtype=np.float32)
            adj_matrix = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
            parents = np.zeros(self.max_nodes, dtype=np.int64)
            existence = np.zeros(self.max_nodes, dtype=np.float32)

            # Organ type indices
            INTERNODE = 0
            LEAF = 2

            # Random branch variation per sample
            b_var = (np.random.rand(3) - 0.5) * 0.04

            # Level 0: Trunk Base (Node 0) -> Trunk Junction (Node 1)
            nodes[0] = self._make_15d_node(
                0.50, 0.92, 0.50,
                length=0.20, radius=0.02,
                pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0,
                organ_type=INTERNODE, shoot_id=0, phytomer_idx=0
            )
            existence[0] = 1.0; parents[0] = 0

            nodes[1] = self._make_15d_node(
                0.50, 0.75, 0.50,
                length=0.17, radius=0.02,
                pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0,
                organ_type=INTERNODE, shoot_id=0, phytomer_idx=1
            )
            existence[1] = 1.0; parents[1] = 0
            adj_matrix[0, 1] = 1.0; adj_matrix[1, 0] = 1.0

            # Level 1: Main Branches from Trunk Junction Node 1 -> Nodes 2, 3, 4
            nodes[2] = self._make_15d_node(
                0.35 + b_var[0], 0.58, 0.65 + b_var[0],
                length=0.16, radius=0.018,
                pitch_deg=0.0, yaw_deg=-45.0, roll_deg=0.0,
                organ_type=INTERNODE, shoot_id=0, phytomer_idx=2
            )
            existence[2] = 1.0; parents[2] = 1
            adj_matrix[1, 2] = 1.0; adj_matrix[2, 1] = 1.0

            nodes[3] = self._make_15d_node(
                0.65 + b_var[1], 0.58, 0.35 + b_var[1],
                length=0.16, radius=0.018,
                pitch_deg=0.0, yaw_deg=45.0, roll_deg=0.0,
                organ_type=INTERNODE, shoot_id=0, phytomer_idx=3
            )
            existence[3] = 1.0; parents[3] = 1
            adj_matrix[1, 3] = 1.0; adj_matrix[3, 1] = 1.0

            nodes[4] = self._make_15d_node(
                0.50 + b_var[2], 0.54, 0.50,
                length=0.18, radius=0.02,
                pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0,
                organ_type=INTERNODE, shoot_id=0, phytomer_idx=4
            )
            existence[4] = 1.0; parents[4] = 1
            adj_matrix[1, 4] = 1.0; adj_matrix[4, 1] = 1.0

            # Level 2: Secondary Sub-Branches / Twigs
            twigs = [
                (5, 2, 0.24, 0.42, 0.72, 0.14, -60.0),
                (6, 2, 0.38, 0.44, 0.56, 0.14, -30.0),
                (7, 3, 0.62, 0.44, 0.44, 0.14, 30.0),
                (8, 3, 0.76, 0.42, 0.28, 0.14, 60.0),
                (9, 4, 0.42, 0.36, 0.50, 0.15, -20.0),
                (10, 4, 0.58, 0.36, 0.50, 0.15, 20.0),
            ]
            for v, u, x, y, z, length, yaw in twigs:
                nodes[v] = self._make_15d_node(
                    x, y, z,
                    length=length, radius=0.015,
                    pitch_deg=0.0, yaw_deg=yaw, roll_deg=0.0,
                    organ_type=INTERNODE, shoot_id=0, phytomer_idx=v
                )
                existence[v] = 1.0; parents[v] = u
                adj_matrix[u, v] = 1.0; adj_matrix[v, u] = 1.0

            # Stem Nodes total = 11 nodes (Nodes 0..10)
            # Terminal End Tip Nodes = Nodes 5, 6, 7, 8, 9, 10

            leaf_idx = 11
            tip_nodes = [5, 6, 7, 8, 9, 10]
            tip_base_angles = [-135.0, -105.0, -75.0, -45.0, -120.0, -60.0]

            for tip, base_deg in zip(tip_nodes, tip_base_angles):
                # 3 Leaves attached per terminal end tip node in skyward fan (-25 deg, 0 deg, +25 deg)
                for fan_offset in [-25.0, 0.0, 25.0]:
                    angle_deg = base_deg + fan_offset
                    rad = math.radians(angle_deg)
                    # direction in normalized [-1, 1] for theta/phi compatibility
                    dir_x = math.cos(rad)
                    dir_y = math.sin(rad)
                    yaw = math.degrees(math.atan2(dir_y, dir_x))
                    pitch = 0.0

                    nodes[leaf_idx] = self._make_15d_node(
                        nodes[tip, 0], nodes[tip, 1], nodes[tip, 2],
                        length=0.10, radius=0.05,
                        pitch_deg=pitch, yaw_deg=yaw, roll_deg=0.0,
                        organ_type=LEAF, shoot_id=0, phytomer_idx=tip
                    )
                    existence[leaf_idx] = 1.0
                    parents[leaf_idx] = tip
                    adj_matrix[tip, leaf_idx] = 1.0; adj_matrix[leaf_idx, tip] = 1.0
                    leaf_idx += 1

            num_active = leaf_idx  # 11 stems + 18 leaves = 29 nodes total!

            # Render 2D Perspective Projection Image of 3D Plant
            img = Image.new("RGB", (self.image_size, self.image_size), (245, 248, 252))
            draw = ImageDraw.Draw(img)

            # Draw 3D Stem Edges projected to 2D
            for v in range(num_active):
                organ_type = int(np.argmax(nodes[v, 8:12]))
                if organ_type == LEAF:
                    continue
                u = parents[v]
                px2, py2 = int(nodes[v, 0] * self.image_size), int(nodes[v, 1] * self.image_size)
                if u != v:
                    px1, py1 = int(nodes[u, 0] * self.image_size), int(nodes[u, 1] * self.image_size)
                    draw.line([px1, py1, px2, py2], fill=(40, 40, 40), width=6)

            # Draw 3D Green Elongated Heart Leaves projected to 2D
            for v in range(num_active):
                organ_type = int(np.argmax(nodes[v, 8:12]))
                if organ_type == LEAF:
                    u = parents[v]
                    px_base = nodes[u, 0] * self.image_size
                    py_base = nodes[u, 1] * self.image_size
                    scale_area = nodes[v, 3]

                    yaw_norm = nodes[v, 6]
                    yaw = (yaw_norm - 0.5) * 360.0
                    rad = math.radians(yaw)
                    cos_a = math.cos(rad)
                    sin_a = math.sin(rad)

                    leaf_len = scale_area * 180
                    leaf_w = leaf_len * 0.55

                    local_pts = [
                        (0.0, 0.0),                          # Base Notch at Node Joint (v=0)
                        (-leaf_w * 0.45, leaf_len * 0.25),   # Left Lobe
                        (-leaf_w * 0.50, leaf_len * 0.55),   # Left Mid
                        (0.0, leaf_len),                     # Sharp Pointed Tip pointing SKYWARDS
                        (leaf_w * 0.50, leaf_len * 0.55),    # Right Mid
                        (leaf_w * 0.45, leaf_len * 0.25),    # Right Lobe
                    ]

                    world_pts = []
                    for u_perp, v_along in local_pts:
                        wx = px_base + v_along * cos_a - u_perp * sin_a
                        wy = py_base + v_along * sin_a + u_perp * cos_a
                        world_pts.append((int(wx), int(wy)))

                    draw.polygon(world_pts, fill=(46, 139, 87), outline=(0, 100, 0))

            organ_types = nodes[:, 8:12].argmax(axis=1).astype(np.int64)

            self.samples.append({
                "image": img,
                "nodes": torch.tensor(nodes, dtype=torch.float32),
                "adj_matrix": torch.tensor(adj_matrix, dtype=torch.float32),
                "parent_indices": torch.tensor(parents, dtype=torch.long),
                "existence_mask": torch.tensor(existence, dtype=torch.float32),
                "organ_types": torch.tensor(organ_types, dtype=torch.long),
                "num_nodes": torch.tensor(int(num_active), dtype=torch.long),
                "camera_pose": torch.tensor([0.0, 0.0], dtype=torch.float32),  # Front view
                "dap": torch.tensor([10.0 / 90.0], dtype=torch.float32),
            })

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        img_tensor = self.transform(sample["image"])
        return {
            "image": img_tensor,
            "raw_image": sample["image"],
            "nodes": sample["nodes"],
            "adj_matrix": sample["adj_matrix"],
            "parent_indices": sample["parent_indices"],
            "existence_mask": sample["existence_mask"],
            "organ_types": sample["organ_types"],
            "num_nodes": sample["num_nodes"],
            "camera_pose": sample["camera_pose"],
            "dap": sample["dap"],
        }
