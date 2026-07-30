import math
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from torchvision import transforms
from typing import Dict, Any, List, Tuple

class Plant3DDataset(Dataset):
    """Dataset generating 3D Botanical Plants (Stems + 3D Leaves) with 2D Perspective Projection Images."""

    def __init__(self, num_samples: int = 100, max_nodes: int = 64, image_size: int = 256):
        self.num_samples = num_samples
        self.max_nodes = max_nodes
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.samples: List[Dict[str, Any]] = []
        self._generate_synthetic_3d_plants()

    def _generate_synthetic_3d_plants(self):
        np.random.seed(42)
        
        for idx in range(self.num_samples):
            nodes = np.zeros((self.max_nodes, 7), dtype=np.float32) # (x, y, z, theta, phi, length, is_leaf)
            adj_matrix = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
            parents = np.zeros(self.max_nodes, dtype=np.int64)
            existence = np.zeros(self.max_nodes, dtype=np.float32)

    def _generate_synthetic_3d_plants(self):
        np.random.seed(42)
        
        for idx in range(self.num_samples):
            nodes = np.zeros((self.max_nodes, 7), dtype=np.float32) # (x, y, z, theta, phi, length, is_leaf)
            adj_matrix = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
            parents = np.zeros(self.max_nodes, dtype=np.int64)
            existence = np.zeros(self.max_nodes, dtype=np.float32)

            # Generate Complex Multi-Level 3D Botanical Plant Architecture
            # Level 0: Trunk Base (Node 0) -> Trunk Junction (Node 1)
            nodes[0] = [0.50, 0.92, 0.50, 0.5, 0.5, 0.20, 0.0]
            existence[0] = 1.0; parents[0] = 0

            nodes[1] = [0.50, 0.75, 0.50, 0.5, 0.5, 0.17, 0.0]
            existence[1] = 1.0; parents[1] = 0
            adj_matrix[0, 1] = 1.0; adj_matrix[1, 0] = 1.0

            # Level 1: Main Branches from Trunk Junction Node 1 -> Nodes 2, 3, 4
            # Random variations per sample for realistic variety
            b_var = (np.random.rand(3) - 0.5) * 0.04
            
            nodes[2] = [0.35 + b_var[0], 0.58, 0.65 + b_var[0], 0.3, 0.6, 0.16, 0.0] # Left-Front Branch
            existence[2] = 1.0; parents[2] = 1
            adj_matrix[1, 2] = 1.0; adj_matrix[2, 1] = 1.0

            nodes[3] = [0.65 + b_var[1], 0.58, 0.35 + b_var[1], 0.7, 0.4, 0.16, 0.0] # Right-Back Branch
            existence[3] = 1.0; parents[3] = 1
            adj_matrix[1, 3] = 1.0; adj_matrix[3, 1] = 1.0

            nodes[4] = [0.50 + b_var[2], 0.54, 0.50, 0.5, 0.5, 0.18, 0.0] # Center-Up Branch
            existence[4] = 1.0; parents[4] = 1
            adj_matrix[1, 4] = 1.0; adj_matrix[4, 1] = 1.0

            # Level 2: Secondary Sub-Branches / Twigs from Level 1 Nodes -> Nodes 5..10
            # Twigs from Node 2 (Left-Front)
            nodes[5] = [0.24, 0.42, 0.72, 0.2, 0.7, 0.14, 0.0]; existence[5] = 1.0; parents[5] = 2; adj_matrix[2, 5] = 1.0; adj_matrix[5, 2] = 1.0
            nodes[6] = [0.38, 0.44, 0.56, 0.4, 0.5, 0.14, 0.0]; existence[6] = 1.0; parents[6] = 2; adj_matrix[2, 6] = 1.0; adj_matrix[6, 2] = 1.0

            # Twigs from Node 3 (Right-Back)
            nodes[7] = [0.62, 0.44, 0.44, 0.6, 0.5, 0.14, 0.0]; existence[7] = 1.0; parents[7] = 3; adj_matrix[3, 7] = 1.0; adj_matrix[7, 3] = 1.0
            nodes[8] = [0.76, 0.42, 0.28, 0.8, 0.3, 0.14, 0.0]; existence[8] = 1.0; parents[8] = 3; adj_matrix[3, 8] = 1.0; adj_matrix[8, 3] = 1.0

            # Twigs from Node 4 (Center-Up)
            nodes[9] = [0.42, 0.36, 0.50, 0.4, 0.5, 0.15, 0.0]; existence[9] = 1.0; parents[9] = 4; adj_matrix[4, 9] = 1.0; adj_matrix[9, 4] = 1.0
            nodes[10] = [0.58, 0.36, 0.50, 0.6, 0.5, 0.15, 0.0]; existence[10] = 1.0; parents[10] = 4; adj_matrix[4, 10] = 1.0; adj_matrix[10, 4] = 1.0

            # Stem Nodes total = 11 nodes (Nodes 0..10)
            # Terminal End Tip Nodes = Nodes 5, 6, 7, 8, 9, 10 (Out-degree 0 stem tip nodes!)
            # Internal Branch Junctions (Nodes 1, 2, 3, 4) HAVE NO LEAVES!

            leaf_idx = 11
            tip_nodes = [5, 6, 7, 8, 9, 10]
            tip_base_angles = [-135.0, -105.0, -75.0, -45.0, -120.0, -60.0]

            for tip, base_deg in zip(tip_nodes, tip_base_angles):
                # 3 Leaves attached per terminal end tip node in skyward fan (-25 deg, 0 deg, +25 deg)
                for fan_offset in [-25.0, 0.0, 25.0]:
                    angle_deg = base_deg + fan_offset
                    rad = math.radians(angle_deg)
                    nodes[leaf_idx] = [
                        nodes[tip, 0], nodes[tip, 1], nodes[tip, 2],
                        math.cos(rad) * 0.5 + 0.5, math.sin(rad) * 0.5 + 0.5,
                        0.10, 1.0 # Leaf type = 1.0
                    ]
                    existence[leaf_idx] = 1.0
                    parents[leaf_idx] = tip
                    adj_matrix[tip, leaf_idx] = 1.0; adj_matrix[leaf_idx, tip] = 1.0
                    leaf_idx += 1

            num_active = leaf_idx # 11 stems + 18 leaves = 29 nodes total!

            # Render 2D Perspective Projection Image of 3D Plant
            img = Image.new("RGB", (self.image_size, self.image_size), (245, 248, 252))
            draw = ImageDraw.Draw(img)

            # Draw 3D Stem Edges projected to 2D
            for v in range(num_active):
                if nodes[v, 6] > 0.5: # Leaf
                    continue
                u = parents[v]
                px2, py2 = int(nodes[v, 0] * self.image_size), int(nodes[v, 1] * self.image_size)
                if u != v:
                    px1, py1 = int(nodes[u, 0] * self.image_size), int(nodes[u, 1] * self.image_size)
                    draw.line([px1, py1, px2, py2], fill=(40, 40, 40), width=6)

            # Draw 3D Green Elongated Heart Leaves projected to 2D (Attached EXACTLY at Terminal End Node Joint)
            for v in range(num_active):
                if nodes[v, 6] > 0.5: # Leaf
                    u = parents[v]
                    # Attached EXACTLY at terminal parent node joint coordinate
                    px_base = nodes[u, 0] * self.image_size
                    py_base = nodes[u, 1] * self.image_size
                    scale_area = nodes[v, 5]

                    # Direction vector (cos_a, sin_a) pointing skywards
                    cos_a = (nodes[v, 3] - 0.5) * 2.0
                    sin_a = (nodes[v, 4] - 0.5) * 2.0
                    norm = math.sqrt(cos_a**2 + sin_a**2) + 1e-5
                    cos_a, sin_a = cos_a / norm, sin_a / norm

                    leaf_len = scale_area * 180
                    leaf_w = leaf_len * 0.55

                    # Cordate Leaf Local Points: (u_perp, v_along)
                    # v_along goes along leaf length [0, leaf_len]
                    # u_perp goes along perpendicular width [-leaf_w, +leaf_w]
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

            self.samples.append({
                "image": img,
                "nodes": torch.tensor(nodes, dtype=torch.float32),
                "adj_matrix": torch.tensor(adj_matrix, dtype=torch.float32),
                "parent_indices": torch.tensor(parents, dtype=torch.long),
                "existence_mask": torch.tensor(existence, dtype=torch.float32),
                "num_nodes": torch.tensor(num_active, dtype=torch.long),
                "camera_pose": torch.tensor([0.0, 0.0], dtype=torch.float32) # Front view (azimuth=0, elevation=0)
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
            "num_nodes": sample["num_nodes"],
            "camera_pose": sample["camera_pose"]
        }
