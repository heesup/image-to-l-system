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

            # Generate 3D Botanical Plant Structure
            # Node 0: Trunk Base (x=0.5, y=0.95, z=0.5)
            nodes[0] = [0.5, 0.95, 0.5, 0.5, 0.5, 0.25, 0.0] # Stem
            existence[0] = 1.0
            parents[0] = 0

            # Trunk top
            nodes[1] = [0.5, 0.70, 0.5, 0.5, 0.5, 0.25, 0.0] # Stem
            existence[1] = 1.0
            parents[1] = 0
            adj_matrix[0, 1] = 1.0; adj_matrix[1, 0] = 1.0

            # Branch 1 (3D Left Branch going forward +z)
            nodes[2] = [0.35, 0.50, 0.65, 0.3, 0.6, 0.20, 0.0] # Stem
            existence[2] = 1.0
            parents[2] = 1
            adj_matrix[1, 2] = 1.0; adj_matrix[2, 1] = 1.0

            # Branch 2 (3D Right Branch going backward -z)
            nodes[3] = [0.65, 0.50, 0.35, 0.7, 0.4, 0.20, 0.0] # Stem
            existence[3] = 1.0
            parents[3] = 1
            adj_matrix[1, 3] = 1.0; adj_matrix[3, 1] = 1.0

            # --- Attach Triple Leaf Cluster (3 Leaves per Terminal Tip Node) ---
            # Terminal Tip Node 2 (Branch 1): 3 Leaves (nodes 4, 5, 6)
            # Terminal Tip Node 3 (Branch 2): 3 Leaves (nodes 7, 8, 9)
            # Terminal Tip Node 1 (Top Trunk): 3 Leaves (nodes 10, 11, 12)
            
            leaf_idx = 4
            tip_nodes = [2, 3, 1]
            tip_base_angles = [-135.0, -45.0, -90.0] # General pointing direction

            for tip, base_deg in zip(tip_nodes, tip_base_angles):
                # 3 Leaves attached in a radial fan: -45 deg, 0 deg, +45 deg relative to tip
                for fan_offset in [-45.0, 0.0, 45.0]:
                    angle_deg = base_deg + fan_offset
                    rad = math.radians(angle_deg)
                    nodes[leaf_idx] = [
                        nodes[tip, 0], nodes[tip, 1], nodes[tip, 2],
                        math.cos(rad) * 0.5 + 0.5, math.sin(rad) * 0.5 + 0.5,
                        0.12, 1.0 # Leaf type = 1.0
                    ]
                    existence[leaf_idx] = 1.0
                    parents[leaf_idx] = tip
                    adj_matrix[tip, leaf_idx] = 1.0; adj_matrix[leaf_idx, tip] = 1.0
                    leaf_idx += 1

            num_active = leaf_idx

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

            # Draw 3D Green Elongated Heart Leaves projected to 2D (Attached EXACTLY at Node Joint)
            for v in range(num_active):
                if nodes[v, 6] > 0.5: # Leaf
                    u = parents[v]
                    # Attached EXACTLY at parent node joint coordinate
                    px_base = int(nodes[u, 0] * self.image_size)
                    py_base = int(nodes[u, 1] * self.image_size)
                    scale_area = nodes[v, 5]

                    # Convert direction angles back to degrees
                    cos_val = (nodes[v, 3] - 0.5) * 2.0
                    sin_val = (nodes[v, 4] - 0.5) * 2.0
                    leaf_angle_deg = math.degrees(math.atan2(sin_val, cos_val))

                    leaf_len = int(scale_area * 180)
                    leaf_w = int(leaf_len * 0.55)
                    rad = math.radians(leaf_angle_deg)
                    cos_a, sin_a = math.cos(rad), math.sin(rad)

                    # Cordate Leaf Polygon: Base Notch starts EXACTLY at (0, 0) (the node joint)
                    local_pts = [
                        (0, 0),                              # Base Notch at Node Joint
                        (-leaf_w * 0.45, -leaf_len * 0.25),  # Left Lobe
                        (-leaf_w * 0.50, -leaf_len * 0.55),  # Left Mid
                        (0, -leaf_len),                      # Sharp Pointed Tip
                        (leaf_w * 0.50, -leaf_len * 0.55),   # Right Mid
                        (leaf_w * 0.45, -leaf_len * 0.25),   # Right Lobe
                    ]

                    world_pts = []
                    for lx, ly in local_pts:
                        wx = px_base + lx * cos_a - ly * sin_a
                        wy = py_base + lx * sin_a + ly * cos_a
                        world_pts.append((int(wx), int(wy)))

                    draw.polygon(world_pts, fill=(46, 139, 87), outline=(0, 100, 0))

            self.samples.append({
                "image": img,
                "nodes": torch.tensor(nodes, dtype=torch.float32),
                "adj_matrix": torch.tensor(adj_matrix, dtype=torch.float32),
                "parent_indices": torch.tensor(parents, dtype=torch.long),
                "existence_mask": torch.tensor(existence, dtype=torch.float32),
                "num_nodes": torch.tensor(num_active, dtype=torch.long)
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
            "num_nodes": sample["num_nodes"]
        }
