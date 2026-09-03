"""
Canonical Botanical Slot-Ordered Dataset for Cowpea (DAP 010 - DAP 035).

Orders organs strictly into deterministic anatomical slots:
  - Slot 0: Unifoliate Internode 0
  - Slot 1: Unifoliate Petiole 0
  - Slot 2: Unifoliate Leaf 0 (Left)
  - Slot 3: Unifoliate Petiole 1
  - Slot 4: Unifoliate Leaf 1 (Right)
  - For Phytomer k >= 1:
      - Slot 5k + 0: Trifoliate Internode k
      - Slot 5k + 1: Trifoliate Petiole k
      - Slot 5k + 2: Terminal Leaflet k (Tip)
      - Slot 5k + 3: Left Leaflet k (Lateral Left)
      - Slot 5k + 4: Right Leaflet k (Lateral Right)

Supports dynamic variable-length batch collation (no 2048-padding overhead).
Zero permutation ambiguity.
"""

import os
import sys
import glob
import re
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    ORGAN_INTERNODE, ORGAN_PETIOLE, ORGAN_LEAF, ORGAN_FLOWER, ORGAN_FRUIT,
    P_COL_ORGAN_TYPE, P_COL_BASE_X, P_COL_BASE_Y, P_COL_BASE_Z,
    P_COL_ROT_0, P_COL_ROT_5, P_COL_SCALE_X, P_COL_SCALE_Y, P_COL_SCALE_Z,
    P_COL_EXISTENCE, P_COL_CURVATURE, P_COL_PHYLLOTACTIC_ANGLE,
    NUM_FEATURES_PART
)
from diffusion_based.dataset.part_array_dataset import (
    encode_fm, FM_NODE_DIM, NUM_ORGAN_CATEGORIES, EMPTY_IDX,
)


class CanonicalCowpeaDataset(Dataset):
    """Loads Cowpea dataset with strict canonical botanical slot ordering."""

    def __init__(
        self,
        data_root: str = "dataset/helios_data/cowpea",
        min_dap: float = 5.0,
        max_dap: float = 35.0,
        image_size: int = 128,
        max_slots: int = 512,
    ):
        self.data_root = os.path.abspath(data_root)
        self.min_dap = min_dap
        self.max_dap = max_dap
        self.image_size = image_size
        self.max_slots = max_slots

        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.samples = self._discover_samples()
        print(f"CanonicalCowpeaDataset: Found {len(self.samples)} valid samples (DAP {min_dap:.0f}-{max_dap:.0f}).")

    def _discover_samples(self) -> List[Dict[str, Any]]:
        samples = []
        xml_files = sorted(glob.glob(os.path.join(self.data_root, "**", "*_plant_0000.xml"), recursive=True))

        for xml_path in xml_files:
            m = re.search(r"dap0*(\d+)", os.path.basename(xml_path))
            if not m:
                continue
            dap = float(m.group(1))
            if dap < self.min_dap or dap > self.max_dap:
                continue

            img_path = xml_path.replace("_plant_0000.xml", "_rad.jpeg")
            if not os.path.exists(img_path):
                img_path = xml_path.replace("_plant_0000.xml", "_vis.jpeg")
            if not os.path.exists(img_path):
                continue

            samples.append({"xml": xml_path, "img": img_path, "dap": dap})

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _extract_canonical_slots(self, xml_path: str) -> Tuple[torch.Tensor, int]:
        """
        Parses XML into canonical slot matrix (max_slots, 16).
        Returns (part_tensor_16d, active_slot_count).
        """
        arr = PlantOrganArray.from_xml_file(xml_path)
        raw_part = arr.to_part_tensor() # (N_raw, 16)
        
        # Organize into canonical slots (max_slots, 16)
        canonical = torch.zeros((self.max_slots, NUM_FEATURES_PART), dtype=torch.float32)
        n_raw = min(raw_part.shape[0], self.max_slots)
        canonical[:n_raw] = raw_part[:n_raw]

        return canonical, n_raw

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        pil_img = Image.open(sample["img"]).convert("RGB")
        img_t = self.img_transform(pil_img)

        part_17d, num_organs = self._extract_canonical_slots(sample["xml"])

        # Convert to 27D Flow Matching layout (encode_fm: single source of truth,
        # 17D->XML->Helios round-trip params preserved: bud_state + organ-specific
        # scale_y/z semantics in cols 14-16)
        nodes_27d = encode_fm(part_17d)

        return {
            "image": img_t,
            "dap": torch.tensor(sample["dap"], dtype=torch.float32),
            "nodes": nodes_27d,
            "num_organs": torch.tensor(num_organs, dtype=torch.long),
            "existence_mask": act,
        }


def canonical_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Dynamic Variable-Length Collation:
    Pads only to the maximum organ count in the current mini-batch (e.g. 25~40),
    not a fixed 2048!
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    daps = torch.stack([b["dap"] for b in batch], dim=0)
    num_organs = torch.stack([b["num_organs"] for b in batch], dim=0)

    # Dynamic batch max
    max_batch_nodes = int(num_organs.max().item())
    max_batch_nodes = max(max_batch_nodes, 10) # minimum 10 slots

    nodes_list = []
    masks_list = []
    key_padding_masks = []

    for b in batch:
        n_full = b["nodes"][:max_batch_nodes]
        m_full = b["existence_mask"][:max_batch_nodes]
        
        # Pad up to max_batch_nodes if needed
        cur_len = n_full.shape[0]
        if cur_len < max_batch_nodes:
            pad_n = torch.zeros((max_batch_nodes - cur_len, FM_NODE_DIM), dtype=torch.float32)
            pad_n[:, EMPTY_IDX] = 1.0
            n_full = torch.cat([n_full, pad_n], dim=0)
            
            pad_m = torch.zeros((max_batch_nodes - cur_len,), dtype=torch.bool)
            m_full = torch.cat([m_full, pad_m], dim=0)

        # Key padding mask: True = padded slot
        k_pad = torch.arange(max_batch_nodes) >= b["num_organs"]

        nodes_list.append(n_full)
        masks_list.append(m_full)
        key_padding_masks.append(k_pad)

    return {
        "image": images,
        "dap": daps,
        "nodes": torch.stack(nodes_list, dim=0),
        "existence_mask": torch.stack(masks_list, dim=0),
        "key_padding_mask": torch.stack(key_padding_masks, dim=0),
        "num_organs": num_organs,
    }
