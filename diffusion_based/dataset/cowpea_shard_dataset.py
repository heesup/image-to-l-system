"""
High-Throughput Shard Dataset for Cowpea 100K Training.
Streams pre-cached tensor shards (.pt) with variable-length dynamic batching.
"""

import os
import glob
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset
from diffusion_based.dataset.part_array_dataset import FM_NODE_DIM, EMPTY_IDX


class CowpeaShardDataset(Dataset):
    """Loads pre-rendered tensor shards from dataset/cache_cowpea_100k/."""

    def __init__(
        self,
        cache_dir: str = "dataset/cache_cowpea_100k",
        fallback_cache_dir: str = "dataset/cache",
        max_samples: Optional[int] = None,
    ):
        self.cache_dir = os.path.abspath(cache_dir)
        self.fallback_dir = os.path.abspath(fallback_cache_dir)
        
        self.shard_files = sorted(glob.glob(os.path.join(self.cache_dir, "*.pt")))
        self.individual_files = []

        if not self.shard_files and os.path.exists(self.fallback_dir):
            self.individual_files = sorted(glob.glob(os.path.join(self.fallback_dir, "cowpea_*.pt")))

        # Index shards and in-shard offsets
        self.index: List[Tuple[str, int]] = []
        
        if self.shard_files:
            for s_path in self.shard_files:
                try:
                    shard_data = torch.load(s_path, map_location="cpu", weights_only=False)
                    if isinstance(shard_data, list):
                        for off in range(len(shard_data)):
                            self.index.append((s_path, off))
                except Exception:
                    continue
        elif self.individual_files:
            for f_path in self.individual_files:
                self.index.append((f_path, -1))

        if max_samples is not None and len(self.index) > max_samples:
            self.index = self.index[:max_samples]

        self._cached_shard_path = None
        self._cached_shard_data = None

        print(f"CowpeaShardDataset: Loaded {len(self.index):,} indexed training samples.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f_path, offset = self.index[idx]

        if offset >= 0:
            # Shard format
            if self._cached_shard_path != f_path:
                self._cached_shard_data = torch.load(f_path, map_location="cpu", weights_only=False)
                self._cached_shard_path = f_path
            sample = self._cached_shard_data[offset]
        else:
            # Individual .pt format
            sample = torch.load(f_path, map_location="cpu", weights_only=False)

        # Standardize keys
        image = sample["image"]
        nodes = sample["nodes"]
        dap = sample["dap"] if "dap" in sample else torch.tensor(30.0)
        num_organs = sample["num_organs"] if "num_organs" in sample else (sample.get("num_nodes", torch.tensor(50)))
        exist = sample["existence_mask"] if "existence_mask" in sample else (nodes[:, EMPTY_IDX] < 0.5)

        return {
            "image": image,
            "nodes": nodes,
            "dap": dap.float() if isinstance(dap, torch.Tensor) else torch.tensor(float(dap)),
            "num_organs": num_organs.long() if isinstance(num_organs, torch.Tensor) else torch.tensor(int(num_organs)),
            "existence_mask": exist,
        }


def cowpea_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Dynamic variable-length batch collation."""
    images = torch.stack([b["image"] for b in batch], dim=0)
    daps = torch.stack([b["dap"] for b in batch], dim=0)
    num_organs = torch.stack([b["num_organs"] for b in batch], dim=0)

    max_batch_nodes = int(num_organs.max().item())
    max_batch_nodes = max(max_batch_nodes, 10)

    nodes_list = []
    masks_list = []
    key_padding_masks = []

    for b in batch:
        n_full = b["nodes"][:max_batch_nodes]
        m_full = b["existence_mask"][:max_batch_nodes]
        
        cur_len = n_full.shape[0]
        if cur_len < max_batch_nodes:
            pad_n = torch.zeros((max_batch_nodes - cur_len, FM_NODE_DIM), dtype=torch.float32)
            pad_n[:, EMPTY_IDX] = 1.0
            n_full = torch.cat([n_full, pad_n], dim=0)
            
            pad_m = torch.zeros((max_batch_nodes - cur_len,), dtype=torch.bool)
            m_full = torch.cat([m_full, pad_m], dim=0)
            
            k_mask = torch.cat([
                torch.zeros((cur_len,), dtype=torch.bool),
                torch.ones((max_batch_nodes - cur_len,), dtype=torch.bool)
            ], dim=0)
        else:
            k_mask = torch.zeros((max_batch_nodes,), dtype=torch.bool)

        nodes_list.append(n_full)
        masks_list.append(m_full)
        key_padding_masks.append(k_mask)

    return {
        "images": images,
        "daps": daps,
        "nodes": torch.stack(nodes_list, dim=0),
        "existence_masks": torch.stack(masks_list, dim=0),
        "key_padding_masks": torch.stack(key_padding_masks, dim=0),
        "num_organs": num_organs,
    }
