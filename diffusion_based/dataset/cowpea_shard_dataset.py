"""
High-Throughput Shard Dataset for Multi-Species Plant Training.
Streams pre-cached tensor shards (.pt) with variable-length dynamic batching.
Automatically discovers pre-rendered shards or falls back to preprocessing raw Helios XMLs.
"""

import os
import glob
from typing import List, Dict, Any, Tuple, Optional

import torch
from torch.utils.data import Dataset
from diffusion_based.dataset.part_array_dataset import FM_NODE_DIM, EMPTY_IDX


class PlantShardDataset(Dataset):
    """Loads pre-rendered tensor shards from dataset/cache_<species>_100k/."""

    def __init__(
        self,
        species: str = "cowpea",
        cache_dir: Optional[str] = None,
        fallback_cache_dir: str = "dataset/cache",
        data_root: str = "dataset/helios_data",
        max_samples: Optional[int] = None,
        auto_preprocess_if_missing: bool = True,
        auto_preprocess_samples: int = 1000,
    ):
        # Support positional directory path as first argument
        if "/" in str(species) or os.path.isdir(str(species)) or str(species).endswith("_shard"):
            cache_dir = species
            self.species = "cowpea"
        else:
            self.species = species

        self.data_root = os.path.abspath(data_root)
        if cache_dir is None:
            primary_shard_dir = os.path.join(self.data_root, f"{self.species}_shard")
            if os.path.exists(primary_shard_dir) and glob.glob(os.path.join(primary_shard_dir, "*.pt")):
                cache_dir = primary_shard_dir
            elif os.path.exists(f"dataset/cache_{self.species}_100k") and glob.glob(f"dataset/cache_{self.species}_100k/*.pt"):
                cache_dir = f"dataset/cache_{self.species}_100k"
            else:
                cache_dir = primary_shard_dir
        self.cache_dir = os.path.abspath(cache_dir)
        self.fallback_dir = os.path.abspath(fallback_cache_dir)

        self.shard_files = sorted(glob.glob(os.path.join(self.cache_dir, "*.pt")))
        self.individual_files = []

        if not self.shard_files and os.path.exists(self.fallback_dir):
            self.individual_files = sorted(glob.glob(os.path.join(self.fallback_dir, f"{species}_*.pt")))

        # If no shards or cache found, optionally auto-generate
        if not self.shard_files and not self.individual_files and auto_preprocess_if_missing:
            species_raw_dir = os.path.join(self.data_root, species)
            raw_xmls = glob.glob(os.path.join(species_raw_dir, "**", "*_plant_*.xml"), recursive=True)
            if raw_xmls:
                print(f"[PlantShardDataset] No cached shards found in '{self.cache_dir}'.")
                print(f"[PlantShardDataset] Found {len(raw_xmls)} raw XMLs. Starting on-the-fly sharding ({auto_preprocess_samples:,} samples)...")
                try:
                    from diffusion_based.dataset.generate_tensor_shards import generate_shards
                    generate_shards(
                        species=self.species,
                        data_root=self.data_root,
                        output_dir=self.cache_dir,
                        total_samples=auto_preprocess_samples,
                        num_workers=1,
                        worker_id=0,
                        shard_size=100,
                    )
                    self.shard_files = sorted(glob.glob(os.path.join(self.cache_dir, "*.pt")))
                except Exception as e:
                    print(f"[PlantShardDataset] Warning: Auto-sharding failed: {e}")
                    print(f"[PlantShardDataset] To generate full 100K shards on SLURM cluster, run:")
                    print(f"    ./slurm_scripts/generate_tensor_shards_jobs.sh --species {species} --submit")
            else:
                print(f"[PlantShardDataset] Notice: Neither pre-computed shards ({self.cache_dir}) nor base XMLs ({species_raw_dir}) were found.")
                print(f"[PlantShardDataset] Please generate base Helios data first via:")
                print(f"    ./slurm_scripts/generate_helios_dataset_jobs.sh --plant-types {species} --seeds 100 --submit")

        # Fast Indexing: Shards are generated in uniform chunks of 100 samples
        self.index: List[Tuple[str, int]] = []
        if self.shard_files:
            # Check the first shard size
            sample_size = 100
            try:
                first_shard = torch.load(self.shard_files[0], map_location="cpu", weights_only=False)
                if isinstance(first_shard, list):
                    sample_size = len(first_shard)
            except Exception:
                sample_size = 100

            for s_path in self.shard_files:
                for off in range(sample_size):
                    self.index.append((s_path, off))
        elif self.individual_files:
            for f_path in self.individual_files:
                self.index.append((f_path, -1))

        if max_samples is not None and len(self.index) > max_samples:
            self.index = self.index[:max_samples]

        self._cached_shard_path = None
        self._cached_shard_data = None

        print(f"PlantShardDataset ({self.species}): Loaded {len(self.index):,} indexed training samples across {len(self.shard_files)} shards.")

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


# Alias for backward compatibility
CowpeaShardDataset = PlantShardDataset


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
        n_tensor = b["nodes"]
        m_tensor = b.get("existence_mask", None)
        if m_tensor is None or not isinstance(m_tensor, torch.Tensor):
            m_tensor = n_tensor[:, EMPTY_IDX] < 0.5
        
        # Pad / Slice nodes to max_batch_nodes
        cur_nodes = n_tensor[:max_batch_nodes]
        cur_len_n = cur_nodes.shape[0]
        if cur_len_n < max_batch_nodes:
            pad_n = torch.zeros((max_batch_nodes - cur_len_n, FM_NODE_DIM), dtype=torch.float32)
            pad_n[:, EMPTY_IDX] = 1.0
            cur_nodes = torch.cat([cur_nodes, pad_n], dim=0)

        # Pad / Slice existence mask to max_batch_nodes
        cur_mask = m_tensor[:max_batch_nodes]
        cur_len_m = cur_mask.shape[0]
        if cur_len_m < max_batch_nodes:
            pad_m = torch.zeros((max_batch_nodes - cur_len_m,), dtype=torch.bool)
            cur_mask = torch.cat([cur_mask, pad_m], dim=0)

        # Key padding mask: True for slots beyond active organ count
        n_org = int(b["num_organs"].item()) if isinstance(b["num_organs"], torch.Tensor) else int(b["num_organs"])
        n_active = min(n_org, max_batch_nodes)
        k_mask = torch.zeros((max_batch_nodes,), dtype=torch.bool)
        if n_active < max_batch_nodes:
            k_mask[n_active:] = True

        nodes_list.append(cur_nodes)
        masks_list.append(cur_mask)
        key_padding_masks.append(k_mask)

    return {
        "images": images,
        "daps": daps,
        "nodes": torch.stack(nodes_list, dim=0),
        "existence_masks": torch.stack(masks_list, dim=0),
        "key_padding_masks": torch.stack(key_padding_masks, dim=0),
        "num_organs": num_organs,
    }
