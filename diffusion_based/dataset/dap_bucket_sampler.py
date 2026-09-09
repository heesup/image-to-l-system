"""DAP-bucketed batch sampler for capacity-homogeneous batches.

Groups samples into DAP buckets so that each mini-batch contains samples of
similar developmental stage. Because the anchor bank slice K is computed from
the batch-max DAP (compute_matryoshka_slice), mixing DAP 1 with DAP 100 in one
batch forces the whole batch through the full 4,096 fine slots. Bucketing makes
the active capacity match the actual botanical content of the batch, cutting
Stage 3 FLOPs/VRAM substantially on young-stage buckets.

Works with DistributedDataParallel (each rank gets a contiguous share of every
bucket, and set_epoch reshuffles within buckets deterministically).
"""

import math
from typing import Iterator, List, Optional

import torch
from torch.utils.data import Sampler


class DAPBucketBatchSampler(Sampler):
    """Batches sample indices such that each batch spans a narrow DAP range.

    Args:
        daps: per-sample DAP values (list/tensor aligned with dataset order).
        batch_size: samples per batch.
        num_buckets: number of equal-width DAP buckets (e.g. 8 -> 12.5-DAP ranges).
        shuffle: shuffle batch order and within-bucket order each epoch.
        world_size/rank: DDP sharding of the batch list.
        seed: base seed for reproducible shuffling.
        drop_last: keep trailing partial batches per bucket (full epoch coverage;
            batch-size variance is acceptable, dropped samples are not).
    """

    def __init__(
        self,
        daps,
        batch_size: int,
        num_buckets: int = 8,
        shuffle: bool = True,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__()
        daps = torch.as_tensor([float(d) for d in daps])
        self.batch_size = int(batch_size)
        self.num_buckets = max(1, int(num_buckets))
        self.shuffle = bool(shuffle)
        self.world_size = max(1, int(world_size))
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        self.dap_min = float(daps.min().item())
        self.dap_max = float(daps.max().item())
        width = (self.dap_max - self.dap_min) / self.num_buckets + 1e-6
        bucket_ids = ((daps - self.dap_min) / width).long().clamp(0, self.num_buckets - 1)

        self.bucket_indices: List[List[int]] = [[] for _ in range(self.num_buckets)]
        for idx, b in enumerate(bucket_ids.tolist()):
            self.bucket_indices[b].append(idx)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _epoch_batches(self) -> List[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        batches: List[List[int]] = []
        for bucket in self.bucket_indices:
            if not bucket:
                continue
            idx = list(bucket)
            if self.shuffle:
                perm = torch.randperm(len(idx), generator=g).tolist()
                idx = [idx[p] for p in perm]
            # DDP: contiguous share of this bucket for this rank
            if self.world_size > 1:
                idx = idx[self.rank::self.world_size]
            if self.batch_size <= 1:
                batches.extend([[i] for i in idx])
                continue
            n = len(idx)
            nb = n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)
            for b in range(nb):
                chunk = idx[b * self.batch_size:(b + 1) * self.batch_size]
                if chunk:
                    batches.append(chunk)
        if self.shuffle:
            order = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[p] for p in order]
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        yield from self._epoch_batches()

    def __len__(self) -> int:
        return len(self._epoch_batches())