"""Differentiable 3D point-cloud loss for plant architecture reconstruction.

Combines:
  - Chamfer distance between predicted and target point clouds.
  - Optional organ-aware Chamfer loss (weighted by organ class).
  - Coordinate normalization utilities to match target PLY dimensions.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_based.models.legacy.helios_geometry_track_a import DifferentiablePlantPointCloud


class PlantPointCloudChamferLoss(nn.Module):
    """End-to-end differentiable loss: 15D nodes -> point cloud -> Chamfer distance.

    This module holds a DifferentiablePlantPointCloud sampler and a Chamfer
    distance loss. It is meant to replace the 2D render loss in the diffusion
    training loop.
    """

    def __init__(
        self,
        n_cylinder_circ: int = 8,
        n_cylinder_axis: int = 4,
        n_leaf_u: int = 6,
        n_leaf_v: int = 10,
        n_ellipsoid_theta: int = 8,
        n_ellipsoid_phi: int = 6,
        organ_weights: Optional[Tuple[float, float, float, float]] = None,
    ):
        super().__init__()
        self.sampler = DifferentiablePlantPointCloud(
            n_cylinder_circ=n_cylinder_circ,
            n_cylinder_axis=n_cylinder_axis,
            n_leaf_u=n_leaf_u,
            n_leaf_v=n_leaf_v,
            n_ellipsoid_theta=n_ellipsoid_theta,
            n_ellipsoid_phi=n_ellipsoid_phi,
        )
        self.organ_weights = organ_weights
        if organ_weights is not None:
            self.register_buffer(
                "organ_w",
                torch.tensor(organ_weights, dtype=torch.float32).view(1, 1, 4),
            )

    def normalize_target(
        self,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return normalized target, center, scale based on target bbox."""
        min_xyz = target.min(dim=1)[0]  # (B, 3)
        max_xyz = target.max(dim=1)[0]  # (B, 3)
        scale = (max_xyz - min_xyz).max(dim=-1)[0].clamp(min=1e-6)  # (B,)
        center = (min_xyz + max_xyz) / 2.0  # (B, 3)
        return (target - center.unsqueeze(1)) / scale.unsqueeze(1).unsqueeze(2), center, scale

    def chamfer(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        chunk: int = 1024,
    ) -> torch.Tensor:
        """Compute symmetric Chamfer distance in tiny chunks to save memory.

        Args:
            pred: (B, M, 3)
            target: (B, K, 3)
            weights: optional (B, M, 1) per-point weights for pred side.
            chunk: max points per cdist dimension (default 1024).
        Returns:
            loss: scalar
        """
        B, M, _ = pred.shape
        K = target.shape[1]

        # pred -> target: chunk target.
        pred_to_target = torch.full((B, M), float("inf"), device=pred.device, dtype=pred.dtype)
        for k_start in range(0, K, chunk):
            k_end = min(k_start + chunk, K)
            tgt_chunk = target[:, k_start:k_end, :]
            for m_start in range(0, M, chunk):
                m_end = min(m_start + chunk, M)
                pred_chunk = pred[:, m_start:m_end, :]
                d_chunk = torch.cdist(pred_chunk, tgt_chunk, p=2)  # (B, m_chunk, k_chunk)
                pred_to_target[:, m_start:m_end] = torch.minimum(
                    pred_to_target[:, m_start:m_end],
                    d_chunk.min(dim=-1)[0],
                )

        # target -> pred: chunk pred.
        target_to_pred = torch.full((B, K), float("inf"), device=target.device, dtype=target.dtype)
        for m_start in range(0, M, chunk):
            m_end = min(m_start + chunk, M)
            pred_chunk = pred[:, m_start:m_end, :]
            for k_start in range(0, K, chunk):
                k_end = min(k_start + chunk, K)
                tgt_chunk = target[:, k_start:k_end, :]
                d_chunk = torch.cdist(pred_chunk, tgt_chunk, p=2)  # (B, m_chunk, k_chunk)
                target_to_pred[:, k_start:k_end] = torch.minimum(
                    target_to_pred[:, k_start:k_end],
                    d_chunk.min(dim=1)[0],
                )

        if weights is not None:
            weights = weights.squeeze(-1)  # (B, M)
            pred_to_target = (pred_to_target * weights).sum(dim=-1) / weights.sum(dim=1).clamp(min=1e-6)

        return (pred_to_target.mean(dim=-1) + target_to_pred.mean(dim=-1)).mean()

    def forward(
        self,
        pred_nodes: torch.Tensor,
        target_xyz: torch.Tensor,
        parent_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute 3D point-cloud loss.

        Args:
            pred_nodes: (B, N, 15) predicted 15D organ graph.
            target_xyz: (B, K, 3) target point cloud (Helios/real PLY) in its own coordinates.
            parent_indices: optional (B, N), not currently used by sampler.

        Returns:
            loss: scalar tensor
            info: dict with auxiliary tensors (pred_xyz, target_norm, etc.)
        """
        B = pred_nodes.shape[0]
        device = pred_nodes.device

        # Build predicted point cloud (differentiable)
        pred_xyz, organ_weights, existence_weights = self.sampler(pred_nodes)

        # Normalize target to its own bounding box, then apply same scale to prediction.
        target_norm, center, scale = self.normalize_target(target_xyz)
        pred_norm = (pred_xyz - center.unsqueeze(1)) / scale.unsqueeze(1).unsqueeze(2)

        # Compute overall Chamfer distance
        loss = self.chamfer(pred_norm, target_norm, weights=existence_weights)

        # Organ-aware weighted loss
        organ_loss = torch.tensor(0.0, device=device)
        if self.organ_weights is not None:
            for organ_idx in range(4):
                w = organ_weights[..., organ_idx].unsqueeze(-1)  # (B, M, 1)
                organ_loss = organ_loss + self.organ_w[0, 0, organ_idx] * self.chamfer(
                    pred_norm, target_norm, weights=w
                )
            loss = loss + organ_loss

        info = {
            "pred_xyz": pred_xyz,
            "pred_norm": pred_norm,
            "target_norm": target_norm,
            "organ_weights": organ_weights,
            "existence_weights": existence_weights,
        }
        return loss, info


def chamfer_distance_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    """Symmetric Chamfer distance between two (M,3) and (K,3) numpy point clouds."""
    from scipy.spatial.distance import cdist
    d = cdist(pred, target, metric="euclidean")
    return float(d.min(axis=1).mean() + d.min(axis=0).mean())


def normalize_point_clouds(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """Normalize both point clouds into the same unit cube based on target bbox."""
    if isinstance(target, torch.Tensor):
        t_np = target.detach().cpu().numpy()
    else:
        t_np = target
    min_xyz = t_np.min(axis=0)
    max_xyz = t_np.max(axis=0)
    scale = (max_xyz - min_xyz).max()
    if scale < 1e-6:
        scale = 1.0
    center = (min_xyz + max_xyz) / 2.0

    def _normalize(cloud):
        if isinstance(cloud, torch.Tensor):
            c = torch.tensor(center, dtype=cloud.dtype, device=cloud.device)
            s = torch.tensor(scale, dtype=cloud.dtype, device=cloud.device)
            return (cloud - c) / s
        return (cloud - center) / scale

    return _normalize(pred), _normalize(target)


def write_ply(path: str, xyz: np.ndarray, colors: np.ndarray, organs: np.ndarray) -> None:
    """Write a binary PLY with x,y,z,r,g,b,organ."""
    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {xyz.shape[0]}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "property uchar organ\n"
            "end_header\n"
        )
        f.write(header.encode())
        for i in range(xyz.shape[0]):
            f.write(xyz[i].astype(np.float32).tobytes())
            f.write(colors[i].astype(np.uint8).tobytes())
            f.write(np.array([organs[i]], dtype=np.uint8).tobytes())


def load_ply_to_tensor(
    path: str,
    opacity_threshold: float = 0.5,
    subsample: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Load a PLY file (Helios or Gaussian splat) into a (1, K, 3) torch tensor.

    For Gaussian-splat PLYs, opacity is sigmoid-applied and points below the
    threshold are filtered out. Colors are ignored.
    """
    import struct

    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.strip() == b"end_header":
                break
        header_str = header.decode()
        n_verts = 0
        props = []
        for line in header_str.split("\n"):
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("property float"):
                props.append(line.split()[-1])

        fmt = "<" + "f" * len(props)
        data = np.fromfile(f, dtype=np.float32, count=len(props) * n_verts).reshape(n_verts, len(props))

    xyz = data[:, :3].astype(np.float32)

    # Filter Gaussian splats by opacity if the property exists
    if "opacity" in props:
        opacity = data[:, props.index("opacity")]
        opacity = 1.0 / (1.0 + np.exp(-opacity))
        mask = opacity >= opacity_threshold
        xyz = xyz[mask]

    # Subsample if requested
    if subsample is not None and xyz.shape[0] > subsample:
        idx = np.random.choice(xyz.shape[0], subsample, replace=False)
        xyz = xyz[idx]

    return torch.from_numpy(xyz).unsqueeze(0).to(device)
