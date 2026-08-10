"""Fully end-to-end differentiable Helios renderer.

Composes `nodes_to_geometry_torch` and `render_torch_geometry` so that a
batch of 15D nodes can be directly back-propagated to RGBA images.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_based.models.helios_geometry import nodes_to_geometry_torch
from diffusion_based.models.helios_rasterizer_3d import HeliosGeometryRasterizer


class DifferentiableHeliosRenderer(nn.Module):
    """Differentiable pipeline: 15D nodes -> RGBA image."""

    def __init__(self, rasterizer: HeliosGeometryRasterizer):
        super().__init__()
        self.rasterizer = rasterizer

    def forward(
        self,
        nodes: torch.Tensor,                     # (B, N, 15)
        parents: Optional[torch.Tensor] = None,
        camera_height: float = 1.0,
        distance_from_center: float = 0.0,
        azimuth_deg: float = 0.0,
        hfov_deg: Optional[float] = None,
        target_center: Optional[torch.Tensor] = None,
        sun_dir: Optional[torch.Tensor] = None,
        focus_plant: bool = False,
        background: Optional[str] = None,
        leaf_sigma: Optional[float] = None,
    ) -> torch.Tensor:
        """Render 15D nodes to RGBA (B, 4, H, W)."""
        (
            tube_verts,
            tube_radii,
            tube_organs,
            leaf_verts,
            leaf_faces,
            leaf_organs,
            bud_centers,
            bud_radii,
            bud_lengths,
            bud_organs,
        ) = nodes_to_geometry_torch(nodes, parents)

        return self.rasterizer.render_torch_geometry(
            tube_verts,
            tube_radii,
            tube_organs,
            leaf_verts,
            leaf_faces,
            leaf_organs,
            bud_centers,
            bud_radii,
            bud_lengths,
            bud_organs,
            camera_height=camera_height,
            distance_from_center=distance_from_center,
            azimuth_deg=azimuth_deg,
            hfov_deg=hfov_deg,
            target_center=target_center,
            sun_dir=sun_dir,
            focus_plant=focus_plant,
            background=background,
            leaf_sigma=leaf_sigma,
        )
