"""Fully end-to-end differentiable Helios renderer.

Composes `nodes_to_geometry_torch` and `render_torch_geometry` so that a
batch of 22D organ nodes can be directly back-propagated to RGBA images.

Node vector layout (25D):
  [0:3]   xyz          - base position (m)
  [3]     length       - organ length (m)
  [4]     radius       - organ radius / leaf width (m)
  [5:14]  R_flat       - 3x3 orientation matrix (row-major), local frame to world
  [14:20] organ_onehot - 6-channel one-hot (INTERNODE, PETIOLE, LEAF,
                          FLORAL_BUD, FLOWER, POD)
  [20]    shoot_id
  [21]    phytomer_idx
  [22]    existence    - confidence [0, 1]
  [23]    head_radius  - flower/pod head radius (m); 0 for non-floral
  [24]    parent_idx   - global parent node index (-1 = root)

Usage:
    parser = HeliosXMLParser(xml_path); parser.parse()
    organ_nodes = parser.get_all_organ_nodes()
    nodes = torch.tensor(
        np.stack([n.to_vec() for n in organ_nodes]), dtype=torch.float32
    ).unsqueeze(0)  # (1, N, 25)

    renderer = DifferentiableHeliosRenderer(rasterizer)
    rgba = renderer(nodes, focus_plant=True, background="black")  # (1, 4, H, W)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from diffusion_based.models.helios_geometry import nodes_to_geometry_torch
from diffusion_based.models.helios_rasterizer_3d import HeliosGeometryRasterizer
from diffusion_based.models.legacy.helios_geometry_legacy import (
    DifferentiableHeliosXMLRenderer,
    build_helios_geometry_from_xml,
)


class DifferentiableHeliosRenderer(nn.Module):
    """Differentiable pipeline: 25D (or 22D/19D/18D) organ nodes → RGBA image.

    The parent graph topology is embedded in the final channel of the node vector,
    so no separate ``parents`` tensor is required.

    For backward-compatibility with older 18D inputs, an optional ``parents``
    argument is still accepted as a fallback.
    """

    def __init__(self, rasterizer: HeliosGeometryRasterizer):
        super().__init__()
        self.rasterizer = rasterizer

    def forward(
        self,
        nodes: torch.Tensor,                     # (B, N, 25) or (B, N, 22/19)
        parents: Optional[torch.Tensor] = None,  # fallback for 18D legacy inputs
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
        """Render 25D nodes to RGBA (B, 4, H, W).

        For 25D/22D/19D nodes, parent_idx is read from the last channel automatically.
        The ``parents`` argument is only used for 18D/legacy inputs.
        """
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
