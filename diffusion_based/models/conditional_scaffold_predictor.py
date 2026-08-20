"""
Conditional Botanical Scaffold Predictor.

Stage-1 Coarse Structural Prior:
Predicts developmental age (DAP), 3D canopy bounding volume (Radius, Height, Leaf Scale),
and active organ count directly from 2D RGB / Depth imagery, then instantiates a
developmentally conditioned 3D Botanical Scaffold x_0(I) for Stage-2 Flow Matching ODE.
"""

import math
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_based.dataset.part_array_dataset import FM_NODE_DIM
from diffusion_based.models.botanical_scaffold import BotanicalScaffoldGenerator


class ConditionalScaffoldPredictor(nn.Module):
    """
    Predicts coarse plant developmental parameters from 2D visual observations (RGB / Depth)
    and constructs a stage-conditioned 3D Botanical Scaffold x_0 in FM feature space.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        max_nodes: int = 512,
        node_dim: int = FM_NODE_DIM,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_dim = node_dim
        self.scaffold_gen = BotanicalScaffoldGenerator(max_nodes=max_nodes, node_dim=node_dim)

        # Lightweight multi-scale convolutional visual backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, embed_dim, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Stage estimation MLP head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 5),  # [dap_norm, radius_norm, height_norm, leaf_scale_norm, active_norm]
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input image tensor (B, C, H, W)
        Returns:
            Dictionary with predicted physical parameters:
              - 'pred_dap': (B, 1) in days [5, 100]
              - 'pred_radius': (B, 1) in meters [0.08, 1.0]
              - 'pred_height': (B, 1) in meters [0.08, 1.0]
              - 'pred_leaf_scale': (B, 1) in meters [0.02, 0.15]
              - 'pred_active_count': (B, 1) integer count [12, max_nodes]
        """
        feats = self.backbone(x)
        raw = torch.sigmoid(self.head(feats))

        dap = 5.0 + raw[:, 0:1] * 90.0
        radius = 0.08 + raw[:, 1:2] * 0.85
        height = 0.08 + raw[:, 2:3] * 0.85
        leaf_scale = 0.025 + raw[:, 3:4] * 0.100
        active_count = 12.0 + raw[:, 4:5] * (self.max_nodes - 12.0)

        return {
            "pred_dap": dap,
            "pred_radius": radius,
            "pred_height": height,
            "pred_leaf_scale": leaf_scale,
            "pred_active_count": active_count,
        }

    def predict_scaffold_batch(
        self,
        x: torch.Tensor,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        """
        Predicts physical parameters for a batch of images and generates the corresponding
        stage-conditioned 3D Botanical Scaffold prior x_0 in FM feature space.
        Returns:
            scaffolds: (B, max_nodes, node_dim)
        """
        preds = self.forward(x)
        B = x.shape[0]
        device = x.device

        scaffolds = []
        for b in range(B):
            r = float(preds["pred_radius"][b].item())
            h = float(preds["pred_height"][b].item())
            ls = float(preds["pred_leaf_scale"][b].item())
            cnt = int(round(float(preds["pred_active_count"][b].item())))

            scaff = self.scaffold_gen.generate_conditioned(
                radius=r,
                height=h,
                leaf_scale=ls,
                active_count=cnt,
                device=device,
            )
            scaffolds.append(scaff)

        scaffolds_t = torch.stack(scaffolds, dim=0)

        if noise_std > 0:
            from diffusion_based.dataset.part_array_dataset import FM_BASE_START, FM_BASE_END, FM_SCALE_START, FM_SCALE_END
            base_noise = torch.randn((B, self.max_nodes, 3), device=device) * noise_std
            scaffolds_t[:, :, FM_BASE_START:FM_BASE_END] += base_noise
            scale_noise = torch.randn((B, self.max_nodes, 3), device=device) * (noise_std * 0.5)
            scaffolds_t[:, :, FM_SCALE_START:FM_SCALE_END] = (scaffolds_t[:, :, FM_SCALE_START:FM_SCALE_END] + scale_noise).clamp(min=1e-4)

        return scaffolds_t
