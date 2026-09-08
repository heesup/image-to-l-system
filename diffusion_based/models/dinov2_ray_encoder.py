"""
DINOv2 Image Encoder with 3D Camera Ray Positional Embedding (PETR-style).

Extracts visual representations using a pretrained DINOv2 ViT-S/14 backbone and
lifts 2D patch tokens into 3D Frustum space using normalized camera ray direction vectors.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2RayEncoder(nn.Module):
    """
    Vision encoder coupling pretrained DINOv2-S/14 with 3D camera ray positional embeddings.

    Args:
        pretrained: Whether to load official DINOv2-S/14 pretrained weights.
        freeze_backbone: Whether to freeze DINOv2 parameters.
        embed_dim: Token embedding dimension (384 for ViT-S/14).
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = 256  # 16x16 grid for 224x224 input with patch size 14

        # 1. Pretrained DINOv2 ViT-S/14 Backbone
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=pretrained)
        self.dino_dim = 384

        # Linear projection if model embed_dim differs from DINOv2 native 384
        if embed_dim != self.dino_dim:
            self.proj = nn.Linear(self.dino_dim, embed_dim)
        else:
            self.proj = nn.Identity()

        # 2. 3D Ray Positional Embedding (PETR / DUSt3R style)
        # Ray direction: v = [x_ndc, y_ndc, 1.0]^T normalized
        self.ray_mlp = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Precompute canonical normalized camera ray directions for a 16x16 patch grid
        # Coordinate mapping: u, v in [0, 1] -> NDC in [-1, 1]
        coords_y = torch.linspace(-1.0, 1.0, 16)
        coords_x = torch.linspace(-1.0, 1.0, 16)
        grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing="ij")  # (16, 16)
        grid_z = torch.ones_like(grid_x)  # Forward camera viewing direction
        rays = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (16, 16, 3)
        rays = rays / torch.norm(rays, dim=-1, keepdim=True)
        self.register_buffer("canonical_rays", rays.view(1, 256, 3))

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, C, H, W) normalized image tensor. Uses channels 0:3 (RGB).
        Returns:
            (B, 257, embed_dim) visual tokens:
                token 0: CLS token
                tokens 1..256: 3D Ray-embedded spatial patch tokens
        """
        # Extract RGB channels (channels 0:3)
        rgb = x[:, :3].float()
        if rgb.shape[-1] != 224 or rgb.shape[-2] != 224:
            rgb = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)

        features = self.backbone.forward_features(rgb)
        patch_tokens = self.proj(features["x_norm_patchtokens"])  # (B, 256, embed_dim)
        cls_token = self.proj(features["x_norm_clstoken"]).unsqueeze(1)  # (B, 1, embed_dim)

        # Inject 3D ray embedding into patch tokens
        ray_embs = self.ray_mlp(self.canonical_rays)  # (1, 256, embed_dim)
        patch_tokens = patch_tokens + ray_embs

        # Prepend CLS token
        return torch.cat([cls_token, patch_tokens], dim=1)
