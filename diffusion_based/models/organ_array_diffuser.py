"""
94D PlantOrganArray Image-to-Graph Diffusion Model.

Conditioned on a single 2D rendered image, denoises a (N, 94) PlantOrganArray tensor
representing the full plant architecture. The existence channel is the last column.
"""

import math
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from typing import Dict


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MultiScaleImageEncoder(nn.Module):
    """ResNet18-based multi-scale image encoder producing a token grid."""

    def __init__(self, out_dim: int = 256, output_tokens: int = 16):
        super().__init__()
        self.output_tokens = output_tokens
        weights = ResNet18_Weights.DEFAULT
        resnet = resnet18(weights=weights)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = resnet.layer1  # 64 ch
        self.layer2 = resnet.layer2  # 128 ch
        self.layer3 = resnet.layer3  # 256 ch

        self.proj1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((output_tokens, output_tokens)),
            nn.Conv2d(64, out_dim // 4, 1),
        )
        self.proj2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((output_tokens, output_tokens)),
            nn.Conv2d(128, out_dim // 2, 1),
        )
        self.proj3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((output_tokens, output_tokens)),
            nn.Conv2d(256, out_dim // 4, 1),
        )
        self.final_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x0 = self.stem(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        p1 = self.proj1(x1)
        p2 = self.proj2(x2)
        p3 = self.proj3(x3)

        feat = torch.cat([p1, p2, p3], dim=1)  # (B, out_dim, H, W)
        B, C, H, W = feat.shape
        tokens = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return self.final_proj(tokens)


class PlantOrganArrayDiffuser(nn.Module):
    """
    Vision-conditioned diffusion model for 94D PlantOrganArray tensors.

    Args:
        max_nodes: maximum number of phytomer nodes (N)
        node_dim: 94 for the full PlantOrganArray tensor
        embed_dim: hidden dimension for node/image features
        num_layers: number of transformer decoder layers
    """

    def __init__(
        self,
        max_nodes: int = 64,
        node_dim: int = 94,
        embed_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_dim = node_dim
        self.embed_dim = embed_dim

        self.image_encoder = MultiScaleImageEncoder(out_dim=embed_dim, output_tokens=16)

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Project noisy node features + noisy existence into embedding space.
        self.node_proj = nn.Linear(node_dim, embed_dim)
        self.node_pos_emb = nn.Embedding(max_nodes, embed_dim)

        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=512,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # Predict the denoised (clean) 94D organ array.
        # Channels 0..92 are normalized continuous features; channel 93 is existence logit.
        self.node_pred_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, node_dim),
        )

    def forward(
        self,
        noisy_nodes: torch.Tensor,
        timesteps: torch.Tensor,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            noisy_nodes: (B, N, 94) noisy organ array tensor.
                Channels 0..92 are normalized continuous features.
                Channel 93 is the continuous existence signal (modeled in [0,1]).
            timesteps: (B,) diffusion timestep.
            images: (B, 3, H, W) condition image.

        Returns:
            Dict with keys:
                - "pred_x0": (B, N, 94) predicted clean organ array
        """
        B, N, _ = noisy_nodes.shape
        device = noisy_nodes.device

        # Image tokens: (B, 256, embed_dim)
        image_tokens = self.image_encoder(images)

        # Time embedding broadcast: (B, embed_dim) -> (B, 1, embed_dim)
        t_emb = self.time_emb(timesteps).unsqueeze(1)

        # Node embeddings
        node_emb = self.node_proj(noisy_nodes)
        positions = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        node_emb = node_emb + self.node_pos_emb(positions)
        node_emb = node_emb + t_emb

        # Transformer decoder: nodes attend to image tokens
        for layer in self.transformer_layers:
            node_emb = layer(node_emb, image_tokens)

        pred_x0 = self.node_pred_head(node_emb)

        return {
            "pred_x0": pred_x0,
        }
