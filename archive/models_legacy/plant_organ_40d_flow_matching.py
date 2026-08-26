"""
Direct 40D Plant Organ Array Flow Matching Model.

Predicts the straight-line velocity field transporting pure Gaussian noise
x_0 ~ N(0, I) in R^{N x 40} directly to the normalized 40D PlantOrganArray x_1,
conditioned on a single RGB observation image via ViT token embeddings.
Zero Ground Truth structure leakage.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_based.models.vit_image_encoder import ViTImageEncoder
from diffusion_based.models.plant_global_vae import OrganFeatureNormalizer


class PlantOrgan40DFlowMatchingModel(nn.Module):
    """
    ViT Image Encoder + Cross-Attention Transformer Decoder predicting 40D velocity field.
    """

    def __init__(
        self,
        max_organs: int = 1200,
        organ_dim: int = 40,
        image_size: int = 128,
        patch_size: int = 8,
        embed_dim: int = 256,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_organs = max_organs
        self.organ_dim = organ_dim
        self.embed_dim = embed_dim

        # 1. ViT Image Encoder (processes RGB condition into patch tokens)
        self.image_encoder = ViTImageEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
            num_layers=encoder_layers,
            num_heads=num_heads,
        )

        # 2. Continuous Timestep Embedding (Sinusoidal + MLP)
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 3. Organ Query Projections
        self.organ_proj = nn.Linear(organ_dim, embed_dim)
        self.organ_pos_emb = nn.Embedding(max_organs, embed_dim)

        # 4. Transformer Decoder with Multi-Head Self-Attention and Image Cross-Attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        # 5. Velocity Prediction Head
        self.velocity_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, organ_dim),
        )

    def _sinusoidal_emb(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        dim = self.embed_dim
        half_dim = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=device).float() / (half_dim - 1))
        args = timesteps.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(
        self,
        noisy_organs: torch.Tensor,
        timesteps: torch.Tensor,
        images: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            noisy_organs: (B, N, 40) interpolated organ tensor x_t.
            timesteps: (B,) continuous flow time t in [0, 1].
            images: (B, 3, H, W) normalized condition image.
            key_padding_mask: (B, N) bool mask (True = padding).
        Returns:
            {'pred_velocity': (B, N, 40)} predicted velocity field v_theta.
        """
        B, N, _ = noisy_organs.shape
        device = noisy_organs.device

        # Encode image condition
        image_tokens = self.image_encoder(images)  # (B, num_patches + 1, embed_dim)

        # Compute time embedding
        t_emb = self.time_embed(self._sinusoidal_emb(timesteps))  # (B, embed_dim)

        # Project noisy organ tokens into query space
        pos_indices = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        queries = self.organ_proj(noisy_organs) + self.organ_pos_emb(pos_indices)  # (B, N, embed_dim)
        queries = queries + t_emb.unsqueeze(1)

        # Transformer cross-attention decoding
        x = self.decoder(
            tgt=queries,
            memory=image_tokens,
            tgt_key_padding_mask=key_padding_mask,
        )

        pred_velocity = self.velocity_head(x)  # (B, N, 40)
        return {"pred_velocity": pred_velocity}
