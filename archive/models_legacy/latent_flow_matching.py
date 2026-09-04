"""
Latent Flow Matching (LFM) Model for 3D Plant Organ Generation.

Trains a ViT image encoder + Transformer decoder to predict the straight-line velocity
field v_theta transporting Gaussian prior noise z_0 ~ N(0, I) to the continuous
512-dimensional plant organ latent manifold z_1 in R^{N x 512}, conditioned on a plant RGB image.

At inference, generates high-fidelity plant latents in 10-20 ODE Euler integration steps,
which are then decoded into full 40D typed organ arrays and Helios XMLs via PlantOrganVAE.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_based.models.vit_image_encoder import ViTImageEncoder


class LatentFlowMatchingModel(nn.Module):
    """
    ViT image encoder + Cross-Attention Transformer decoder predicting velocity field in 512D latent space.
    """

    def __init__(
        self,
        max_organs: int = 2048,
        latent_dim: int = 512,
        image_size: int = 128,
        patch_size: int = 8,
        embed_dim: int = 512,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_organs = max_organs
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        # 1. ViT Image Encoder (extracts visual condition tokens)
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

        # 3. Latent Organ Token Projections
        self.latent_proj = nn.Linear(latent_dim, embed_dim)
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
            nn.Linear(embed_dim, latent_dim),
        )

    def _sinusoidal_emb(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Computes sinusoidal positional embeddings for continuous timesteps t in [0, 1].
        """
        device = timesteps.device
        dim = self.embed_dim
        half_dim = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=device).float() / (half_dim - 1))
        args = timesteps.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        images: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            noisy_latents: (B, N, latent_dim) interpolated latent tensor z_t.
            timesteps: (B,) continuous flow time t in [0, 1].
            images: (B, 3, H, W) normalized condition image.
            key_padding_mask: (B, N) bool mask (True = ignore/padding).
        Returns:
            {'pred_velocity': (B, N, latent_dim)} predicted velocity field v_theta.
        """
        B, N, _ = noisy_latents.shape
        device = noisy_latents.device

        # Encode image condition
        image_tokens = self.image_encoder(images)  # (B, num_patches, embed_dim)

        # Compute time embedding
        t_emb = self.time_embed(self._sinusoidal_emb(timesteps))  # (B, embed_dim)

        # Project latents and add organ order positional embedding + time conditioning
        pos_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        queries = self.latent_proj(noisy_latents) + self.organ_pos_emb(pos_ids)
        queries = queries + t_emb.unsqueeze(1)

        # Transformer Cross-Attention Decoder
        hidden = self.decoder(
            tgt=queries,
            memory=image_tokens,
            tgt_key_padding_mask=key_padding_mask,
        )

        # Predict velocity in 512D latent space
        pred_velocity = self.velocity_head(hidden)

        return {"pred_velocity": pred_velocity}
