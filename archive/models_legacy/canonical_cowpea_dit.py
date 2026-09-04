"""
Canonical Cowpea Diffusion Transformer (DiT) Flow Matching Model.

Features:
  1. Conditioned on (RGB Image + Continuous DAP Age Embedding).
  2. Variable-length Transformer Decoder with Attention Key-Padding Masks (No fixed 2048 padding).
  3. Auxiliary Organ Count Predictor: predicts exact organ count N_hat for variable-length generation.
  4. Straight-line Velocity Field Prediction for 26D Canonical Botanical Part Slots.
"""

import math
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_based.models.vit_image_encoder import ViTImageEncoder


class CanonicalCowpeaDiTModel(nn.Module):
    """
    DiT Model with DAP conditioning and variable-length dynamic decoding.
    """

    def __init__(
        self,
        max_slots: int = 60,
        node_dim: int = 26,
        image_size: int = 128,
        patch_size: int = 8,
        embed_dim: int = 384,
        encoder_layers: int = 8,
        decoder_layers: int = 6,
        num_heads: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_slots = max_slots
        self.node_dim = node_dim
        self.embed_dim = embed_dim

        # 1. ViT Image Encoder (Condition Tokens)
        self.image_encoder = ViTImageEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
            num_layers=encoder_layers,
            num_heads=num_heads,
        )

        # 2. DAP (Plant Age) Continuous Embedding
        self.dap_embed = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

        # 3. Timestep Continuous Embedding (Sinusoidal + MLP)
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 4. Canonical Slot Queries & Learned Botanical Positional Embeddings
        self.slot_proj = nn.Linear(node_dim, embed_dim)
        self.slot_pos_emb = nn.Embedding(max_slots, embed_dim)

        # 5. Transformer Decoder with Multi-Head Self-Attention and Image Cross-Attention
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

        # 6. Velocity Prediction Head
        self.velocity_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, node_dim),
        )

        # 7. Auxiliary Organ Count Predictor (predicts exact number of organs)
        self.count_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def _sinusoidal(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        dim = self.embed_dim
        half_dim = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=device).float() / (half_dim - 1))
        args = timesteps.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(
        self,
        noisy_slots: torch.Tensor,
        timesteps: torch.Tensor,
        images: torch.Tensor,
        daps: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            noisy_slots: (B, N_batch, 26) interpolated part tensor x_t.
            timesteps: (B,) continuous flow time t in [0, 1].
            images: (B, 3, H, W) normalized condition image.
            daps: (B,) continuous plant age in DAP.
            key_padding_mask: (B, N_batch) bool mask (True = padding).
        Returns:
            {'pred_velocity': (B, N_batch, 26), 'pred_count': (B, 1)}
        """
        B, N, _ = noisy_slots.shape
        device = noisy_slots.device

        # Encode image tokens
        img_tokens = self.image_encoder(images)  # (B, num_patches + 1, embed_dim)

        # Encode DAP age token
        dap_token = self.dap_embed(daps.view(B, 1, 1) / 30.0)  # (B, 1, embed_dim)
        memory = torch.cat([img_tokens, dap_token], dim=1)      # (B, num_patches + 2, embed_dim)

        # Predict total organ count from global context token
        pred_count = self.count_head(memory[:, 0])  # (B, 1)

        # Time embedding
        t_emb = self.time_embed(self._sinusoidal(timesteps))  # (B, embed_dim)

        # Slot queries with learned anatomical position embeddings
        pos_indices = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        queries = self.slot_proj(noisy_slots) + self.slot_pos_emb(pos_indices)
        queries = queries + t_emb.unsqueeze(1)

        # Transformer cross-attention decoding with dynamic key padding mask
        x = self.decoder(
            tgt=queries,
            memory=memory,
            tgt_key_padding_mask=key_padding_mask,
        )

        pred_velocity = self.velocity_head(x)  # (B, N, 26)
        return {
            "pred_velocity": pred_velocity,
            "pred_count": pred_count,
        }
