"""
DiT-Large (150M Parameters) Architecture for Canonical Botanical Crop Generation.
Features:
  - 16-layer ViT Image Encoder (embed_dim=768, 16 heads)
  - 12-layer Transformer Decoder (embed_dim=768, 16 heads)
  - DAP continuous age conditioning + Camera viewpoint conditioning
  - Dynamic Variable-Length attention key padding masking
  - Auxiliary organ count regression head
"""

import math
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed2D(nn.Module):
    """2D Image to Patch Embedding."""
    def __init__(self, img_size: int = 128, patch_size: int = 8, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embedding for continuous diffusion/flow time t in [0, 1]."""
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return emb


class CanonicalCowpeaDiTLargeModel(nn.Module):
    """
    150M-parameter Large Diffusion Transformer for Variable-Length Botanical Crop Flow Matching.
    """
    def __init__(
        self,
        max_slots: int = 4096,
        node_dim: int = 26,
        image_size: int = 128,
        patch_size: int = 8,
        embed_dim: int = 768,
        encoder_layers: int = 16,
        decoder_layers: int = 12,
        num_heads: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_slots = max_slots
        self.node_dim = node_dim
        self.embed_dim = embed_dim
        self.image_size = image_size

        # 1. Condition Encoders
        self.patch_embed = PatchEmbed2D(img_size=image_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.img_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # ViT Image Encoder (16 Layers)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.image_encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers)

        # Continuous DAP Age & Viewpoint Conditioners
        self.time_embed = SinusoidalTimestepEmbedding(embed_dim)
        self.dap_mlp = nn.Sequential(
            SinusoidalTimestepEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.viewpoint_mlp = nn.Sequential(
            nn.Linear(5, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 2. Flow Matching Target Tokens & Positional Embeddings
        self.node_in_proj = nn.Linear(node_dim, embed_dim)
        self.slot_pos_embed = nn.Parameter(torch.zeros(1, max_slots, embed_dim))

        # Transformer Decoder (12 Layers with Cross-Attention to ViT tokens)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_layers)

        # 3. Output Velocity Head
        self.out_norm = nn.LayerNorm(embed_dim)
        self.vel_head = nn.Linear(embed_dim, node_dim)

        # Auxiliary Organ Count Regressor Head
        self.count_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.img_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.slot_pos_embed, std=0.02)
        nn.init.constant_(self.vel_head.weight, 0.0)
        nn.init.constant_(self.vel_head.bias, 0.0)

    def _get_slot_pos_embed(self, n_slots: int, device: torch.device) -> torch.Tensor:
        """Returns slot positional embedding up to n_slots with dynamic sinusoidal fallback if n_slots > max_slots."""
        if n_slots <= self.max_slots:
            return self.slot_pos_embed[:, :n_slots, :]
        # For canopies larger than max_slots, blend learned embedding with sinusoidal continuation
        learned = self.slot_pos_embed
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=device) / half_dim
        )
        extra_pos = torch.arange(start=self.max_slots, end=n_slots, dtype=torch.float32, device=device).unsqueeze(1)
        args = extra_pos * freqs.unsqueeze(0)
        extra_sinusoidal = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).unsqueeze(0)
        return torch.cat([learned, extra_sinusoidal], dim=1)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        images: torch.Tensor,
        dap: torch.Tensor,
        viewpoints: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        x_t: (B, N_slots, 26) - Noisy continuous organ parameter tokens
        t: (B,) - Flow Matching continuous time in [0, 1]
        images: (B, 3, 128, 128) - RGB Conditioning
        dap: (B,) - Continuous DAP age
        viewpoints: (B, 5) - [caz, cel, cam_h, saz, sel] optional viewpoints
        key_padding_mask: (B, N_slots) - True for empty padding slots
        """
        B, N, _ = x_t.shape
        device = x_t.device

        # 1. Encode RGB Image with 16-layer ViT
        patch_tokens = self.patch_embed(images) + self.img_pos_embed
        memory = self.image_encoder(patch_tokens) # (B, num_patches, embed_dim)

        # 2. Encode DAP Age & Flow Time Conditioning
        t_emb = self.time_embed(t.clamp(0.0, 1.0))
        dap_norm = (dap / 100.0).clamp(0.0, 1.0)
        dap_emb = self.dap_mlp(dap_norm)

        cond_token = (t_emb + dap_emb).unsqueeze(1) # (B, 1, embed_dim)
        if viewpoints is not None:
            vp_emb = self.viewpoint_mlp(viewpoints).unsqueeze(1)
            cond_token = cond_token + vp_emb

        memory = torch.cat([cond_token, memory], dim=1) # (B, 1 + num_patches, embed_dim)

        # 3. Project Target Tokens & Add Scalable Positional Embeddings
        pos_emb = self._get_slot_pos_embed(N, device)
        tgt = self.node_in_proj(x_t) + pos_emb

        # 4. Decode with Dynamic Variable-Length Masking
        dec_out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_key_padding_mask=key_padding_mask
        ) # (B, N, embed_dim)

        dec_out = self.out_norm(dec_out)
        pred_velocity = self.vel_head(dec_out) # (B, N, 26)

        # Predict total active organ count from global ViT memory
        pred_count = self.count_head(memory[:, 0, :]).squeeze(-1) # (B,)

        return {
            "pred_velocity": pred_velocity,
            "pred_count": pred_count,
        }
