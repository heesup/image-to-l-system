"""
Standalone Vision Transformer image encoder (shared by 14D and legacy models).

Encodes a rendered plant image with a Vision Transformer (patch embedding +
positional encoding + transformer encoder) into a grid of feature tokens.
"""

import torch
import torch.nn as nn


class ViTImageEncoder(nn.Module):
    """
    Standard Vision Transformer encoder producing a grid of feature tokens.

    Args:
        image_size: input square image resolution (padded to patch multiple).
        patch_size: patch size.
        in_channels: 3 for RGB (1 is expanded).
        embed_dim: token embedding dimension.
        num_layers: number of transformer encoder blocks.
        num_heads: attention heads.
        mlp_ratio: hidden dim / embed dim ratio.
        dropout: dropout rate.
    """

    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalized image.
        Returns:
            (B, num_patches + 1, embed_dim) feature tokens (cls prepended).
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        B = x.shape[0]
        x = self.patch_embed(x)                      # (B, D, Hp, Wp)
        x = x.flatten(2).transpose(1, 2)             # (B, Np, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)        # (B, Np+1, D)
        x = x + self.pos_embed
        x = self.encoder(x)
        return self.norm(x)
