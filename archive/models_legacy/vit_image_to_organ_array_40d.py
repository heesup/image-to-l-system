"""
ViT-based Image -> PlantOrganArray inverse rendering model.

Encodes a rendered plant image with a Vision Transformer (patch embedding +
positional encoding + transformer encoder), then decodes with a fixed set of
learnable node queries (max_nodes) that cross-attend to the image tokens to
predict the full (N, 40) typed organ array.

Prediction heads:
  - continuous columns 0..38 (excluding categorical organ_type col 11): linear
  - organ_type (col 11): per-node categorical logits (num_organ_types classes)
  - existence (col 39): per-node logit

Trained with a combination of
  1. supervised organ-array regression (masked MSE + existence BCE + type CE)
  2. an image-space loss: render the predicted array through the differentiable
     HeliosPyTorchRenderer and compare against the target image.

Interface mirrors PlantOrganArrayDiffuser.forward so the two can be swapped.
"""

import math
import torch
import torch.nn as nn
from typing import Dict


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


class OrganArrayDecoder(nn.Module):
    """
    Set-prediction decoder: max_nodes learnable queries cross-attend to the
    image tokens and predict the (N, 40) organ array.

    Args:
        max_nodes: fixed number of node queries.
        node_dim: 40 for the typed organ array.
        embed_dim: hidden dimension.
        num_layers: number of transformer decoder layers.
        num_heads: attention heads.
        num_organ_types: categorical classes for column 11.
    """

    def __init__(
        self,
        max_nodes: int = 256,
        node_dim: int = 40,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        num_organ_types: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_dim = node_dim
        self.embed_dim = embed_dim

        self.query_embed = nn.Parameter(torch.zeros(1, max_nodes, embed_dim))
        nn.init.trunc_normal_(self.query_embed, std=0.02)
        self.pos_embed = nn.Embedding(max_nodes, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * 4.0),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.continuous_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, node_dim),
        )
        self.organ_type_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_organ_types),
        )
        self.existence_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        image_tokens: torch.Tensor,
        t_emb: torch.Tensor,
        node_queries: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            image_tokens: (B, T, embed_dim) tokens from the ViT encoder.
            t_emb: (B, embed_dim) optional conditioning (0 for non-diffusion).
            node_queries: optional (B, N, embed_dim) external node queries
                (e.g. projected noisy organ nodes for DDIM). If None, learned
                query embeddings are used (regression mode).
        Returns:
            Dict with 'pred_x0' (B, N, node_dim), 'organ_type_logits' (B, N, K),
            'existence_logits' (B, N).
        """
        B, T, D = image_tokens.shape
        device = image_tokens.device

        if node_queries is None:
            queries = self.query_embed.expand(B, -1, -1) + self.pos_embed(
                torch.arange(self.max_nodes, device=device).unsqueeze(0).expand(B, -1)
            )
        else:
            queries = node_queries
        if t_emb is not None and t_emb.numel() > 0:
            queries = queries + t_emb.unsqueeze(1)

        x = self.decoder(queries, image_tokens)

        pred_x0 = self.continuous_head(x)
        organ_type_logits = self.organ_type_head(x)
        existence_logits = self.existence_head(x).squeeze(-1)

        return {
            "pred_x0": pred_x0,
            "organ_type_logits": organ_type_logits,
            "existence_logits": existence_logits,
        }


class ViTImageToOrganArray(nn.Module):
    """
    Full ViT image -> (N, 40) organ array inverse rendering model.

    Args:
        max_nodes: number of organ-node slots.
        node_dim: 40 for the typed layout.
        image_size: input image resolution.
        patch_size: ViT patch size.
        embed_dim: hidden dimension.
        encoder_layers: ViT encoder depth.
        decoder_layers: transformer decoder depth.
        num_heads: attention heads.
        num_organ_types: classes for column 11.
    """

    def __init__(
        self,
        max_nodes: int = 256,
        node_dim: int = 40,
        image_size: int = 128,
        patch_size: int = 8,
        embed_dim: int = 256,
        encoder_layers: int = 6,
        decoder_layers: int = 4,
        num_heads: int = 8,
        num_organ_types: int = 8,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_dim = node_dim
        self.embed_dim = embed_dim
        self.num_organ_types = num_organ_types

        self.image_encoder = ViTImageEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
            num_layers=encoder_layers,
            num_heads=num_heads,
        )
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self._node_proj = nn.Linear(node_dim, embed_dim)
        self.decoder = OrganArrayDecoder(
            max_nodes=max_nodes,
            node_dim=node_dim,
            embed_dim=embed_dim,
            num_layers=decoder_layers,
            num_heads=num_heads,
            num_organ_types=num_organ_types,
        )

    def forward(
        self,
        images: torch.Tensor,
        timesteps: torch.Tensor = None,
        noisy_nodes: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Image-conditional forward pass.

        Args:
            images: (B, 3, H, W) normalized image.
            timesteps: optional (B,) timesteps for diffusion-style conditioning
                (None/zero -> no time conditioning).
            noisy_nodes: optional (B, N, node_dim) noisy nodes. If provided and
                timesteps is not None, they are used to add a node-conditioning
                embedding so the module can also be used as the denoiser in a
                DDIM pipeline (keeps the diffuser interface compatible).

        Returns:
            Dict with 'pred_x0', 'organ_type_logits', 'existence_logits'.
        """
        image_tokens = self.image_encoder(images)

        t_emb = None
        if timesteps is not None and timesteps.numel() > 0 and (timesteps.max() != 0 or timesteps.min() != 0):
            t_emb = self.time_embed(self._sinusoidal(timesteps))

        if noisy_nodes is not None and t_emb is not None:
            # Node conditioning used by diffusion training: summarize noisy
            # nodes (mean over nodes) and add to the time embedding so the
            # model can also act as a denoiser (diffuser-compatible interface).
            t_emb = t_emb + nn.functional.linear(
                noisy_nodes.mean(dim=1), self._node_proj.weight
            )

        return self.decoder(image_tokens, t_emb)

    def _sinusoidal(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        dim = self.embed_dim
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        args = timesteps.float()[:, None] * emb[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ViTOrganArrayDiffuser(ViTImageToOrganArray):
    """
    DDIM-compatible wrapper matching the PlantOrganArrayDiffuser interface.

    forward(noisy_nodes, timesteps, images) -> {'pred_x0', 'organ_type_logits',
    'existence_logits'}. Noisy nodes are projected into the decoder query
    space so the ViT acts as a denoiser conditioned on the image.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Project (B, N, node_dim) noisy nodes -> (B, N, embed_dim) decoder queries
        self.node_query_proj = nn.Linear(self.node_dim, self.embed_dim)
        self.node_pos_emb = nn.Embedding(self.max_nodes, self.embed_dim)

    def forward(
        self,
        noisy_nodes: torch.Tensor,
        timesteps: torch.Tensor,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Diffuser-compatible forward: (B, N, node_dim) noisy nodes, (B,) timesteps,
        (B, 3, H, W) condition image.
        """
        image_tokens = self.image_encoder(images)
        t_emb = self.time_embed(self._sinusoidal(timesteps))

        B, N, _ = noisy_nodes.shape
        device = noisy_nodes.device
        queries = self.node_query_proj(noisy_nodes) + self.node_pos_emb(
            torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        )
        return self.decoder(image_tokens, t_emb, node_queries=queries)
