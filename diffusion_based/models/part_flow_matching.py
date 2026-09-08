"""
Part-centric Flow-Matching Model.

ViT image encoder + transformer decoder that predicts a velocity field v_theta
transporting a Gaussian prior to a plant organ array, conditioned on a rendered
plant image.

The target vector for each organ is an N-dimensional part descriptor whose layout
is defined by the dataset / renderer (e.g. organ type, base position, continuous
rotation, scale, existence). The exact dimension is configurable via node_dim so
the architecture can accommodate future extensions such as curvature parameters.

The model predicts the full part velocity (continuous). Organ type and existence
are treated as continuous during flow matching and discretized at inference
(round organ type, threshold existence).
"""

import math
import torch
import torch.nn as nn
from typing import Dict

from diffusion_based.models.vit_image_encoder import ViTImageEncoder


class PartFlowMatchingModel(nn.Module):
    """ViT encoder + transformer decoder predicting velocity field and organ-type logits.

    Two-stage architecture:
      Stage 1: Continuous geometry velocity field (13D: base, rot6d, scale, curv).
      Stage 2: Discrete organ-type classifier head (13 classes) conditioned on both
               the decoder token features and the predicted geometry features (cascaded).
    """

    def __init__(
        self,
        max_nodes: int = 2048,
        node_dim: int = 13,
        num_classes: int = 13,
        image_size: int = 128,
        patch_size: int = 8,
        embed_dim: int = 256,
        encoder_layers: int = 6,
        decoder_layers: int = 4,
        num_heads: int = 8,
        classify_types: bool = True,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_dim = node_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.classify_types = classify_types

        self.image_encoder = ViTImageEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=4,  # RGB(3) + CHM depth(1); 16-ch pyramid is averaged per-zoom in forward
            embed_dim=embed_dim,
            num_layers=encoder_layers,
            num_heads=num_heads,
        )

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.dap_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Project noisy part nodes into decoder query space
        self.node_query_proj = nn.Linear(node_dim, embed_dim)
        self.node_pos_emb = nn.Embedding(max_nodes, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        # Stage 1: Continuous geometry velocity head
        self.velocity_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, node_dim),
        )

        # Stage 2: Cascaded discrete organ-type classifier head (conditioned on geometry)
        if self.classify_types:
            geom_proj_dim = embed_dim // 2
            self.geom_proj = nn.Sequential(
                nn.Linear(node_dim, geom_proj_dim),
                nn.GELU(),
                nn.Linear(geom_proj_dim, geom_proj_dim),
            )
            self.type_classifier_head = nn.Sequential(
                nn.Linear(embed_dim + geom_proj_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, num_classes),
            )

    def _sinusoidal(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        dim = self.embed_dim
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        args = timesteps.float()[:, None] * emb[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(
        self,
        noisy_nodes: torch.Tensor,
        timesteps: torch.Tensor,
        images: torch.Tensor,
        daps: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            noisy_nodes: (B, N, D) interpolated part tensor x_t.
            timesteps: (B,) flow time t in [0, 1].
            images: (B, 3, H, W) normalized condition image.
            daps: Optional (B,) float tensor of Days After Planting.
        Returns:
            Dict containing:
                'pred_velocity': (B, N, D) predicted continuous geometry velocity.
                'pred_type_logits': (B, N, num_classes) discrete organ-type logits (if classify_types).
        """
        image_tokens = self.image_encoder(images)  # (B, T, D)
        t_emb = self.time_embed(self._sinusoidal(timesteps))  # (B, D)

        if daps is not None:
            # Normalize DAP ~ [0, 100] days
            dap_norm = daps.float() / 100.0
            d_emb = self.dap_embed(self._sinusoidal(dap_norm))
            cond_emb = t_emb + d_emb
        else:
            cond_emb = t_emb

        B, N, _ = noisy_nodes.shape
        device = noisy_nodes.device
        queries = self.node_query_proj(noisy_nodes) + self.node_pos_emb(
            torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        )
        queries = queries + cond_emb.unsqueeze(1)

        x = self.decoder(queries, image_tokens)
        pred_velocity = self.velocity_head(x)

        out = {"pred_velocity": pred_velocity}

        if self.classify_types:
            # Cascaded coupling: classifier takes decoder token features + predicted geometry
            geom_feat = self.geom_proj(pred_velocity)
            combined = torch.cat([x, geom_feat], dim=-1)
            pred_type_logits = self.type_classifier_head(combined)
            out["pred_type_logits"] = pred_type_logits

        return out

