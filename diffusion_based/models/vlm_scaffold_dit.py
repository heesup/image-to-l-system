"""
Botanical Multi-Modal Diffusion Transformer (Botanical MM-DiT).
Pure Single-Stage End-to-End Rectified Flow Matching from Multi-Modal 2D Visual Observations (RGB + Depth)
to Ultra-High-Dimensional (4,096 x 26 = 106,496D) 3D Functional-Structural Plant Models (FSPMs).

Features:
1. 4-Channel Vision Prompt (RGB 3ch + Canopy Height Depth 1ch) encoded via DINOv3 ViT.
2. Global Macro Phenotyping Heads (DAP, Height, Crown Radius, Organ Count).
3. Pure Single-Stage Continuous Rectified Flow Matching starting from standard Gaussian Prior x_0 ~ N(0, I).
4. Classifier-Free Guidance (CFG) conditioning dropout and extrapolation.
5. End-to-End Differentiable Rendering Self-Consistency Supervision.
"""

from typing import Dict, Tuple, Optional, Any, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_based.dataset.part_array_dataset import (
    FM_NODE_DIM,
    FM_OT_END,
    FM_BASE_START,
    FM_BASE_END,
    FM_ROT_START,
    FM_ROT_END,
    FM_SCALE_START,
    FM_SCALE_END,
    FM_CURV_IDX,
    FM_PHYLLO_IDX,
    ORGAN_CATEGORIES,
    EMPTY_IDX,
    BASE_SCALE,
    SCALE_SCALE,
    CURVATURE_SCALE,
    PHYLLOTACTIC_SCALE,
)
from diffusion_based.models.vlm_vision_tower import DINOv3VisionTower


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.embed_dim // 2
        emb_scale = math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb_scale)
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.embed_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class BotanicalMMDiTModel(nn.Module):
    """
    Pure Single-Stage End-to-End Multi-Modal Diffusion Transformer for 3D Plant Generation.
    Directly maps x_0 ~ N(0, I) -> x_1 (3D Organ Array) conditioned on RGB+Depth visual prompt tokens.
    """

    def __init__(
        self,
        dinov3_model: str = "facebook/dinov3-small-patch14",
        max_slots: int = 4096,
        node_dim: int = FM_NODE_DIM,
        embed_dim: int = 768,
        in_channels: int = 4,
        decoder_layers: int = 12,
        num_heads: int = 12,
        dropout: float = 0.0,
        pretrained: bool = True,
        freeze_vision_backbone: bool = False,
    ):
        super().__init__()
        self.max_slots = max_slots
        self.node_dim = node_dim
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        # 1. Multi-Modal Vision Tower (RGB 3ch + Depth 1ch = 4ch)
        self.vision_tower = DINOv3VisionTower(
            model_name=dinov3_model,
            embed_dim=embed_dim,
            in_channels=in_channels,
            pretrained=pretrained,
            freeze_backbone=freeze_vision_backbone,
        )

        # 2. Global Macro Phenotype Prediction Heads (Top-Down Biological Prior)
        self.dap_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
        self.height_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
        self.radius_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
        self.count_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

        # 3. Continuous Time & Slot Position Embeddings
        self.time_emb = SinusoidalTimeEmbedding(embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.macro_cond_proj = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.slot_pos_embed = nn.Parameter(torch.randn(1, max_slots, embed_dim) * 0.02)
        self.node_in_proj = nn.Linear(node_dim, embed_dim)

        # 4. Multi-Modal Transformer Decoder Engine (Self-Attention + Visual Cross-Attention)
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

        # 5. Output Velocity Head for Continuous Rectified Flow Matching
        self.out_norm = nn.LayerNorm(embed_dim)
        self.vel_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, node_dim),
        )

        # 6. Null Embeddings for Classifier-Free Guidance (CFG)
        self.null_spatial_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.null_macro_emb = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

    def extract_vision_tokens(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts spatial visual patch tokens and global pooled token from DINOv3."""
        v_out = self.vision_tower(img)
        return v_out["spatial_tokens"], v_out["global_token"]

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        img: Optional[torch.Tensor] = None,
        spatial_tokens: Optional[torch.Tensor] = None,
        global_token: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        cond_drop_prob: float = 0.0,
        force_uncond: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Unified Pure Single-Stage Forward Pass for Flow Matching with Classifier-Free Guidance.
        """
        B, N, _ = x_t.shape
        device = x_t.device

        # 1. Vision Feature Extraction
        if (spatial_tokens is None or global_token is None) and img is not None:
            spatial_tokens, global_token = self.extract_vision_tokens(img)

        # 2. Top-Down Macro Phenotype Prediction
        if global_token is not None:
            pred_dap = F.relu(self.dap_head(global_token).squeeze(-1))
            pred_h = torch.clamp(F.softplus(self.height_head(global_token).squeeze(-1)), min=0.08, max=0.85)
            pred_rad = torch.clamp(F.softplus(self.radius_head(global_token).squeeze(-1)), min=0.05, max=0.80)
            pred_cnt = F.relu(self.count_head(global_token).squeeze(-1))

            macro_vec = torch.stack([
                pred_dap / 100.0,
                pred_h,
                pred_rad,
                pred_cnt / 1000.0,
            ], dim=-1)
            macro_emb = self.macro_cond_proj(macro_vec)
        else:
            pred_dap = torch.zeros(B, device=device)
            pred_h = torch.zeros(B, device=device)
            pred_rad = torch.zeros(B, device=device)
            pred_cnt = torch.zeros(B, device=device)
            macro_emb = self.null_macro_emb.expand(B, -1)

        # 3. Classifier-Free Guidance (CFG) conditioning drop
        if force_uncond:
            S = spatial_tokens.shape[1] if spatial_tokens is not None else 1
            spatial_tokens = self.null_spatial_token.expand(B, S, -1)
            macro_emb = self.null_macro_emb.expand(B, -1)
        elif self.training and cond_drop_prob > 0.0:
            drop_mask = (torch.rand(B, 1, 1, device=device) < cond_drop_prob)
            drop_mask_macro = drop_mask.squeeze(-1)
            S = spatial_tokens.shape[1]
            spatial_tokens = torch.where(drop_mask, self.null_spatial_token.expand(B, S, -1), spatial_tokens)
            macro_emb = torch.where(drop_mask_macro, self.null_macro_emb.expand(B, -1), macro_emb)

        # 4. DiT Conditioning Injection
        t_emb = self.time_mlp(self.time_emb(t))
        cond = (t_emb + macro_emb).unsqueeze(1)

        # Node Projection + Canonical Slot Positional Embeddings + Time/Macro Condition
        h = self.node_in_proj(x_t) + self.slot_pos_embed[:, :N] + cond

        # 5. Transformer Decoder (Self-Attention + Spatial Visual Cross-Attention)
        h_out = self.decoder(
            tgt=h,
            memory=spatial_tokens,
            tgt_key_padding_mask=key_padding_mask,
        )
        h_out = self.out_norm(h_out)

        v_pred = self.vel_head(h_out)

        # Instant 3D Endpoint Estimation: x_1_hat = x_t + (1 - t) * v_pred
        t_expand = t.view(B, 1, 1)
        x_1_hat = x_t + (1.0 - t_expand) * v_pred

        return {
            "pred_velocity": v_pred,
            "x_1_hat": x_1_hat,
            "pred_dap": pred_dap,
            "pred_height": pred_h,
            "pred_radius": pred_rad,
            "pred_active_count": pred_cnt,
            "spatial_tokens": spatial_tokens,
            "global_token": global_token,
        }

    @torch.no_grad()
    def sample_plant(
        self,
        img: torch.Tensor,
        num_steps: int = 15,
        guidance_scale: float = 2.0,
        target_slots: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, Any]:
        """
        Pure Single-Stage Flow Matching Inference with CFG Extrapolation:
        Starts from pure Gaussian prior x_0 ~ N(0, I) and integrates 10-15 Euler ODE steps.
        """
        self.eval()
        img = img.to(device)
        B = img.shape[0]

        # 1. Visual Token Extraction & Macro Traits
        spatial_tokens, global_token = self.extract_vision_tokens(img)
        pred_dap = F.relu(self.dap_head(global_token).squeeze(-1))
        pred_h = torch.clamp(F.softplus(self.height_head(global_token).squeeze(-1)), min=0.08, max=0.85)
        pred_rad = torch.clamp(F.softplus(self.radius_head(global_token).squeeze(-1)), min=0.05, max=0.80)
        pred_cnt = F.relu(self.count_head(global_token).squeeze(-1))

        N = target_slots if target_slots is not None else self.max_slots

        # 2. Initialize from standard Gaussian prior x_0 ~ N(0, I)
        x_t = torch.randn(B, N, self.node_dim, device=device)
        dt = 1.0 / num_steps

        # 3. Euler ODE Integration
        for s in range(num_steps):
            t_val = torch.full((B,), s * dt, device=device)

            if guidance_scale > 1.0:
                out_cond = self.forward(
                    x_t=x_t,
                    t=t_val,
                    spatial_tokens=spatial_tokens,
                    global_token=global_token,
                    force_uncond=False,
                )
                v_cond = out_cond["pred_velocity"]

                out_uncond = self.forward(
                    x_t=x_t,
                    t=t_val,
                    spatial_tokens=spatial_tokens,
                    global_token=global_token,
                    force_uncond=True,
                )
                v_uncond = out_uncond["pred_velocity"]

                v_total = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                out = self.forward(
                    x_t=x_t,
                    t=t_val,
                    spatial_tokens=spatial_tokens,
                    global_token=global_token,
                    force_uncond=False,
                )
                v_total = out["pred_velocity"]

            x_t = x_t + v_total * dt

        return {
            "x_gen": x_t,
            "pred_dap": pred_dap,
            "pred_height": pred_h,
            "pred_radius": pred_rad,
            "pred_active_count": pred_cnt,
        }


# Backwards compatibility alias
VLMScaffoldDiTModel = BotanicalMMDiTModel
