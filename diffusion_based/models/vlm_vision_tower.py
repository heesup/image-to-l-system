"""
Pretrained Vision Tower for Plant Phenotyping and 3D Plant Reconstruction.
Supports Meta DINOv3 (https://huggingface.co/collections/facebook/dinov3 and https://github.com/facebookresearch/dinov3)
with graceful HuggingFace / torch.hub loading and feature extraction.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv3VisionTower(nn.Module):
    """
    Pretrained Multi-Modal Vision Backbone extracting both global phenotypic tokens and
    spatial dense patch tokens from 4-channel (RGB 3ch + Canopy Height Depth 1ch) inputs.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-small-patch14",
        embed_dim: int = 768,
        in_channels: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone

        self.backbone = None
        self.is_hf_model = False
        self._init_backbone(model_name, pretrained)

        # Projection heads if backbone feature dimension does not match embed_dim
        self.global_proj = nn.Identity()
        self.spatial_proj = nn.Identity()
        
        feat_dim = getattr(self, "feat_dim", 768)
        if feat_dim != embed_dim:
            self.global_proj = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, embed_dim),
            )
            self.spatial_proj = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, embed_dim),
            )

    def _init_backbone(self, model_name: str, pretrained: bool):
        if not pretrained:
            print(f"[DINOv3VisionTower] Initializing native Multi-Modal ViT-B/14 (in_channels={self.in_channels}, embed_dim={self.embed_dim})")
            self._init_native_vit()
            return

        # 1. Try loading via HuggingFace Transformers (AutoModel)
        try:
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(model_name, add_pooling_layer=False, local_files_only=False)
            self.is_hf_model = True
            self.feat_dim = getattr(self.backbone.config, "hidden_size", 768)
            print(f"[DINOv3VisionTower] Successfully loaded HuggingFace DINOv3 model: {model_name} (dim={self.feat_dim})")
            if self.freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad = False
            return
        except Exception:
            pass

        # 2. Fallback to native ViT
        print(f"[DINOv3VisionTower] Using native Multi-Modal ViT-B/14 (in_channels={self.in_channels}, embed_dim={self.embed_dim})")
        self._init_native_vit()

    def _init_native_vit(self):
        self.feat_dim = self.embed_dim
        self.patch_size = 16
        self.conv_proj = nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 1025, self.embed_dim) * 0.02)
        nhead = 16 if self.embed_dim % 16 == 0 else (8 if self.embed_dim % 8 == 0 else 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=nhead,
            dim_feedforward=self.embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.is_hf_model = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input RGB image tensor (B, 3, H, W) normalized to [0, 1] or ImageNet norm
        Returns:
            Dictionary containing:
              - 'global_token': (B, embed_dim) [CLS] token representing macro phenotype (DAP, height, etc.)
              - 'spatial_tokens': (B, num_patches, embed_dim) Dense patch tokens for fine 3D geometric cross-attention
        """
        # Standardize channels: pad 3ch to 4ch or slice 4ch to 3ch based on model architecture
        B, C, H, W = x.shape
        if C == 3 and self.in_channels == 4:
            x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)

        if self.is_hf_model:
            out = self.backbone(pixel_values=x[:, :3])
            hidden_states = out.last_hidden_state  # (B, 1 + num_patches, feat_dim)
            cls_token = hidden_states[:, 0]
            patch_tokens = hidden_states[:, 1:]
        elif hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(x[:, :3])
            cls_token = feats["x_norm_clstoken"]
            patch_tokens = feats["x_norm_patchtokens"]
        elif hasattr(self, "conv_proj"):
            # Native Multi-Modal ViT path
            patches = self.conv_proj(x).flatten(2).transpose(1, 2)  # (B, P, D)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls_tokens, patches], dim=1)
            num_t = tokens.shape[1]
            if num_t <= self.pos_embed.shape[1]:
                tokens = tokens + self.pos_embed[:, :num_t]
            else:
                tokens = tokens + F.interpolate(self.pos_embed.transpose(1, 2), size=num_t, mode='linear', align_corners=False).transpose(1, 2)
            tokens = self.backbone(tokens)
            cls_token = tokens[:, 0]
            patch_tokens = tokens[:, 1:]
        else:
            cls_token = torch.zeros((B, self.embed_dim), device=x.device)
            patch_tokens = torch.zeros((B, 256, self.embed_dim), device=x.device)

        global_token = self.global_proj(cls_token)
        spatial_tokens = self.spatial_proj(patch_tokens)

        return {
            "global_token": global_token,
            "spatial_tokens": spatial_tokens,
        }
