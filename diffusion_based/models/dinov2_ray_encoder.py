"""
DINO Backbone Encoder with 3D Camera Ray Positional Embedding (PETR-style).

The backbone family is swappable for scaling A/B experiments:

    dinov2_vits14       22M   patch 14   input 224   (control / current)
    dinov2_vitb14       87M   patch 14   input 224
    dinov2_vitl14      300M   patch 14   input 224
    dinov3_vits16       21M   patch 16   input 256   (web LVD-1689M, gated)
    dinov3_vitb16       86M   patch 16   input 256   (web LVD-1689M, gated)
    dinov3_vitl16      300M   patch 16   input 256   (web LVD-1689M, gated)
    dinov3_vitl16_sat  300M   patch 16   input 256   (SAT-493M satellite, cached)

Both families expose the same `forward_features()` dict keys
(`x_norm_clstoken`, `x_norm_patchtokens`), and both produce a 16x16 = 256
patch-token grid at their configured input size, so the 3D ray embedding and
every downstream tensor shape are identical across backbones.

DINOv3 web weights are license-gated: download them manually and point
`DINOV3_WEIGHTS` (env or `weights_path`) at the local file or URL. The
SAT-493M checkpoint is resolved from the torch hub cache by default.
"""

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class BackboneSpec:
    family: str          # "dinov2" | "dinov3"
    name: str            # torch.hub entrypoint
    dim: int             # native token dimension
    patch_size: int
    input_size: int      # native token grid = (input_size // patch_size)^2
    sat: bool = False    # SAT-493M pretrained DINOv3 variant


BACKBONES = {
    "dinov2_vits14": BackboneSpec("dinov2", "dinov2_vits14", 384, 14, 224),
    "dinov2_vitb14": BackboneSpec("dinov2", "dinov2_vitb14", 768, 14, 224),
    "dinov2_vitl14": BackboneSpec("dinov2", "dinov2_vitl14", 1024, 14, 224),
    "dinov3_vits16": BackboneSpec("dinov3", "dinov3_vits16", 384, 16, 256),
    "dinov3_vitb16": BackboneSpec("dinov3", "dinov3_vitb16", 768, 16, 256),
    "dinov3_vitl16": BackboneSpec("dinov3", "dinov3_vitl16", 1024, 16, 256),
    "dinov3_vitl16_sat": BackboneSpec("dinov3", "dinov3_vitl16", 1024, 16, 256, sat=True),
}

_DINOV3_DEFAULT_REPO = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov3_main")
_DINOV3_SAT_WEIGHTS = os.path.expanduser(
    "~/.cache/torch/hub/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")


def _load_backbone(spec: BackboneSpec, pretrained: bool, weights_path: Optional[str] = None):
    if spec.family == "dinov2":
        return torch.hub.load("facebookresearch/dinov2", spec.name, pretrained=pretrained)

    repo = os.environ.get("DINOV3_REPO", _DINOV3_DEFAULT_REPO)
    if not os.path.isdir(repo):
        raise FileNotFoundError(
            f"DINOv3 repo not found at {repo}. Clone facebookresearch/dinov3 or set DINOV3_REPO.")
    weights = weights_path or os.environ.get("DINOV3_WEIGHTS") or None
    if weights is None and spec.sat:
        weights = _DINOV3_SAT_WEIGHTS
    if weights is None:
        raise FileNotFoundError(
            f"DINOv3 web weights for '{spec.name}' are license-gated. Download the checkpoint and "
            f"set DINOV3_WEIGHTS=/path/to/checkpoint.pth (or pass weights_path).")
    if not (weights.startswith("http") or os.path.exists(weights)):
        raise FileNotFoundError(f"DINOv3 weights not found: {weights}")
    return torch.hub.load(repo, spec.name, source="local", weights=weights)


class DINORayEncoder(nn.Module):
    """
    Vision encoder coupling a pretrained DINO backbone with 3D camera ray
    positional embeddings.

    Args:
        backbone: Key from BACKBONES (default: dinov2_vits14, the original).
        pretrained: Whether to load pretrained backbone weights.
        freeze_backbone: Whether to freeze the backbone parameters.
        embed_dim: Output token dimension (projected from the native backbone dim).
        weights_path: Optional explicit DINOv3 checkpoint path/URL.
    """

    def __init__(
        self,
        backbone: str = "dinov2_vits14",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        embed_dim: int = 384,
        weights_path: Optional[str] = None,
    ):
        super().__init__()
        if backbone not in BACKBONES:
            raise KeyError(f"Unknown backbone '{backbone}'. Available: {sorted(BACKBONES)}")
        self.backbone_name = backbone
        self.spec = BACKBONES[backbone]
        self.embed_dim = embed_dim
        self.input_size = self.spec.input_size
        grid = self.spec.input_size // self.spec.patch_size
        self.num_patches = grid * grid

        self.backbone = _load_backbone(self.spec, pretrained, weights_path)
        self.dino_dim = self.spec.dim

        if embed_dim != self.dino_dim:
            self.proj = nn.Linear(self.dino_dim, embed_dim)
        else:
            self.proj = nn.Identity()

        # 3D Ray Positional Embedding (PETR / DUSt3R style):
        # ray direction v = [x_ndc, y_ndc, 1.0]^T normalized per patch.
        self.ray_mlp = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        coords = torch.linspace(-1.0, 1.0, grid)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        grid_z = torch.ones_like(grid_x)
        rays = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (grid, grid, 3)
        rays = rays / torch.norm(rays, dim=-1, keepdim=True)
        self.register_buffer("canonical_rays", rays.view(1, self.num_patches, 3))

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, C, H, W) normalized image tensor. Uses channels 0:3 (RGB).
        Returns:
            (B, 1 + num_patches, embed_dim) visual tokens:
                token 0: CLS token
                tokens 1..N: 3D Ray-embedded spatial patch tokens
        """
        rgb = x[:, :3].float()
        if rgb.shape[-1] != self.input_size or rgb.shape[-2] != self.input_size:
            rgb = F.interpolate(rgb, size=(self.input_size, self.input_size),
                                mode="bilinear", align_corners=False)

        features = self.backbone.forward_features(rgb)
        patch_tokens = self.proj(features["x_norm_patchtokens"])  # (B, N, embed_dim)
        cls_token = self.proj(features["x_norm_clstoken"]).unsqueeze(1)  # (B, 1, embed_dim)

        if patch_tokens.shape[1] != self.num_patches:
            raise RuntimeError(
                f"Backbone '{self.backbone_name}' produced {patch_tokens.shape[1]} patch tokens "
                f"at input {self.input_size}, expected {self.num_patches}.")

        patch_tokens = patch_tokens + self.ray_mlp(self.canonical_rays.to(patch_tokens.dtype))
        return torch.cat([cls_token, patch_tokens], dim=1)


# Backward-compatible alias (original class name).
DINOv2RayEncoder = DINORayEncoder
