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

import math
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
        num_levels: int = 1,
        use_depth: bool = False,
    ):
        super().__init__()
        # num_levels > 1: the input carries the cache's zoom pyramid (4 channels per level, RGB + depth,
        # zooms 1x/2x/4x/8x centred on the plant); every level's RGB goes through the backbone and the
        # tokens come back level-major after the level-0 CLS: [CLS_0 | patches_0 | patches_1 | ...].
        # Level identity is added by the consumers (trainable), the ray embedding is shared.
        self.num_levels = int(num_levels)
        self.use_depth = bool(use_depth)
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

        coords = torch.linspace(-1.0, 1.0, grid)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        grid_z = torch.ones_like(grid_x)
        rays = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (grid, grid, 3)
        rays = rays / torch.norm(rays, dim=-1, keepdim=True)
        rays_flat = rays.view(1, self.num_patches, 3)
        self.register_buffer("canonical_rays", rays_flat)

        # 3D Ray Positional Embedding (Sinusoidal Fourier Features, NeRF / Transformer style):
        # Maps 3D unit ray directions deterministically into embed_dim using multi-frequency sine/cosine bands.
        # 100% deterministic, 0 random weights, 0 trainable parameters, perfectly continuous 3D coordinate frame.
        num_freqs = embed_dim // 6
        freq_bands = (2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)) * math.pi
        args = rays_flat.unsqueeze(-1) * freq_bands.view(1, 1, 1, num_freqs)
        sin_feats = torch.sin(args)
        cos_feats = torch.cos(args)
        sinusoidal_ray_embed = torch.cat([sin_feats, cos_feats], dim=-1).flatten(start_dim=-2)
        if sinusoidal_ray_embed.shape[-1] < embed_dim:
            pad = embed_dim - sinusoidal_ray_embed.shape[-1]
            sinusoidal_ray_embed = F.pad(sinusoidal_ray_embed, (0, pad))
        self.register_buffer("canonical_ray_embed", sinusoidal_ray_embed.view(1, self.num_patches, embed_dim))

        # Depth adaptor (--use_depth): the design specifies an RGB-D input but the backbone is RGB-only
        # (see 20260918-stage3-conditioning-settled §5). Rather than rebuild DINO's patch embedding for a
        # 4th channel, encode each level's CHM/depth channel through a small stem into a per-patch embedding
        # and ADD it to the RGB patch tokens. Depth is plant-normalised (per-sample shift-scale over the
        # level's own spatial extent) so RELATIVE height is preserved and absolute depth scale (the rig's
        # depth is inaccurate) cannot dominate -- per-node normalisation would subtract each node's own
        # height, which is the entire signal a canopy height map carries.
        if self.use_depth:
            # Kernel 1x1: commute the conv with the average pool into the patch grid -- pool the
            # normalised DEPTH to (grid, grid) first, then conv. The two orders are identical in
            # value (avg-pool then 1x1 conv == 1x1 conv then avg-pool) and this one keeps the
            # activation (L*B, 1, g, g) instead of (L*B, embed_dim, 224, 224), ~200x less memory kept
            # for backward -- the first Gate B attempt (2026-09-18, job 38430694) died with an
            # illegal memory access at the first backward of a batch-48 render step with the
            # un-pooled order stacked on top.
            self.depth_adapt = nn.Conv2d(1, embed_dim, kernel_size=1)
        else:
            self.depth_adapt = None

        if freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False
            if self.depth_adapt is not None:
                # the depth adaptor is the new, untrained RGB-D pathway; it must stay live even though the
                # pretrained backbone is frozen.
                for p in self.depth_adapt.parameters():
                    p.requires_grad = True

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
        B = x.shape[0]
        L = self.num_levels if (self.num_levels > 1 and x.shape[1] >= 4 * self.num_levels) else 1
        if L == 1:
            rgb = x[:, :3].float()
            depth = x[:, 3:4].float() if (self.use_depth and x.shape[1] >= 4) else None
        else:
            rgb = torch.cat([x[:, 4 * l:4 * l + 3] for l in range(L)], dim=0).float()   # (L*B, 3, H, W), level-major
            depth = (torch.cat([x[:, 4 * l + 3:4 * l + 4] for l in range(L)], dim=0).float()
                     if self.use_depth else None)                                      # (L*B, 1, H, W)
        if rgb.shape[-1] != self.input_size or rgb.shape[-2] != self.input_size:
            rgb = F.interpolate(rgb, size=(self.input_size, self.input_size),
                                mode="bilinear", align_corners=False)

        features = self.backbone.forward_features(rgb)
        patch_tokens = self.proj(features["x_norm_patchtokens"])  # (L*B, N, embed_dim)
        cls_token = self.proj(features["x_norm_clstoken"])[:B].unsqueeze(1)  # level-0 CLS (B, 1, embed_dim)

        if patch_tokens.shape[1] != self.num_patches:
            raise RuntimeError(
                f"Backbone '{self.backbone_name}' produced {patch_tokens.shape[1]} patch tokens "
                f"at input {self.input_size}, expected {self.num_patches}.")

        # Depth pathway (--use_depth): per-sample shift-scale normalisation (plant-relative height), then
        # pooled to the patch grid and 1x1-conv'd into embed_dim, ADDED to the RGB patch tokens. Depth is
        # appearance-independent, so it is the channel that survives when RGB is degraded (real frames /
        # Helios). Off by default -- the frozen checkpoints were trained RGB-only and load unchanged.
        if depth is not None:
            if depth.shape[-1] != self.input_size or depth.shape[-2] != self.input_size:
                depth = F.interpolate(depth, size=(self.input_size, self.input_size),
                                      mode="bilinear", align_corners=False)
            # per-sample shift-scale (plant-normalised): preserve relative height, drop absolute scale.
            d_flat = depth.flatten(1)                                   # (L*B, H*W)
            d_mu = d_flat.mean(dim=1, keepdim=True)
            d_sd = d_flat.std(dim=1, keepdim=True).clamp(min=1e-4)
            depth_n = ((depth - d_mu.view(-1, 1, 1, 1)) / d_sd.view(-1, 1, 1, 1))
            grid = int(math.isqrt(self.num_patches))
            # pool BEFORE the 1x1 conv (identical in value, ~200x less activation memory; see __init__)
            d_pool = F.adaptive_avg_pool2d(depth_n, (grid, grid))       # (L*B, 1, g, g)
            d_tok = self.depth_adapt(d_pool).flatten(2).transpose(1, 2)  # (L*B, N, embed_dim)
            patch_tokens = patch_tokens + d_tok

        patch_tokens = patch_tokens + self.canonical_ray_embed.to(dtype=patch_tokens.dtype)
        if L > 1:
            N, C = patch_tokens.shape[1], patch_tokens.shape[2]
            patch_tokens = patch_tokens.view(L, B, N, C).permute(1, 0, 2, 3).reshape(B, L * N, C)
        return torch.cat([cls_token, patch_tokens], dim=1)


# Backward-compatible alias (original class name).
DINOv2RayEncoder = DINORayEncoder
