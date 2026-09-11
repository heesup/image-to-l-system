"""
Hierarchical Matryoshka Botanical Flow Matching Model.

Decouples macroscopic plant skeleton/phytomer topology (Coarse Stage 1)
from microscopic trifoliolate leaflet and organ geometry (Fine Stage 2)
using Matryoshka power-of-2 nested queries.
"""

import math
from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_based.models.dinov2_ray_encoder import DINOv2RayEncoder
from diffusion_based.dataset.part_array_dataset import (
    BASE_SCALE,
    SCALE_SCALE,
    CURV_SCALE,
    GEOM_NODE_DIM,
    NUM_ORGAN_TYPES,
    FM_BASE_START,
    FM_ROT_START,
    FM_ROT_END,
    decode_fm,
)
from diffusion_based.dataset.phytomer_packets import (
    apply_reference_rotation,
    assemble_packets,
    denormalize_packet_scales,
)


# =============================================================================
# ANCHOR CAPACITY CURVE (calibrated 2026-09-08 evening on cowpea_curv26 cache)
# =============================================================================
# GT phytomer cluster counts per plant (petiole bases + standalone internodes,
# matcher-consistent counting in HierarchicalBotanicalMatcher) were sampled
# per DAP from dataset/cache/cowpea_curv26 (100-300 samples per DAP bucket,
# 59,441-file scan) and a logistic curve was fitted to the per-DAP
# rolling-max envelope of the OBSERVED SAMPLE MAX (not the p97.5 quantile),
# so a DAP->K lookup covers 100% of observed samples including worst-case
# branching tails (max 342 clusters @ DAP 92):
#   phytomers_max(dap) = L / (1 + exp(-k*(dap - x0))) + N0
# Curve constants (max-envelope fit; regenerate via tools/calibrate_anchor_capacity.py):
ANCHOR_CURVE_L = 297.2      # asymptotic capacity (phytomers)
ANCHOR_CURVE_K = 0.1841     # logistic growth rate (1/day)
ANCHOR_CURVE_X0 = 30.5      # midpoint DAP (days)
ANCHOR_CURVE_N0 = 15.53     # offset at DAP -> -inf
# Per-sample capacity = ceil(curve(DAP) * ANCHOR_MARGIN + ANCHOR_MARGIN_FLAT).
# ANCHOR_MARGIN guards the residual individual variance (branching differences)
# on top of the max-envelope fit; ANCHOR_MARGIN_FLAT adds absolute headroom
# for seedlings where relative variance is largest.
# Coverage on the 2026-09-08 evening scan: p97.5 100% | observed-max 100%
# (DAP 92 max 342 -> K=350; cache organ rows max 2,625 -> slots 2,800).
ANCHOR_MARGIN = 1.10
ANCHOR_MARGIN_FLAT = 6.0
ANCHOR_MARGIN_MIN = 1.05


# =============================================================================
# PHYTOMER-LEVEL FLOW TARGET LAYOUT (Stage 3 refinement of base + rot + latent)
# =============================================================================
# Per-anchor Stage-3 flow target. Stage 2 predicts a deterministic 3D node
# scaffold (+ coarse scale); Stage 3 FLOW-REFINES the node pose (base + rot +
# scale) jointly with the per-phytomer latent, rather than treating the pose
# as fixed conditioning.
#   z_1[anchor] = [ node_base_xyz(3) | node_rot_6d(6) | anchor_scale(3) | phytomer_latent(D) ]
# The anchor scale s_a (petiole length row [len, radius, unused], v3 packet
# format) is explicit so the flow refines phytomer size directly (GS-style
# primitive scale); the latent carries only scale-NORMALIZED relative geometry.
# Layout indices (relative to z_1's first dim):
PHYTO_FLOW_BASE_START = 0          # xyz (3)  — refined anchor base position (metres)
PHYTO_FLOW_BASE_END = 3
PHYTO_FLOW_ROT_START = 3           # rot6d (6) — refined anchor rotation (also the
PHYTO_FLOW_ROT_END = 9             #              reference frame for the packet)
PHYTO_FLOW_SCALE_START = 9         # scale (3) — refined anchor scale (petiole row)
PHYTO_FLOW_SCALE_END = 12
PHYTO_FLOW_LATENT_START = 12       # latent (D) — VAE latent of the relative packet


def build_phytomer_flow_target(
    anchor_pos: torch.Tensor,
    anchor_rot: torch.Tensor,
    anchor_scale: torch.Tensor,
    phytomer_latent: torch.Tensor,
) -> torch.Tensor:
    """Concatenates per-anchor pose + scale + latent into the Stage-3 flow target (B, K, 12+D).

    Args:
        anchor_pos: (B, K, 3) refined anchor base positions (metres).
        anchor_rot: (B, K, 6) refined anchor 6D rotation (reference frame).
        anchor_scale: (B, K, 3) GT anchor scale (petiole scale row, FM units).
        phytomer_latent: (B, K, D) VAE latent of the anchor-relative packets.

    Returns:
        (B, K, 12 + D) flow target.
    """
    return torch.cat([anchor_pos, anchor_rot, anchor_scale, phytomer_latent], dim=-1)


def split_phytomer_flow_target(z: torch.Tensor, latent_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Splits a Stage-3 flow vector back into (anchor_pos, anchor_rot, anchor_scale, latent)."""
    pos = z[..., PHYTO_FLOW_BASE_START:PHYTO_FLOW_BASE_END]
    rot = z[..., PHYTO_FLOW_ROT_START:PHYTO_FLOW_ROT_END]
    scl = z[..., PHYTO_FLOW_SCALE_START:PHYTO_FLOW_SCALE_END]
    lat = z[..., PHYTO_FLOW_LATENT_START:PHYTO_FLOW_LATENT_START + latent_dim]
    return pos, rot, scl, lat


def apply_ref_for_flow(
    packet_hat: torch.Tensor,
    refined_pos: torch.Tensor,
    refined_rot: torch.Tensor,
    base_scale: float = BASE_SCALE,
) -> torch.Tensor:
    """Re-anchors (B, K, M, 26) relative packets using the REFINED anchor pose.

    Option 1 semantics: the refined anchor rotation is the reference frame for
    the packet; the refined anchor base position re-adds absolute placement.

    NOTE: builds fresh tensors and concatenates (no inplace writes into the
    grad-captured packet tensor — avoids autograd version conflicts in the
    differentiable render path).
    """
    B, K, S, _ = packet_hat.shape
    pos = refined_pos.unsqueeze(-2) * base_scale  # (B, K, 1, 3)
    base = packet_hat[..., FM_BASE_START:FM_BASE_START + 3] + pos
    rot = apply_reference_rotation(
        packet_hat[..., FM_ROT_START:FM_ROT_START + 6],
        refined_rot.unsqueeze(-2).expand(B, K, S, 6),
    )
    return torch.cat(
        [packet_hat[..., :FM_BASE_START], base, rot, packet_hat[..., FM_ROT_END:]],
        dim=-1,
    )


def compute_matryoshka_slice(
    dap: Optional[torch.Tensor] = None,
    num_phytomers: Optional[torch.Tensor] = None,
    max_anchors: int = 512,
    margin: float = ANCHOR_MARGIN,
) -> int:
    """Computes upper-bound active anchor count based on biological growth or predicted phytomers.

    If num_phytomers is provided:
        K_upper = min(max_anchors, ceil(num_phytomers * margin + flat))
    Else if DAP is provided:
        K_upper(t) = min(max_anchors, ceil(ANCHOR_L / (1 + exp(-ANCHOR_K * (t - ANCHOR_X0))) + ANCHOR_N0)
                         * ANCHOR_MARGIN + ANCHOR_FLAT)

    2026-09-08 recalibration (evening rescan, 100-300 samples/DAP): the previous
    p97.5-envelope fit left 11% of DAP buckets with at least one observed sample
    beyond capacity (worst 342 clusters @ DAP 92). The fit now targets the
    rolling-max envelope of observed sample maxima -> 100% observed-max coverage.
    Continuous (non power-of-2) capacity avoids tier-snap quantization waste.
    """
    margin = max(float(margin), ANCHOR_MARGIN_MIN)
    if num_phytomers is not None:
        val = float(num_phytomers.max().item())
        val = max(0.0, val)
        k = math.ceil(val * margin + ANCHOR_MARGIN_FLAT)
    elif dap is not None:
        dap_val = float(dap.max().item())
        dap_val = max(0.0, min(100.0, dap_val))
        base = (
            ANCHOR_CURVE_L
            / (1.0 + math.exp(-ANCHOR_CURVE_K * (dap_val - ANCHOR_CURVE_X0)))
            + ANCHOR_CURVE_N0
        )
        k = math.ceil(max(0.0, base) * margin + ANCHOR_MARGIN_FLAT)
    else:
        return max_anchors

    return int(min(max(k, 8), max_anchors))


def estimate_anchor_capacity(
    dap: Optional[torch.Tensor],
    margin: float = ANCHOR_MARGIN,
    flat: float = ANCHOR_MARGIN_FLAT,
    max_anchors: int = 512,
) -> torch.Tensor:
    """Per-sample anchor capacity from DAP via the calibrated logistic phytomer curve.

    Returns a (B,) long tensor of per-sample anchor capacities. Used to build
    per-sample capacity-aware losses (e.g. clamping GT existence targets to the
    active anchor slice) without collapsing the batch to a single max-K.
    """
    if dap is None:
        return torch.full((1,), max_anchors, dtype=torch.long)
    d = dap.float().view(-1)
    base = (
        ANCHOR_CURVE_L / (1.0 + torch.exp(-ANCHOR_CURVE_K * (d - ANCHOR_CURVE_X0)))
        + ANCHOR_CURVE_N0
    )
    k = torch.ceil(torch.clamp(base, min=0.0) * margin + flat)
    return k.clamp(min=8, max=max_anchors).long()


def estimate_organ_budget(dap: Optional[torch.Tensor], margin: float = 1.35, max_slots: int = 4096) -> torch.Tensor:
    """Estimates expected active organ capacity using fitted botanical sigmoid growth curve.

    Fitted Logistic Model for Cowpea:
      N(dap) = clamp( (L / (1 + exp(-k * (dap - x0))) + N0) * margin, min=24, max=max_slots )
    where L=816.92, k=0.0679, x0=41.84, N0=-43.60.
    With margin=1.35 (+35% upper bound):
      DAP 05 -> ~25 organs
      DAP 10 -> ~55 organs
      DAP 20 -> ~145 organs
      DAP 30 -> ~282 organs
      DAP 40 -> ~458 organs
      DAP 50 -> ~642 organs
      DAP 60 -> ~795 organs
      DAP 80 -> ~967 organs
    """
    if dap is None:
        return torch.tensor([min(500, max_slots)], dtype=torch.long)
    d = dap.float()
    base = 816.92 / (1.0 + torch.exp(-0.0679 * (d - 41.84))) - 43.60
    budget = torch.clamp(base * margin, min=14.0, max=float(max_slots))
    return budget.long()


class MacroBiologicalHead(nn.Module):
    """Stage 1: Macro Biological Prior Head.

    Predicts:
    1. Plant Age (DAP in [0, 100])
    2. Phytomer Count (N_phytomer >= 0.0)
    3. Soft Tapering Existence Prior (w_k in [0, 1]) with additive logit bias
       for direct, differentiable gradient flow from existence loss to macro topology.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        default_margin: float = 1.0,
        default_tau: float = 0.8,
        init_phytomer_count: float = 50.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.default_margin = default_margin
        self.default_tau = default_tau

        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.GELU(),
        )
        self.dap_head = nn.Linear(embed_dim // 2, 1)
        self.phy_head = nn.Linear(embed_dim // 2, 1)
        # Bias-init the count head so pred_num starts near the dataset mean instead
        # of the activation floor (elu(x)+1 min = 1.0). elu is positive-leaning above
        # 0, so bias b gives pred ~= elu(b)+1 ~= b+1; b = ln(mean) puts pred ~ mean.
        # Without this the pred-count prior (init_logits) suppresses all anchors
        # beyond k~2 for the first epochs -> nothing gets rendered (IoU 0%).
        with torch.no_grad():
            self.phy_head.bias.fill_(math.log(max(init_phytomer_count, 1.1) - 1.0))

    def forward(
        self,
        cls_token: torch.Tensor,
        max_k: int,
        margin: Optional[float] = None,
        tau: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        m = margin if margin is not None else self.default_margin
        t = tau if tau is not None else self.default_tau

        h = self.net(cls_token)
        pred_dap = F.relu(self.dap_head(h)) * 100.0  # (B, 1)
        pred_num_phytomers = F.elu(self.phy_head(h)) + 1.0  # (B, 1) >= 0.0

        # Soft tapering margin schedule across slots 0..max_k-1
        # w_k = sigmoid((N_phy + margin - k) / tau)
        # Additive logit bias: init_logits = (N_phy + margin - k) / tau
        k_indices = torch.arange(max_k, device=cls_token.device, dtype=torch.float32).unsqueeze(0)  # (1, K)
        # Numerical safety clamp: limits logits to [-15.0, 15.0] to strictly prevent IEEE 754 Float32
        # exp() overflow in binary_cross_entropy_with_logits backward pass.
        init_logits = torch.clamp((pred_num_phytomers + m - k_indices) / t, min=-15.0, max=15.0)  # (B, K)
        soft_margin_weights = torch.sigmoid(init_logits)  # (B, K) in [0, 1]

        return {
            "pred_dap": pred_dap,
            "pred_num_phytomers": pred_num_phytomers,
            "soft_margin_weights": soft_margin_weights,
            "init_logits": init_logits.unsqueeze(-1),  # (B, K, 1)
        }


class CoarseSkeletalTransformer(nn.Module):
    """Stage 2: Deterministic Set Transformer predicting coarse 3D skeletal node point cloud scaffold.

    Predicts 3D base coordinates, 6D orientation, and existence logits for K anchors.
    Uses DETR3D / PETR style 3D reference coordinates, 3D positional encoding,
    and incorporates Stage 1 MacroBiologicalHead soft margin prior.
    """

    def __init__(
        self,
        max_anchors: int = 512,
        embed_dim: int = 384,
        num_heads: int = 8,
        num_layers: int = 4,
        init_phytomer_count: float = 50.0,
        dap_clue: bool = True,
    ):
        super().__init__()
        self.max_anchors = max_anchors
        self.embed_dim = embed_dim
        self.dap_clue = dap_clue

        # Content query
        self.anchor_queries = nn.Parameter(torch.randn(max_anchors, embed_dim) * 0.02)

        # 3D Reference Points (DETR3D / PETR style): initialized across canonical plant envelope
        # x, y in [-0.5, 0.5], z vertically ascending in [0.0, 1.0]
        init_ref = torch.randn(max_anchors, 3) * 0.15
        init_ref[:, 2] = torch.linspace(0.0, 1.0, max_anchors)  # Vertical upward growth prior
        self.ref_points = nn.Parameter(init_ref)

        # 3D Positional Encoder for reference points
        self.ref_pos_mlp = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Decoder layers for cross-attention to 3D visual tokens
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Stage 1: Macro Biological Head
        self.macro_head = MacroBiologicalHead(embed_dim=embed_dim,
                                              init_phytomer_count=init_phytomer_count)

        # DAP clue embedding: broadcasts the predicted plant age to every anchor
        # query so Stage 2/3 condition on the developmental stage (like the macro
        # count prior). Normalized DAP -> MLP; zero-init out layer keeps the
        # initial behavior identical to the unconditioned model.
        self.dap_embed = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        nn.init.zeros_(self.dap_embed[-1].weight)
        nn.init.zeros_(self.dap_embed[-1].bias)

        # Stage 2 Prediction Heads
        # 1. Delta 3D position relative to 3D reference points
        self.pos_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),
        )

        # 2. Anchor 6D continuous rotation
        self.rot_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 6),
        )

        # 2b. Coarse anchor scale (petiole scale row [len, radius, unused], FM
        # units): bridge-prior seed for the 76D flow's scale dims. Softplus keeps
        # it positive; the flow refines it against the GT anchor scale.
        self.scale_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),
        )

        # 3. Anchor existence probability logit (delta relative to soft margin prior)
        self.exist_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

        # Edge-biased anchor self-attention (soft graph prior): nearby anchors
        # (small 3D distance) attend more strongly, enforcing spatial coherence
        # along the shoot axis (kinematic-chain locality). GraphFormer-style
        # distance bias added to the attention scores.
        self.anchor_self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.05, batch_first=True,
        )
        self.anchor_norm = nn.LayerNorm(embed_dim)
        self.edge_bias_temp = 0.15

    def forward(
        self,
        image_tokens: torch.Tensor,
        active_k: Optional[int] = None,
        capacity_mode: str = "given",
        margin: float = ANCHOR_MARGIN,
        flat: float = ANCHOR_MARGIN_FLAT,
        pred_dap: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Stage 1 + Stage 2 forward pass.

        Args:
            image_tokens: (B, T, embed_dim) 3D-aware visual tokens from DINOv2RayEncoder.
            active_k: Sliced anchor count. Required when capacity_mode='given'.
            capacity_mode:
                'given'      - use `active_k` as-is (GT-DAP teacher forcing path).
                'pred_phyto' - two-pass: run MacroBiologicalHead first on the CLS token,
                               read `pred_num_phytomers`, then slice the anchor bank to
                               K = ceil(max_sample * margin + flat). The count prediction
                               itself stays fully differentiable (loss path), while the
                               slicing index is a discrete capacity decision (no gradient
                               needed — the soft-margin prior carries the botanical prior).
                'full'       - use the entire anchor bank (max_anchors).
        Returns:
            Dict containing:
                'anchor_pos': (B, K, 3) predicted 3D anchor base positions.
                'anchor_rot': (B, K, 6) predicted 6D rotation.
                'anchor_logits': (B, K, 1) existence logits with soft margin bias.
                'anchor_features': (B, K, embed_dim) anchor latent representations.
                'pred_dap': (B, 1) auxiliary estimated DAP.
                'pred_num_phytomers': (B, 1) predicted phytomer count.
                'soft_margin_weights': (B, K) soft tapering existence prior.
        """
        B = image_tokens.shape[0]
        cls_token = image_tokens[:, 0]

        # Stage 1 (Macro Biological Prior) FIRST: it lives on the CLS token and needs
        # the full max_k width for the soft-margin schedule. In 'pred_phyto' mode this
        # pass also determines the anchor bank slicing.
        macro_out_full = self.macro_head(cls_token, max_k=self.max_anchors)

        if capacity_mode == "given":
            if active_k is None:
                active_k = self.max_anchors
            K = int(active_k)
        elif capacity_mode == "pred_phyto":
            # Discrete capacity from the (differentiable) phytomer count prediction.
            # The count tensor's autograd graph is preserved in macro_out_full; the
            # .max()/.item() read below only forks the scalar used for slicing.
            K = compute_matryoshka_slice(
                num_phytomers=macro_out_full["pred_num_phytomers"].detach(),
                max_anchors=self.max_anchors,
                margin=margin,
            )
            K = max(int(K), 8)
        elif capacity_mode == "full":
            K = self.max_anchors
        else:
            raise ValueError(f"Unknown capacity_mode: {capacity_mode}")

        # Query = Content embedding + 3D Positional Encoding + DAP clue.
        # The predicted DAP (from Stage 1) is broadcast to all anchors as a
        # developmental-stage prior; zero-init keeps the start state unchanged
        # while loss_dap gradients teach the head, and dap_embed learns to use it.
        q_content = self.anchor_queries[:K].unsqueeze(0).expand(B, -1, -1)
        q_pos = self.ref_pos_mlp(self.ref_points[:K]).unsqueeze(0).expand(B, -1, -1)
        q = q_content + q_pos
        if self.dap_clue and pred_dap is not None:
            d = (pred_dap.float().view(B, 1) / 100.0)  # normalize to [0, 1]
            q = q + self.dap_embed(d).unsqueeze(1).expand(-1, K, -1)

        # Cross-attend with 3D-aware image tokens
        anchor_features = self.decoder(q, image_tokens)

        # Edge-biased anchor self-attention (soft graph prior): distance-based
        # attention bias so spatially adjacent anchors (kinematic chain) share
        # context. Bias = -dist / temp, added to the attention scores via the
        # attn_mask slot (additive, per-head broadcast). Distances come from the
        # ordered 3D reference points (z-ascending = chain order), so the prior
        # is static and available before the decoder.
        ref_k = self.ref_points[:K]  # (K, 3)
        dist_k = torch.cdist(ref_k, ref_k)  # (K, K)
        edge_bias = -dist_k / self.edge_bias_temp  # (K, K)
        attn_out, _ = self.anchor_self_attn(
            anchor_features, anchor_features, anchor_features,
            attn_mask=edge_bias,
            need_weights=False,
        )
        anchor_features = self.anchor_norm(anchor_features + attn_out)

        # Stage 2 Heads: predict coordinate offset from 3D reference points (physically bounded to +/- 0.5m)
        delta_pos = torch.tanh(self.pos_head(anchor_features)) * 0.5
        anchor_pos = self.ref_points[:K].unsqueeze(0) + delta_pos

        # Continuous 6D rotation with Gram-Schmidt orthonormalization (Zhou et al., CVPR 2019)
        # Prevents norm explosion (Rot: 205) and enforces valid SO(3) frame
        raw_rot = self.rot_head(anchor_features)
        v1 = raw_rot[..., :3]
        v2 = raw_rot[..., 3:]
        e1 = F.normalize(v1, dim=-1, eps=1e-6)
        u2 = v2 - (e1 * v2).sum(dim=-1, keepdim=True) * e1
        e2 = F.normalize(u2, dim=-1, eps=1e-6)
        anchor_rot = torch.cat([e1, e2], dim=-1)

        anchor_scale = (F.softplus(self.scale_head(anchor_features)) + 1e-4).clamp(max=2.0)  # (B, K, 3)

        # Re-slice the soft margin prior to the active width (computed once, full width)
        macro_out = {
            "pred_dap": macro_out_full["pred_dap"],
            "pred_num_phytomers": macro_out_full["pred_num_phytomers"],
            "soft_margin_weights": macro_out_full["soft_margin_weights"][:, :K],
        }

        # Predict anchor existence logits, strictly bounded to [-8.0, 8.0]
        # (Numerical guarantee: BCE loss can never exceed 8.0, preventing Ext loss explosion)
        anchor_logits = torch.tanh(self.exist_head(anchor_features)) * 8.0

        return {
            "anchor_pos": anchor_pos,
            "anchor_rot": anchor_rot,
            "anchor_scale": anchor_scale,
            "anchor_logits": anchor_logits,
            "anchor_features": anchor_features,
            "pred_dap": macro_out["pred_dap"],
            "pred_num_phytomers": macro_out["pred_num_phytomers"],
            "soft_margin_weights": macro_out["soft_margin_weights"],
            "active_k": K,
        }


# =============================================================================
# CANONICAL PHYTOMER ROLES FOR M=8 FINE SLOTS
# =============================================================================
ROLE_INTERNODE = 0      # Slot 0: Main/lateral stem segment
ROLE_PETIOLE = 1        # Slot 1: Leaf stalk (connects node to leaflets)
ROLE_LEAF = 2           # Slots 2, 3, 4: Trifoliate leaflets (terminal, left, right)
ROLE_PEDUNCLE = 3       # Slot 5: Inflorescence stem (flower stalk)
ROLE_REPRODUCTIVE = 4   # Slots 6, 7: Flowers (closed/open), Pods, or Dormant Buds
NUM_PHYTOMER_ROLES = 5

SLOT_ROLE_MAPPING = [0, 1, 2, 2, 2, 3, 4, 4]
SLOT_SUB_ROLE_MAPPING = [0, 0, 0, 1, 2, 0, 0, 1]  # Sub-index within role


class AnchorVisualProjector(nn.Module):
    """Hybrid 3D-to-2D Anchor Visual Projector (PETR / Point-Query style).

    Projects 3D anchor scaffold coordinates (B, K, 3) onto the 2D DINOv2 patch
    token grid (e.g. 16x16 = 256 patches), bilinearly samples local visual features,
    and fuses them with 3D height/depth cues via a zero-initialized residual MLP.
    """

    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.embed_dim = embed_dim

        # Project 3D position [x, y, z] to 2D normalized image grid [u, v] in [-1, 1]
        # In drone top-down view: x in [-0.5, 0.5], y in [-0.5, 0.5] map across the image.
        self.pos_to_uv = nn.Linear(3, 2)
        with torch.no_grad():
            self.pos_to_uv.weight.zero_()
            self.pos_to_uv.weight[0, 0] = 2.0  # x -> u
            self.pos_to_uv.weight[1, 1] = 2.0  # y -> v
            self.pos_to_uv.bias.zero_()

        # Fusion MLP: combines sampled 2D visual feature (C) with 3D coordinate (3)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Zero-initialize the final projection so initial output is 0 (residual identity)
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, image_tokens: torch.Tensor, anchor_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_tokens: (B, 1 + N_patches, C) with N_patches = H * W (e.g. 256 for 16x16)
            anchor_pos: (B, K, 3) 3D anchor scaffold coordinates
        Returns:
            (B, K, C) localized visual feature embedding for each anchor
        """
        B, K, _ = anchor_pos.shape
        # Strip CLS token (token 0) to leave the 2D spatial patch tokens
        patch_tokens = image_tokens[:, 1:]  # (B, N_patches, C)
        N_patches = patch_tokens.shape[1]
        grid_size = int(math.isqrt(N_patches))
        if grid_size * grid_size != N_patches:
            return torch.zeros(B, K, self.embed_dim, device=anchor_pos.device, dtype=anchor_pos.dtype)

        # Reshape to 2D spatial feature map: (B, H, W, C) -> (B, C, H, W)
        feat_map = patch_tokens.view(B, grid_size, grid_size, self.embed_dim).permute(0, 3, 1, 2)

        # Compute 2D normalized grid coordinates (u, v) in [-1, 1]
        uv = torch.tanh(self.pos_to_uv(anchor_pos.float()))  # (B, K, 2) in [-1, 1]
        grid = uv.unsqueeze(1)  # (B, 1, K, 2)

        # Bilinear sampling from patch feature map
        sampled = F.grid_sample(
            feat_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled = sampled.squeeze(2).transpose(1, 2)  # (B, K, C)

        # Fuse sampled local visual features with 3D anchor coordinates
        fused = self.fusion(torch.cat([sampled, anchor_pos], dim=-1))  # (B, K, C)
        return fused


class FineBotanicalFlowMatchingDecoder(nn.Module):
    """Stage 2: Flow Matching Decoder predicting microscopic organ velocity field.

    Dispatches M fine slots per active anchor (M=8: 1 stem + 1 petiole + 3 leaflets + 1 peduncle + 2 reproductive).
    Performs joint intra-block kinematic attention within each phytomer, followed by global cross-attention.
    """

    def __init__(
        self,
        slots_per_anchor: int = 10,
        node_dim: int = 16,  # 16D latent vector from OrganLatentVAE
        num_classes: int = NUM_ORGAN_TYPES,  # 13 organ categories
        embed_dim: int = 384,
        num_heads: int = 8,
        num_layers: int = 6,
    ):
        super().__init__()
        self.slots_per_anchor = slots_per_anchor
        self.node_dim = node_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Canonical Phytomer Functional Role Embeddings (Stem, Petiole, Leaf, Peduncle, Reproductive)
        self.functional_role_emb = nn.Embedding(NUM_PHYTOMER_ROLES, embed_dim)
        self.slot_pos_emb = nn.Embedding(slots_per_anchor, embed_dim)
        self.role_emb = self.slot_pos_emb  # Backward compatibility alias

        # Hybrid 3D-to-2D Anchor Visual Projector (Point-Query sampling)
        self.anchor_projector = AnchorVisualProjector(embed_dim=embed_dim)

        # Continuous geometry projection
        self.geom_proj = nn.Linear(node_dim, embed_dim)

        # Sinusoidal timestep embedding for continuous flow time t in [0, 1]
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Intra-Phytomer Block Multi-Head Self-Attention:
        # Allows the M=8 organs within each local phytomer block to coordinate joint kinematics
        # (petiole pitch, leaflet spread, stem orientation) prior to global image conditioning
        self.block_norm = nn.LayerNorm(embed_dim)
        self.block_self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.05,
            batch_first=True,
        )

        # Cross-layer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 3D Node Scaffold Positional & Rotational Encoders (Stage 2 -> Stage 3 Conditioning)
        self.node_pos_mlp = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.node_rot_mlp = nn.Sequential(
            nn.Linear(6, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Continuous geometry velocity head (v_theta predicting d/dt of 16D latent)
        self.velocity_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, node_dim),
        )

        # Fine slot existence logit head (determines active vs pruned slot)
        self.exist_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
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
        noisy_fine_nodes: torch.Tensor,
        timesteps: torch.Tensor,
        anchor_features: torch.Tensor,
        image_tokens: torch.Tensor,
        anchor_pos: Optional[torch.Tensor] = None,
        anchor_rot: Optional[torch.Tensor] = None,
        existence_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            noisy_fine_nodes: (B, N_fine, node_dim) noisy geometry x_t where N_fine = K * M.
            timesteps: (B,) flow time t in [0, 1].
            anchor_features: (B, K, embed_dim) from Stage 2 CoarseSkeletalTransformer.
            image_tokens: (B, T, embed_dim) from ViT.
            anchor_pos: Optional (B, K, 3) predicted 3D node scaffold coordinates.
            anchor_rot: Optional (B, K, 6) predicted 3D node orientations.
            existence_mask: Optional (B, N_fine) boolean mask of active slots.
        Returns:
            Dict containing:
                'pred_velocity': (B, N_fine, node_dim) geometry velocity vector field.
                'pred_exist_logits': (B, N_fine, 1) fine slot existence logits.
        """
        B, N_fine, _ = noisy_fine_nodes.shape
        K = anchor_features.shape[1]
        M = self.slots_per_anchor
        assert N_fine == K * M, f"Fine node count ({N_fine}) must equal K*M ({K}*{M})"
        device = noisy_fine_nodes.device

        # 1. Expand anchor features across their M children
        # (B, K, D) -> (B, K, 1, D) -> (B, K, M, D) -> (B, N_fine, D)
        expanded_anchors = anchor_features.unsqueeze(2).expand(-1, -1, M, -1).reshape(B, N_fine, self.embed_dim)

        # 2. Stage 2 -> 3: 3D Node Scaffold Positional, Visual Projector & Rotational Condition Embeddings
        if anchor_pos is not None:
            node_pos_emb = self.node_pos_mlp(anchor_pos).unsqueeze(2).expand(-1, -1, M, -1).reshape(B, N_fine, self.embed_dim)
            proj_feat = self.anchor_projector(image_tokens, anchor_pos)  # (B, K, C)
            proj_emb = proj_feat.unsqueeze(2).expand(-1, -1, M, -1).reshape(B, N_fine, self.embed_dim)
        else:
            node_pos_emb = 0.0
            proj_emb = 0.0

        if anchor_rot is not None:
            node_rot_emb = self.node_rot_mlp(anchor_rot).unsqueeze(2).expand(-1, -1, M, -1).reshape(B, N_fine, self.embed_dim)
        else:
            node_rot_emb = 0.0

        # 3. Canonical Phytomer Role + Positional Embeddings
        # Map slot positions [0..M-1] to botanical functional roles [0..4]
        role_map_tensor = torch.as_tensor(SLOT_ROLE_MAPPING[:M], dtype=torch.long, device=device)
        slot_pos_tensor = torch.arange(M, device=device)
        block_role_embs = self.functional_role_emb(role_map_tensor) + self.slot_pos_emb(slot_pos_tensor)  # (M, D)
        role_embs = block_role_embs.unsqueeze(0).expand(K, -1, -1).reshape(1, N_fine, self.embed_dim).expand(B, -1, -1)

        # 4. Geometry projection
        geom_embs = self.geom_proj(noisy_fine_nodes)

        # 5. Time conditioning
        t_emb = self.time_embed(self._sinusoidal(timesteps)).unsqueeze(1)  # (B, 1, D)

        # Composite query representation (Conditioned on 3D Scaffold + Anchor Latents + Roles + Time + Hybrid Projector)
        queries = geom_embs + expanded_anchors + node_pos_emb + proj_emb + node_rot_emb + role_embs + t_emb

        # 6. Intra-Phytomer Block Multi-Head Self-Attention:
        # Reshape to (B * K, M, D) so local organs directly communicate joint kinematics
        q_block = queries.view(B * K, M, self.embed_dim)
        q_norm = self.block_norm(q_block)
        block_attn_out, _ = self.block_self_attn(q_norm, q_norm, q_norm)
        queries = (q_block + block_attn_out).view(B, N_fine, self.embed_dim)

        # 7. Global Decoder cross-attention to image tokens & anchor memory
        # Concatenate image tokens with anchor features as memory for full multi-scale conditioning
        memory = torch.cat([image_tokens, anchor_features], dim=1)

        x = self.decoder(queries, memory)

        # 8. Heads
        pred_velocity = self.velocity_head(x)
        pred_exist_logits = torch.tanh(self.exist_head(x)) * 8.0

        return {
            "pred_velocity": pred_velocity,
            "pred_exist_logits": pred_exist_logits,
        }


class PhytomerFlowMatchingDecoder(nn.Module):
    """Stage 3 (phytomer mode): flow-matches base + rot + latent per anchor.

    Unlike FineBotanicalFlowMatchingDecoder, which dispatches M organ slots per
    anchor (and uses pose only as conditioning), this decoder flow-matches ONE
    12+D vector per anchor:
        z_1 = [ node_base_xyz(3) | node_rot_6d(6) | phytomer_latent(D) ]
    so Stage 2's 3D node scaffold (base + rot) is REFINED by flow, not fixed.
    The refined node_rot is also the reference frame for the anchor-relative
    packet (Option 1). Existence stays a separate gating head (per-slot, Kx8);
    it is never flow-matched (a 0/1 gate regresses to its mean under velocity
    matching).

    Tokens: K anchors (vs M*K slots) -> Mx fewer, O(K^2) self-attn -> M^2x cheaper.
    No intra-block (M) self-attention: the joint latent is per-phytomer.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        base_dim: int = 3,
        rot_dim: int = 6,
        scale_dim: int = 3,
        num_classes: int = NUM_ORGAN_TYPES,
        embed_dim: int = 384,
        num_heads: int = 8,
        num_layers: int = 6,
        slots_per_phytomer: int = 10,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_dim = base_dim
        self.rot_dim = rot_dim
        self.scale_dim = scale_dim
        self.slots_per_phytomer = slots_per_phytomer
        self.node_flow_dim = base_dim + rot_dim + scale_dim + latent_dim  # 12 + D
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Continuous flow vector projection (12 + D -> embed)
        self.geom_proj = nn.Linear(self.node_flow_dim, embed_dim)

        # Sinusoidal timestep embedding for continuous flow time t in [0, 1]
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Decoder: cross-attention query -> image tokens + anchor memory
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 3D Node Scaffold Positional, Rotational & Scale Encoders (Stage 2 -> Stage 3 conditioning)
        self.node_pos_mlp = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.node_rot_mlp = nn.Sequential(
            nn.Linear(6, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.node_scale_mlp = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Hybrid 3D-to-2D Anchor Visual Projector (Point-Query sampling)
        self.anchor_projector = AnchorVisualProjector(embed_dim=embed_dim)

        # Velocity head: predicts d/dt of the flow vector (64D in decoupled mode)
        self.velocity_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, self.node_flow_dim),
        )

        # Per-slot existence logits (B, K, M) — separate gating head, NOT flow-matched.
        # Predicts which of the M canonical organs are present in this phytomer.
        self.exist_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, self.slots_per_phytomer),
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
        noisy_flow: torch.Tensor,
        timesteps: torch.Tensor,
        anchor_features: torch.Tensor,
        image_tokens: torch.Tensor,
        anchor_pos: Optional[torch.Tensor] = None,
        anchor_rot: Optional[torch.Tensor] = None,
        anchor_scale: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Flow-matches a per-anchor latent vector (64D in decoupled mode).

        Args:
            noisy_flow: (B, K, node_flow_dim) interpolated x_t.
            timesteps: (B,) flow time t in [0, 1].
            anchor_features: (B, K, embed) from Stage 2 CoarseSkeletalTransformer.
            image_tokens: (B, T, embed) from ViT.
            anchor_pos: Optional (B, K, 3) Stage-2 scaffold base positions.
            anchor_rot: Optional (B, K, 6) Stage-2 scaffold rotations.
            anchor_scale: Optional (B, K, 3) Stage-2 scaffold scales.
        Returns:
            'pred_velocity': (B, K, node_flow_dim) velocity field.
            'pred_slot_exist_logits': (B, K, M) per-slot existence logits.
        """
        B, K, _ = noisy_flow.shape
        device = noisy_flow.device

        # Continuous flow vector projection + sinusoidal timestep embedding
        geom_embs = self.geom_proj(noisy_flow)  # (B, K, embed)
        t_emb = self.time_embed(self._sinusoidal(timesteps)).unsqueeze(1)  # (B, 1, embed)
        queries = geom_embs + t_emb

        # Stage 2 3D Scaffold Conditioning
        if anchor_pos is not None:
            queries = queries + self.node_pos_mlp(anchor_pos)
            # Hybrid Projection: inject local 2D-projected visual token into query
            queries = queries + self.anchor_projector(image_tokens, anchor_pos)
        if anchor_rot is not None:
            queries = queries + self.node_rot_mlp(anchor_rot)
        if anchor_scale is not None:
            queries = queries + self.node_scale_mlp(anchor_scale)

        # Global decoder cross-attention to image tokens + anchor memory
        memory = torch.cat([image_tokens, anchor_features], dim=1)
        x = self.decoder(queries, memory)

        pred_velocity = self.velocity_head(x)
        pred_slot_exist_logits = torch.tanh(self.exist_head(x)) * 8.0

        return {
            "pred_velocity": pred_velocity,
            "pred_slot_exist_logits": pred_slot_exist_logits,
        }


class HierarchicalPartFlowMatchingModel(nn.Module):
    """End-to-End 3-Stage Cascaded Hierarchical Botanical Flow Matching System.

    Stage 1: Macro Biological Prior (Plant Age DAP & Phytomer Count N_phy with Soft Margin).
    Stage 2: Coarse 3D Node Point Cloud Scaffold (Anchors along shoot axis).
    Stage 3: Intra-Phytomer Canonical Flow Matching (Organ geometry relative to node).
    """

    def __init__(
        self,
        max_anchors: int = 512,
        slots_per_anchor: int = 10,
        node_dim: int = 16,
        num_classes: int = NUM_ORGAN_TYPES,
        image_size: int = 128,
        patch_size: int = 8,
        embed_dim: int = 384,
        vit_layers: int = 8,
        vit_heads: int = 8,
        coarse_layers: int = 4,
        fine_layers: int = 6,
        flow_granularity: str = "organ",
        phytomer_latent_dim: int = 64,
        backbone: str = "dinov2_vits14",
        freeze_backbone: bool = False,
        init_phytomer_count: float = 50.0,
    ):
        super().__init__()
        self.max_anchors = max_anchors
        self.slots_per_anchor = slots_per_anchor
        self.max_fine_slots = max_anchors * slots_per_anchor  # 512 * 10 = 5,120
        self.node_dim = node_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.flow_granularity = flow_granularity
        self.phytomer_latent_dim = phytomer_latent_dim

        # 1. Pretrained DINO Backbone with 3D Camera Ray Positional Embedding (PETR style).
        #    Swappable via `backbone` for scaling A/B (dinov2_vits14 / vitb14 / vitl14 /
        #    dinov3_vits16 / vitb16 / vitl16 / vitl16_sat).
        self.image_encoder = DINOv2RayEncoder(
            backbone=backbone,
            pretrained=True,
            freeze_backbone=freeze_backbone,
            embed_dim=embed_dim,
        )

        # Semantic color palette (13, 3): learnable per-organ-type RGB used by the
        # differentiable render path (c_v = probs_v @ palette). Initialized from
        # the geometry builder's hardcoded table so behavior starts identical;
        # the cos-color loss then gives it gradient (previously the constant
        # colors carried none). Lives on the model so the optimizer tracks it.
        _pal = torch.zeros(13, 3)
        _pal[0] = torch.tensor([0.0, 0.0, 0.0])      # NONE
        _pal[1] = torch.tensor([0.20, 0.15, 0.10])   # ROOT_META
        _pal[2] = torch.tensor([0.20, 0.15, 0.10])   # SHOOT_META
        _pal[3] = torch.tensor([0.22, 0.45, 0.15])   # INTERNODE (COLOR_STEM)
        _pal[4] = torch.tensor([0.25, 0.50, 0.18])   # PETIOLE
        _pal[5] = torch.tensor([0.25, 0.62, 0.18])   # LEAF
        _pal[6] = torch.tensor([0.55, 0.52, 0.25])   # PEDUNCLE
        _pal[7] = torch.tensor([0.25, 0.45, 0.15])   # BUD_DORMANT
        _pal[8] = torch.tensor([0.30, 0.50, 0.18])   # BUD_ACTIVE
        _pal[9] = torch.tensor([0.98, 0.85, 0.15])   # FLOWER_CLOSED
        _pal[10] = torch.tensor([0.98, 0.85, 0.15])  # FLOWER_OPEN
        _pal[11] = torch.tensor([0.85, 0.65, 0.13])  # FRUIT (COLOR_POD)
        _pal[12] = torch.tensor([0.40, 0.30, 0.15])  # BUD_ABORTED
        self.register_buffer("color_palette", _pal)

        # 2. Stage 1 & 2: Coarse Skeletal Transformer with MacroBiologicalHead
        self.coarse_stage = CoarseSkeletalTransformer(
            max_anchors=max_anchors,
            embed_dim=embed_dim,
            num_heads=vit_heads,
            num_layers=coarse_layers,
            init_phytomer_count=init_phytomer_count,
        )

        # 3. Stage 3: Fine Botanical Flow Matching Decoder conditioned on 3D Scaffold.
        #    flow_granularity:
        #      'organ'   - per-organ slots (8K x node_dim), pose as conditioning.
        #      'phytomer'- per-anchor 64D VAE latent flow vector (Hybrid Decoupled Architecture:
        #                  pure standard Gaussian prior z_0 ~ N(0, I_64), pose/scale as conditioning).
        if flow_granularity == "phytomer":
            self.fine_stage = PhytomerFlowMatchingDecoder(
                latent_dim=phytomer_latent_dim,
                base_dim=0,
                rot_dim=0,
                scale_dim=0,
                num_classes=num_classes,
                embed_dim=embed_dim,
                num_heads=vit_heads,
                num_layers=fine_layers,
                slots_per_phytomer=slots_per_anchor,
            )
        else:
            self.fine_stage = FineBotanicalFlowMatchingDecoder(
                slots_per_anchor=slots_per_anchor,
                node_dim=node_dim,
                num_classes=num_classes,
                embed_dim=embed_dim,
                num_heads=vit_heads,
                num_layers=fine_layers,
            )

    def probe_pred_dap(self, image_tokens: torch.Tensor) -> torch.Tensor:
        """Two-pass helper: reads Stage 1's predicted DAP from the CLS token (no grad
        needed for the clue itself — the clue is an input, loss_dap keeps the head
        supervised)."""
        cls_token = image_tokens[:, 0]
        with torch.no_grad():
            return self.coarse_stage.macro_head(cls_token, max_k=self.max_anchors)["pred_dap"]

    def forward(
        self,
        noisy_fine_nodes: torch.Tensor,
        timesteps: torch.Tensor,
        images: torch.Tensor,
        daps: Optional[torch.Tensor] = None,
        teacher_anchor_pos: Optional[torch.Tensor] = None,
        num_phytomers: Optional[torch.Tensor] = None,
        capacity_mode: str = "given",
        image_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Unified joint forward pass across 3 cascaded stages.

        Args:
            noisy_fine_nodes: (B, N_fine, node_dim) interpolated geometry x_t.
            timesteps: (B,) flow time t in [0, 1].
            images: (B, 4, H, W) or (B, 16, H, W) drone RGB+CHM imagery.
            daps: Optional (B,) plant age tensor for Matryoshka slicing (teacher forcing).
            teacher_anchor_pos: Optional (B, K, 3) ground-truth anchor positions for teacher forcing.
            num_phytomers: Optional (B,) ground-truth phytomer count for Matryoshka slicing.
            capacity_mode:
                'given'      - Stage 2 slices the anchor bank from (GT) DAP/phytomer hints
                               via compute_matryoshka_slice. Teacher-forcing training path.
                'pred_phyto' - Stage 2 slices from the predicted phytomer count (two-pass).
                               Inference / self-conditioning path.
        """
        # 1. Vision perception: 3D-aware multi-scale tokens.
        # The caller may pass precomputed tokens (probe reuse: the training loop
        # already ran the encoder once for capacity slicing — passing them in
        # saves one full DINOv2 forward per step).
        if image_tokens is None:
            image_tokens = self.image_encoder(images)

        # 2. Anchor bank slicing. The predicted DAP is passed as a stage clue so
        #    Stage 2/3 condition on the developmental stage (dap_embed); in the
        #    GT-DAP teacher-forcing path the GT value substitutes for reliability.
        if capacity_mode == "given":
            active_k = compute_matryoshka_slice(dap=daps, num_phytomers=num_phytomers, max_anchors=self.max_anchors)
            clue_dap = daps  # GT DAP as the clue during teacher forcing
            coarse_out = self.coarse_stage(image_tokens, active_k=active_k, pred_dap=clue_dap)
        else:
            clue = self.probe_pred_dap(image_tokens)
            coarse_out = self.coarse_stage(image_tokens, capacity_mode=capacity_mode, pred_dap=clue)

        anchor_features = coarse_out["anchor_features"]
        anchor_pos = coarse_out["anchor_pos"]
        anchor_rot = coarse_out["anchor_rot"]
        active_k = int(coarse_out["active_k"])

        # If noisy nodes exceed active fine slots, slice accordingly; if fewer, pad
        # (the anchor bank slice is authoritative — K_out drives Stage 3's width).
        # In phytomer mode, noisy_fine_nodes is already (B, K, 12+D) — no 8K slicing.
        if self.flow_granularity == "phytomer":
            if noisy_fine_nodes.shape[1] > active_k:
                noisy_fine_nodes = noisy_fine_nodes[:, :active_k]
            elif noisy_fine_nodes.shape[1] < active_k:
                pad_n = active_k - noisy_fine_nodes.shape[1]
                noisy_fine_nodes = F.pad(noisy_fine_nodes, (0, 0, 0, pad_n))
        else:
            active_fine = active_k * self.slots_per_anchor
            if noisy_fine_nodes.shape[1] > active_fine:
                noisy_fine_nodes = noisy_fine_nodes[:, :active_fine]
            elif noisy_fine_nodes.shape[1] < active_fine:
                pad_n = active_fine - noisy_fine_nodes.shape[1]
                noisy_fine_nodes = F.pad(noisy_fine_nodes, (0, 0, 0, pad_n))

        # 4. Stage 3: Fine Botanical Flow Matching conditioned on 3D Scaffold
        # anchor_features, anchor_pos, anchor_rot, and anchor_scale are detached as conditioning inputs so the velocity
        # loss refines latent geometry without backpropagating into Stage-2 scaffold heads or features.
        anchor_pos = coarse_out["anchor_pos"]
        anchor_rot = coarse_out["anchor_rot"]
        anchor_scale = coarse_out.get("anchor_scale")
        if self.flow_granularity == "phytomer":
            fine_out = self.fine_stage(
                noisy_flow=noisy_fine_nodes,
                timesteps=timesteps,
                anchor_features=anchor_features.detach(),
                image_tokens=image_tokens,
                anchor_pos=anchor_pos.detach() if anchor_pos is not None else None,
                anchor_rot=anchor_rot.detach() if anchor_rot is not None else None,
                anchor_scale=anchor_scale.detach() if anchor_scale is not None else None,
            )
            pred_velocity = fine_out["pred_velocity"]
        else:
            fine_out = self.fine_stage(
                noisy_fine_nodes=noisy_fine_nodes,
                timesteps=timesteps,
                anchor_features=anchor_features.detach(),
                image_tokens=image_tokens,
                anchor_pos=anchor_pos.detach() if anchor_pos is not None else None,
                anchor_rot=anchor_rot.detach() if anchor_rot is not None else None,
            )
            pred_velocity = fine_out["pred_velocity"]

        return {
            # Stage 1: Macro Biological Prior
            "pred_dap": coarse_out["pred_dap"],
            "pred_num_phytomers": coarse_out["pred_num_phytomers"],
            "soft_margin_weights": coarse_out["soft_margin_weights"],
            # Stage 2: 3D Node Point Cloud Scaffold
            "pred_anchor_pos": coarse_out["anchor_pos"],
            "pred_anchor_rot": coarse_out["anchor_rot"],
            "pred_anchor_scale": coarse_out["anchor_scale"],
            "pred_anchor_logits": coarse_out["anchor_logits"],
            "active_k": active_k,
            # Stage 3: Intra-Phytomer Canonical Flow Matching
            "pred_velocity": pred_velocity,
            "pred_fine_exist_logits": (
                fine_out["pred_slot_exist_logits"]
                if self.flow_granularity == "phytomer"
                else fine_out["pred_exist_logits"]
            ),
            "flow_granularity": self.flow_granularity,
        }

    @torch.no_grad()
    def sample_ode(
        self,
        images: torch.Tensor,
        daps: Optional[torch.Tensor] = None,
        num_steps: int = 20,
        guidance_scale: float = 1.0,
        vae: Optional[nn.Module] = None,
        phytomer_vae: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """2nd-Order Heun Predictor-Corrector ODE Sampling.

        Generates full 3D plant organ array from an input condition image.
        Integrates velocity field in 16D regularized latent space.
        Takes <50ms end-to-end on GPU.

        flow_granularity='phytomer': integrates the (B, K, 12+D) per-anchor vector
        [base | rot | latent]; the refined anchor rot is the reference frame for
        the relative packet (Option 1). `phytomer_vae` decodes the latent part.
        """
        B = images.shape[0]
        device = images.device

        # 1. Encode image & predict Stage 1 Macro Prior + Stage 2 3D Node Scaffold
        # Inference capacity: two-pass with predicted phytomer count. If GT DAP is
        # provided (self-consistency evaluation), take the max of both signals so the
        # capacity covers the more generous estimate without falling back to full 512.
        image_tokens = self.image_encoder(images)
        clue_dap = self.probe_pred_dap(image_tokens)
        coarse_pred = self.coarse_stage(image_tokens, capacity_mode="pred_phyto", pred_dap=clue_dap)
        k_pred = int(coarse_pred["active_k"])
        if daps is not None:
            k_dap = compute_matryoshka_slice(dap=daps, max_anchors=self.max_anchors)
            active_k = max(k_pred, int(k_dap))
            # Re-run Stage 2 with the wider slice (macro head result is unaffected; the
            # soft margin prior is resliced deterministically inside forward). The clue
            # is the PREDICTED DAP at inference (no GT leakage); when GT DAP is given
            # the clue is the GT (same as the teacher-forcing path).
            clue = daps
            coarse_out = self.coarse_stage(image_tokens, active_k=active_k, pred_dap=clue)
            coarse_out["pred_num_phytomers"] = coarse_pred["pred_num_phytomers"]
            coarse_out["pred_dap"] = coarse_pred["pred_dap"]
        else:
            coarse_out = coarse_pred
            active_k = k_pred

        anchor_pos = coarse_out["anchor_pos"]      # (B, K, 3)
        anchor_rot = coarse_out["anchor_rot"]      # (B, K, 6)
        anchor_features = coarse_out["anchor_features"]
        soft_margin_weights = coarse_out["soft_margin_weights"]
        anchor_existence = torch.sigmoid(coarse_out["anchor_logits"]).squeeze(-1) * soft_margin_weights  # (B, K)

        # 2. Construct Prior x_0.
        #    organ mode: standard Gaussian (B, K*8, 16) — exact legacy behavior.
        #    phytomer mode: standard Gaussian (B, K, D) — pure 64D VAE latent flow (Hybrid Decoupled).
        M = self.slots_per_anchor
        if self.flow_granularity == "phytomer":
            D = self.phytomer_latent_dim
            N_fine = active_k  # K anchors
            x = torch.randn(B, active_k, D, device=device)
        else:
            N_fine = active_k * M
            x = torch.randn(B, N_fine, self.node_dim, device=device)

        # 3. 2nd-Order Heun ODE Integration from t=0 to t=1 (Stage 3 Flow Matching)
        dt = 1.0 / num_steps
        anchor_scale = coarse_out.get("anchor_scale")
        forward_kwargs = dict(
            anchor_features=anchor_features,
            image_tokens=image_tokens,
            anchor_pos=anchor_pos,
            anchor_rot=anchor_rot,
            anchor_scale=anchor_scale,
        )
        for step in range(num_steps):
            t_curr = step * dt
            t_next = (step + 1) * dt
            t_tensor = torch.full((B,), t_curr, device=device)

            if self.flow_granularity == "phytomer":
                out_1 = self.fine_stage(noisy_flow=x, timesteps=t_tensor, **forward_kwargs)
            else:
                out_1 = self.fine_stage(noisy_fine_nodes=x, timesteps=t_tensor, **forward_kwargs)
            v_1 = out_1["pred_velocity"]

            # Predictor step (Euler)
            x_pred = x + v_1 * dt

            # Corrector step (Heun)
            if step < num_steps - 1:
                t_next_tensor = torch.full((B,), t_next, device=device)
                if self.flow_granularity == "phytomer":
                    out_2 = self.fine_stage(noisy_flow=x_pred, timesteps=t_next_tensor, **forward_kwargs)
                else:
                    out_2 = self.fine_stage(noisy_fine_nodes=x_pred, timesteps=t_next_tensor, **forward_kwargs)
                v_2 = out_2["pred_velocity"]
                v_eff = 0.5 * (v_1 + v_2)
            else:
                v_eff = v_1

            x = x + v_eff * dt

        # Final evaluation of existence at t=1
        if self.flow_granularity == "phytomer":
            final_out = self.fine_stage(
                noisy_flow=x, timesteps=torch.ones((B,), device=device), **forward_kwargs)
            pred_slot_exist_logits = final_out["pred_slot_exist_logits"]        # (B, K, M)
            pred_slot_exist = torch.sigmoid(pred_slot_exist_logits)             # (B, K, M)
            combined_prob = pred_slot_exist * anchor_existence.unsqueeze(-1)    # (B, K, M)
            slot_active = (combined_prob > 0.35).float().reshape(B, active_k * M)
            pred_latent = x          # (B, K, D) latent flow vector (caller reference)
            pred_fine_exist_logits = pred_slot_exist_logits
            N_fine = active_k * M    # flat slot surface for legacy callers
        else:
            final_out = self.fine_stage(
                noisy_fine_nodes=x, timesteps=torch.ones((B,), device=device), **forward_kwargs)
            pred_fine_exist_logits = final_out["pred_exist_logits"]  # (B, N_fine, 1)
            fine_slot_prob = torch.sigmoid(pred_fine_exist_logits).squeeze(-1)  # (B, N_fine)

            # Compute combined existence confidence (anchor_existence * fine_slot_existence)
            expanded_anchor_exist = anchor_existence.unsqueeze(2).expand(-1, -1, M).reshape(B, N_fine)
            combined_prob = fine_slot_prob * expanded_anchor_exist  # (B, N_fine)

            # Dynamic Top-K selection based on fitted botanical sigmoid growth curve with margin
            dap_use = daps if daps is not None else coarse_out.get("pred_dap", None)
            if dap_use is not None:
                budgets = estimate_organ_budget(dap_use.view(-1), margin=1.35, max_slots=N_fine).to(device)
            else:
                budgets = torch.full((B,), min(400, N_fine), dtype=torch.long, device=device)

            # Fully vectorized dynamic Top-K and threshold selection across batch B
            k_max = min(int(budgets.max().item()), N_fine)
            top_k_indices = torch.topk(combined_prob, k_max, dim=-1).indices  # (B, k_max)
            col_idx = torch.arange(k_max, device=device).unsqueeze(0).expand(B, -1)
            budget_mask = col_idx < budgets.clamp(max=N_fine).unsqueeze(1)
            gathered_prob = torch.gather(combined_prob, 1, top_k_indices)
            valid = budget_mask & (gathered_prob > 0.15)
            slot_active = torch.zeros((B, N_fine), dtype=torch.float32, device=device)
            slot_active.scatter_(1, top_k_indices, valid.float())
            slot_active = torch.clamp(slot_active + (combined_prob > 0.35).float(), 0.0, 1.0)
            pred_latent = x

        res = {
            "pred_latent": pred_latent,
            "pred_fine_exist_logits": pred_fine_exist_logits,
            "slot_active": slot_active,
            "anchor_pos": anchor_pos,
            "anchor_rot": anchor_rot,
            "anchor_scale": coarse_out.get("anchor_scale"),
            "anchor_existence": anchor_existence,
            "pred_num_phytomers": coarse_out["pred_num_phytomers"],
            "pred_dap": coarse_out["pred_dap"],
            "soft_margin_weights": soft_margin_weights,
            "active_fine_count": N_fine,
            "active_k": active_k,
            "flow_granularity": self.flow_granularity,
        }

        if self.flow_granularity == "phytomer":
            # Decoupled flow vector is pure 64D phytomer VAE latent
            latent = x
            res["refined_anchor_pos"] = anchor_pos
            res["refined_anchor_rot"] = anchor_rot
            res["refined_anchor_scale"] = anchor_scale
            res["phytomer_latent"] = latent
            res["pred_slot_exist_logits"] = final_out["pred_slot_exist_logits"]
            if phytomer_vae is not None:
                out = phytomer_vae.decode(latent.reshape(-1, self.phytomer_latent_dim))  # (B*K, M, 26)
                probs = F.softmax(out["cls_logits"], dim=-1)  # (B*K, M, 13)
                pred_cls = out["cls_logits"].argmax(-1)
                packet_hat = out["recon_packets"].reshape(B, active_k, M, 26)
                # v3: the decoded scales are NORMALIZED (VAE trained on normalized
                # packets); restore absolute scale with anchor scale
                packet_hat = denormalize_packet_scales(
                    packet_hat.reshape(-1, M, 26),
                    anchor_scale.reshape(-1, 3) if anchor_scale is not None else torch.ones((B*active_k, 3), device=device),
                ).reshape(B, active_k, M, 26)
                # Structurally assemble slot bases from the petiole geometry
                # (deterministic), then re-anchor with the anchor pose.
                packet_hat = assemble_packets(
                    packet_hat.reshape(-1, M, 26),
                    anchor_rot.reshape(-1, 6),
                ).reshape(B, active_k, M, 26)
                abs_packets = apply_ref_for_flow(packet_hat, anchor_pos, anchor_rot)
                flat_abs = abs_packets.reshape(B, active_k * M, 26)
                keep = pred_cls.reshape(B, active_k * M) > 0
                # Vectorized 1-shot decoding across all batch items (B, N, 26) -> (B, N, 14)
                res["part_14d"] = decode_fm(flat_abs)
                res["organ_probs"] = probs
                res["pred_cls"] = pred_cls.reshape(B, active_k * M)
                res["pred_cls_logits"] = out["cls_logits"].reshape(B, active_k * M, -1)
                res["pred_geometry"] = flat_abs[..., FM_BASE_START:]
            else:
                res["pred_geometry"] = x
                res["pred_cls"] = torch.zeros((B, active_k * M), dtype=torch.long, device=device)
                res["pred_cls_logits"] = torch.zeros((B, active_k * M, self.num_classes), device=device)
        else:
            # If VAE is provided, decode 16D latent into physical 14D part tensor and geometry
            if vae is not None:
                part_14d, probs = vae.decode_to_part_tensor(pred_latent)
                decoded = vae.decode(pred_latent)
                res["part_14d"] = part_14d
                res["pred_geometry"] = decoded["recon_26d"][:, :, FM_BASE_START:]
                res["organ_probs"] = probs
                res["pred_cls"] = decoded["cls_logits"].argmax(dim=-1)
                res["pred_cls_logits"] = decoded["cls_logits"]
            else:
                res["pred_geometry"] = pred_latent
                res["pred_cls"] = torch.zeros((B, N_fine), dtype=torch.long, device=device)
                res["pred_cls_logits"] = torch.zeros((B, N_fine, self.num_classes), device=device)

        return res
