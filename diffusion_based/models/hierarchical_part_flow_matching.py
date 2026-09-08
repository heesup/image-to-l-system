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
)


def compute_matryoshka_slice(
    dap: Optional[torch.Tensor] = None,
    num_phytomers: Optional[torch.Tensor] = None,
    max_anchors: int = 512,
    margin: float = 4.0,
) -> int:
    """Computes upper-bound active anchor count based on biological growth or predicted phytomers.

    If num_phytomers is provided:
        K_upper = min(max_anchors, ceil(num_phytomers + margin))
    Else if DAP is provided:
        K_upper(t) = min(max_anchors, ceil(8 * 2^(t / 8.5)))
    """
    if num_phytomers is not None:
        val = float(num_phytomers.max().item())
        k = math.ceil(val + margin)
    elif dap is not None:
        dap_val = float(dap.max().item())
        # Doubling every 8.5 days starting from 8 at DAP 0
        k = math.ceil(8.0 * (2.0 ** (max(0.0, dap_val) / 8.5)))
    else:
        return max_anchors

    # Snap to nearest power-of-2 tier
    tiers = [8, 16, 32, 64, 128, 256, max_anchors]
    for tier in tiers:
        if k <= tier:
            return min(tier, max_anchors)
    return max_anchors


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
        init_logits = (pred_num_phytomers + m - k_indices) / t  # (B, K)
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
    ):
        super().__init__()
        self.max_anchors = max_anchors
        self.embed_dim = embed_dim

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
        self.macro_head = MacroBiologicalHead(embed_dim=embed_dim)

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

        # 3. Anchor existence probability logit (delta relative to soft margin prior)
        self.exist_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(
        self,
        image_tokens: torch.Tensor,
        active_k: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            image_tokens: (B, T, embed_dim) 3D-aware visual tokens from DINOv2RayEncoder.
            active_k: Optional sliced anchor count (e.g. 16, 64, 128, 512).
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
        K = active_k if active_k is not None else self.max_anchors

        # Query = Content embedding + 3D Positional Encoding of reference points
        q_content = self.anchor_queries[:K].unsqueeze(0).expand(B, -1, -1)
        q_pos = self.ref_pos_mlp(self.ref_points[:K]).unsqueeze(0).expand(B, -1, -1)
        q = q_content + q_pos

        # Cross-attend with 3D-aware image tokens
        anchor_features = self.decoder(q, image_tokens)

        # Stage 1: Macro Biological Head from global CLS token (token 0)
        cls_token = image_tokens[:, 0]
        macro_out = self.macro_head(cls_token, max_k=K)

        # Stage 2 Heads: predict coordinate offset from 3D reference points
        delta_pos = self.pos_head(anchor_features)
        anchor_pos = self.ref_points[:K].unsqueeze(0) + delta_pos
        anchor_rot = self.rot_head(anchor_features)

        # Combine learned delta logits with differentiable soft margin logit prior
        delta_logits = self.exist_head(anchor_features)
        anchor_logits = delta_logits + macro_out["init_logits"]

        return {
            "anchor_pos": anchor_pos,
            "anchor_rot": anchor_rot,
            "anchor_logits": anchor_logits,
            "anchor_features": anchor_features,
            "pred_dap": macro_out["pred_dap"],
            "pred_num_phytomers": macro_out["pred_num_phytomers"],
            "soft_margin_weights": macro_out["soft_margin_weights"],
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


class FineBotanicalFlowMatchingDecoder(nn.Module):
    """Stage 2: Flow Matching Decoder predicting microscopic organ velocity field.

    Dispatches M fine slots per active anchor (M=8: 1 stem + 1 petiole + 3 leaflets + 1 peduncle + 2 reproductive).
    Performs joint intra-block kinematic attention within each phytomer, followed by global cross-attention.
    """

    def __init__(
        self,
        slots_per_anchor: int = 8,
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

        # 2. Stage 2 -> 3: 3D Node Scaffold Positional & Rotational Condition Embeddings
        if anchor_pos is not None:
            node_pos_emb = self.node_pos_mlp(anchor_pos).unsqueeze(2).expand(-1, -1, M, -1).reshape(B, N_fine, self.embed_dim)
        else:
            node_pos_emb = 0.0

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

        # Composite query representation (Conditioned on 3D Scaffold + Anchor Latents + Roles + Time)
        queries = geom_embs + expanded_anchors + node_pos_emb + node_rot_emb + role_embs + t_emb

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
        pred_exist_logits = self.exist_head(x)

        return {
            "pred_velocity": pred_velocity,
            "pred_exist_logits": pred_exist_logits,
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
        slots_per_anchor: int = 8,
        node_dim: int = 16,
        num_classes: int = NUM_ORGAN_TYPES,
        image_size: int = 128,
        patch_size: int = 8,
        embed_dim: int = 384,
        vit_layers: int = 8,
        vit_heads: int = 8,
        coarse_layers: int = 4,
        fine_layers: int = 6,
    ):
        super().__init__()
        self.max_anchors = max_anchors
        self.slots_per_anchor = slots_per_anchor
        self.max_fine_slots = max_anchors * slots_per_anchor  # 512 * 8 = 4,096
        self.node_dim = node_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 1. Pretrained DINOv2 Backbone with 3D Camera Ray Positional Embedding (PETR style)
        self.image_encoder = DINOv2RayEncoder(
            pretrained=True,
            freeze_backbone=False,
            embed_dim=embed_dim,
        )

        # 2. Stage 1 & 2: Coarse Skeletal Transformer with MacroBiologicalHead
        self.coarse_stage = CoarseSkeletalTransformer(
            max_anchors=max_anchors,
            embed_dim=embed_dim,
            num_heads=vit_heads,
            num_layers=coarse_layers,
        )

        # 3. Stage 3: Fine Botanical Flow Matching Decoder conditioned on 3D Scaffold
        self.fine_stage = FineBotanicalFlowMatchingDecoder(
            slots_per_anchor=slots_per_anchor,
            node_dim=node_dim,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_heads=vit_heads,
            num_layers=fine_layers,
        )

    def forward(
        self,
        noisy_fine_nodes: torch.Tensor,
        timesteps: torch.Tensor,
        images: torch.Tensor,
        daps: Optional[torch.Tensor] = None,
        teacher_anchor_pos: Optional[torch.Tensor] = None,
        num_phytomers: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Unified joint forward pass across 3 cascaded stages.

        Args:
            noisy_fine_nodes: (B, N_fine, node_dim) interpolated geometry x_t.
            timesteps: (B,) flow time t in [0, 1].
            images: (B, 4, H, W) or (B, 16, H, W) drone RGB+CHM imagery.
            daps: Optional (B,) plant age tensor for Matryoshka slicing.
            teacher_anchor_pos: Optional (B, K, 3) ground-truth anchor positions for teacher forcing.
            num_phytomers: Optional (B,) ground-truth phytomer count for Matryoshka slicing.
        """
        # 1. Vision perception: 3D-aware multi-scale tokens
        image_tokens = self.image_encoder(images)

        # 2. Dynamic Matryoshka Slicing
        # Compute active K based on DAP or phytomer budget
        active_k = compute_matryoshka_slice(dap=daps, num_phytomers=num_phytomers, max_anchors=self.max_anchors)

        # 3. Stage 1 & 2: Macro Prior + Coarse 3D Node Scaffold Prediction
        coarse_out = self.coarse_stage(image_tokens, active_k=active_k)
        anchor_features = coarse_out["anchor_features"]
        anchor_pos = coarse_out["anchor_pos"]
        anchor_rot = coarse_out["anchor_rot"]

        # If noisy nodes exceed active fine slots, slice accordingly
        active_fine = active_k * self.slots_per_anchor
        if noisy_fine_nodes.shape[1] > active_fine:
            noisy_fine_nodes = noisy_fine_nodes[:, :active_fine]

        # 4. Stage 3: Fine Botanical Flow Matching conditioned on 3D Scaffold
        fine_out = self.fine_stage(
            noisy_fine_nodes=noisy_fine_nodes,
            timesteps=timesteps,
            anchor_features=anchor_features,
            image_tokens=image_tokens,
            anchor_pos=anchor_pos,
            anchor_rot=anchor_rot,
        )

        return {
            # Stage 1: Macro Biological Prior
            "pred_dap": coarse_out["pred_dap"],
            "pred_num_phytomers": coarse_out["pred_num_phytomers"],
            "soft_margin_weights": coarse_out["soft_margin_weights"],
            # Stage 2: 3D Node Point Cloud Scaffold
            "pred_anchor_pos": coarse_out["anchor_pos"],
            "pred_anchor_rot": coarse_out["anchor_rot"],
            "pred_anchor_logits": coarse_out["anchor_logits"],
            "active_k": active_k,
            # Stage 3: Intra-Phytomer Canonical Flow Matching
            "pred_velocity": fine_out["pred_velocity"],
            "pred_fine_exist_logits": fine_out["pred_exist_logits"],
        }

    @torch.no_grad()
    def sample_ode(
        self,
        images: torch.Tensor,
        daps: Optional[torch.Tensor] = None,
        num_steps: int = 20,
        guidance_scale: float = 1.0,
        vae: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """2nd-Order Heun Predictor-Corrector ODE Sampling.

        Generates full 3D plant organ array from an input condition image.
        Integrates velocity field in 16D regularized latent space.
        Takes <50ms end-to-end on GPU.
        """
        B = images.shape[0]
        device = images.device

        # 1. Encode image & predict Stage 1 Macro Prior + Stage 2 3D Node Scaffold
        image_tokens = self.image_encoder(images)
        active_k = compute_matryoshka_slice(daps, max_anchors=self.max_anchors)
        coarse_out = self.coarse_stage(image_tokens, active_k=active_k)

        anchor_pos = coarse_out["anchor_pos"]      # (B, K, 3)
        anchor_rot = coarse_out["anchor_rot"]      # (B, K, 6)
        anchor_features = coarse_out["anchor_features"]
        soft_margin_weights = coarse_out["soft_margin_weights"]
        anchor_existence = torch.sigmoid(coarse_out["anchor_logits"]).squeeze(-1) * soft_margin_weights  # (B, K)

        # 2. Construct Prior x_0 strictly matching standard Gaussian N(0, I)
        M = self.slots_per_anchor
        N_fine = active_k * M
        x = torch.randn(B, N_fine, self.node_dim, device=device)

        # 3. 2nd-Order Heun ODE Integration from t=0 to t=1 (Stage 3 Flow Matching)
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_curr = step * dt
            t_next = (step + 1) * dt
            t_tensor = torch.full((B,), t_curr, device=device)

            out_1 = self.fine_stage(
                noisy_fine_nodes=x,
                timesteps=t_tensor,
                anchor_features=anchor_features,
                image_tokens=image_tokens,
                anchor_pos=anchor_pos,
                anchor_rot=anchor_rot,
            )
            v_1 = out_1["pred_velocity"]

            # Predictor step (Euler)
            x_pred = x + v_1 * dt

            # Corrector step (Heun)
            if step < num_steps - 1:
                t_next_tensor = torch.full((B,), t_next, device=device)
                out_2 = self.fine_stage(
                    noisy_fine_nodes=x_pred,
                    timesteps=t_next_tensor,
                    anchor_features=anchor_features,
                    image_tokens=image_tokens,
                    anchor_pos=anchor_pos,
                    anchor_rot=anchor_rot,
                )
                v_2 = out_2["pred_velocity"]
                v_eff = 0.5 * (v_1 + v_2)
            else:
                v_eff = v_1

            x = x + v_eff * dt

        # Final evaluation of fine slot existence at t=1
        final_out = self.fine_stage(
            noisy_fine_nodes=x,
            timesteps=torch.ones((B,), device=device),
            anchor_features=anchor_features,
            image_tokens=image_tokens,
            anchor_pos=anchor_pos,
            anchor_rot=anchor_rot,
        )

        pred_latent = x  # (B, N_fine, node_dim)
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

        slot_active = torch.zeros((B, N_fine), dtype=torch.float32, device=device)
        for b in range(B):
            k_b = min(int(budgets[b].item()), N_fine)
            top_k_indices = torch.topk(combined_prob[b], k_b).indices
            # Only keep top-k slots that have meaningful probability (> 0.15) to prevent dormant ghost organs
            valid_topk = top_k_indices[combined_prob[b, top_k_indices] > 0.15]
            slot_active[b, valid_topk] = 1.0
            # Also keep high-confidence slots (e.g. > 0.35)
            slot_active[b, combined_prob[b] > 0.35] = 1.0

        res = {
            "pred_latent": pred_latent,
            "pred_fine_exist_logits": pred_fine_exist_logits,
            "slot_active": slot_active,
            "anchor_pos": anchor_pos,
            "anchor_rot": anchor_rot,
            "anchor_existence": anchor_existence,
            "pred_num_phytomers": coarse_out["pred_num_phytomers"],
            "pred_dap": coarse_out["pred_dap"],
            "soft_margin_weights": soft_margin_weights,
            "active_fine_count": N_fine,
        }

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
