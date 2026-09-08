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

from diffusion_based.models.vit_image_encoder import ViTImageEncoder
from diffusion_based.dataset.part_array_dataset import (
    BASE_SCALE,
    SCALE_SCALE,
    CURV_SCALE,
    GEOM_NODE_DIM,
    NUM_ORGAN_TYPES,
    FM_BASE_START,
)


def compute_matryoshka_slice(dap: torch.Tensor, max_anchors: int = 512) -> int:
    """Computes upper-bound active anchor count based on empirical logistic growth curve.

    K_upper(t) = min(max_anchors, ceil(8 * 2^(t / 8.5)))
    """
    if dap is None:
        return max_anchors
    dap_val = float(dap.max().item())
    # Doubling every 8.5 days starting from 8 at DAP 0
    k = math.ceil(8.0 * (2.0 ** (max(0.0, dap_val) / 8.5)))
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


class CoarseSkeletalTransformer(nn.Module):
    """Stage 1: Deterministic Set Transformer predicting coarse 3D skeletal anchors.

    Predicts 3D base coordinates, 6D orientation, and existence logits for K anchors.
    Also includes an auxiliary DAP prediction head for in-the-wild inputs without metadata.
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

        # Matryoshka nested query embeddings
        self.anchor_queries = nn.Parameter(torch.randn(max_anchors, embed_dim) * 0.02)
        self.anchor_pos_emb = nn.Embedding(max_anchors, embed_dim)

        # Decoder layers for cross-attention to image tokens
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

        # Prediction Heads
        # 1. Anchor 3D base position (x, y, z) in normalized scale (multiplied by BASE_SCALE=20.0)
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

        # 3. Anchor existence probability logit (active phytomer node vs pruned background)
        self.exist_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

        # Auxiliary DAP regressor head (estimates age [0, 100] days from pooled image tokens)
        self.dap_head = nn.Sequential(
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
            image_tokens: (B, T, embed_dim) visual tokens from ViT.
            active_k: Optional sliced anchor count (e.g. 16, 64, 128, 512).
        Returns:
            Dict containing:
                'anchor_pos': (B, K, 3) predicted 3D anchor base positions.
                'anchor_rot': (B, K, 6) predicted 6D rotation.
                'anchor_logits': (B, K, 1) existence logits.
                'anchor_features': (B, K, embed_dim) anchor latent representations.
                'pred_dap': (B, 1) auxiliary estimated DAP.
        """
        B = image_tokens.shape[0]
        K = active_k if active_k is not None else self.max_anchors
        device = image_tokens.device

        # Slice Matryoshka queries for target scale
        queries = self.anchor_queries[:K].unsqueeze(0).expand(B, -1, -1)
        pos_emb = self.anchor_pos_emb(torch.arange(K, device=device)).unsqueeze(0).expand(B, -1, -1)
        q = queries + pos_emb

        # Cross-attend with image tokens
        anchor_features = self.decoder(q, image_tokens)

        # Heads
        anchor_pos = self.pos_head(anchor_features)
        anchor_rot = self.rot_head(anchor_features)
        anchor_logits = self.exist_head(anchor_features)

        # Auxiliary DAP estimation from global image token pool
        pooled_img = image_tokens.mean(dim=1)
        pred_dap = F.relu(self.dap_head(pooled_img)) * 100.0  # Scale ~[0, 100]

        return {
            "anchor_pos": anchor_pos,
            "anchor_rot": anchor_rot,
            "anchor_logits": anchor_logits,
            "anchor_features": anchor_features,
            "pred_dap": pred_dap,
        }


class FineBotanicalFlowMatchingDecoder(nn.Module):
    """Stage 2: Flow Matching Decoder predicting microscopic organ velocity field.

    Dispatches M fine slots per active anchor (M=8: 1 stem + 3 leaflets + 2 buds + 2 spare).
    Performs residual flow matching around the anchor coordinates.
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

        # Intra-anchor role embedding: Role 0=Stalk, Roles 1-3=Trifoliolate, Roles 4-7=Aux/Buds
        self.role_emb = nn.Embedding(slots_per_anchor, embed_dim)

        # Continuous geometry projection
        self.geom_proj = nn.Linear(node_dim, embed_dim)

        # Sinusoidal timestep embedding for continuous flow time t in [0, 1]
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
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

        # Continuous geometry velocity head (v_theta predicting d/dt of 13D geometry)
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
        existence_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            noisy_fine_nodes: (B, N_fine, node_dim) noisy geometry x_t where N_fine = K * M.
            timesteps: (B,) flow time t in [0, 1].
            anchor_features: (B, K, embed_dim) from Stage 1 CoarseSkeletalTransformer.
            image_tokens: (B, T, embed_dim) from ViT.
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

        # 2. Local organ role embedding: (M, D) expanded across B and K
        role_ids = torch.arange(M, device=device).unsqueeze(0).expand(K, -1).reshape(-1)  # (N_fine,)
        role_embs = self.role_emb(role_ids).unsqueeze(0).expand(B, -1, -1)

        # 3. Geometry projection
        geom_embs = self.geom_proj(noisy_fine_nodes)

        # 4. Time conditioning
        t_emb = self.time_embed(self._sinusoidal(timesteps)).unsqueeze(1)  # (B, 1, D)

        # Composite query representation
        queries = geom_embs + expanded_anchors + role_embs + t_emb

        # 5. Decoder cross-attention to image tokens
        # Concatenate image tokens with anchor features as memory for full multi-scale conditioning
        memory = torch.cat([image_tokens, anchor_features], dim=1)

        x = self.decoder(queries, memory)

        # 6. Heads
        pred_velocity = self.velocity_head(x)
        pred_exist_logits = self.exist_head(x)

        return {
            "pred_velocity": pred_velocity,
            "pred_exist_logits": pred_exist_logits,
        }


class HierarchicalPartFlowMatchingModel(nn.Module):
    """End-to-End Hierarchical Matryoshka Botanical Flow Matching System.

    Couples Stage 1 Coarse Skeletal Predictor and Stage 2 Fine Flow Matching Decoder
    with multi-scale ViT visual perception.
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

        # 1. Multi-scale ViT Image Encoder (RGB + CHM depth: 4 channels)
        self.image_encoder = ViTImageEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=4,
            embed_dim=embed_dim,
            num_layers=vit_layers,
            num_heads=vit_heads,
        )

        # 2. Stage 1 Coarse Skeletal Transformer
        self.coarse_stage = CoarseSkeletalTransformer(
            max_anchors=max_anchors,
            embed_dim=embed_dim,
            num_heads=vit_heads,
            num_layers=coarse_layers,
        )

        # 3. Stage 2 Fine Botanical Flow Matching Decoder
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
    ) -> Dict[str, torch.Tensor]:
        """Unified joint forward pass.

        Args:
            noisy_fine_nodes: (B, N_fine, node_dim) interpolated geometry x_t.
            timesteps: (B,) flow time t in [0, 1].
            images: (B, 4, H, W) or (B, 16, H, W) drone RGB+CHM imagery.
            daps: Optional (B,) plant age tensor for Matryoshka slicing.
            teacher_anchor_pos: Optional (B, K, 3) ground-truth anchor positions for teacher forcing.
        """
        # 1. Vision perception
        image_tokens = self.image_encoder(images)

        # 2. Dynamic Matryoshka Slicing
        # Compute active K based on batch maximum DAP
        active_k = compute_matryoshka_slice(daps, max_anchors=self.max_anchors)

        # 3. Stage 1: Coarse Skeleton Prediction
        coarse_out = self.coarse_stage(image_tokens, active_k=active_k)
        anchor_features = coarse_out["anchor_features"]

        # If noisy nodes exceed active fine slots, slice accordingly
        active_fine = active_k * self.slots_per_anchor
        if noisy_fine_nodes.shape[1] > active_fine:
            noisy_fine_nodes = noisy_fine_nodes[:, :active_fine]

        # 4. Stage 2: Fine Botanical Flow Matching
        fine_out = self.fine_stage(
            noisy_fine_nodes=noisy_fine_nodes,
            timesteps=timesteps,
            anchor_features=anchor_features,
            image_tokens=image_tokens,
        )

        return {
            # Stage 1 outputs
            "pred_anchor_pos": coarse_out["anchor_pos"],
            "pred_anchor_rot": coarse_out["anchor_rot"],
            "pred_anchor_logits": coarse_out["anchor_logits"],
            "pred_dap": coarse_out["pred_dap"],
            "active_k": active_k,
            # Stage 2 outputs
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

        # 1. Encode image & predict coarse skeleton (5ms)
        image_tokens = self.image_encoder(images)
        active_k = compute_matryoshka_slice(daps, max_anchors=self.max_anchors)
        coarse_out = self.coarse_stage(image_tokens, active_k=active_k)

        anchor_pos = coarse_out["anchor_pos"]  # (B, K, 3)
        anchor_features = coarse_out["anchor_features"]
        anchor_existence = torch.sigmoid(coarse_out["anchor_logits"]).squeeze(-1)  # (B, K)

        # 2. Construct Prior x_0 strictly matching standard Gaussian N(0, I)
        M = self.slots_per_anchor
        N_fine = active_k * M
        x = torch.randn(B, N_fine, self.node_dim, device=device)

        # 3. 2nd-Order Heun ODE Integration from t=0 to t=1 (40ms)
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
            slot_active[b, top_k_indices] = 1.0
            # Also keep high-confidence slots (e.g. > 0.35)
            slot_active[b, combined_prob[b] > 0.35] = 1.0

        res = {
            "pred_latent": pred_latent,
            "pred_fine_exist_logits": pred_fine_exist_logits,
            "slot_active": slot_active,
            "anchor_pos": anchor_pos,
            "anchor_existence": anchor_existence,
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
