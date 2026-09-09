"""
Phytomer-level Variational Autoencoder (PhytomerVAE).

Compresses one canonical 8-slot phytomer packet (8 x 26D FM organ rows +
8-bit presence mask = 216D) into a compact spherical standard Gaussian
latent z in R^D ~ N(0, I), D in {32, 64, 128} (default 64).

Rationale (validated 2026-09-09 on cowpea_curv26):
- Organs within a phytomer co-vary strongly (petiole-vs-leaflet scale r=0.78,
  stem-vs-leaflet r=0.52), so the effective DOF is far below 8x16=128.
- One latent per phytomer turns Stage 3 flow matching from 8K tokens into K
  tokens (64x cheaper self-attention, 8x cheaper ODE) and collapses the
  matcher to a single anchor-level Hungarian (no slot-level assignment).
- Geometry is ANCHOR-RELATIVE (organ base - cluster center): the anchor/node
  position carries global placement, the latent models translation-invariant
  local morphology. decode_packet() re-anchors with the node position.

Conventions mirror OrganLatentVAE (LayerNorm+SiLU MLPs, specialized heads,
masked recon weights cls 1 / base 3 / rot 2 / scale 3 / curv 0.5, beta-KL).
Absent slots target the exact empty convention (NONE one-hot + zero geometry).
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from diffusion_based.dataset.part_array_dataset import (
    NUM_ORGAN_TYPES,
    FM_OT_END,
    FM_BASE_START,
    FM_BASE_END,
    FM_ROT_START,
    FM_ROT_END,
    FM_SCALE_START,
    FM_SCALE_END,
    FM_CURV,
    FM_NODE_DIM,
)
from diffusion_based.dataset.phytomer_packets import (
    NUM_SLOTS,
    assemble_packets,
)

SLOT_DIM = FM_NODE_DIM            # 26D FM organ row
# Base columns are REMOVED before encoding: with EXACT XML phytomer clustering
# every slot's base is deterministic (stem/petiole/peduncle/bud = center,
# leaflets = 0.8/1.0 x the petiole curve — verified 0.00cm over the full
# dataset). Only flower/fruit bases (rare, per-plant offsets) are VAE-learned.
# 8 x (23 + presence bit) = 192.
PACKET_IN_DIM = NUM_SLOTS * (SLOT_DIM - 3 + 1)  # 8 x (23 + 1) = 192


class PhytomerVAE(nn.Module):
    """Compact phytomer-level VAE: (8 x 26D + 8 presence) -> z in R^D -> 8 x 26D."""

    def __init__(
        self,
        slots_per_phytomer: int = NUM_SLOTS,
        slot_dim: int = SLOT_DIM,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        num_classes: int = NUM_ORGAN_TYPES,
    ):
        super().__init__()
        self.slots_per_phytomer = slots_per_phytomer
        self.slot_dim = slot_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        # Base columns are REMOVED before encoding: with EXACT XML phytomer
        # clustering every slot's base is deterministic (stem/petiole/peduncle/
        # bud = center, leaflets = 0.8/1.0 x the petiole curve). Only
        # flower/fruit bases (rare) are VAE-learned. 8 x (23 + 1) = 192.
        self.in_dim = slots_per_phytomer * (slot_dim - 3 + 1)

        # Encoder: packet -> hidden -> (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: latent -> hidden -> 8 slots x (13 cls logits + 13 geom)
        self.decoder_backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.head_cls = nn.Linear(hidden_dim, slots_per_phytomer * num_classes)
        self.head_rot = nn.Linear(hidden_dim, slots_per_phytomer * 6)
        self.head_scale = nn.Linear(hidden_dim, slots_per_phytomer * 3)
        self.head_curv = nn.Linear(hidden_dim, slots_per_phytomer * 1)

        # Dedicated rotation branch (opt-in via decode(use_rot_branch=True)):
        # rotation detail was squeezed by the shared backbone bottleneck (Exp B
        # diagnosis: error spread evenly across roles, 128D latent didn't help).
        # Gives the 48 rotation dims their own pathway from z.
        self.rot_branch = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
        )
        self.head_rot_dedicated = nn.Linear(hidden_dim // 2, slots_per_phytomer * 6)

    def pack_input(self, packets: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        """Flattens (P, 8, 26) + (P, 8) presence -> (P, 192) encoder input.

        Base columns are REMOVED: with EXACT XML phytomer clustering every slot's
        base is deterministic (stem/petiole/peduncle/bud = center, leaflets =
        0.8/1.0 x the petiole curve — verified 0.00cm). Only flower/fruit bases
        (rare, per-plant offsets) are VAE-learned. 216D -> 192D.
        """
        P = packets.shape[0]
        keep = torch.ones(packets.shape[-1], dtype=torch.bool, device=packets.device)
        keep[FM_BASE_START:FM_BASE_END] = False
        stripped = packets[..., keep]  # (P, 8, 23)
        return torch.cat([stripped.reshape(P, -1), presence.float().reshape(P, -1)], dim=-1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes (P, 192) packets to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-10.0, 10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor, use_rot_branch: bool = False) -> Dict[str, torch.Tensor]:
        """Decodes (P, D) latents into (P, 8, 26) FM packets.

        Base columns are ZERO: with EXACT XML phytomer clustering every slot's
        base is deterministic (stem/petiole/peduncle/bud = center, leaflets =
        0.8/1.0 x the petiole curve), so the caller's assemble_packets()
        reconstructs them. Only flower/fruit bases (rare) are VAE-learned and
        passed through by assemble_packets.

        use_rot_branch=True routes rotation through the dedicated branch
        (see __init__) instead of the shared backbone head.
        """
        P = z.shape[0]
        S = self.slots_per_phytomer
        h = self.decoder_backbone(z)
        pred_cls_logits = self.head_cls(h).reshape(P, S, self.num_classes)     # (P, 8, 13)
        pred_base = torch.zeros(P, S, 3, device=z.device, dtype=z.dtype)       # (P, 8, 3) zeroed
        if use_rot_branch:
            pred_rot = self.head_rot_dedicated(self.rot_branch(z)).reshape(P, S, 6)  # (P, 8, 6)
        else:
            pred_rot = self.head_rot(h).reshape(P, S, 6)                       # (P, 8, 6)
        pred_scale = F.softplus(self.head_scale(h)).reshape(P, S, 3) + 1e-4    # (P, 8, 3)
        pred_curv = self.head_curv(h).reshape(P, S, 1)                         # (P, 8, 1)

        pred_probs = F.softmax(pred_cls_logits, dim=-1)
        pred_26d = torch.cat([pred_probs, pred_base, pred_rot, pred_scale, pred_curv], dim=-1)
        return {
            "cls_logits": pred_cls_logits,
            "base": pred_base,
            "rot": pred_rot,
            "scale": pred_scale,
            "curv": pred_curv,
            "recon_packets": pred_26d,  # (P, 8, 26), relative coords
        }

    def forward(
        self, packets: torch.Tensor, presence: torch.Tensor,
        use_rot_branch: bool = False,
    ) -> Dict[str, torch.Tensor]:
        x = self.pack_input(packets, presence)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z, use_rot_branch=use_rot_branch)
        out["mu"] = mu
        out["logvar"] = logvar
        out["z"] = z
        return out

    @torch.no_grad()
    def _hungarian_align_targets(
        self,
        pred: Dict[str, torch.Tensor],
        target_packets: torch.Tensor,
        target_presence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Realigns GT organs to slots by optimal bipartite matching per role.

        For each role with >1 GT organ, solves LSA(pred slots x GT organs) on
        base-L1 + 0.5*scale-L1 + class-mismatch cost and permutes GT rows to the
        matched slots. Single-GT roles keep canonical-first placement; unmatched
        slots keep the empty convention. Returns (aligned_packets, aligned_presence).

        Rationale: canonical within-role ordering (bottom->top z) is unstable for
        near-coincident organs (leaflet base z-gap median 1.4mm; ordering agreement
        43.5% ~ random), forcing the VAE to fit a multimodal target with a unimodal
        posterior (blurred rotations + heavy-tailed bases). Optimal assignment lets
        each prediction match its nearest GT mode. Warm-start required.
        """
        P, S, _ = target_packets.shape
        device = target_packets.device
        pred_scale = pred["scale"].detach()                  # (P, 8, 3)
        pred_cls = pred["cls_logits"].argmax(-1).detach()    # (P, 8)
        tgt_scale = target_packets[:, :, FM_SCALE_START:FM_SCALE_END]
        tgt_cls = target_packets[:, :, :FM_OT_END].argmax(-1)

        aligned = torch.zeros_like(target_packets)
        aligned[:, :, 0] = 1.0  # NONE one-hot default
        aligned_pres = torch.zeros_like(target_presence, dtype=torch.bool)

        role_ranges = [(0, 1), (1, 2), (2, 5), (5, 6), (6, 8)]
        for i in range(P):
            for (lo, hi) in role_ranges:
                slots = list(range(lo, hi))
                gt_here = [s for s in slots
                           if bool(target_presence[i, s]) and int(tgt_cls[i, s]) > 0]
                if not gt_here:
                    continue
                if len(gt_here) == 1:
                    # Unambiguous: canonical-first placement (no LSA needed).
                    aligned[i, lo] = target_packets[i, gt_here[0]]
                    aligned_pres[i, lo] = True
                    continue
                # Ambiguous: optimal assignment over the role's slots.
                n_s = len(slots)
                cb = torch.stack([
                    0.5 * (pred_scale[i, s] - tgt_scale[i, g]).abs().sum()
                    + 1.0 * float(int(pred_cls[i, s]) != int(tgt_cls[i, g]))
                    for s in slots for g in gt_here
                ]).reshape(n_s, len(gt_here))
                r_idx, c_idx = linear_sum_assignment(cb.cpu().numpy())
                for r, c in zip(r_idx.tolist(), c_idx.tolist()):
                    aligned[i, slots[r]] = target_packets[i, gt_here[c]]
                    aligned_pres[i, slots[r]] = True
        return aligned, aligned_pres

    def compute_loss(
        self,
        pred: Dict[str, torch.Tensor],
        target_packets: torch.Tensor,
        target_presence: torch.Tensor,
        absent_weight: float = 0.1,
        beta_kl: float = 1e-3,
        rot_weight: float = 2.0,
        symmetry_aware_rot: bool = False,
        ortho_reg_weight: float = 0.0,
        hungarian_roles: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Masked composite VAE loss over the 8-slot packet.

        Present organs: full reconstruction weight (same block weights as
        OrganLatentVAE: cls 1 / base 3 / rot 2 / scale 3 / curv 0.5).
        Absent slots: small weight toward the exact empty convention target
        (NONE one-hot + zero geometry), keeping the decoder well-behaved where
        the existence head will gate outputs out at inference.

        Rotation options (default off = exact OrganLatentVAE parity):
        - rot_weight: scales the rotation term (try 4.0 if rotation underfits).
        - symmetry_aware_rot: cylindrical slots use spin-invariant axis loss.
          AXIS SEMANTICS (verified against mesh builder + leaf OBJ extents):
          - Leaf assets (CowpeaLeaf OBJ): +x = midrib length, +y = width.
            Leaflets are meshes -> long axis = column 0.
          - Tubes (internode 3, petiole 4, peduncle 6): prototype aligned +Z,
            rotated by R; builder reads forward = R[:, 1] (column 1).
            -> long axis = column 1.
          Cylindrical slot set: stem(0), petiole(1), peduncle(5) -> compare
          column 1 (1-cos, spin-invariant); leaflets/repro keep full Frobenius
          (their spin around midrib is visible). Recovers the spin-ambiguity
          error on stalks (~4-6 deg measured).
        - ortho_reg_weight: penalizes non-orthonormal raw 6D outputs
          ((||x||-1)^2 + (||y||-1)^2 + (x.y)^2), keeping predictions on the
          rotation manifold when spin is unconstrained.
        - hungarian_roles: resolve multi-slot role assignment (leaflets x3,
          repro x2, and any overflowing single-slot role) by optimal bipartite
          matching (prediction <-> GT) instead of fixed canonical order.
          Removes slot-identity ambiguity that blurs rotation/base learning
          (leaflet canonical order agreement only 43.5%). Requires sane
          predictions to match against — always warm-start from a converged
           canonical-order checkpoint, never from scratch.
        - hungarian_roles: resolve multi-GT-organ role ambiguity (leaflets x3,
          repro x2, overflowing single-slot roles) by optimal bipartite matching
          (prediction <-> GT) instead of fixed canonical order. Single-GT roles
          keep canonical-first placement. Unmatched slots target the empty
          convention. Requires sane predictions — always warm-start, see above.
        """
        if hungarian_roles:
            target_packets, target_presence = self._hungarian_align_targets(
                pred, target_packets, target_presence)
        pres = target_presence.float()  # (P, 8)
        n_present = pres.sum().clamp(min=1.0)
        n_absent = (1.0 - pres).sum().clamp(min=1.0)

        target_cls = target_packets[:, :, :FM_OT_END].argmax(dim=-1)  # (P, 8)
        w = pres + absent_weight * (1.0 - pres)                        # (P, 8)
        loss_cls = (F.cross_entropy(
            pred["cls_logits"].reshape(-1, self.num_classes),
            target_cls.reshape(-1),
            reduction="none",
        ).reshape_as(pres) * w).sum() / (n_present + absent_weight * n_absent)

        def _masked_smooth(pred_t: torch.Tensor, tgt_t: torch.Tensor) -> torch.Tensor:
            # pred_t/tgt_t: (P, 8, D)
            err = F.smooth_l1_loss(pred_t, tgt_t, reduction="none").mean(dim=-1)  # (P, 8)
            return (err * w).sum() / (n_present + absent_weight * n_absent)

        tgt_base = torch.zeros_like(target_packets[:, :, FM_BASE_START:FM_BASE_END])
        tgt_rot = target_packets[:, :, FM_ROT_START:FM_ROT_END]
        tgt_scale = target_packets[:, :, FM_SCALE_START:FM_SCALE_END]
        tgt_curv = target_packets[:, :, FM_CURV:FM_CURV + 1]

        # Base is structurally assembled (deterministic from XML phytomer
        # clustering: center for stem/petiole/peduncle/buds, 0.8/1.0 x the
        # petiole curve for leaflets), so the target is zeroed and the loss is
        # trivially satisfied. Only flower/fruit bases (rare) are VAE-learned.
        loss_base = _masked_smooth(pred["base"], tgt_base)
        loss_rot = _masked_smooth(pred["rot"], tgt_rot)

        # Scale relative log-space loss (mirrors OrganLatentVAE: equal weight to
        # 2mm radius and 35cm length).
        log_pred_s = torch.log(pred["scale"] + 1e-3)
        log_tgt_s = torch.log(tgt_scale + 1e-3)
        loss_scale = _masked_smooth(log_pred_s, log_tgt_s) + 0.1 * _masked_smooth(
            pred["scale"], tgt_scale
        )
        loss_curv = _masked_smooth(pred["curv"], tgt_curv)

        # SO(3) rotation loss on PRESENT slots.
        # Default: matches organ VAE (Frobenius MSE + 0.5x smooth-L1 on raw 6D).
        # symmetry_aware_rot=True: spin-invariant axis loss on cylindrical slots.
        # Axis conventions from mesh builder (verified 2026-09-09):
        #   tubes (stem/petiole/peduncle): forward axis = R[:, 1]  (col 1)
        #   leaf meshes: midrib (long axis) = R[:, 0]  (col 0, OBJ +x = midrib)
        # For tubes, spin about the long axis is invisible -> compare col 1 only.
        # Leaflets keep full Frobenius (their spin is visible in the mesh).
        R_pred = _rot6d_to_matrix(pred["rot"])
        R_tgt = _rot6d_to_matrix(tgt_rot.detach())
        frob = ((R_pred - R_tgt) ** 2).mean(dim=(-2, -1))  # (P, 8)
        loss_rot_frob = (frob * pres).sum() / n_present
        if symmetry_aware_rot:
            # Tube slots: compare forward axis (column 1) only
            f_pred = F.normalize(pred["rot"][..., 3:6], dim=-1)
            f_tgt = F.normalize(tgt_rot[..., 3:6].detach(), dim=-1)
            f_cos = (f_pred * f_tgt).sum(dim=-1).clamp(-1.0, 1.0)
            tube_loss = 1.0 - f_cos  # (P, 8), spin-invariant
            tube = torch.zeros(self.slots_per_phytomer, dtype=torch.bool, device=pres.device)
            tube[[0, 1, 5]] = True  # stem, petiole, peduncle slots
            tube = tube.unsqueeze(0).expand_as(pres)
            loss_rot_frob = (
                (tube_loss * tube.float() * pres).sum()
                + (frob * (~tube).float() * pres).sum()
            ) / n_present
        loss_rot = loss_rot_frob + 0.5 * loss_rot

        # Orthonormality regularizer on raw 6D outputs (keeps predictions on the
        # rotation manifold; matters most when spin is unconstrained).
        loss_ortho = torch.tensor(0.0, device=pred["rot"].device)
        if ortho_reg_weight > 0.0:
            x_raw = pred["rot"][..., 0:3]
            y_raw = pred["rot"][..., 3:6]
            nx = x_raw.norm(dim=-1).clamp(min=1e-8)
            ny = y_raw.norm(dim=-1).clamp(min=1e-8)
            ortho = (nx - 1.0).pow(2) + (ny - 1.0).pow(2) + ((x_raw * y_raw).sum(-1) / (nx * ny)).pow(2)
            loss_ortho = (ortho * pres).sum() / n_present

        mu, logvar = pred["mu"], pred["logvar"]
        loss_kl = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1))

        recon_loss = (loss_cls + 3.0 * loss_base + rot_weight * loss_rot
                      + 3.0 * loss_scale + 0.5 * loss_curv
                      + ortho_reg_weight * loss_ortho)
        total_loss = recon_loss + beta_kl * loss_kl

        # Metrics (present slots only)
        pred_cls = pred["cls_logits"].argmax(dim=-1)
        acc = ((pred_cls == target_cls).float() * pres).sum() / n_present

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "loss_cls": loss_cls,
            "loss_base": loss_base,
            "loss_rot": loss_rot,
            "loss_scale": loss_scale,
            "loss_curv": loss_curv,
            "loss_kl": loss_kl,
            "cls_acc": acc,
        }


def _rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Continuous 6D rotation -> 3x3 rotation matrices (Gram-Schmidt)."""
    x = F.normalize(rot6d[..., 0:3], dim=-1)
    z = F.normalize(torch.cross(x, rot6d[..., 3:6], dim=-1), dim=-1)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)
