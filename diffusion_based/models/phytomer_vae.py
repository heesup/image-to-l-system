"""
Phytomer-level Variational Autoencoder (PhytomerVAE).

Compresses one canonical 10-slot phytomer packet (10 x 26D FM organ rows +
10-bit presence mask = 240D) into a HYBRID latent z in R^D, D = coarse_dim +
slots_per_phytomer x residual_dim (default 48 + 10x8 = 128):
- z[:coarse_dim]  — one shared "coarse" channel for class/base/scale/curv
  (the affine morphology that co-varies strongly across a phytomer's organs).
- z[coarse_dim:]  — slots_per_phytomer independent "residual" channels
  (residual_dim each), one per organ slot, dedicated to that organ's rotation.

Rationale for splitting rotation out (measured 2026-09-08/09/11 on
cowpea_curv26 across many ablations, see docs/ongoing/20260909_phytomer_latent_and_local_matching.md
section 2): a single SHARED 64D or 128D latent floors rotation error around
5-6 deg no matter how it is decoded — widening it 64D->128D changed nothing,
and a FROZEN shared latent + a much wider decoder still floored at 5.74 deg.
That rules out both encoder and decoder capacity as the bottleneck; what is
left is ALLOCATION: organs within a phytomer share strong affine correlation
(petiole-vs-leaflet scale r=0.78, stem-vs-leaflet r=0.52) but their rotations
do not share a comparable common axis, so a shared latent's KL budget is spent
on the strongly-correlated shape channels and starves rotation. Giving each
organ slot its own small residual channel — encoded directly from that slot's
own rot6d, decoded by a shared-weight per-slot head conditioned on
[coarse | that slot's residual] — routes rotation around the shared bottleneck
without losing the shape-sharing benefit or the token-count win described below.

Other rationale (validated 2026-09-09 on cowpea_curv26):
- One latent per phytomer turns Stage 3 flow matching from 8K tokens into K
  tokens (64x cheaper self-attention, 8x cheaper ODE) and collapses the
  matcher to a single phytomer-level Hungarian (no slot-level assignment).
  The coarse+residual split keeps this: the residual channels are extra width
  on the SAME per-phytomer token, not extra tokens.
- Geometry is PHYTOMER-RELATIVE (organ base - cluster center): the phytomer/node
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
# EVERY slot's base is deterministic (stem/petiole/peduncle/bud = center,
# leaflets = 0.8/1.0 x the petiole curve, flowers/fruit = curved peduncle tip —
# all verified over the full dataset). Nothing is VAE-learned (v2).
# 10 x (23 + presence bit) = 240.
PACKET_IN_DIM = NUM_SLOTS * (SLOT_DIM - 3 + 1)  # 10 x (23 + 1) = 240


class PhytomerVAE(nn.Module):
    """Hybrid phytomer-level VAE: (10 x 26D + 10 presence) -> z in R^D -> 10 x 26D.

    z splits into a shared coarse channel (class/base/scale/curv) and one
    residual channel per organ slot (rotation only) — see module docstring.
    """

    def __init__(
        self,
        slots_per_phytomer: int = NUM_SLOTS,
        slot_dim: int = SLOT_DIM,
        latent_dim: int = 128,
        residual_dim: int = 8,
        hidden_dim: int = 256,
        num_classes: int = NUM_ORGAN_TYPES,
    ):
        super().__init__()
        self.slots_per_phytomer = slots_per_phytomer
        self.slot_dim = slot_dim
        self.residual_dim = residual_dim
        self.coarse_dim = latent_dim - slots_per_phytomer * residual_dim
        if self.coarse_dim <= 0:
            raise ValueError(
                f"latent_dim={latent_dim} leaves no room for the coarse channel "
                f"once {slots_per_phytomer} slots x residual_dim={residual_dim} "
                f"={slots_per_phytomer * residual_dim} residual dims are removed; "
                f"raise latent_dim or lower residual_dim.")
        self.latent_dim = latent_dim  # = coarse_dim + slots_per_phytomer * residual_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        # Base columns are REMOVED before encoding: with EXACT XML phytomer
        # clustering every slot's base is deterministic (stem/petiole/peduncle/
        # bud = center, leaflets = 0.8/1.0 x the petiole curve, flowers/fruit =
        # curved peduncle tip). 10 x (23 + 1) = 240.
        self.in_dim = slots_per_phytomer * (slot_dim - 3 + 1)
        # Width of one stripped slot's geometry block (23 = 26 - 3 base cols),
        # and where rot6d sits inside it once base is removed (rot immediately
        # follows base in the original 26D layout, so it slides back by exactly
        # the base width — see encode()).
        self._slot_geom_dim = slot_dim - 3
        self._rot_off_in_slot = FM_ROT_START - (FM_BASE_END - FM_BASE_START)
        self._rot_width = FM_ROT_END - FM_ROT_START

        # Shared trunk: packet -> hidden. Both the coarse and residual heads
        # read from this (residual also reads each slot's own raw rotation).
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
        # Coarse channel: shared morphology (class/base/scale/curv id est.
        # everything except rotation). Organs co-vary strongly here
        # (petiole-vs-leaflet scale r=0.78), so a shared bottleneck is the
        # right fit and matches the old (pre-hybrid) single-latent design.
        self.fc_mu = nn.Linear(hidden_dim, self.coarse_dim)
        self.fc_logvar = nn.Linear(hidden_dim, self.coarse_dim)

        # Residual channel: ONE small latent per slot, dedicated to that
        # slot's rotation. Encoded from the shared trunk context PLUS that
        # slot's own rot6d (not pooled away), so rotation detail the trunk
        # would otherwise discard survives into the latent.
        self.residual_encoder = nn.Sequential(
            nn.Linear(hidden_dim + self._rot_width, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.SiLU(),
        )
        self.fc_mu_residual = nn.Linear(hidden_dim // 2, residual_dim)
        self.fc_logvar_residual = nn.Linear(hidden_dim // 2, residual_dim)

        # Decoder: coarse latent -> hidden -> everything EXCEPT rotation.
        self.decoder_backbone = nn.Sequential(
            nn.Linear(self.coarse_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # One output layer per quantity. Each emits values for ALL slots_per_phytomer
        # organs at once, hence the plural: the 3 leaflets of one phytomer have
        # genuinely different scales/curvatures, and reproducing that per-organ
        # variation from the coarse latent is exactly what these layers exist for.
        self.organ_cls = nn.Linear(hidden_dim, slots_per_phytomer * num_classes)
        self.organ_scales = nn.Linear(hidden_dim, slots_per_phytomer * 3)
        self.organ_curvs = nn.Linear(hidden_dim, slots_per_phytomer * 1)

        # Rotation decoder: per-slot, conditioned on [coarse | that slot's
        # residual]. Weights are SHARED across slots (applied via a 3D
        # reshape) so parameter count does not grow with slots_per_phytomer;
        # the coarse half of the input still lets rotation be informed by
        # organ class/shape context without living in the same bottleneck.
        self.organ_rot_decoder = nn.Sequential(
            nn.Linear(self.coarse_dim + residual_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 6),
        )

    def pack_input(self, packets: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        """Flattens (P, 10, 26) + (P, 10) presence -> (P, 240) encoder input.

        v3 scale normalization: slot scales are divided by the packet's phytomer
        scale s_a (petiole scale row) FIRST, so the encoder sees scale-INVARIANT
        relative geometry. The 76D flow state carries s_a explicitly; decode
        multiplies it back (denormalize_packet_scales) BEFORE assemble_packets.

        Base columns are REMOVED: with EXACT XML phytomer clustering every slot's
        base is deterministic (stem/petiole/peduncle/bud = center, leaflets =
        0.8/1.0 x the petiole curve, flowers/fruit = curved peduncle tip).
        """
        from diffusion_based.dataset.phytomer_packets import (
            phytomer_scale, normalize_packet_scales)
        pk = normalize_packet_scales(packets, phytomer_scale(packets))
        P = pk.shape[0]
        keep = torch.ones(pk.shape[-1], dtype=torch.bool, device=pk.device)
        keep[FM_BASE_START:FM_BASE_END] = False
        stripped = pk[..., keep]  # (P, 10, 23)
        return torch.cat([stripped.reshape(P, -1), presence.float().reshape(P, -1)], dim=-1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes (P, in_dim) packets to (mu, logvar), each (P, latent_dim).

        Layout: [:coarse_dim] is the shared coarse channel; the remaining
        slots_per_phytomer x residual_dim is one residual_dim block per slot,
        concatenated in slot order (slot s at
        [coarse_dim + s*residual_dim : coarse_dim + (s+1)*residual_dim]).
        """
        S = self.slots_per_phytomer
        P = x.shape[0]
        h = self.encoder(x)  # (P, hidden)
        mu_c = self.fc_mu(h)
        logvar_c = self.fc_logvar(h).clamp(-10.0, 10.0)

        # Recover each slot's own rot6d from the base-stripped input (base was
        # removed by pack_input, so rot6d slides back by the base width — see
        # __init__ for _rot_off_in_slot/_slot_geom_dim).
        geom = x[:, :S * self._slot_geom_dim].reshape(P, S, self._slot_geom_dim)
        slot_rot = geom[:, :, self._rot_off_in_slot:self._rot_off_in_slot + self._rot_width]

        h_exp = h.unsqueeze(1).expand(P, S, self.hidden_dim)
        hr = self.residual_encoder(torch.cat([h_exp, slot_rot], dim=-1))  # (P, S, hidden//2)
        mu_r = self.fc_mu_residual(hr).reshape(P, S * self.residual_dim)
        logvar_r = self.fc_logvar_residual(hr).clamp(-10.0, 10.0).reshape(P, S * self.residual_dim)

        mu = torch.cat([mu_c, mu_r], dim=-1)
        logvar = torch.cat([logvar_c, logvar_r], dim=-1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decodes (P, D) hybrid latents into (P, 10, 26) FM packets.

        z[:coarse_dim] drives class/base/scale/curv (shared, all slots at
        once); z[coarse_dim + s*residual_dim : ...] drives slot s's rotation
        only, via a per-slot head shared across slots (see __init__).

        Base columns are ZERO: with EXACT XML phytomer clustering every slot's
        base is deterministic (stem/petiole/peduncle/bud = center, leaflets =
        0.8/1.0 x the petiole curve), so the caller's assemble_packets()
        reconstructs them. Only flower/fruit bases (rare) are VAE-learned and
        passed through by assemble_packets.
        """
        P = z.shape[0]
        S = self.slots_per_phytomer
        z_c = z[:, :self.coarse_dim]
        z_r = z[:, self.coarse_dim:].reshape(P, S, self.residual_dim)

        h = self.decoder_backbone(z_c)
        pred_cls_logits = self.organ_cls(h).reshape(P, S, self.num_classes)   # (P, 10, 13)
        pred_base = torch.zeros(P, S, 3, device=z.device, dtype=z.dtype)       # (P, 10, 3) zeroed
        pred_scale = F.softplus(self.organ_scales(h)).reshape(P, S, 3) + 1e-4    # (P, 10, 3)
        pred_curv = self.organ_curvs(h).reshape(P, S, 1)                         # (P, 10, 1)

        z_c_exp = z_c.unsqueeze(1).expand(P, S, self.coarse_dim)
        pred_rot = self.organ_rot_decoder(torch.cat([z_c_exp, z_r], dim=-1))     # (P, 10, 6)

        pred_probs = F.softmax(pred_cls_logits, dim=-1)
        pred_26d = torch.cat([pred_probs, pred_base, pred_rot, pred_scale, pred_curv], dim=-1)
        return {
            "cls_logits": pred_cls_logits,
            "base": pred_base,
            "rot": pred_rot,
            "scale": pred_scale,
            "curv": pred_curv,
            "recon_packets": pred_26d,  # (P, 10, 26), relative coords
        }

    def forward(
        self, packets: torch.Tensor, presence: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        x = self.pack_input(packets, presence)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
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
        pred_scale = pred["scale"].detach()                  # (P, 10, 3)
        pred_cls = pred["cls_logits"].argmax(-1).detach()    # (P, 8)
        tgt_scale = target_packets[:, :, FM_SCALE_START:FM_SCALE_END]
        tgt_cls = target_packets[:, :, :FM_OT_END].argmax(-1)

        aligned = torch.zeros_like(target_packets)
        aligned[:, :, 0] = 1.0  # NONE one-hot default
        aligned_pres = torch.zeros_like(target_presence, dtype=torch.bool)

        from diffusion_based.dataset.phytomer_packets import ROLE_SLOT_RANGES
        role_ranges = list(ROLE_SLOT_RANGES.values())
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
        """Masked composite VAE loss over the 10-slot packet.

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
            # pred_t/tgt_t: (P, 10, D)
            err = F.smooth_l1_loss(pred_t, tgt_t, reduction="none").mean(dim=-1)  # (P, 8)
            return (err * w).sum() / (n_present + absent_weight * n_absent)

        tgt_base = torch.zeros_like(target_packets[:, :, FM_BASE_START:FM_BASE_END])
        tgt_rot = target_packets[:, :, FM_ROT_START:FM_ROT_END]
        # v3: scale targets are NORMALIZED by the packet phytomer scale (matching
        # pack_input, which normalizes the encoder input). Decode output scales
        # are normalized; callers denormalize by s_a before assemble_packets.
        from diffusion_based.dataset.phytomer_packets import (
            phytomer_scale as _phytomer_scale, normalize_packet_scales as _norm_scales)
        tgt_packets_norm = _norm_scales(
            target_packets, _phytomer_scale(target_packets))
        tgt_scale = tgt_packets_norm[:, :, FM_SCALE_START:FM_SCALE_END]
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
            tube_loss = 1.0 - f_cos  # (P, 10), spin-invariant
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
