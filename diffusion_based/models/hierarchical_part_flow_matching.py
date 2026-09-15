"""
Hierarchical Matryoshka Botanical Flow Matching Model.

Decouples macroscopic plant skeleton/phytomer topology (Coarse Stage 1)
from microscopic trifoliolate leaflet and organ geometry (Fine Stage 2)
using Matryoshka power-of-2 nested queries.
"""

import math
import os
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
    matrix_to_rot6d,
)
from diffusion_based.dataset.phytomer_topology import chain_phytomers
from diffusion_based.dataset.phytomer_roll import derive_forward, roll_to_matrix


# =============================================================================
# PHYTOMER CAPACITY CURVE (calibrated 2026-09-08 evening on cowpea_curv26 cache)
# =============================================================================
# GT phytomer cluster counts per plant (petiole bases + standalone internodes,
# matcher-consistent counting in HierarchicalBotanicalMatcher) were sampled
# per DAP from dataset/cache/cowpea_curv26 (100-300 samples per DAP bucket,
# 59,441-file scan) and a logistic curve was fitted to the per-DAP
# rolling-max envelope of the OBSERVED SAMPLE MAX (not the p97.5 quantile),
# so a DAP->K lookup covers 100% of observed samples including worst-case
# branching tails (max 342 clusters @ DAP 92):
#   phytomers_max(dap) = L / (1 + exp(-k*(dap - x0))) + N0
# Curve constants (max-envelope fit; regenerate via tools/calibrate_phytomer_capacity.py):
PHYTOMER_CURVE_L = 297.2      # asymptotic capacity (phytomers)
PHYTOMER_CURVE_K = 0.1841     # logistic growth rate (1/day)
PHYTOMER_CURVE_X0 = 30.5      # midpoint DAP (days)
PHYTOMER_CURVE_N0 = 15.53     # offset at DAP -> -inf
# Per-sample capacity = ceil(curve(DAP) * PHYTOMER_MARGIN + PHYTOMER_MARGIN_FLAT).
# PHYTOMER_MARGIN guards the residual individual variance (branching differences)
# on top of the max-envelope fit; PHYTOMER_MARGIN_FLAT adds absolute headroom
# for seedlings where relative variance is largest.
# Coverage on the 2026-09-08 evening scan: p97.5 100% | observed-max 100%
# (DAP 92 max 342 -> K=350; cache organ rows max 2,625 -> slots 2,800).
PHYTOMER_MARGIN = 1.10
PHYTOMER_MARGIN_FLAT = 6.0
PHYTOMER_MARGIN_MIN = 1.05

# Ceiling on the Stage-2 phytomer scale head, in FM units (x SCALE_SCALE = 50),
# so 25.0 = 50 cm. Ground-truth petiole length tops out near 4.0 FM units.
SCALE_CEIL = 25.0


_GRAD_PROBE = os.environ.get("FM_GRAD_PROBE", "0") in ("1", "2")
# FM_ACT_PROBE=1: forward probe on Stage 2's decoder LayerNorm inputs. A pre-LN
# block's backward carries a 1/std factor per token through each LayerNorm, so
# a token whose 384 features have collapsed to near-constant amplifies whatever
# gradient reaches it (the per-layer amplification _make_layer_probe measures).
# This reports the smallest per-token input std seen by each norm, once per
# probe interval, so the collapse can be watched before the canary fires.
_ACT_PROBE = os.environ.get("FM_ACT_PROBE", "0") == "1"
_ACT_PROBE_EVERY = int(os.environ.get("FM_ACT_PROBE_EVERY", "50"))
_act_probe_state: dict = {}


def _make_act_probe(label: str):
    def _pre_hook(module, args):
        x = args[0]
        if not torch.is_tensor(x) or x.dim() < 2:
            return
        with torch.no_grad():
            std = x.float().std(dim=-1)                      # per token
            st = _act_probe_state.setdefault(label, {"n": 0, "min": float("inf"), "n_tiny": 0, "tok": 0})
            st["n"] += 1
            st["min"] = min(st["min"], float(std.min()))
            st["n_tiny"] += int((std < 1e-3).sum())
            st["tok"] += int(std.numel())
            if st["n"] % _ACT_PROBE_EVERY == 0:
                print(f"  [ActProbe] {label}: min token std {st['min']:.3e} | tokens with std<1e-3: "
                      f"{st['n_tiny']}/{st['tok']} | median this call {float(std.median()):.3e}", flush=True)
                st["min"] = float("inf"); st["n_tiny"] = 0; st["tok"] = 0
    return _pre_hook
_PROBE_THRESHOLD = 1e9
_probe_seen: set = set()


def _probe_grad(t, label: str) -> None:
    if t is None:
        return
    """Report the first backward pass in which `t` carries a gradient past
    _PROBE_THRESHOLD. Used to find which consumer of a shared tensor is the
    source of a gradient explosion, since the parameter-level canary in the
    training loop can only say where it landed. One report per label."""
    if not t.requires_grad:
        return

    def _hook(g):
        if label in _probe_seen:
            return
        finite = g[torch.isfinite(g)].abs()
        mx = float(finite.max()) if finite.numel() else float("inf")
        n_nan = int(torch.isnan(g).sum())
        if mx > _PROBE_THRESHOLD or n_nan > 0:
            _probe_seen.add(label)
            print(f"  [GradProbe] {label}: max_finite={mx:.3e} nan={n_nan} "
                  f"shape={list(g.shape)}", flush=True)

    t.register_hook(_hook)


def _make_layer_probe(label: str):
    """full-backward hook reporting a module's own gradient amplification.

    The intermediate probe showed the gradient entering Stage 2's decoder
    output under 1e9 and leaving its input at ~1e13, so ~1e4 of amplification
    happens inside those layers. Reasoning about which op does it has a poor
    track record here (seven architectural hypotheses tested and refuted on
    2026-09-12), so this measures each layer's out->in ratio directly and
    reports the first layer to exceed 100x. One report per label."""

    def _hook(module, grad_input, grad_output):
        if label in _probe_seen:
            return

        def _mx(ts):
            best = 0.0
            for t in ts:
                if t is None:
                    continue
                f = t[torch.isfinite(t)].abs()
                if f.numel():
                    best = max(best, float(f.max()))
            return best

        g_in, g_out = _mx(grad_input), _mx(grad_output)
        if g_out > 0 and g_in / g_out > 100.0:
            _probe_seen.add(label)
            print(f"  [LayerProbe] {label}: grad_out={g_out:.3e} -> grad_in={g_in:.3e} "
                  f"(amplification {g_in / g_out:.1f}x)", flush=True)

    return _hook


STAGE3_GEOM_DIM = 8   # [dpos(3) * BASE_SCALE | roll (cos, sin) | scale(3), FM units]


def split_flow_state(x: torch.Tensor, latent_dim: int):
    """Stage 3 flow state -> (geometry block or None, latent block).

    With `stage3_geometry` the per-phytomer state is
        [ (pos - parent_pos) * BASE_SCALE (3) | roll (2) | scale (3) | latent (D) ]
    so the child's position is generated RELATIVE TO ITS FIXED PARENT (design doc
    §2.1; the 2026-09-14 GT-substitution ablation put node position first among the
    per-node errors, rotation and latent next). Without it the state is the latent alone.
    """
    if x.shape[-1] == latent_dim:
        return None, x
    return x[..., :STAGE3_GEOM_DIM], x[..., -latent_dim:]


def geometry_from_flow(geom: torch.Tensor, parent_pos: torch.Tensor):
    """(B, K, 8) geometry block + (B, K, 3) parent position (NaN -> origin) ->
    (pos (B, K, 3) metres, roll (B, K, 2) unit, scale (B, K, 3) FM units)."""
    parent = torch.nan_to_num(parent_pos, nan=0.0)
    pos = parent + geom[..., :3] / BASE_SCALE
    roll = safe_normalize(geom[..., 3:5], eps=1e-3)
    scale = geom[..., 5:8]
    return pos, roll, scale


def parent_relative(pos: torch.Tensor, parent_pos: torch.Tensor,
                    has_parent: Optional[torch.Tensor]) -> torch.Tensor:
    """(B, K, 4) [parent - self, has_parent] for Stage 3's (parent, self) conditioning.

    parent_pos may carry NaN where a node has no parent (reconstruct_phytomer_rot's
    convention); those rows become [0, 0, 0, 0]. K may differ between pos and
    parent_pos (the phytomer bank slice vs the GT-padded width): the parent
    tensors are sliced or zero-padded to pos's K.
    """
    B, K, _ = pos.shape
    pp = parent_pos.to(pos.dtype)
    if has_parent is None:
        has = torch.isfinite(pp).all(dim=-1)
    else:
        has = has_parent.bool()
    if pp.shape[1] != K:
        if pp.shape[1] > K:
            pp, has = pp[:, :K], has[:, :K]
        else:
            pad = K - pp.shape[1]
            pp = F.pad(pp, (0, 0, 0, pad)); has = F.pad(has, (0, pad))
    rel = torch.nan_to_num(pp, nan=0.0) - pos
    rel = torch.where(has.unsqueeze(-1), rel, torch.zeros_like(rel))
    return torch.cat([rel, has.to(pos.dtype).unsqueeze(-1)], dim=-1)


def _make_act_recorder(store: dict, name: str):
    def _fwd(module, inputs, output):
        try:
            outs = output if isinstance(output, (tuple, list)) else (output,)
            m = 0.0
            for o in outs:
                if torch.is_tensor(o) and o.numel():
                    f = o.detach()[torch.isfinite(o.detach())].abs()
                    if f.numel():
                        m = max(m, float(f.max()))
            ins = [i for i in inputs if torch.is_tensor(i) and i.numel()]
            mi = max((float(i.detach()[torch.isfinite(i.detach())].abs().max()) for i in ins if torch.isfinite(i.detach()).any()), default=0.0)
            store[name] = (mi, m)
        except Exception:
            pass
    return _fwd


_op_probe_seen: set = set()


def _make_op_probe(store: dict, name: str, threshold: float = 1e12):
    def _hook(module, grad_input, grad_output):
        if name in _op_probe_seen:
            return

        def _mx(ts):
            best = 0.0
            for t in ts:
                if t is None:
                    continue
                f = t[torch.isfinite(t)].abs()
                if f.numel():
                    best = max(best, float(f.max()))
            return best

        g_in, g_out = _mx(grad_input), _mx(grad_output)
        if g_in > threshold or g_out > threshold:
            _op_probe_seen.add(name)
            a_in, a_out = store.get(name, (float("nan"), float("nan")))
            print(f"  [OpProbe] {name}: grad_out={g_out:.3e} -> grad_in={g_in:.3e} "
                  f"(x{(g_in / g_out) if g_out > 0 else float('inf'):.1f}) | fwd max|in|={a_in:.3e} max|out|={a_out:.3e}", flush=True)

    return _hook


def safe_normalize(v: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Safely normalizes vectors along the last dimension with bounded gradient.
    Detaching the denominator norm prevents collinear/zero division where d/dv(v/||v||)
    involves 1/||v||^3 terms that explode to 10^9+ in backward pass.
    """
    norm = torch.clamp(torch.norm(v, p=2, dim=-1, keepdim=True), min=eps)
    return v / norm.detach()


@torch.no_grad()
def reconstruct_phytomer_rot(
    pos: torch.Tensor,
    roll: torch.Tensor,
    ordinal: torch.Tensor,
    is_base_logits: torch.Tensor,
    exist: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full node rotation (B, K, 6) from predicted position + roll, deriving
    the forward axis from resolved topology instead of predicting it.

    Topology resolution (chain_phytomers) is a discrete, non-differentiable
    argmin over a distance/ordinal cost -- there is no gradient to preserve
    here regardless of caller context, hence @no_grad. Called at inference
    (sample_ode) and, per-render-sample only, from the training loop's
    differentiable render block (position/roll themselves keep their own
    gradients from wherever this function's OUTPUT feeds back into, e.g.
    assemble_packets' `centers` argument -- only the topology lookup itself
    is detached).

    Args:
        pos: (B, K, 3) node positions.
        roll: (B, K, 2) predicted (cos, sin) roll.
        ordinal: (B, K) predicted position along the shoot.
        is_base_logits: (B, K) predicted is-shoot-base logit (>0 = base).
        exist: optional (B, K) existence probability. **Pass it at inference.**
            Stage 2 predicts K slots but only some are real phytomers; the rest
            sit wherever the head left them, which on the epoch-10 checkpoint
            meant 20 cm below the soil and 30 cm off to the side. Chained over
            all K slots, such a stray node is the lowest of everything, so the
            height rule leaves it parentless, it becomes a "root" whose parent
            is the origin, and a 36 cm internode gets drawn from the origin to
            it -- while real nodes near it pick it as THEIR parent. Chaining
            only the live slots (chain_phytomers' own exist>0.5 gate) removes
            both. None (the training render block) chains everything, which is
            fine there because that path pairs nodes through the GT-derived
            matched-only map instead.
    Returns:
        rot6d: (B, K, 6) reconstructed rotation.
        parent_pos: (B, K, 3) each node's parent position under the resolved
            chain: the ORIGIN for the root, and a non-live row's OWN position
            (zero gap, so assemble_packets draws it with zero length rather
            than a NaN-triggered fallback). Hand this to
            assemble_packets(parent_pos=..., centers=pos) so the internode is
            the parent->node segment, as the training render block already
            does; without it inference used the decoded slot-0 length.
    """
    B, K, _ = pos.shape
    out = torch.zeros(B, K, 6, device=pos.device, dtype=pos.dtype)
    parent_pos = pos.clone()
    for b in range(B):
        parent_idx, _, _ = chain_phytomers(
            pos[b], ordinal=ordinal[b], is_base=(is_base_logits[b] > 0).float(),
            exist=None if exist is None else exist[b])
        # A parentless LIVE row is the plant root, whose parent is the origin.
        live = (torch.ones(K, dtype=torch.bool, device=pos.device) if exist is None
                else exist[b] > 0.5)
        fwd = derive_forward(pos[b], parent_idx)
        R = roll_to_matrix(fwd, roll[b])
        out[b] = matrix_to_rot6d(R)
        has_parent = parent_idx >= 0
        parent_pos[b] = torch.where(
            has_parent.unsqueeze(-1), pos[b][parent_idx.clamp(min=0)],
            torch.where(live.unsqueeze(-1), torch.zeros_like(pos[b]), pos[b]))
    return out, parent_pos


# =============================================================================
# PHYTOMER-LEVEL FLOW TARGET LAYOUT (base + roll + scale + latent, combined-flow
# configuration -- NOT what HierarchicalPartFlowMatchingModel currently wires
# up: it constructs PhytomerFlowMatchingDecoder with base_dim=rot_dim=scale_dim=0,
# so the ACTIVE flow target is the 128D VAE latent alone, and base/roll/scale
# are Stage-2-only outputs supervised by their own direct losses, never part of
# the ODE state -- see the "Hybrid Decoupled" comments in
# HierarchicalPartFlowMatchingModel.__init__ and PhytomerFlowMatchingDecoder's
# own docstring. These two helpers exist for the alternative, larger combined
# flow vector this class's constructor arguments still support, kept in sync
# with the 2026-09-11 roll change for whenever that configuration is used.)
# =============================================================================
#   z_1[phytomer] = [ node_base_xyz(3) | node_roll(2) | phytomer_scale(3) | phytomer_latent(D) ]
# The phytomer scale s_a (petiole length row [len, radius, unused], v3 packet
# format) is explicit so the flow refines phytomer size directly (GS-style
# primitive scale); the latent carries only scale-NORMALIZED relative geometry.
# Layout indices (relative to z_1's first dim):
PHYTO_FLOW_BASE_START = 0          # xyz (3)  — refined phytomer base position (metres)
PHYTO_FLOW_BASE_END = 3
PHYTO_FLOW_ROLL_START = 3          # (cos, sin) (2) — forward axis is derived from
PHYTO_FLOW_ROLL_END = 5            #                  position, not part of this vector
PHYTO_FLOW_SCALE_START = 5         # scale (3) — refined phytomer scale (petiole row)
PHYTO_FLOW_SCALE_END = 8
PHYTO_FLOW_LATENT_START = 8        # latent (D) — VAE latent of the relative packet


def build_phytomer_flow_target(
    phytomer_pos: torch.Tensor,
    phytomer_roll: torch.Tensor,
    phytomer_scale: torch.Tensor,
    phytomer_latent: torch.Tensor,
) -> torch.Tensor:
    """Concatenates per-phytomer pose + scale + latent into the Stage-3 flow target (B, K, 8+D).

    Args:
        phytomer_pos: (B, K, 3) refined phytomer base positions (metres).
        phytomer_roll: (B, K, 2) refined phytomer (cos, sin) roll.
        phytomer_scale: (B, K, 3) GT phytomer scale (petiole scale row, FM units).
        phytomer_latent: (B, K, D) VAE latent of the phytomer-relative packets.

    Returns:
        (B, K, 8 + D) flow target.
    """
    return torch.cat([phytomer_pos, phytomer_roll, phytomer_scale, phytomer_latent], dim=-1)


def split_phytomer_flow_target(z: torch.Tensor, latent_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Splits a Stage-3 flow vector back into (phytomer_pos, phytomer_roll, phytomer_scale, latent)."""
    pos = z[..., PHYTO_FLOW_BASE_START:PHYTO_FLOW_BASE_END]
    roll = z[..., PHYTO_FLOW_ROLL_START:PHYTO_FLOW_ROLL_END]
    scl = z[..., PHYTO_FLOW_SCALE_START:PHYTO_FLOW_SCALE_END]
    lat = z[..., PHYTO_FLOW_LATENT_START:PHYTO_FLOW_LATENT_START + latent_dim]
    return pos, roll, scl, lat


def apply_ref_for_flow(
    packet_hat: torch.Tensor,
    refined_pos: torch.Tensor,
    refined_rot: torch.Tensor,
    base_scale: float = BASE_SCALE,
) -> torch.Tensor:
    """Re-phytomers (B, K, M, 26) relative packets using the REFINED phytomer pose.

    Option 1 semantics: the refined phytomer rotation is the reference frame for
    the packet; the refined phytomer base position re-adds absolute placement.

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
    max_phytomers: int = 512,
    margin: float = PHYTOMER_MARGIN,
) -> int:
    """Computes upper-bound active phytomer count based on biological growth or predicted phytomers.

    If num_phytomers is provided:
        K_upper = min(max_phytomers, ceil(num_phytomers * margin + flat))
    Else if DAP is provided:
        K_upper(t) = min(max_phytomers, ceil(PHYTOMER_L / (1 + exp(-PHYTOMER_K * (t - PHYTOMER_X0))) + PHYTOMER_N0)
                         * PHYTOMER_MARGIN + PHYTOMER_FLAT)

    2026-09-08 recalibration (evening rescan, 100-300 samples/DAP): the previous
    p97.5-envelope fit left 11% of DAP buckets with at least one observed sample
    beyond capacity (worst 342 clusters @ DAP 92). The fit now targets the
    rolling-max envelope of observed sample maxima -> 100% observed-max coverage.
    Continuous (non power-of-2) capacity avoids tier-snap quantization waste.
    """
    margin = max(float(margin), PHYTOMER_MARGIN_MIN)
    if num_phytomers is not None:
        val = float(num_phytomers.max().item())
        val = max(0.0, val)
        k = math.ceil(val * margin + PHYTOMER_MARGIN_FLAT)
    elif dap is not None:
        dap_val = float(dap.max().item())
        dap_val = max(0.0, min(100.0, dap_val))
        base = (
            PHYTOMER_CURVE_L
            / (1.0 + math.exp(-PHYTOMER_CURVE_K * (dap_val - PHYTOMER_CURVE_X0)))
            + PHYTOMER_CURVE_N0
        )
        k = math.ceil(max(0.0, base) * margin + PHYTOMER_MARGIN_FLAT)
    else:
        return max_phytomers

    return int(min(max(k, 8), max_phytomers))


def estimate_phytomer_capacity(
    dap: Optional[torch.Tensor],
    margin: float = PHYTOMER_MARGIN,
    flat: float = PHYTOMER_MARGIN_FLAT,
    max_phytomers: int = 512,
) -> torch.Tensor:
    """Per-sample phytomer capacity from DAP via the calibrated logistic phytomer curve.

    Returns a (B,) long tensor of per-sample phytomer capacities. Used to build
    per-sample capacity-aware losses (e.g. clamping GT existence targets to the
    active phytomer slice) without collapsing the batch to a single max-K.
    """
    if dap is None:
        return torch.full((1,), max_phytomers, dtype=torch.long)
    d = dap.float().view(-1)
    base = (
        PHYTOMER_CURVE_L / (1.0 + torch.exp(-PHYTOMER_CURVE_K * (d - PHYTOMER_CURVE_X0)))
        + PHYTOMER_CURVE_N0
    )
    k = torch.ceil(torch.clamp(base, min=0.0) * margin + flat)
    return k.clamp(min=8, max=max_phytomers).long()


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
        # Without this the pred-count prior (init_logits) suppresses all phytomers
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

    Predicts 3D base coordinates, 6D orientation, and existence logits for K phytomers.
    Uses DETR3D / PETR style 3D reference coordinates, 3D positional encoding,
    and incorporates Stage 1 MacroBiologicalHead soft margin prior.
    """

    def __init__(
        self,
        max_phytomers: int = 512,
        embed_dim: int = 384,
        num_heads: int = 8,
        num_layers: int = 4,
        init_phytomer_count: float = 50.0,
        dap_clue: bool = True,
        num_levels: int = 1,
        node_token_window: int = 1,
    ):
        super().__init__()
        self.max_phytomers = max_phytomers
        self.embed_dim = embed_dim
        self.dap_clue = dap_clue
        self.num_levels = int(num_levels)
        self.node_token_window = int(node_token_window)
        if self.num_levels > 1:
            self.level_embed = nn.Parameter(torch.zeros(self.num_levels, embed_dim))   # multizoom token identity

        # Content query
        self.phytomer_queries = nn.Parameter(torch.randn(max_phytomers, embed_dim) * 0.02)

        # 3D Reference Points (DETR3D / PETR style): initialized across canonical plant envelope
        # x, y in [-0.5, 0.5], z vertically ascending in [0.0, 1.0]
        init_ref = torch.randn(max_phytomers, 3) * 0.15
        init_ref[:, 2] = torch.linspace(0.0, 1.0, max_phytomers)  # Vertical upward growth prior
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
        # A pre-LN (norm_first) decoder leaves its output as the raw residual
        # stream, which in the Stage 2 burst state (2026-09-14 dissection)
        # reached magnitude ~1e3 and was fed as query/key/value into the bf16
        # phytomer self-attention below, where the backward amplified the
        # incoming gradient ~6000x on every batch. Pre-LN transformers close
        # with a final LayerNorm for exactly this reason; FM_DECODER_FINAL_NORM=0
        # rebuilds the old architecture for checkpoints that predate it.
        _final_norm = nn.LayerNorm(embed_dim) if os.environ.get("FM_DECODER_FINAL_NORM", "1") == "1" else None
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers, norm=_final_norm)
        if _ACT_PROBE:
            for _i, _layer in enumerate(self.decoder.layers):
                for _nm in ("norm1", "norm2", "norm3"):
                    getattr(_layer, _nm).register_forward_pre_hook(_make_act_probe(f"decoder.layer{_i}.{_nm}"))
        if _GRAD_PROBE:
            # The probe on the intermediates showed the gradient entering this
            # decoder's output under 1e9 and the gradient leaving its input at
            # ~1e13, so the ~1e4 amplification is inside these layers. These
            # per-layer hooks report each layer's own output->input ratio,
            # which says WHICH layer rather than just that it is in here.
            for _i, _layer in enumerate(self.decoder.layers):
                _layer.register_full_backward_hook(_make_layer_probe(f"decoder.layer{_i}"))
        if os.environ.get("FM_GRAD_PROBE", "0") == "2":
            # FM_GRAD_PROBE=2 (2026-09-14): every leaf submodule of this stage
            # reports, once, the first backward pass in which its grad_input
            # exceeds 1e12 (with the max |activation| it produced in the
            # matching forward), so a burst can be pinned to an op, not a layer.
            self._probe_act = {}
            _thr = float(os.environ.get("FM_OP_PROBE_THRESHOLD", "1e12"))
            for _name, _mod in self.named_modules():
                if _name and not isinstance(_mod, (nn.Sequential, nn.ModuleList, nn.TransformerDecoder)):
                    _mod.register_forward_hook(_make_act_recorder(self._probe_act, _name))
                    _mod.register_full_backward_hook(_make_op_probe(self._probe_act, _name, _thr))

        # Stage 1: Macro Biological Head
        self.macro_head = MacroBiologicalHead(embed_dim=embed_dim,
                                              init_phytomer_count=init_phytomer_count)

        # DAP clue embedding: broadcasts the predicted plant age to every phytomer
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

        # 2. Phytomer roll (1 DOF, (cos, sin) representation). The forward axis
        # (the other 2 of rotation's 3 DOF) is DERIVED post-hoc from resolved
        # topology (this_node_pos - parent_node_pos, see phytomer_roll.py) --
        # predicting it independently was redundant with position. Measured
        # 2026-09-11: chain_phytomers loses <=2.5 points of parent-recovery
        # accuracy without a rotation-based directional cue once the ordinal
        # cue below is present (97.5/94.5/97.7% vs 100.0/95.4/98.9% at DAP
        # 10/50/90), which is what makes topology resolvable from position
        # alone and rotation reconstructable only afterward.
        self.roll_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2),
        )

        # 2b. Coarse phytomer scale (petiole scale row [len, radius, unused], FM
        # units): bridge-prior seed for the 76D flow's scale dims. Softplus keeps
        # it positive; the flow refines it against the GT phytomer scale.
        self.scale_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),
        )

        # 2c. Where this phytomer sits on its shoot: a normalised position along
        # the shoot, and whether it is the shoot's base. Geometry alone cannot
        # recover the stem chain reliably enough — measured, a chain built from
        # positions and rotations puts 200 of 201 phytomers in the right shoot,
        # and that single mistake relocates an entire branch and costs ~50
        # points of rendered IoU, because the XML export places a shoot by
        # continuing its chain. Predicting the ordinal directly gives the
        # chaining a cue that does not degrade with node spacing (DAP 10 nodes
        # sit 0.47 cm apart). Ordinal + base flag rather than a shoot id: ids
        # are permutation-arbitrary and would need their own matching.
        self.order_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2),   # [normalised ordinal, is_shoot_base logit]
        )

        # 3. Phytomer existence probability logit (delta relative to soft margin prior)
        self.exist_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

        # Edge-biased phytomer self-attention (soft graph prior): nearby phytomers
        # (small 3D distance) attend more strongly, enforcing spatial coherence
        # along the shoot axis (kinematic-chain locality). GraphFormer-style
        # distance bias added to the attention scores.
        self.phytomer_self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.05, batch_first=True,
        )
        self.phytomer_norm = nn.LayerNorm(embed_dim)
        self.edge_bias_temp = 0.15

    def forward(
        self,
        image_tokens: torch.Tensor,
        active_k: Optional[int] = None,
        capacity_mode: str = "given",
        margin: float = PHYTOMER_MARGIN,
        flat: float = PHYTOMER_MARGIN_FLAT,
        pred_dap: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Stage 1 + Stage 2 forward pass.

        Args:
            image_tokens: (B, T, embed_dim) 3D-aware visual tokens from DINOv2RayEncoder.
            active_k: Sliced phytomer count. Required when capacity_mode='given'.
            capacity_mode:
                'given'      - use `active_k` as-is (GT-DAP teacher forcing path).
                'pred_phyto' - two-pass: run MacroBiologicalHead first on the CLS token,
                               read `pred_num_phytomers`, then slice the phytomer bank to
                               K = ceil(max_sample * margin + flat). The count prediction
                               itself stays fully differentiable (loss path), while the
                               slicing index is a discrete capacity decision (no gradient
                               needed — the soft-margin prior carries the botanical prior).
                'full'       - use the entire phytomer bank (max_phytomers).
        Returns:
            Dict containing:
                'phytomer_pos': (B, K, 3) predicted 3D phytomer base positions.
                'phytomer_roll': (B, K, 2) predicted (cos, sin) roll -- the forward
                    axis (the other 2 of rotation's 3 DOF) is not part of this
                    output; derive it from resolved topology (phytomer_roll.py).
                'phytomer_logits': (B, K, 1) existence logits with soft margin bias.
                'phytomer_features': (B, K, embed_dim) phytomer latent representations.
                'pred_dap': (B, 1) auxiliary estimated DAP.
                'pred_num_phytomers': (B, 1) predicted phytomer count.
                'soft_margin_weights': (B, K) soft tapering existence prior.
        """
        B = image_tokens.shape[0]
        cls_token = image_tokens[:, 0]

        # Stage 1 (Macro Biological Prior) FIRST: it lives on the CLS token and needs
        # the full max_k width for the soft-margin schedule. In 'pred_phyto' mode this
        # pass also determines the phytomer bank slicing.
        macro_out_full = self.macro_head(cls_token, max_k=self.max_phytomers)

        if capacity_mode == "given":
            if active_k is None:
                active_k = self.max_phytomers
            K = int(active_k)
        elif capacity_mode == "pred_phyto":
            # Discrete capacity from the (differentiable) phytomer count prediction.
            # The count tensor's autograd graph is preserved in macro_out_full; the
            # .max()/.item() read below only forks the scalar used for slicing.
            K = compute_matryoshka_slice(
                num_phytomers=macro_out_full["pred_num_phytomers"].detach(),
                max_phytomers=self.max_phytomers,
                margin=margin,
            )
            K = max(int(K), 8)
        elif capacity_mode == "full":
            K = self.max_phytomers
        else:
            raise ValueError(f"Unknown capacity_mode: {capacity_mode}")

        # Query = Content embedding + 3D Positional Encoding + DAP clue.
        # The predicted DAP (from Stage 1) is broadcast to all phytomers as a
        # developmental-stage prior; zero-init keeps the start state unchanged
        # while loss_dap gradients teach the head, and dap_embed learns to use it.
        if self.num_levels > 1:
            image_tokens = add_level_embed(image_tokens, self.level_embed)
        q_content = self.phytomer_queries[:K].unsqueeze(0).expand(B, -1, -1)
        q_pos = self.ref_pos_mlp(self.ref_points[:K]).unsqueeze(0).expand(B, -1, -1)
        q = q_content + q_pos
        if self.dap_clue and pred_dap is not None:
            d = (pred_dap.float().view(B, 1) / 100.0)  # normalize to [0, 1]
            q = q + self.dap_embed(d).unsqueeze(1).expand(-1, K, -1)

        # Gradient barrier at the query boundary, same pattern already used on
        # the render path's pos_all / fine-exist logits.
        #
        # Measured 2026-09-12: the gradient reaching this decoder's OUTPUT stays
        # under 1e9 while the gradient leaving its INPUT reaches ~1e13, and
        # decoder layer 0 alone amplifies ~107x at healthy magnitudes. Every
        # leaf that feeds this query -- phytomer_queries, ref_points and
        # ref_pos_mlp -- therefore hangs off the end of a large amplifier, and
        # an intermittent explosion landed on exactly those (ref_points first,
        # one element at 1.3e15, no NaN). Healthy gradients here are ~1e-2, so a
        # +-5 clamp never binds in normal training; it only stops a pathological
        # step from destroying the reference-point prior, which then takes the
        # whole run down (job 38240070: every step of epoch 4 skipped).
        #
        # This is a barrier, NOT a root cause. Seven architectural hypotheses
        # were tested and refuted (see the 2026-09-12 handoff doc §1.9); the
        # amplifier itself is still unexplained, and the canary is deliberately
        # left in place to report if it fires anyway.
        if q.requires_grad:
            q.register_hook(lambda g: torch.nan_to_num(g.clamp(-5.0, 5.0), nan=0.0))

        # Cross-attend with 3D-aware image tokens
        if os.environ.get("FM_DECODER_FP32", "0") == "1":
            with torch.autocast(device_type="cuda", enabled=False):
                phytomer_features = self.decoder(q.float(), image_tokens.float()).to(q.dtype)
        else:
            phytomer_features = self.decoder(q, image_tokens)

        # Edge-biased phytomer self-attention (soft graph prior): distance-based
        # attention bias so spatially adjacent phytomers (kinematic chain) share
        # context. Bias = -dist / temp, added to the attention scores via the
        # attn_mask slot (additive, per-head broadcast). Distances come from the
        # ordered 3D reference points (z-ascending = chain order), so the prior
        # is static and available before the decoder.
        ref_k = self.ref_points[:K]  # (K, 3)
        dist_k = torch.cdist(ref_k, ref_k)  # (K, K)
        edge_bias = -dist_k / self.edge_bias_temp  # (K, K)
        # FM_SELFATTN_FP32 (default on since 2026-09-14): this block runs outside
        # autocast in float32. Measured on the frozen burst state: bf16 here
        # amplified the gradient entering the decoder from ~3e8 to ~2e12 on
        # every batch (8/8 steps tripped the canary); in fp32 0-2 of 8 did.
        # FM_NO_EDGE_BIAS=1 is a diagnostic only (dropping the mask made it worse).
        if os.environ.get("FM_NO_EDGE_BIAS", "0") == "1":
            edge_bias = None
        if os.environ.get("FM_SELFATTN_FP32", "1") == "1":
            with torch.autocast(device_type="cuda", enabled=False):
                _f32 = phytomer_features.float()
                attn_out, _ = self.phytomer_self_attn(
                    _f32, _f32, _f32,
                    attn_mask=edge_bias.float() if edge_bias is not None else None,
                    need_weights=False,
                )
                phytomer_features = self.phytomer_norm(_f32 + attn_out).to(phytomer_features.dtype)
        else:
            attn_out, _ = self.phytomer_self_attn(
                phytomer_features, phytomer_features, phytomer_features,
                attn_mask=edge_bias,
                need_weights=False,
            )
            phytomer_features = self.phytomer_norm(phytomer_features + attn_out)

        # Stage 2 Heads: predict coordinate offset from 3D reference points (physically bounded to +/- 0.5m)
        delta_pos = torch.tanh(self.pos_head(phytomer_features)) * 0.5
        phytomer_pos = self.ref_points[:K].unsqueeze(0) + delta_pos

        # Opt-in backward probe (FM_GRAD_PROBE=1). The gradient canary names the
        # PARAMETER a leak lands on, but ref_points is fed by three separate
        # paths (this direct add, ref_pos_mlp, and the cdist edge bias), so the
        # parameter alone does not say which consumer produced it. These hooks
        # report on the intermediates instead, which does. Off by default:
        # hooks on every forward are not free.
        if _GRAD_PROBE:
            _probe_grad(q_pos, "q_pos (ref_pos_mlp out)")
            _probe_grad(edge_bias, "edge_bias (cdist out)")
            _probe_grad(phytomer_features, "phytomer_features")
            _probe_grad(delta_pos, "delta_pos (pos_head out)")
            _probe_grad(phytomer_pos, "phytomer_pos")

        # Roll (1 DOF, (cos, sin)). Forward axis is derived elsewhere (after
        # topology is resolved) from position alone -- see phytomer_roll.py.
        phytomer_roll = safe_normalize(self.roll_head(phytomer_features), eps=1e-3)

        # Soft ceiling, not clamp(max=2.0). That clamp capped s_a at 4 cm while
        # the target (the petiole scale row, x SCALE_SCALE=50) averages 6.23 cm:
        # measured 2026-09-11, 88.6% of ground-truth phytomers sat above the
        # ceiling (83.5% at DAP 50, 98.5% at DAP 90). clamp has zero gradient
        # past its limit, so the head could not learn its way out, and since
        # denormalize_packet_scales multiplies every slot by s_a, that shrank
        # whole mature phytomers. SCALE_CEIL = 25.0 is 50 cm, well above the
        # observed maximum, and tanh keeps it saturating smoothly rather than
        # killing the gradient.
        phytomer_scale = SCALE_CEIL * torch.tanh(
            (F.softplus(self.scale_head(phytomer_features)) + 1e-4) / SCALE_CEIL)  # (B, K, 3)

        # Option B Gradient Isolation Barrier for Differentiable Rendering:
        order_raw = self.order_head(phytomer_features)              # (B, K, 2)
        phytomer_ordinal = F.softplus(order_raw[..., 0])           # >= 0, unbounded above
        phytomer_base_logits = order_raw[..., 1]

        # Render loss (depth & dice) directly trains pos_head, roll_head, and scale_head weights
        # to ground the 3D plant in drone camera space, but gradients STOP at feat_render (detached),
        # completely shielding the 4-layer Transformer decoder, self-attention, and phytomer queries
        # from rasterizer boundary noise and gradient explosions.
        feat_render = phytomer_features.detach()
        delta_pos_r = torch.tanh(self.pos_head(feat_render)) * 0.5
        phytomer_pos_render = self.ref_points[:K].detach().unsqueeze(0) + delta_pos_r

        phytomer_roll_render = safe_normalize(self.roll_head(feat_render), eps=1e-3)

        phytomer_scale_render = SCALE_CEIL * torch.tanh(
            (F.softplus(self.scale_head(feat_render)) + 1e-4) / SCALE_CEIL)

        # Re-slice the soft margin prior to the active width (computed once, full width)
        macro_out = {
            "pred_dap": macro_out_full["pred_dap"],
            "pred_num_phytomers": macro_out_full["pred_num_phytomers"],
            "soft_margin_weights": macro_out_full["soft_margin_weights"][:, :K],
        }

        # Predict phytomer existence logits, strictly bounded to [-8.0, 8.0]
        # (Numerical guarantee: BCE loss can never exceed 8.0, preventing Ext loss explosion)
        phytomer_logits = torch.tanh(self.exist_head(phytomer_features)) * 8.0

        return {
            "phytomer_pos": phytomer_pos,
            "phytomer_roll": phytomer_roll,
            "phytomer_scale": phytomer_scale,
            "phytomer_pos_render": phytomer_pos_render,
            "phytomer_roll_render": phytomer_roll_render,
            "phytomer_scale_render": phytomer_scale_render,
            "phytomer_ordinal": phytomer_ordinal,
            "phytomer_base_logits": phytomer_base_logits,
            "phytomer_logits": phytomer_logits,
            "phytomer_features": phytomer_features,
            "pred_dap": macro_out["pred_dap"],
            "pred_num_phytomers": macro_out["pred_num_phytomers"],
            "soft_margin_weights": macro_out["soft_margin_weights"],
            "active_k": K,
        }


# =============================================================================
# CANONICAL PHYTOMER ROLES FOR M=10 FINE SLOTS (v2/v3 packet contract)
# =============================================================================
ROLE_INTERNODE = 0      # Slot 0: Main/lateral stem segment
ROLE_PETIOLE = 1        # Slot 1: Leaf stalk (connects node to leaflets)
ROLE_LEAF = 2           # Slots 2, 3, 4: Trifoliate leaflets (terminal, left, right)
ROLE_PEDUNCLE = 3       # Slot 5: Inflorescence stem (flower stalk)
ROLE_REPRODUCTIVE = 4   # Slots 6..9: Flowers (closed/open), Pods, or Dormant Buds
NUM_PHYTOMER_ROLES = 5

# v3 10-slot layout (matches ROLE_SLOT_RANGES in phytomer_packets.py exactly):
# slot 0 = stem, slot 1 = petiole, slots 2..4 = 3 leaflets, slot 5 = peduncle,
# slots 6..9 = 4 reproductive (flowers/pods/buds — the 3.29% of phytomers with
# 3+ repro organs are no longer truncated).
SLOT_ROLE_MAPPING = [0, 1, 2, 2, 2, 3, 4, 4, 4, 4]
SLOT_SUB_ROLE_MAPPING = [0, 0, 0, 1, 2, 0, 0, 1, 2, 3]  # Sub-index within role


def add_level_embed(image_tokens: torch.Tensor, level_embed: torch.Tensor) -> torch.Tensor:
    """Adds a per-zoom-level embedding to level-major pyramid tokens [CLS | L x N patches] (multizoom)."""
    B, T, C = image_tokens.shape
    L = level_embed.shape[0]
    N = (T - 1) // L
    if L * N != T - 1:
        return image_tokens
    patches = image_tokens[:, 1:].view(B, L, N, C) + level_embed.view(1, L, 1, C).to(image_tokens.dtype)
    return torch.cat([image_tokens[:, :1], patches.reshape(B, L * N, C)], dim=1)


class PhytomerVisualProjector(nn.Module):
    """Hybrid 3D-to-2D Phytomer Visual Projector (PETR / Point-Query style).

    Projects 3D phytomer scaffold coordinates (B, K, 3) onto the 2D DINOv2 patch
    token grid (e.g. 16x16 = 256 patches), bilinearly samples local visual features,
    and fuses them with 3D height/depth cues via a zero-initialized residual MLP.
    """

    def __init__(self, embed_dim: int = 384, num_levels: int = 1, zooms=(1.0, 2.0, 4.0, 8.0), window: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        # window > 1: the node's local token is the mean over a window x window block of tokens around the
        # sampled point instead of one bilinear sample -- a 4-6 cm node error at 7.5 cm per token otherwise
        # reads the neighbouring patch (design doc §2.7 follow-up, 2026-09-15).
        self.window = int(window)
        # multizoom: the token grid holds num_levels plant-centred zoom levels (level-major); a node
        # samples the FINEST level that still contains it (uv scaled by the level's zoom), so a seedling
        # that is 3 px at 1x is read from the 8x crop.
        self.num_levels = int(num_levels)
        # plain attribute (not a buffer): its length differs between multizoom and single-level models and
        # must not enter the state dict, so checkpoints load across both configurations.
        self.zooms = tuple(float(z) for z in list(zooms)[:max(self.num_levels, 1)])

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

    def forward(self, image_tokens: torch.Tensor, phytomer_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_tokens: (B, 1 + N_patches, C) with N_patches = H * W (e.g. 256 for 16x16)
            phytomer_pos: (B, K, 3) 3D phytomer scaffold coordinates
        Returns:
            (B, K, C) localized visual feature embedding for each phytomer
        """
        B, K, _ = phytomer_pos.shape
        # Strip CLS token (token 0) to leave the 2D spatial patch tokens
        patch_tokens = image_tokens[:, 1:]  # (B, L * N_patches, C)
        L = self.num_levels if (self.num_levels > 1 and patch_tokens.shape[1] % self.num_levels == 0) else 1
        N_patches = patch_tokens.shape[1] // L
        grid_size = int(math.isqrt(N_patches))
        if grid_size * grid_size != N_patches:
            return torch.zeros(B, K, self.embed_dim, device=phytomer_pos.device, dtype=phytomer_pos.dtype)

        # Reshape to 2D spatial feature maps: (B*L, C, H, W)
        feat_map = patch_tokens.reshape(B * L, grid_size, grid_size, self.embed_dim).permute(0, 3, 1, 2)

        # 2D normalized grid coordinates (u, v) in [-1, 1] per level: the level-0 map scaled by the zoom
        uv0 = self.pos_to_uv(phytomer_pos.float())                                   # (B, K, 2)
        zoom = torch.tensor(self.zooms[:L], device=uv0.device, dtype=uv0.dtype).view(1, L, 1, 1)
        uv_pre = uv0.unsqueeze(1) * zoom                                              # (B, L, K, 2) crop-normalized
        uv = torch.tanh(uv_pre)
        grid = uv.reshape(B * L, 1, K, 2)

        # Bilinear sampling from each level's patch feature map (mean over a window of token offsets if window > 1)
        if self.window > 1:
            step = 2.0 / max(grid_size - 1, 1)                                        # one token in [-1, 1] units
            r = (self.window - 1) // 2
            offs = torch.tensor([(dx * step, dy * step) for dy in range(-r, r + 1) for dx in range(-r, r + 1)],
                                device=grid.device, dtype=grid.dtype)                  # (W*W, 2)
            grid_w = (grid.unsqueeze(2) + offs.view(1, 1, -1, 1, 2)).reshape(B * L, 1, K * offs.shape[0], 2)
            sampled = F.grid_sample(feat_map, grid_w, mode="bilinear", padding_mode="border", align_corners=True)
            sampled = sampled.squeeze(2).transpose(1, 2).reshape(B * L, K, offs.shape[0], self.embed_dim).mean(dim=2)
        else:
            sampled = F.grid_sample(
                feat_map,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).squeeze(2).transpose(1, 2)
        sampled = sampled.reshape(B, L, K, self.embed_dim)  # (B, L, K, C)
        if L > 1:
            inside = (uv_pre.abs() < 1.0).all(dim=-1)                                # (B, L, K): inside level l's crop
            inside[:, 0] = True                                                       # level 0 always holds the node
            lvl = (inside.float() * torch.arange(L, device=uv.device, dtype=uv.dtype).view(1, L, 1)).argmax(dim=1)  # finest inside
            sampled = sampled.gather(1, lvl.view(B, 1, K, 1).expand(-1, 1, -1, self.embed_dim)).squeeze(1)
        else:
            sampled = sampled[:, 0]

        # Fuse sampled local visual features with 3D phytomer coordinates
        fused = self.fusion(torch.cat([sampled, phytomer_pos], dim=-1))  # (B, K, C)
        return fused


class FineBotanicalFlowMatchingDecoder(nn.Module):
    """Stage 2: Flow Matching Decoder predicting microscopic organ velocity field.

    Dispatches M fine slots per active phytomer (M=8: 1 stem + 1 petiole + 3 leaflets + 1 peduncle + 2 reproductive).
    Performs joint intra-block kinematic attention within each phytomer, followed by global cross-attention.
    """

    def __init__(
        self,
        slots_per_phytomer: int = 10,
        node_dim: int = 16,  # 16D latent vector from OrganLatentVAE
        num_classes: int = NUM_ORGAN_TYPES,  # 13 organ categories
        embed_dim: int = 384,
        num_heads: int = 8,
        num_layers: int = 6,
    ):
        super().__init__()
        self.slots_per_phytomer = slots_per_phytomer
        self.node_dim = node_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Canonical Phytomer Functional Role Embeddings (Stem, Petiole, Leaf, Peduncle, Reproductive)
        self.functional_role_emb = nn.Embedding(NUM_PHYTOMER_ROLES, embed_dim)
        self.slot_pos_emb = nn.Embedding(slots_per_phytomer, embed_dim)
        self.role_emb = self.slot_pos_emb  # Backward compatibility alias

        # Hybrid 3D-to-2D Phytomer Visual Projector (Point-Query sampling); organ mode has no zoom pyramid
        self.phytomer_projector = PhytomerVisualProjector(embed_dim=embed_dim, num_levels=1)

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
        self.node_roll_mlp = nn.Sequential(
            nn.Linear(2, embed_dim),
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
        phytomer_features: torch.Tensor,
        image_tokens: torch.Tensor,
        phytomer_pos: Optional[torch.Tensor] = None,
        phytomer_roll: Optional[torch.Tensor] = None,
        existence_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            noisy_fine_nodes: (B, N_fine, node_dim) noisy geometry x_t where N_fine = K * M.
            timesteps: (B,) flow time t in [0, 1].
            phytomer_features: (B, K, embed_dim) from Stage 2 CoarseSkeletalTransformer.
            image_tokens: (B, T, embed_dim) from ViT.
            phytomer_pos: Optional (B, K, 3) predicted 3D node scaffold coordinates.
            phytomer_roll: Optional (B, K, 2) predicted node (cos, sin) roll.
            existence_mask: Optional (B, N_fine) boolean mask of active slots.
        Returns:
            Dict containing:
                'pred_velocity': (B, N_fine, node_dim) geometry velocity vector field.
                'pred_exist_logits': (B, N_fine, 1) fine slot existence logits.
        """
        B, N_fine, _ = noisy_fine_nodes.shape
        K = phytomer_features.shape[1]
        M = self.slots_per_phytomer
        assert N_fine == K * M, f"Fine node count ({N_fine}) must equal K*M ({K}*{M})"
        device = noisy_fine_nodes.device

        # 1. Expand phytomer features across their M children
        # (B, K, D) -> (B, K, 1, D) -> (B, K, M, D) -> (B, N_fine, D)
        expanded_phytomers = phytomer_features.unsqueeze(2).expand(-1, -1, M, -1).reshape(B, N_fine, self.embed_dim)

        # 2. Stage 2 -> 3: 3D Node Scaffold Positional, Visual Projector & Rotational Condition Embeddings
        if phytomer_pos is not None:
            node_pos_emb = self.node_pos_mlp(phytomer_pos).unsqueeze(2).expand(-1, -1, M, -1).reshape(B, N_fine, self.embed_dim)
            proj_feat = self.phytomer_projector(image_tokens, phytomer_pos)  # (B, K, C)
            proj_emb = proj_feat.unsqueeze(2).expand(-1, -1, M, -1).reshape(B, N_fine, self.embed_dim)
        else:
            node_pos_emb = 0.0
            proj_emb = 0.0

        if phytomer_roll is not None:
            node_rot_emb = self.node_roll_mlp(phytomer_roll).unsqueeze(2).expand(-1, -1, M, -1).reshape(B, N_fine, self.embed_dim)
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

        # Composite query representation (Conditioned on 3D Scaffold + Phytomer Latents + Roles + Time + Hybrid Projector)
        queries = geom_embs + expanded_phytomers + node_pos_emb + proj_emb + node_rot_emb + role_embs + t_emb

        # 6. Intra-Phytomer Block Multi-Head Self-Attention:
        # Reshape to (B * K, M, D) so local organs directly communicate joint kinematics
        q_block = queries.view(B * K, M, self.embed_dim)
        q_norm = self.block_norm(q_block)
        block_attn_out, _ = self.block_self_attn(q_norm, q_norm, q_norm)
        queries = (q_block + block_attn_out).view(B, N_fine, self.embed_dim)

        # 7. Global Decoder cross-attention to image tokens & phytomer memory
        # Concatenate image tokens with phytomer features as memory for full multi-scale conditioning
        memory = torch.cat([image_tokens, phytomer_features], dim=1)

        x = self.decoder(queries, memory)

        # 8. Heads
        pred_velocity = self.velocity_head(x)
        pred_exist_logits = torch.tanh(self.exist_head(x)) * 8.0

        return {
            "pred_velocity": pred_velocity,
            "pred_exist_logits": pred_exist_logits,
        }


class PhytomerFlowMatchingDecoder(nn.Module):
    """Stage 3 (phytomer mode): flow-matches base + rot + latent per phytomer.

    Unlike FineBotanicalFlowMatchingDecoder, which dispatches M organ slots per
    phytomer (and uses pose only as conditioning), this decoder flow-matches ONE
    12+D vector per phytomer:
        z_1 = [ node_base_xyz(3) | node_rot_6d(6) | phytomer_latent(D) ]
    so Stage 2's 3D node scaffold (base + rot) is REFINED by flow, not fixed.
    The refined node_rot is also the reference frame for the phytomer-relative
    packet (Option 1). Existence stays a separate gating head (per-slot, Kx8);
    it is never flow-matched (a 0/1 gate regresses to its mean under velocity
    matching).

    Tokens: K phytomers (vs M*K slots) -> Mx fewer, O(K^2) self-attn -> M^2x cheaper.
    No intra-block (M) self-attention: the joint latent is per-phytomer.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        base_dim: int = 3,
        rot_dim: int = 2,
        scale_dim: int = 3,
        num_classes: int = NUM_ORGAN_TYPES,
        embed_dim: int = 384,
        num_heads: int = 8,
        num_layers: int = 6,
        slots_per_phytomer: int = 10,
        num_levels: int = 1,
        node_token_window: int = 1,
    ):
        super().__init__()
        self.num_levels = int(num_levels)
        self.node_token_window = int(node_token_window)
        if self.num_levels > 1:
            self.level_embed = nn.Parameter(torch.zeros(self.num_levels, embed_dim))   # multizoom token identity
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

        # Decoder: cross-attention query -> image tokens + phytomer memory
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
        self.node_roll_mlp = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.node_scale_mlp = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # (parent, self) conditioning (2026-09-12 redesign, docs/ongoing/20260912_* §2.1):
        # the node's parent -- held FIXED at GT during training, with noise, and
        # taken from the resolved chain at inference -- enters as
        # [parent - self (3, metres), has_parent (1)]. The output layer starts at
        # zero so checkpoints trained without it are unchanged until trained.
        # FM_PARENT_COND=0 builds the model WITHOUT this module (identical
        # parameter set to checkpoints from before 2026-09-12 evening), which an
        # exact resume -- optimizer moments included -- of such a checkpoint needs.
        if os.environ.get("FM_PARENT_COND", "1") == "1":
            self.node_parent_mlp = nn.Sequential(
                nn.Linear(4, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            nn.init.zeros_(self.node_parent_mlp[2].weight)
            nn.init.zeros_(self.node_parent_mlp[2].bias)
        else:
            self.node_parent_mlp = None

        # Hybrid 3D-to-2D Phytomer Visual Projector (Point-Query sampling)
        self.phytomer_projector = PhytomerVisualProjector(embed_dim=embed_dim, num_levels=self.num_levels, window=self.node_token_window)

        # Velocity head: predicts d/dt of the flow vector (node_flow_dim = 12 + latent_dim)
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
        phytomer_features: torch.Tensor,
        image_tokens: torch.Tensor,
        phytomer_pos: Optional[torch.Tensor] = None,
        phytomer_roll: Optional[torch.Tensor] = None,
        phytomer_scale: Optional[torch.Tensor] = None,
        phytomer_parent_rel: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Flow-matches a per-phytomer latent vector (node_flow_dim-D, pure
        VAE latent in the currently wired Hybrid Decoupled configuration --
        base_dim=rot_dim=scale_dim=0 at construction, see
        HierarchicalPartFlowMatchingModel.__init__).
        With multizoom the image tokens are the level-major pyramid and get this stage's own
        level embedding before the projector and the cross-attention memory read them.

        Args:
            noisy_flow: (B, K, node_flow_dim) interpolated x_t.
            timesteps: (B,) flow time t in [0, 1].
            phytomer_features: (B, K, embed) from Stage 2 CoarseSkeletalTransformer.
            image_tokens: (B, T, embed) from ViT.
            phytomer_pos: Optional (B, K, 3) Stage-2 scaffold base positions.
            phytomer_roll: Optional (B, K, 2) Stage-2 scaffold (cos, sin) roll.
            phytomer_scale: Optional (B, K, 3) Stage-2 scaffold scales.
            phytomer_parent_rel: Optional (B, K, 4) [parent - self (metres), has_parent].
        Returns:
            'pred_velocity': (B, K, node_flow_dim) velocity field.
            'pred_slot_exist_logits': (B, K, M) per-slot existence logits.
        """
        if self.num_levels > 1:
            image_tokens = add_level_embed(image_tokens, self.level_embed)
        B, K, _ = noisy_flow.shape
        device = noisy_flow.device

        # Continuous flow vector projection + sinusoidal timestep embedding
        geom_embs = self.geom_proj(noisy_flow)  # (B, K, embed)
        t_emb = self.time_embed(self._sinusoidal(timesteps)).unsqueeze(1)  # (B, 1, embed)
        queries = geom_embs + t_emb

        # Stage 2 3D Scaffold Conditioning
        if phytomer_pos is not None:
            queries = queries + self.node_pos_mlp(phytomer_pos)
            # Hybrid Projection: inject local 2D-projected visual token into query
            queries = queries + self.phytomer_projector(image_tokens, phytomer_pos)
        if phytomer_roll is not None:
            queries = queries + self.node_roll_mlp(phytomer_roll)
        if phytomer_scale is not None:
            queries = queries + self.node_scale_mlp(phytomer_scale)
        if phytomer_parent_rel is not None and self.node_parent_mlp is not None:
            queries = queries + self.node_parent_mlp(phytomer_parent_rel)

        # Global decoder cross-attention to image tokens + phytomer memory
        memory = torch.cat([image_tokens, phytomer_features], dim=1)
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
    Stage 2: Coarse 3D Node Point Cloud Scaffold (Phytomers along shoot axis).
    Stage 3: Intra-Phytomer Canonical Flow Matching (Organ geometry relative to node).
    """

    def __init__(
        self,
        max_phytomers: int = 512,
        slots_per_phytomer: int = 10,
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
        phytomer_latent_dim: int = 128,
        backbone: str = "dinov2_vits14",
        freeze_backbone: bool = False,
        init_phytomer_count: float = 50.0,
        stage3_geometry: bool = False,
        multizoom: bool = False,
        node_token_window: int = 1,
    ):
        super().__init__()
        self.stage3_geometry = bool(stage3_geometry)
        self.node_token_window = int(node_token_window)
        # multizoom (design doc §2.7): all four cache zoom levels (1x/2x/4x/8x, plant-centred) go through
        # the frozen backbone; Stage 2/3 attend over all levels' tokens (with a learned level embedding)
        # and each node's local token is read from the finest level containing it.
        self.multizoom = bool(multizoom)
        self.num_levels = 4 if self.multizoom else 1
        self.max_phytomers = max_phytomers
        self.slots_per_phytomer = slots_per_phytomer
        self.max_fine_slots = max_phytomers * slots_per_phytomer  # 512 * 10 = 5,120
        self.node_dim = node_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.flow_granularity = flow_granularity
        self.phytomer_latent_dim = phytomer_latent_dim
        # --latent_norm: the flow matches (z - mu) / sigma of the VAE latent instead of z, so every
        # latent dim is O(1) against the unit noise (the raw 128D latent has per-dim std ~0.4, which
        # left the velocity loss ~85% noise prediction and the conditioning unused -- 2026-09-15).
        # Identity (mu 0, sigma 1) unless set_latent_norm() is called; saved with the state dict.
        self.register_buffer("latent_mu", torch.zeros(phytomer_latent_dim))
        self.register_buffer("latent_sigma", torch.ones(phytomer_latent_dim))

        # 1. Pretrained DINO Backbone with 3D Camera Ray Positional Embedding (PETR style).
        #    Swappable via `backbone` for scaling A/B (dinov2_vits14 / vitb14 / vitl14 /
        #    dinov3_vits16 / vitb16 / vitl16 / vitl16_sat).
        self.image_encoder = DINOv2RayEncoder(
            backbone=backbone,
            pretrained=True,
            freeze_backbone=freeze_backbone,
            embed_dim=embed_dim,
            num_levels=self.num_levels,
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
            max_phytomers=max_phytomers,
            embed_dim=embed_dim,
            num_levels=self.num_levels,
            node_token_window=self.node_token_window,
            num_heads=vit_heads,
            num_layers=coarse_layers,
            init_phytomer_count=init_phytomer_count,
        )

        # 3. Stage 3: Fine Botanical Flow Matching Decoder conditioned on 3D Scaffold.
        #    flow_granularity:
        #      'organ'   - per-organ slots (8K x node_dim), pose as conditioning.
        #      'phytomer'- per-phytomer hybrid (coarse+residual) VAE latent flow vector
        #                  (Hybrid Decoupled Architecture: pure standard Gaussian prior
        #                  z_0 ~ N(0, I_D), pose/scale as conditioning).
        if flow_granularity == "phytomer":
            # stage3_geometry: the child's position (relative to its fixed parent),
            # roll and scale ride in the flow state with the latent (see
            # split_flow_state); otherwise the state is the pure VAE latent.
            self.fine_stage = PhytomerFlowMatchingDecoder(
                latent_dim=phytomer_latent_dim,
                num_levels=self.num_levels,
                node_token_window=self.node_token_window,
                base_dim=3 if self.stage3_geometry else 0,
                rot_dim=2 if self.stage3_geometry else 0,
                scale_dim=3 if self.stage3_geometry else 0,
                num_classes=num_classes,
                embed_dim=embed_dim,
                num_heads=vit_heads,
                num_layers=fine_layers,
                slots_per_phytomer=slots_per_phytomer,
            )
        else:
            self.fine_stage = FineBotanicalFlowMatchingDecoder(
                slots_per_phytomer=slots_per_phytomer,
                node_dim=node_dim,
                num_classes=num_classes,
                embed_dim=embed_dim,
                num_heads=vit_heads,
                num_layers=fine_layers,
            )
        # Width of the per-phytomer flow state Stage 3 integrates.
        self.flow_dim = (self.fine_stage.node_flow_dim if flow_granularity == "phytomer" else node_dim)

    def probe_pred_dap(self, image_tokens: torch.Tensor) -> torch.Tensor:
        """Two-pass helper: reads Stage 1's predicted DAP from the CLS token (no grad
        needed for the clue itself — the clue is an input, loss_dap keeps the head
        supervised)."""
        cls_token = image_tokens[:, 0]
        with torch.no_grad():
            return self.coarse_stage.macro_head(cls_token, max_k=self.max_phytomers)["pred_dap"]

    def forward(
        self,
        noisy_fine_nodes: torch.Tensor,
        timesteps: torch.Tensor,
        images: torch.Tensor,
        daps: Optional[torch.Tensor] = None,
        teacher_phytomer_pos: Optional[torch.Tensor] = None,
        num_phytomers: Optional[torch.Tensor] = None,
        capacity_mode: str = "given",
        image_tokens: Optional[torch.Tensor] = None,
        phytomer_parent_pos: Optional[torch.Tensor] = None,
        phytomer_has_parent: Optional[torch.Tensor] = None,
        stage3_cond_override: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Unified joint forward pass across 3 cascaded stages.

        stage3_cond_override: optional {"pos": (B,K,3), "roll": (B,K,2), "scale": (B,K,3)}
            used INSTEAD of Stage 2's outputs as Stage 3's conditioning (teacher
            forcing with ground-truth nodes, --stage3_gt_nodes): measures what the
            latent path can do when the geometry it is conditioned on is right.
            Stage 2's own outputs and losses are unaffected.

        Args:
            noisy_fine_nodes: (B, N_fine, node_dim) interpolated geometry x_t.
            timesteps: (B,) flow time t in [0, 1].
            images: (B, 4, H, W) or (B, 16, H, W) drone RGB+CHM imagery.
            daps: Optional (B,) plant age tensor for Matryoshka slicing (teacher forcing).
            teacher_phytomer_pos: Optional (B, K, 3) ground-truth phytomer positions for teacher forcing.
            num_phytomers: Optional (B,) ground-truth phytomer count for Matryoshka slicing.
            capacity_mode:
                'given'      - Stage 2 slices the phytomer bank from (GT) DAP/phytomer hints
                               via compute_matryoshka_slice. Teacher-forcing training path.
                'pred_phyto' - Stage 2 slices from the predicted phytomer count (two-pass).
                               Inference / self-conditioning path.
            phytomer_parent_pos / phytomer_has_parent: Optional (B, K, 3) / (B, K) the
                fixed parent position of each predicted node (GT parent with noise
                during training; the resolved chain's parent at inference) and
                whether it has one. Stage 3 sees [parent - self, has_parent].
        """
        # 1. Vision perception: 3D-aware multi-scale tokens.
        # The caller may pass precomputed tokens (probe reuse: the training loop
        # already ran the encoder once for capacity slicing — passing them in
        # saves one full DINOv2 forward per step).
        if image_tokens is None:
            image_tokens = self.image_encoder(images)

        # 2. Phytomer bank slicing. The predicted DAP is passed as a stage clue so
        #    Stage 2/3 condition on the developmental stage (dap_embed); in the
        #    GT-DAP teacher-forcing path the GT value substitutes for reliability.
        if capacity_mode == "given":
            active_k = compute_matryoshka_slice(dap=daps, num_phytomers=num_phytomers, max_phytomers=self.max_phytomers)
            clue_dap = daps  # GT DAP as the clue during teacher forcing
            coarse_out = self.coarse_stage(image_tokens, active_k=active_k, pred_dap=clue_dap)
        else:
            clue = self.probe_pred_dap(image_tokens)
            coarse_out = self.coarse_stage(image_tokens, capacity_mode=capacity_mode, pred_dap=clue)

        phytomer_features = coarse_out["phytomer_features"]
        phytomer_pos = coarse_out["phytomer_pos"]
        phytomer_roll = coarse_out["phytomer_roll"]
        active_k = int(coarse_out["active_k"])

        # If noisy nodes exceed active fine slots, slice accordingly; if fewer, pad
        # (the phytomer bank slice is authoritative — K_out drives Stage 3's width).
        # In phytomer mode, noisy_fine_nodes is already (B, K, 12+D) — no 8K slicing.
        if self.flow_granularity == "phytomer":
            if noisy_fine_nodes.shape[1] > active_k:
                noisy_fine_nodes = noisy_fine_nodes[:, :active_k]
            elif noisy_fine_nodes.shape[1] < active_k:
                pad_n = active_k - noisy_fine_nodes.shape[1]
                noisy_fine_nodes = F.pad(noisy_fine_nodes, (0, 0, 0, pad_n))
        else:
            active_fine = active_k * self.slots_per_phytomer
            if noisy_fine_nodes.shape[1] > active_fine:
                noisy_fine_nodes = noisy_fine_nodes[:, :active_fine]
            elif noisy_fine_nodes.shape[1] < active_fine:
                pad_n = active_fine - noisy_fine_nodes.shape[1]
                noisy_fine_nodes = F.pad(noisy_fine_nodes, (0, 0, 0, pad_n))

        # 4. Stage 3: Fine Botanical Flow Matching conditioned on 3D Scaffold
        # phytomer_features, phytomer_pos, phytomer_roll, and phytomer_scale are detached as conditioning inputs so the velocity
        # loss refines latent geometry without backpropagating into Stage-2 scaffold heads or features.
        phytomer_pos = coarse_out["phytomer_pos"]
        phytomer_roll = coarse_out["phytomer_roll"]
        phytomer_scale = coarse_out.get("phytomer_scale")
        if stage3_cond_override is not None:
            phytomer_pos = stage3_cond_override.get("pos", phytomer_pos)
            phytomer_roll = stage3_cond_override.get("roll", phytomer_roll)
            phytomer_scale = stage3_cond_override.get("scale", phytomer_scale)
        if self.flow_granularity == "phytomer":
            fine_out = self.fine_stage(
                noisy_flow=noisy_fine_nodes,
                timesteps=timesteps,
                phytomer_features=phytomer_features.detach(),
                image_tokens=image_tokens,
                phytomer_pos=phytomer_pos.detach() if phytomer_pos is not None else None,
                phytomer_roll=phytomer_roll.detach() if phytomer_roll is not None else None,
                phytomer_scale=phytomer_scale.detach() if phytomer_scale is not None else None,
                phytomer_parent_rel=parent_relative(
                    phytomer_pos.detach(), phytomer_parent_pos, phytomer_has_parent)
                if (phytomer_pos is not None and phytomer_parent_pos is not None) else None,
            )
            pred_velocity = fine_out["pred_velocity"]
        else:
            fine_out = self.fine_stage(
                noisy_fine_nodes=noisy_fine_nodes,
                timesteps=timesteps,
                phytomer_features=phytomer_features.detach(),
                image_tokens=image_tokens,
                phytomer_pos=phytomer_pos.detach() if phytomer_pos is not None else None,
                phytomer_roll=phytomer_roll.detach() if phytomer_roll is not None else None,
            )
            pred_velocity = fine_out["pred_velocity"]

        return {
            # Stage 1: Macro Biological Prior
            "pred_dap": coarse_out["pred_dap"],
            "pred_num_phytomers": coarse_out["pred_num_phytomers"],
            "soft_margin_weights": coarse_out["soft_margin_weights"],
            # Stage 2: 3D Node Point Cloud Scaffold
            "pred_phytomer_pos": coarse_out["phytomer_pos"],
            "pred_phytomer_roll": coarse_out["phytomer_roll"],
            "pred_phytomer_scale": coarse_out["phytomer_scale"],
            "pred_phytomer_pos_render": coarse_out.get("phytomer_pos_render", coarse_out["phytomer_pos"]),
            "pred_phytomer_roll_render": coarse_out.get("phytomer_roll_render", coarse_out["phytomer_roll"]),
            "pred_phytomer_scale_render": coarse_out.get("phytomer_scale_render", coarse_out["phytomer_scale"]),
            "pred_phytomer_ordinal": coarse_out["phytomer_ordinal"],
            "pred_phytomer_base_logits": coarse_out["phytomer_base_logits"],
            "pred_phytomer_logits": coarse_out["phytomer_logits"],
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
    def set_latent_norm(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        # sigma floor 0.05: the VAE has near-constant (KL-collapsed) dims with sigma ~1e-3 (smoke 2026-09-15:
        # min 0.001, mean 0.18, max 1.44); dividing by their true sigma would turn their noise into O(1)
        # targets the flow must fit for nothing. Floored, they stay ~0 in the flow state and ~mu on decode.
        self.latent_mu.copy_(mu.to(self.latent_mu)); self.latent_sigma.copy_(sigma.to(self.latent_sigma).clamp(min=0.05))

    def latent_norm_active(self) -> bool:
        return bool((self.latent_sigma != 1).any() or (self.latent_mu != 0).any())

    def normalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        """VAE latent -> flow-state latent block."""
        return (z - self.latent_mu.to(z)) / self.latent_sigma.to(z)

    def denormalize_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Flow-state latent block -> VAE latent (what the VAE decoder and every consumer expects)."""
        return self.latent_mu.to(x) + self.latent_sigma.to(x) * x

    def sample_ode(
        self,
        images: torch.Tensor,
        daps: Optional[torch.Tensor] = None,
        num_steps: int = 20,
        guidance_scale: float = 1.0,
        vae: Optional[nn.Module] = None,
        phytomer_vae: Optional[nn.Module] = None,
        cond_override: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """2nd-Order Heun Predictor-Corrector ODE Sampling.

        cond_override: {"pos", "roll", "scale"} (B, K, ·) replacing Stage 2's node
        outputs as Stage 3's conditioning AND as the plant's node geometry
        (teacher forcing, see forward's stage3_cond_override).

        Generates full 3D plant organ array from an input condition image.
        Integrates velocity field in 16D regularized latent space.
        Takes <50ms end-to-end on GPU.

        flow_granularity='phytomer': integrates the (B, K, 12+D) per-phytomer vector
        [base | rot | latent]; the refined phytomer rot is the reference frame for
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
            k_dap = compute_matryoshka_slice(dap=daps, max_phytomers=self.max_phytomers)
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

        phytomer_pos = coarse_out["phytomer_pos"]      # (B, K, 3)
        phytomer_roll = coarse_out["phytomer_roll"]    # (B, K, 2)
        if cond_override is not None:
            phytomer_pos = cond_override.get("pos", phytomer_pos)
            phytomer_roll = cond_override.get("roll", phytomer_roll)
            if cond_override.get("scale") is not None:
                coarse_out = dict(coarse_out); coarse_out["phytomer_scale"] = cond_override["scale"]
        phytomer_ordinal = coarse_out["phytomer_ordinal"]        # (B, K)
        phytomer_base_logits = coarse_out["phytomer_base_logits"]  # (B, K)
        phytomer_features = coarse_out["phytomer_features"]
        soft_margin_weights = coarse_out["soft_margin_weights"]
        phytomer_existence = torch.sigmoid(coarse_out["phytomer_logits"]).squeeze(-1) * soft_margin_weights  # (B, K)

        # 2. Construct Prior x_0.
        #    organ mode: standard Gaussian (B, K*8, 16) — exact legacy behavior.
        #    phytomer mode: standard Gaussian (B, K, D) — pure hybrid VAE latent flow (Hybrid Decoupled).
        M = self.slots_per_phytomer
        if self.flow_granularity == "phytomer":
            D = self.phytomer_latent_dim
            N_fine = active_k  # K phytomers
            x = torch.randn(B, active_k, self.flow_dim, device=device)
        else:
            N_fine = active_k * M
            x = torch.randn(B, N_fine, self.node_dim, device=device)

        # 3. 2nd-Order Heun ODE Integration from t=0 to t=1 (Stage 3 Flow Matching)
        dt = 1.0 / num_steps
        phytomer_scale = coarse_out.get("phytomer_scale")
        forward_kwargs = dict(
            phytomer_features=phytomer_features,
            image_tokens=image_tokens,
            phytomer_pos=phytomer_pos,
            phytomer_roll=phytomer_roll,
        )
        if self.flow_granularity == "phytomer":
            forward_kwargs["phytomer_scale"] = phytomer_scale
            # (parent, self) conditioning at inference: the parent is the
            # resolved chain's, over the live slots only -- the same call that
            # derives the node rotation after the ODE, run once up front.
            _, chain_parent_pos = reconstruct_phytomer_rot(
                phytomer_pos, phytomer_roll, phytomer_ordinal, phytomer_base_logits,
                exist=phytomer_existence)
            if self.stage3_geometry:
                # Training conditions the root on the ORIGIN as its parent
                # (gt_parent_links), so do the same here instead of the
                # parentless [0, 0, 0, 0] row.
                chain_parent_pos = torch.nan_to_num(chain_parent_pos, nan=0.0)
                live = (phytomer_existence > 0.5).float()
                forward_kwargs["phytomer_parent_rel"] = parent_relative(phytomer_pos, chain_parent_pos, live)
            else:
                forward_kwargs["phytomer_parent_rel"] = parent_relative(phytomer_pos, chain_parent_pos, None)
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
            combined_prob = pred_slot_exist * phytomer_existence.unsqueeze(-1)    # (B, K, M)
            slot_active = (combined_prob > 0.35).float().reshape(B, active_k * M)
            geom_block, pred_latent = split_flow_state(x, D)   # (B, K, D) latent flow vector (caller reference)
            pred_fine_exist_logits = pred_slot_exist_logits
            N_fine = active_k * M    # flat slot surface for legacy callers
            stage2_pos, stage2_roll, stage2_scale = phytomer_pos, phytomer_roll, phytomer_scale
            if geom_block is not None:
                # Stage 3 refined the child against its (chain) parent: these are
                # the node position / roll / scale the plant is built from now.
                phytomer_pos, phytomer_roll, phytomer_scale = geometry_from_flow(geom_block, chain_parent_pos)
        else:
            final_out = self.fine_stage(
                noisy_fine_nodes=x, timesteps=torch.ones((B,), device=device), **forward_kwargs)
            pred_fine_exist_logits = final_out["pred_exist_logits"]  # (B, N_fine, 1)
            fine_slot_prob = torch.sigmoid(pred_fine_exist_logits).squeeze(-1)  # (B, N_fine)

            # Compute combined existence confidence (phytomer_existence * fine_slot_existence)
            expanded_phytomer_exist = phytomer_existence.unsqueeze(2).expand(-1, -1, M).reshape(B, N_fine)
            combined_prob = fine_slot_prob * expanded_phytomer_exist  # (B, N_fine)

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

        # Full node rotation is DERIVED here (forward axis from resolved
        # topology + position, roll from Stage 2's head) rather than predicted
        # directly -- see phytomer_roll.py. Valid post-ODE since position and
        # ordinal are Stage-2-only outputs, never touched by Stage 3's ODE in
        # this Hybrid Decoupled config (only the VAE latent is flow-matched).
        if self.flow_granularity == "phytomer":
            # Chain only the live slots: stray non-existent slots otherwise
            # become "roots" below the plant and draw long internodes (see
            # reconstruct_phytomer_rot's `exist` note).
            phytomer_rot, phytomer_parent_pos = reconstruct_phytomer_rot(
                phytomer_pos, phytomer_roll, phytomer_ordinal, phytomer_base_logits,
                exist=phytomer_existence)
        else:
            phytomer_rot, phytomer_parent_pos = None, None

        res = {
            # raw VAE latent (denormalized under --latent_norm) -- what every consumer decodes
            "pred_latent": (self.denormalize_latent(pred_latent) if self.flow_granularity == "phytomer" else pred_latent),
            "pred_fine_exist_logits": pred_fine_exist_logits,
            "slot_active": slot_active,
            "phytomer_pos": phytomer_pos,
            "phytomer_roll": phytomer_roll,
            "phytomer_rot": phytomer_rot,
            "phytomer_scale": (phytomer_scale if self.flow_granularity == "phytomer" else coarse_out.get("phytomer_scale")),
            "phytomer_pos_stage2": (stage2_pos if self.flow_granularity == "phytomer" else phytomer_pos),
            "phytomer_existence": phytomer_existence,
            "pred_num_phytomers": coarse_out["pred_num_phytomers"],
            "pred_dap": coarse_out["pred_dap"],
            "soft_margin_weights": soft_margin_weights,
            "active_fine_count": N_fine,
            "active_k": active_k,
            "flow_granularity": self.flow_granularity,
        }

        if self.flow_granularity == "phytomer":
            # The VAE latent block of the flow state (the whole state without
            # stage3_geometry; its last D dims with it -- see split_flow_state).
            latent = self.denormalize_latent(pred_latent)   # identity unless --latent_norm
            res["refined_phytomer_pos"] = phytomer_pos
            res["refined_phytomer_rot"] = phytomer_rot
            res["refined_phytomer_scale"] = phytomer_scale
            res["phytomer_latent"] = latent
            res["pred_slot_exist_logits"] = final_out["pred_slot_exist_logits"]
            if phytomer_vae is not None:
                out = phytomer_vae.decode(latent.reshape(-1, self.phytomer_latent_dim))  # (B*K, M, 26)
                probs = F.softmax(out["cls_logits"], dim=-1)  # (B*K, M, 13)
                pred_cls = out["cls_logits"].argmax(-1)
                packet_hat = out["recon_packets"].reshape(B, active_k, M, 26)
                # v3: the decoded scales are NORMALIZED (VAE trained on normalized
                # packets); restore absolute scale with phytomer scale
                packet_hat = denormalize_packet_scales(
                    packet_hat.reshape(-1, M, 26),
                    phytomer_scale.reshape(-1, 3) if phytomer_scale is not None else torch.ones((B*active_k, 3), device=device),
                ).reshape(B, active_k, M, 26)
                # Structurally assemble slot bases from the petiole geometry
                # (deterministic), then place into the world frame with the phytomer pose.
                # The internode is the parent->node segment of the resolved chain
                # (the training render block already does this); the decoded
                # slot-0 length is only a fallback and was drawing metre-long
                # stems on seedlings when used here.
                if os.environ.get("FM_ASSEMBLY_PROBE") == "1":
                    # Which predicted nodes even exist, and where. Stray nodes
                    # (K > true count) that survive into the chain drag a long
                    # internode with them; this shows whether that is happening.
                    _ex = phytomer_existence
                    _p0 = phytomer_pos[0]
                    _pp0 = phytomer_parent_pos[0]
                    print(f"  [AssemblyProbe] batch0: active_k={active_k}  "
                          f"exist_prob={[round(float(v), 2) for v in _ex[0, :12].reshape(-1)]}",
                          flush=True)
                    for r in range(min(_p0.shape[0], 12)):
                        print(f"  [AssemblyProbe] node {r:2d} pos_cm="
                              f"({float(_p0[r, 0])*100:6.1f},{float(_p0[r, 1])*100:6.1f},{float(_p0[r, 2])*100:6.1f}) "
                              f"parent_cm=({float(_pp0[r, 0])*100:6.1f},{float(_pp0[r, 1])*100:6.1f},{float(_pp0[r, 2])*100:6.1f})",
                              flush=True)
                packet_hat = assemble_packets(
                    packet_hat.reshape(-1, M, 26),
                    phytomer_rot.reshape(-1, 6),
                    parent_pos=phytomer_parent_pos.reshape(-1, 3),
                    centers=phytomer_pos.reshape(-1, 3),
                ).reshape(B, active_k, M, 26)
                abs_packets = apply_ref_for_flow(packet_hat, phytomer_pos, phytomer_rot)
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
