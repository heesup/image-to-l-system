"""
Canonical phytomer packet builder (shared util).

Groups per-organ 26D FM nodes into fixed 8-slot phytomer packets with canonical
role ordering, for phytomer-level latent modeling (PhytomerVAE) and anchor-only
flow supervision.

Packet layout (matches HierarchicalBotanicalMatcher role semantics exactly):
    slot 0:     Stem / Internode        (organ types 1..3, at most 1)
    slot 1:     Petiole                 (organ type 4, at most 1)
    slots 2-4:  Trifoliate leaflets     (organ type 5, at most 3)
    slot 5:     Peduncle                (organ type 6, at most 1)
    slots 6-7:  Reproductive            (organ types 7..12, at most 2)

Clustering rule is IDENTICAL to HierarchicalBotanicalMatcher:
petiole bases define cluster centers (+ standalone internodes >4cm away,
seedling fallback to internodes). Within each role, organs take slots in
canonical order (bottom -> top by z, then azimuth), mirroring
canonical_sort_nodes. Roles that overflow drop extras (measured 2026-09-09 on
cowpea_curv26, organ-level drop: stem 23.6%, petiole 1.8%, leaflet 29.8%,
repro 15.3%, 21.1% overall — overwhelmingly neighbor-phytomer organs
misassigned by nearest-center on dense canopies; identical truncation to
the current 8-slot anchors, so no supervision regression vs status quo).

Positions in packets are ANCHOR-RELATIVE (organ base - cluster center, same
BASE_SCALE units): the anchor/node position carries global placement, so the
latent models translation-invariant local morphology. decode_packet() re-adds
the center. Absent slots encode as NONE one-hot + zero geometry (the encode_fm
empty convention) with presence bit 0.

ROTATIONS ARE ANCHOR-RELATIVE (2026-09-09): each organ's absolute world-frame
6D rotation was a large co-variance the shared latent had to absorb (a leaf at
azimuth 90 vs 270 has entirely different absolute 6D). Packets now store
rot6d RELATIVE to the phytomer reference frame = the internode slot's rotation
(slot 0; fallback = first present slot, else identity). Rotation is the
reference-inverse product R_ref^T @ R_org, so slot 0 encodes ~identity and the
remaining 7 slots encode pose DIFFERENCE. decode_packet() re-applies the
reference: R_org = R_ref @ R_rel. This is a pure frame change — exact
roundtrip (numerically verified), but it dramatically reduces the per-packet
rotation variance, freeing 64D latent capacity for the affine morphology.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from diffusion_based.dataset.part_array_dataset import (
    EMPTY_IDX,
    FM_OT_END,
    FM_BASE_START,
    FM_BASE_END,
    FM_NODE_DIM,
    NUM_ORGAN_TYPES,
    SCALE_SCALE,
    FM_CURV,
    CURV_SCALE,
)

# ROT6D indices within the 26D FM row.
FM_ROT_START = FM_BASE_END      # 16
FM_ROT_END = FM_ROT_START + 6   # 22
FM_SCALE_START = FM_ROT_END     # 22
FM_SCALE_END = FM_SCALE_START + 3  # 25

# STRUCTURAL ASSEMBLY (2026-09-09, XML-phytomer re-verification):
# With EXACT XML phytomer membership (cache field `phytomer_ids`), every slot's
# base is DETERMINISTIC (verified over the full dataset):
#   slot 0 stem (ROOT_META/SHOOT_META):  base = center
#   slot 1 petiole:                      base = center (p99 = 0.000cm)
#   slots 2-3 lateral leaflets:          base = 0.8 x petiole CURVE (arc frac)
#   slot 4 terminal leaflet:             base = 1.0 x petiole CURVE (tip)
#   slot 5 peduncle:                     base = center
#   slot 6/7 BUD (dormant/active/aborted, types 7,8,12): base = center
#
# NOT deterministic (VAE must learn their base):
#   slots 6/7 flowers & fruit (types 9-11): attach at the CURVED peduncle tip
#     with per-plant flower_offset (rare, ~3% of packets) — VAE-learned.
#
# ASSEMBLY_TYPE: base == center (zeroed) or petiole-curve point (computed),
# else base is VAE-learned.
DETERMINISTIC_ORGAN_TYPES = {1, 2, 3, 4, 6, 7, 8, 12}  # stem, petiole, peduncle, buds
# Leaflet attach arc-fractions along the petiole curve (XML-phytomer verified:
# slot 2/3 = 0.800, slot 4 = 0.991-1.000, error <= 0.005).
LEAFLET_ATTACH_FRAC = {2: 0.8, 3: 0.8, 4: 1.0}


def _petiole_curve_points(
    R_pet_world: torch.Tensor,
    pet_len: float,
    pet_curv_deg_m: float,
    n_seg: int = 6,
) -> torch.Tensor:
    """Curved petiole centerline (renderer convention: gravitropic bend about
    the horizontal axis perpendicular to the petiole's vertical plane).

    Matches the geometry builder's tube path (n_seg_curv=6, Rodrigues rotation
    about cross(cur_axis, z_world)). Returns (n_seg+1, 3) points starting at
    the petiole base (origin, relative to the cluster center).
    """
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=R_pet_world.device)
    dr = pet_len / n_seg
    cur_axis = R_pet_world[:, 1].clone()
    pts = [torch.zeros(3, device=R_pet_world.device)]
    for _ in range(n_seg):
        h = torch.linalg.cross(cur_axis, z_axis)
        hn = h.norm()
        h_u = h / (hn + 1e-8) if hn > 1e-4 else torch.tensor([1.0, 0.0, 0.0], device=R_pet_world.device)
        theta = torch.deg2rad(pet_curv_deg_m) * dr
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        kcv = torch.linalg.cross(h_u, cur_axis)
        kdv = (h_u * cur_axis).sum()
        rot_v = cur_axis * cos_t + kcv * sin_t + h_u * kdv * (1 - cos_t)
        rot_v = rot_v / (rot_v.norm() + 1e-8)
        pts.append(pts[-1] + rot_v * dr)
        cur_axis = rot_v
    return torch.stack(pts)


def strip_base(packets: torch.Tensor) -> torch.Tensor:
    """Zeroes the base columns of DETERMINISTIC slots (structural assembly).

    Stem/petiole/peduncle/bud bases are exactly the cluster center and leaflet
    bases are exactly 0.8/1.0 x the petiole curve (XML-phytomer verified), so
    the latent never needs to predict them. Only flower/fruit bases (rare,
    per-plant offsets) stay in the latent. The 24 base dims shrink the VAE
    input 216D -> 192D.
    """
    out = packets.clone()
    ot = out[:, :, :FM_OT_END].argmax(dim=-1)  # (P, 8)
    det = torch.zeros_like(ot, dtype=torch.bool)
    for t in DETERMINISTIC_ORGAN_TYPES:
        det |= ot == t
    out[:, :, FM_BASE_START:FM_BASE_END] = torch.where(
        det.unsqueeze(-1), torch.zeros_like(out[:, :, FM_BASE_START:FM_BASE_END]),
        out[:, :, FM_BASE_START:FM_BASE_END])
    return out


def assemble_packets(
    packets: torch.Tensor,
    reference_rots: torch.Tensor,
    base_scale: float = 20.0,
) -> torch.Tensor:
    """Reconstructs DETERMINISTIC slot bases from assembly rules; keeps VAE-learned
    bases for flowers/fruit.

    GT-verified deterministic rules (XML-phytomer clustering, full cowpea_curv26):
      stem/petiole/peduncle/bud base = cluster center (error <= 0.07cm p99).
      lateral leaflets (slots 2-3) = 0.8 x the CURVED petiole centerline.
      terminal leaflet (slot 4)    = 1.0 x the CURVED petiole centerline (tip).
    Flower/fruit bases (rare, per-plant offsets) are passed through unchanged.

    Base positions are world-frame vectors relative to the cluster center (the
    packet convention); the caller re-applies the anchor center + reference
    rotation (decode_packets / apply_ref_for_flow).

    NOTE: builds the base block in a fresh tensor and concatenates (no inplace
    writes into the grad-captured packet tensor — avoids autograd version
    conflicts in the differentiable render path).
    """
    out = packets.clone()
    P = out.shape[0]
    ot = out[:, :, :FM_OT_END].argmax(dim=-1)  # (P, 8)
    det = torch.zeros_like(ot, dtype=torch.bool)
    for t in DETERMINISTIC_ORGAN_TYPES:
        det |= ot == t
    # Fresh tensor (NOT a view of out): inplace writes below must not bump the
    # autograd version of out's slices (AsStridedBackward0 conflict).
    base = out[:, :, FM_BASE_START:FM_BASE_END].clone()
    base = torch.where(det.unsqueeze(-1), torch.zeros_like(base), base)
    # Leaflet bases: arc-fraction point on the CURVED petiole centerline.
    R_ref = rot6d_to_matrix(reference_rots)  # (P, 3, 3)
    for p in range(P):
        if not bool(det[p, 1]):
            continue
        R_pet = R_ref[p] @ rot6d_to_matrix(out[p, 1, FM_ROT_START:FM_ROT_END])
        pet_len = out[p, 1, FM_SCALE_START] / SCALE_SCALE  # metres
        pet_curv = out[p, 1, FM_CURV] / CURV_SCALE  # deg/m
        if pet_len < 1e-4:
            continue
        curve = _petiole_curve_points(R_pet, pet_len, pet_curv)
        for s, frac in LEAFLET_ATTACH_FRAC.items():
            if bool(det[p, s]):
                idx_f = frac * (len(curve) - 1)
                i0 = int(idx_f)
                t = idx_f - i0
                i1 = min(i0 + 1, len(curve) - 1)
                cpt = curve[i0] * (1 - t) + curve[i1] * t
                base[p, s] = cpt * base_scale
    return torch.cat(
        [out[..., :FM_BASE_START], base, out[..., FM_BASE_END:]], dim=-1
    )


def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """6D (Zhou et al.) -> 3x3 rotation matrices, COLUMNS convention.

    Matches the render geometry builder (R_mats = stack([r1, r2, r3], dim=-1))
    and phytomer_vae._rot6d_to_matrix. d6: (..., 6) -> (..., 3, 3).
    """
    x = F.normalize(d6[..., 0:3], dim=-1)
    z = F.normalize(torch.cross(x, d6[..., 3:6], dim=-1), dim=-1)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """Inverse of rot6d_to_matrix: 3x3 (orthonormal) -> 6D (first two COLUMNS).

    Matches the columns convention used by rot6d_to_matrix and the render
    geometry builder (R_mats = stack([r1, r2, r3], -1)).
    """
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def rotation_relative_to_reference(rot6d: torch.Tensor, ref_rot6d: torch.Tensor) -> torch.Tensor:
    """Converts absolute world-frame rot6d to reference-relative rot6d.

    R_rel = R_ref^T @ R_org. If one of the inputs is (..., 6) and the other a
    single (6,), broadcast applies.
    """
    R_org = rot6d_to_matrix(rot6d)
    R_ref = rot6d_to_matrix(ref_rot6d)
    if R_org.dim() == R_ref.dim() - 1:
        R_ref = R_ref.unsqueeze(0)
    R_rel = R_ref.transpose(-1, -2) @ R_org
    return matrix_to_rot6d(R_rel)


def apply_reference_rotation(rot6d: torch.Tensor, ref_rot6d: torch.Tensor) -> torch.Tensor:
    """Inverse of rotation_relative_to_reference: R_org = R_ref @ R_rel."""
    R_rel = rot6d_to_matrix(rot6d)
    R_ref = rot6d_to_matrix(ref_rot6d)
    R_org = R_ref @ R_rel
    return matrix_to_rot6d(R_org)

# Canonical slot spans per functional role: role -> (start, end) slot indices.
ROLE_SLOT_RANGES = {
    0: (0, 1),   # stem
    1: (1, 2),   # petiole
    2: (2, 5),   # leaflets x3
    3: (5, 6),   # peduncle
    4: (6, 8),   # reproductive x2
}
NUM_SLOTS = 8

# Organ type (t_label) -> role, half-open [lo, hi). Mirrors matcher semantics.
LABEL_ROLE_RANGES = {
    0: (1, 4),    # types 1,2,3
    1: (4, 5),    # type 4
    2: (5, 6),    # type 5
    3: (6, 7),    # type 6
    4: (7, 13),   # types 7..12
}

STANDALONE_INTERNODE_DIST = 0.04  # meters, matcher-consistent


def _role_of_labels(t_label: torch.Tensor) -> torch.Tensor:
    """Maps organ type labels to functional roles in {-1, 0..4} (-1 = NONE/other)."""
    role = torch.full_like(t_label, -1)
    for r, (lo, hi) in LABEL_ROLE_RANGES.items():
        role[(t_label >= lo) & (t_label < hi)] = r
    return role


def cluster_organs(
    positions: torch.Tensor,
    labels: torch.Tensor,
    standalone_threshold: float = STANDALONE_INTERNODE_DIST,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Groups organs into phytomer clusters (matcher-identical rule).

    Args:
        positions: (M, 3) organ base positions in meters.
        labels: (M,) int organ type labels.
        standalone_threshold: internodes farther than this from every petiole
            base become their own cluster centers.

    Returns:
        cluster_centers: (C, 3) node positions.
        assignments: (M,) cluster index per organ (nearest center).
    """
    petiole_idx = torch.nonzero(labels == 4, as_tuple=True)[0]
    if len(petiole_idx) > 0:
        cluster_centers = positions[petiole_idx]
        internode_idx = torch.nonzero(labels == 3, as_tuple=True)[0]
        if len(internode_idx) > 0:
            dist = torch.cdist(positions[internode_idx], cluster_centers).min(dim=1).values
            standalone = internode_idx[dist > standalone_threshold]
            if len(standalone) > 0:
                cluster_centers = torch.cat([cluster_centers, positions[standalone]], dim=0)
    else:
        internode_idx = torch.nonzero(labels == 3, as_tuple=True)[0]
        if len(internode_idx) > 0:
            cluster_centers = positions[internode_idx]
        else:
            cluster_centers = positions[:1]
    assignments = torch.cdist(positions, cluster_centers).argmin(dim=1)
    return cluster_centers, assignments


def _canonical_order_key(nodes_26d: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Canonical within-role order: bottom -> top (z), then azimuth. Mirrors canonical_sort_nodes."""
    import math as _math
    z = nodes_26d[indices, FM_BASE_START + 2]
    az = torch.atan2(nodes_26d[indices, FM_BASE_START + 1], nodes_26d[indices, FM_BASE_START])
    return torch.argsort(z * 10.0 + az / (2.0 * _math.pi), stable=True)


def build_phytomer_packets(
    nodes_26d: torch.Tensor,
    existence_mask: Optional[torch.Tensor] = None,
    base_scale: float = 20.0,
    drop_stats: Optional[Dict[str, int]] = None,
    reference_rot: Optional[torch.Tensor] = None,
    phytomer_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packs active organs into canonical 8-slot phytomer packets.

    Args:
        nodes_26d: (N, 26) FM-layout organ rows (absolute base positions).
        existence_mask: optional (N,) float mask; if None, derived from one-hot.
        base_scale: normalization divisor for base coords (BASE_SCALE=20.0).
        drop_stats: optional dict incremented with 'dropped_organs' / 'total_organs'.
        reference_rot: OPTIONAL explicit anchor-frame 6D rotation (Option 1 —
            the node frame, e.g. Stage-2 anchor_rot). Shape (6,) broadcast to all
            packets, or (P, 6) per-packet. When None, defaults to the packet's
            own internode frame (slot-0 internode rot, else first present slot,
            else identity) — botanically the node frame, so equivalent when the
            anchor frame IS the internode. Rotation is relativized to this frame
            for ALL present slots (slot 0 becomes identity when it is the ref).
        phytomer_ids: OPTIONAL (N, 2) int64 (shoot_id, phytomer_idx) per organ
            row, from the XML (cache field `phytomer_ids`). When given, organs
            are grouped by EXACT XML phytomer membership instead of nearest-center
            clustering — eliminating the ~77% mature-plant leaflet mis-assignment
            (nearest-center puts leaflets on the wrong petiole in dense canopies).

    Returns:
        packets: (P, 8, 26) FM rows with ANCHOR-RELATIVE base positions AND
                 ANCHOR-RELATIVE (reference-frame) rot6d.
        presence: (P, 8) bool, True where a real organ occupies the slot.
        centers: (P, 3) cluster center positions in meters (for re-anchoring).
        reference_rot: (P, 6) ABSOLUTE 6D rotation of each packet's reference
                 frame (the internode slot, or first present slot, or identity).
                 Stored in absolute coords so decode_packet() can re-apply it.
    """
    device = nodes_26d.device
    ot = nodes_26d[:, :FM_OT_END].argmax(dim=-1)
    if existence_mask is None:
        active = (ot != EMPTY_IDX) & (nodes_26d[:, EMPTY_IDX] < 0.5)
    elif existence_mask.shape[0] != nodes_26d.shape[0]:
        # Compact cache format: mask over the leading prefix (see
        # generate_tensor_shards: existence_mask has length num_organs and aligns
        # with nodes[:num_organs], which holds all actives first).
        active = torch.zeros(nodes_26d.shape[0], dtype=torch.bool, device=device)
        n = min(existence_mask.shape[0], nodes_26d.shape[0])
        active[:n] = existence_mask[:n].to(device) > 0.5
        active = active & (ot != EMPTY_IDX)
    else:
        active = (existence_mask.to(device) > 0.5) & (ot != EMPTY_IDX)

    packets: List[torch.Tensor] = []
    presence: List[torch.Tensor] = []
    centers: List[torch.Tensor] = []
    ref_rots: List[torch.Tensor] = []

    # External reference (Option 1: explicit anchor frame). Broadcast a (6,)
    # vector to match the number of packets once clusters are known.
    ext_ref = None
    if reference_rot is not None:
        ext_ref = reference_rot.to(device)
        if ext_ref.ndim == 1:
            ext_ref = ext_ref.unsqueeze(0)

    if int(active.sum().item()) == 0:
        return (
            torch.zeros((0, NUM_SLOTS, FM_NODE_DIM), device=device, dtype=nodes_26d.dtype),
            torch.zeros((0, NUM_SLOTS), device=device, dtype=torch.bool),
            torch.zeros((0, 3), device=device, dtype=nodes_26d.dtype),
            torch.zeros((0, 6), device=device, dtype=nodes_26d.dtype),
        )

    act_idx = torch.nonzero(active, as_tuple=True)[0]
    positions = nodes_26d[act_idx][:, FM_BASE_START:FM_BASE_END] / base_scale  # meters
    labels = ot[act_idx]

    if phytomer_ids is not None:
        # EXACT XML phytomer membership: group by (shoot_id, phytomer_idx).
        # Meta rows (ROOT/SHOOT_META, id == -1) are dropped from packets.
        ids = phytomer_ids.to(device)
        if ids.shape[0] != nodes_26d.shape[0]:
            ids = torch.cat([ids, torch.full((nodes_26d.shape[0] - ids.shape[0], 2), -1,
                                             dtype=torch.int64, device=device)], dim=0)
        ids_act = ids[act_idx]
        valid = (ids_act[:, 0] >= 0) & (ids_act[:, 1] >= 0)
        # Unique phytomer keys in first-appearance order (XML order).
        keys = ids_act[valid]
        uniq_keys, uniq_inv = torch.unique(keys, dim=0, return_inverse=True)
        # Reorder unique keys by first appearance (XML order).
        first_pos = torch.zeros(uniq_keys.shape[0], dtype=torch.long, device=device)
        seen = torch.zeros(uniq_keys.shape[0], dtype=torch.bool, device=device)
        order = torch.zeros(uniq_keys.shape[0], dtype=torch.long, device=device)
        cnt = 0
        for i in range(keys.shape[0]):
            k = int(uniq_inv[i].item())
            if not bool(seen[k].item()):
                seen[k] = True
                order[cnt] = k
                cnt += 1
        uniq_keys = uniq_keys[order]
        remap = torch.zeros(uniq_keys.shape[0], dtype=torch.long, device=device)
        remap[order] = torch.arange(uniq_keys.shape[0], device=device)
        cluster_ids = torch.full((act_idx.shape[0],), -1, dtype=torch.long, device=device)
        cluster_ids[valid] = remap[uniq_inv]
        n_clusters = uniq_keys.shape[0]
        # Cluster center = petiole base if present, else internode base, else
        # the mean of member bases (XML phytomer has exactly one petiole).
        cluster_centers = torch.zeros(n_clusters, 3, device=device)
        for c in range(n_clusters):
            pos_idx = torch.nonzero(cluster_ids == c, as_tuple=True)[0]  # into positions
            members = act_idx[pos_idx]
            pet = pos_idx[ot[members] == 4]
            if len(pet) > 0:
                cluster_centers[c] = positions[pet[0]]
            else:
                ino = pos_idx[ot[members] == 3]
                if len(ino) > 0:
                    cluster_centers[c] = positions[ino[0]]
                else:
                    cluster_centers[c] = positions[pos_idx].mean(dim=0)
        assignments = cluster_ids
    else:
        cluster_centers, assignments = cluster_organs(positions, labels)

    if drop_stats is not None:
        drop_stats["total_organs"] = drop_stats.get("total_organs", 0) + int(act_idx.numel())

    for c in range(cluster_centers.shape[0]):
        members = act_idx[assignments == c]
        if len(members) == 0:
            continue
        center = cluster_centers[c]
        packet = torch.zeros((NUM_SLOTS, FM_NODE_DIM), device=device, dtype=nodes_26d.dtype)
        packet[:, EMPTY_IDX] = 1.0  # default: NONE
        pres = torch.zeros((NUM_SLOTS,), device=device, dtype=torch.bool)
        member_roles = _role_of_labels(ot[members])
        for r in range(5):
            in_role = members[member_roles == r]
            if len(in_role) == 0:
                continue
            # Canonical within-role order (bottom -> top, then azimuth)
            order = _canonical_order_key(nodes_26d, in_role)
            in_role = in_role[order]
            lo, hi = ROLE_SLOT_RANGES[r]
            n_take = min(len(in_role), hi - lo)
            if len(in_role) > n_take and drop_stats is not None:
                drop_stats["dropped_organs"] = drop_stats.get("dropped_organs", 0) + int(len(in_role) - n_take)
            for s, src in enumerate(in_role[:n_take]):
                packet[lo + s] = nodes_26d[src]
                pres[lo + s] = True
        # Relativize base positions of PRESENT slots to the cluster center
        # (same normalized units). Absent slots keep the exact empty convention
        # (NONE one-hot + zero geometry) so the VAE never sees center leakage.
        packet[pres, FM_BASE_START:FM_BASE_END] = (
            packet[pres, FM_BASE_START:FM_BASE_END] - (center * base_scale)
        )
        # Relativize ROTATIONS to the packet reference frame.
        # Reference precedence: explicit anchor frame (Option 1, ext_ref) >
        # internode slot 0 > first present slot > identity. When an explicit
        # reference is given, ALL present slots (incl. slot 0) are relativized to
        # it; when using the internal fallback, slot 0 is the reference (-> identity).
        present_slots = torch.nonzero(pres, as_tuple=True)[0]
        if ext_ref is not None:
            ref_rot = ext_ref[c] if ext_ref.shape[0] > 1 else ext_ref[0]
        elif pres[0]:
            ref_rot = packet[0, FM_ROT_START:FM_ROT_END].clone()
        elif present_slots.numel() > 0:
            ref_rot = packet[present_slots[0], FM_ROT_START:FM_ROT_END].clone()
        else:
            ref_rot = torch.zeros(6, device=device)
        # Apply R_ref^T @ R_org to every present slot (slot 0 -> identity when it
        # is the reference; when ext_ref is given slot 0 is also relativized).
        packet[pres, FM_ROT_START:FM_ROT_END] = rotation_relative_to_reference(
            packet[pres, FM_ROT_START:FM_ROT_END], ref_rot.unsqueeze(0).expand(pres.sum(), -1)
        )
        packets.append(packet)
        presence.append(pres)
        centers.append(center)
        ref_rots.append(ref_rot)

    return torch.stack(packets), torch.stack(presence), torch.stack(centers), torch.stack(ref_rots)


def decode_packets(
    packets: torch.Tensor,
    centers: torch.Tensor,
    presence: torch.Tensor,
    reference_rots: torch.Tensor,
    base_scale: float = 20.0,
) -> torch.Tensor:
    """Re-anchors a batch of packets to absolute coordinates (vectorized).

    Args:
        packets: (P, 8, 26) anchor-relative FM rows.
        centers: (P, 3) absolute cluster centers (meters).
        presence: (P, 8) bool slot mask.
        reference_rots: (P, 6) absolute reference-frame 6D rotations.

    Returns:
        (P, 8, 26) FM rows in absolute (world) frame.
    """
    P = packets.shape[0]
    out = packets.clone()
    # Base: add center (only present slots; absent keep empty convention).
    center = centers.unsqueeze(1) * base_scale  # (P, 1, 3)
    out[:, :, FM_BASE_START:FM_BASE_END] += center
    # Rotation: R_org = R_ref @ R_rel.
    if reference_rots is not None:
        out[:, :, FM_ROT_START:FM_ROT_END] = apply_reference_rotation(
            out[:, :, FM_ROT_START:FM_ROT_END],
            reference_rots.unsqueeze(1).expand(P, 8, 6),
        )
    return out


def decode_packet(
    packet_26d: torch.Tensor,
    center: torch.Tensor,
    presence: Optional[torch.Tensor] = None,
    reference_rot: Optional[torch.Tensor] = None,
    base_scale: float = 20.0,
) -> torch.Tensor:
    """Re-anchors a packet to absolute coordinates (inverse of build_phytomer_packets).

    Only present slots are shifted/rotated back to absolute; absent slots keep
    the empty convention. Requires the packet's reference rotation (absolute),
    stored at build time. If reference_rot is None, rotations are returned as-is
    (legacy absolute packets).
    """
    out = packet_26d.clone()
    if presence is None:
        presence = torch.ones((packet_26d.shape[0],), dtype=torch.bool, device=packet_26d.device)
    outc = out[presence]
    outc[:, FM_BASE_START:FM_BASE_END] = (
        outc[:, FM_BASE_START:FM_BASE_END] + (center * base_scale)
    )
    if reference_rot is not None:
        ref = reference_rot if reference_rot.ndim == 2 else reference_rot.unsqueeze(0)
        outc[:, FM_ROT_START:FM_ROT_END] = apply_reference_rotation(
            outc[:, FM_ROT_START:FM_ROT_END], ref
        )
    out[presence] = outc
    return out
