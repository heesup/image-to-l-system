"""Stem inverse kinematics for the 40D export: make Helios's FK follow the nodes.

Why this exists (measured 2026-09-12, see
docs/ongoing/20260912_stage2_stage3_boundary_and_remaining_redundancy.md §2.4):
`PartTensorTo40DConverter` never uses our internode directions. Helios rebuilds
a shoot by forward kinematics from per-node parameters -- a species-constant
pitch (20 deg), a phyllotactic angle the converter derives from consecutive
petiole azimuths, and the per-segment curvature/yaw perturbations it copies
from the 14D curvature column -- so the stem's shape is integrated from those
per-node angles and a small error at one node moves every node above it. With
a VAE that reproduces petiole rotations to 1.4 deg the Helios round-trip still
lost ten points to that integration (DAP 50: 82.9% as decoded, 92.7% with exact
petiole rotations), and zeroing the decoded internode curvature was worth
another three.

This module closes the loop the other way round: it treats the 14D internode
rows (base -> tip, i.e. the parent-node -> node chords that Stage 2 predicts
and the identity path has exactly) as the target and solves the stored
perturbations, lengths, phyllotactic angles and petiole pitches so that the
same FK (`HeliosPlantGeometryBuilder.extract_part_tensor`, the Python mirror
of Helios's phytomer builder) lands each internode tip on its 14D tip and each
petiole on its 14D direction. It is a black-box fixed-point iteration on that
FK rather than a re-implementation of it, so it cannot drift from the FK:

    per iteration: run the FK, then for every internode with an index > 0 in
    its shoot (Helios applies no perturbation to a shoot's first phytomer)
        curvature_pert_0  += elevation(dir_14d) - elevation(chord_fk)
        yaw_pert_0        += azimuth  (dir_14d) - azimuth  (chord_fk)
    and for every petiole
        pitch             += angle(target, axis_fk) - angle(realized, axis_fk)
        phyllotactic      += signed azimuth of target vs realized about axis_fk

    Each internode is fitted to its OWN 14D direction (its chord, for the
    packet path), never to an absolute tip from wherever the FK base currently
    is: a first version did the latter and diverged, because a millimetre of
    accumulated base error at a 5 mm shoot-tip internode is a 100-degree
    "correction". And because Helios builds each internode RELATIVE to the
    previous one (the pitch deflection is applied to the previous axis), a
    direction error at node i is inherited by every node above it, so the
    step applied to node i is the error of node i minus the error of its
    predecessor -- the part of the deviation that node i itself introduced.
    A second version applied the full error at every node simultaneously and
    also diverged (each node was corrected once for itself and once for each
    ancestor).

The curvature perturbation rotates about Helios's `shoot_bending_axis` =
cross(axis, z), which is horizontal and perpendicular to the axis, so it moves
elevation only; the yaw perturbation rotates about z and moves azimuth only.
Each update is therefore a unit-gain step on an almost-diagonal map, and the
loop converges in a handful of iterations (the only coupling is that a node's
own correction shifts the base of every node above it, which the next
iteration sees). Nothing here is learned or species-fitted; on ground-truth
14D input the FK already matches, so the corrections are zero and the export is
unchanged.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from diffusion_based.models.plant_organ_array import (
    ORGAN_INTERNODE,
    ORGAN_PETIOLE,
    ORGAN_SHOOT_META,
    T_COL_YAW,
    P_COL_ORGAN_TYPE,
    T_COL_CURV_PERT_0,
    T_COL_EXISTENCE,
    T_COL_LENGTH,
    T_COL_LENGTH_MAX,
    T_COL_ORGAN_TYPE,
    T_COL_PARENT_PETIOLE_IDX,
    T_COL_PHYLLOTACTIC_ANGLE,
    T_COL_PHYTOMER_IDX,
    T_COL_PITCH,
    T_COL_SHOOT_ID,
    T_COL_YAW_PERT_0,
    PlantOrganArray,
)

_RAD2DEG = 180.0 / math.pi


def _elev_az(v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Elevation and azimuth (radians) of (..., 3) vectors."""
    n = v.norm(dim=-1).clamp(min=1e-9)
    elev = torch.asin((v[..., 2] / n).clamp(-1.0, 1.0))
    az = torch.atan2(v[..., 1], v[..., 0])
    return elev, az


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def _signed_azimuth_about(axis: torch.Tensor, v_from: torch.Tensor, v_to: torch.Tensor) -> torch.Tensor:
    """Signed angle from v_from to v_to measured in the plane perpendicular to axis."""
    a = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    p1 = v_from - (v_from * a).sum(-1, keepdim=True) * a
    p2 = v_to - (v_to * a).sum(-1, keepdim=True) * a
    n1 = p1.norm(dim=-1); n2 = p2.norm(dim=-1)
    ok = (n1 > 1e-6) & (n2 > 1e-6)
    cross = torch.cross(p1, p2, dim=-1)
    ang = torch.atan2((cross * a).sum(-1), (p1 * p2).sum(-1))
    return torch.where(ok, ang, torch.zeros_like(ang))


def refine_stem_to_part_tensor(
    arr_40d: torch.Tensor,
    part_14d: torch.Tensor,
    n_iter: int = 8,
    tol_m: float = 5e-4,
    petiole_correction: bool = True,
    max_step_deg: float = 30.0,
    parallel_guard_deg: float = 10.0,
    damping: float = 0.9,
    base_correction: bool = True,
    verbose: bool = False,
    stats: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """Adjusts arr_40d (row-aligned with part_14d) so the FK reproduces the 14D stem.

    Args:
        arr_40d: (N, 40) typed rows from PartTensorTo40DConverter, same row order
            as part_14d.
        part_14d: (N, 14) canonical part tensor the 40D was converted from.
        n_iter: fixed-point iterations (each one FK pass).
        tol_m: stop when every internode tip is within this distance (metres).
        petiole_correction: also correct petiole pitch and the node's
            phyllotactic angle from the realized vs target petiole directions.
        stats: optional dict receiving the max/mean internode tip error before
            and after (metres) and the iterations used.
    Returns a new (N, 40) tensor.
    """
    from diffusion_based.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder

    arr = arr_40d.detach().clone().float().cpu()
    p14 = part_14d.detach().float().cpu()
    N = arr.shape[0]
    if N == 0:
        return arr

    ot40 = arr[:, T_COL_ORGAN_TYPE].round().long()
    ot14 = p14[:, P_COL_ORGAN_TYPE].round().long()
    exist = arr[:, T_COL_EXISTENCE] > 0.5
    inode = (ot40 == ORGAN_INTERNODE) & (ot14 == ORGAN_INTERNODE) & exist
    inode_idx = torch.nonzero(inode).flatten()
    if inode_idx.numel() == 0:
        return arr

    # Index of each internode within its shoot (Helios applies perturbations
    # and the phyllotactic rotation only from the second phytomer on).
    sid = arr[:, T_COL_SHOOT_ID].round().long()
    pidx = arr[:, T_COL_PHYTOMER_IDX].round().long()
    rank = torch.zeros(N, dtype=torch.long)
    for s in sid[inode_idx].unique().tolist():
        rows = inode_idx[sid[inode_idx] == s]
        order = torch.argsort(pidx[rows])
        rank[rows[order]] = torch.arange(rows.numel())
    adjustable = inode_idx[rank[inode_idx] > 0]
    # Predecessor internode row along the shoot (-1 for a shoot's first node).
    predecessor = torch.full((N,), -1, dtype=torch.long)
    for s in sid[inode_idx].unique().tolist():
        rows = inode_idx[sid[inode_idx] == s]
        rows = rows[torch.argsort(pidx[rows])]
        predecessor[rows[1:]] = rows[:-1]

    # 14D targets: internode tip = base + forward * length (rot6d columns 4:10,
    # forward axis is column 1 of the matrix, i.e. elements (1, 4, 7) of the
    # row-major 3x3 -- reuse the same helper the converter uses).
    from diffusion_based.models.part_tensor_to_40d import rotation_6d_to_matrix
    R = rotation_6d_to_matrix(p14[:, 4:10]).transpose(1, 2)   # converter convention
    fwd = R[:, :, 1]
    fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    tip_target = p14[:, 1:4] + fwd * p14[:, 10:11].abs()

    # Petiole rows -> their node's internode row and petiole slot.
    pet_rows = torch.nonzero((ot40 == ORGAN_PETIOLE) & (ot14 == ORGAN_PETIOLE) & exist).flatten()
    node_of: Dict[Tuple[int, int], int] = {(int(sid[i]), int(pidx[i])): int(i) for i in inode_idx.tolist()}
    pet_node = torch.full((N,), -1, dtype=torch.long)
    for j in pet_rows.tolist():
        pet_node[j] = node_of.get((int(sid[j]), int(pidx[j])), -1)
    pet_ok = pet_rows[pet_node[pet_rows] >= 0]
    pet_slot = arr[:, T_COL_PARENT_PETIOLE_IDX].round().long().clamp(0, 1)

    builder = HeliosPlantGeometryBuilder()

    def run_fk(a):
        return builder.extract_part_tensor(PlantOrganArray(a.clone()), return_node_poses=True, stem_only=True)[1]

    # Processing order: shoots by id (the emitter numbers a parent before its
    # children), nodes by index within the shoot. Each node is corrected with
    # the FK re-run after the previous correction, so every step sees exactly
    # the unit-gain response measured by finite differences; a simultaneous
    # update cannot, because Helios builds each internode relative to the
    # previous one's final axis but NOT to its perturbations' effect on the
    # petiole frame, which makes upstream corrections propagate downstream
    # with alternating sign.
    order = []
    for s_ in sorted(sid[inode_idx].unique().tolist()):
        rows = inode_idx[sid[inode_idx] == s_]
        order.extend(rows[torch.argsort(pidx[rows])].tolist())
    pet_of_node: Dict[int, list] = {}
    for j in pet_ok.tolist():
        pet_of_node.setdefault(int(pet_node[j]), []).append(j)

    # A shoot's first internode gets no perturbation in Helios: its direction
    # comes from the shoot's base pitch/yaw (SHOOT_META row). Those two angles
    # are solved from a finite-difference Jacobian of the FK chord, which is
    # cheaper to get right than the closed form (yaw is about the PARENT
    # internode axis, pitch about an axis built from the parent petiole).
    meta_of_shoot: Dict[int, int] = {}
    for r_ in torch.nonzero(ot40 == ORGAN_SHOOT_META).flatten().tolist():
        meta_of_shoot.setdefault(int(sid[r_]), r_)

    def base_step(i: int, poses) -> None:
        m = meta_of_shoot.get(int(sid[i]))
        if m is None:
            return
        chord = poses["tip"][i] - poses["base"][i]
        n = float(chord.norm())
        if n < 1e-6:
            return
        c = chord / n
        t = fwd[i]
        ang_err = math.degrees(math.acos(float((c * t).sum().clamp(-1, 1))))
        if ang_err > 0.05:
            resid = t - c                                        # small-angle direction error
            cols = []
            for col in (T_COL_PITCH, T_COL_YAW):
                trial = arr.clone(); trial[m, col] += 1.0
                pz = run_fk(trial)
                ch = pz["tip"][i] - pz["base"][i]
                cols.append((ch / ch.norm().clamp(min=1e-9) - c))    # d(dir)/d(deg)
            J = torch.stack(cols, dim=1)                             # (3, 2)
            # Skip a degenerate Jacobian (an angle the FK does not respond
            # to here, e.g. yaw about a parent axis the child is parallel to)
            # rather than let least squares ask for a huge step.
            sv = torch.linalg.svdvals(J)
            if float(sv.min()) > 2e-4:
                sol = torch.linalg.lstsq(J, resid.unsqueeze(1)).solution.flatten()
                cap = min(max_step_deg, 1.5 * ang_err)
                sol = sol.clamp(-cap, cap)
                if torch.isfinite(sol).all():
                    arr[m, T_COL_PITCH] += float(sol[0])
                    arr[m, T_COL_YAW] += float(sol[1])
        # The base internode's length is the one length the chain does not
        # fix (assemble_packets overrides chained lengths with the parent ->
        # node chord but a shoot base keeps its decoded length), so take it
        # from where the shoot actually starts to where its first node is.
        L = float((tip_target[i] - poses["base"][i]).norm())
        if L > 1e-4:
            arr[i, T_COL_LENGTH] = L
            arr[i, T_COL_LENGTH_MAX] = L

    def node_step(i: int, poses) -> None:
        # internode direction (nodes beyond the shoot base only)
        if rank[i] > 0:
            chord = poses["tip"][i] - poses["base"][i]
            if float(chord.norm()) > 1e-6:
                e_t, a_t = _elev_az(fwd[i]); e_c, a_c = _elev_az(chord)
                arr[i, T_COL_CURV_PERT_0] += float(((e_t - e_c) * _RAD2DEG).clamp(-max_step_deg, max_step_deg))
                arr[i, T_COL_YAW_PERT_0] += float((_wrap(a_t - a_c) * _RAD2DEG).clamp(-max_step_deg, max_step_deg))
        if not petiole_correction:
            return
        axis = poses["axis"][i]
        axis = axis / axis.norm().clamp(min=1e-9)
        for j in pet_of_node.get(i, []):
            slot = int(pet_slot[j])
            if float(poses["has_petiole"][i, slot]) < 0.5:
                continue
            realized = poses["petiole_axes"][i, slot]
            target = fwd[j]
            cos_t = float((target * axis).sum().clamp(-1, 1)); cos_r = float((realized * axis).sum().clamp(-1, 1))
            d_pitch = (math.acos(cos_t) - math.acos(cos_r)) * _RAD2DEG
            d_pitch = max(-max_step_deg, min(max_step_deg, d_pitch))
            sign = 1.0 if float(arr[j, T_COL_PITCH]) >= 0 else -1.0
            arr[j, T_COL_PITCH] = sign * min(179.0, max(0.0, abs(float(arr[j, T_COL_PITCH])) + d_pitch))
            ang_t = math.degrees(math.acos(cos_t))
            if slot == 0 and rank[i] > 0 and parallel_guard_deg < ang_t < 180.0 - parallel_guard_deg:
                d_az = float(_signed_azimuth_about(axis.unsqueeze(0), realized.unsqueeze(0), target.unsqueeze(0))[0]) * _RAD2DEG
                d_az = max(-max_step_deg, min(max_step_deg, d_az))
                arr[i, T_COL_PHYLLOTACTIC_ANGLE] = (float(arr[i, T_COL_PHYLLOTACTIC_ANGLE]) + d_az) % 360.0

    err_before = None
    prev_max = None
    used = 0
    for it in range(n_iter):
        poses = run_fk(arr)
        err = (poses["tip"][inode_idx] - tip_target[inode_idx]).norm(dim=-1)
        if err_before is None:
            err_before = err.clone()
        if verbose:
            print(f"  [StemIK] sweep {it}: tip error mean {err.mean()*100:.3f} cm max {err.max()*100:.3f} cm")
        used = it
        cur_max = float(err.max())
        # One sweep reaches the fixed point on every plant measured so far; a
        # further sweep only pays for itself while it still moves the residual
        # (the shoot-base internodes, which Helios never perturbs, set a floor).
        if cur_max < tol_m or (prev_max is not None and cur_max > 0.9 * prev_max):
            break
        prev_max = cur_max
        for i in order:
            if base_correction and rank[i] == 0:
                base_step(i, poses)
                poses = run_fk(arr)
            node_step(i, poses)
            poses = run_fk(arr)

    if stats is not None:
        _, poses = builder.extract_part_tensor(PlantOrganArray(arr.clone()), return_node_poses=True)
        err_after = (poses["tip"][inode_idx] - tip_target[inode_idx]).norm(dim=-1)
        stats.update({
            "tip_err_before_mean_m": float(err_before.mean()), "tip_err_before_max_m": float(err_before.max()),
            "tip_err_after_mean_m": float(err_after.mean()), "tip_err_after_max_m": float(err_after.max()),
            "iterations": used + 1,
        })
    return arr
