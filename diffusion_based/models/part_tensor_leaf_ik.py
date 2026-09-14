"""Leaf orientation inverse for the 40D export: make the FK reproduce each leaf's 14D rotation.

Why this exists (measured 2026-09-14 on dataset plants, see
docs/ongoing/20260912_stage2_stage3_boundary_and_remaining_redundancy.md §2.6):
`PartTensorTo40DConverter` writes leaf pitch/yaw/roll as constants (pitch 2.54,
roll -15, yaw +10/0/-10 by leaflet index). Those are the species defaults the
exact_gt_renders test plants were generated with, so fig14 never noticed; the
training dataset perturbs every leaf, and under the FK the exported leaves come
out 13-15 degrees (mean) from their 14D rotation, which costs the Helios
round-trip 14 points of foreground IoU on a DAP 15 seedling (81.8% vs 95.7%).

The FK composes a leaf's rotation as

    R_leaf = Rz(azimuth) . Rz(yaw_rot) . Ry(-pitch_rot) . Rx(roll_rot)

where `azimuth` comes from the petiole tip and, per leaf kind,
    lateral leaflet : yaw_rot = yaw, pitch_rot = pitch, roll_rot = (asin(pz) + roll) * sign
    terminal leaflet: yaw_rot = 0,   pitch_rot = pitch + asin(pz), roll_rot = 0
    single leaf     : yaw_rot = 0,   pitch_rot = pitch + asin(pz), roll_rot = acos(iz) - roll
(pz = petiole tip z, iz = internode tip z, sign = +-1 by side). The FK returns
`azimuth`, asin(pz), acos(iz), sign and the kind per leaf row
(`extract_part_tensor(return_node_poses=True)["leaf_frame"/"leaf_kind"]`), so
the inverse is closed-form: M = Rz(-azimuth) . R_target is decomposed as
Rz(a) Ry(b) Rx(g) and mapped back through the rules above. A lateral leaflet
has three free angles and is matched exactly; the terminal leaflet has one
(pitch) and the single leaf two, so their residual is whatever the petiole tip
frame leaves -- which is also what Helios itself can represent, since the
dataset was generated under the same rules.

Runs after `part_tensor_stem_ik` (the leaf frames depend on the solved
petioles) and, like it, is a black-box step on the FK: two FK passes, one to
read the frames and one to verify.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray, rotation_6d_to_matrix, ORGAN_LEAF,
    P_COL_ORGAN_TYPE, T_COL_ORGAN_TYPE, T_COL_EXISTENCE, T_COL_PITCH, T_COL_YAW, T_COL_ROLL,
)

_RAD2DEG = 180.0 / math.pi


def _rz(a: float) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _zyx(M: torch.Tensor):
    """M = Rz(a) Ry(b) Rx(g) with the FK's rotr_* conventions -> (a, b, g)."""
    b = -math.asin(max(-1.0, min(1.0, float(M[2, 0]))))
    a = math.atan2(float(M[1, 0]), float(M[0, 0]))
    g = math.atan2(float(M[2, 1]), float(M[2, 2]))
    return a, b, g


def _wrap_deg(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


def leaf_rotation_error_deg(R_a: torch.Tensor, R_b: torch.Tensor) -> torch.Tensor:
    tr = (R_a.transpose(1, 2) @ R_b).diagonal(dim1=1, dim2=2).sum(-1)
    return torch.rad2deg(torch.acos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0)))


def refine_leaf_orientation(
    arr_40d: torch.Tensor,
    part_14d: torch.Tensor,
    verbose: bool = False,
    stats: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """Adjusts leaf pitch/yaw/roll in arr_40d (row-aligned with part_14d) so the FK
    reproduces each leaf's 14D rotation. Returns a new (N, 40) tensor."""
    from diffusion_based.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder

    arr = arr_40d.detach().clone().float().cpu()
    p14 = part_14d.detach().float().cpu()
    if arr.shape[0] == 0:
        return arr
    ot40 = arr[:, T_COL_ORGAN_TYPE].round().long()
    ot14 = p14[:, P_COL_ORGAN_TYPE].round().long()
    exist = arr[:, T_COL_EXISTENCE] > 0.5
    idx = torch.nonzero((ot40 == ORGAN_LEAF) & (ot14 == ORGAN_LEAF) & exist).flatten()
    if idx.numel() == 0:
        return arr

    builder = HeliosPlantGeometryBuilder()

    def fk(a):
        return builder.extract_part_tensor(PlantOrganArray(a.clone()), return_node_poses=True)

    # 14D and FK rows store rot6d as the first two COLUMNS of R (helios_pytorch_geometry
    # _make_row_rot); rotation_6d_to_matrix stacks them as rows, hence the transpose.
    R_t = rotation_6d_to_matrix(p14[:, 4:10]).transpose(1, 2)
    out, poses = fk(arr)
    frame, kind = poses["leaf_frame"], poses["leaf_kind"]
    realized = idx[kind[idx] >= 0]
    R_r = rotation_6d_to_matrix(out[:, 4:10]).transpose(1, 2)
    err_before = leaf_rotation_error_deg(R_t[realized], R_r[realized])

    for j in realized.tolist():
        k = int(kind[j])
        az, asin_pz, acos_iz, sign = frame[j].tolist()
        M = _rz(-az) @ R_t[j]
        a, b, g = _zyx(M)
        if k == 1:                                   # lateral leaflet: exact
            arr[j, T_COL_YAW] = _wrap_deg(a * _RAD2DEG)
            arr[j, T_COL_PITCH] = -b * _RAD2DEG
            arr[j, T_COL_ROLL] = _wrap_deg((g * sign - asin_pz) * _RAD2DEG)
        elif k == 2:                                 # terminal leaflet: pitch only
            # best Ry(bb) fit to M: maximise trace(Ry(bb)^T M)
            bb = math.atan2(float(M[0, 2] - M[2, 0]), float(M[0, 0] + M[2, 2]))
            arr[j, T_COL_PITCH] = (-bb - asin_pz) * _RAD2DEG
        else:                                        # single leaf: pitch and roll
            # ind_from_tip == 0 here too, so the FK adds asin(pz) to the pitch
            arr[j, T_COL_PITCH] = (-b - asin_pz) * _RAD2DEG
            arr[j, T_COL_ROLL] = _wrap_deg((acos_iz - g) * _RAD2DEG)

    out2, _ = fk(arr)
    R_r2 = rotation_6d_to_matrix(out2[:, 4:10]).transpose(1, 2)
    err_after = leaf_rotation_error_deg(R_t[realized], R_r2[realized])
    if verbose:
        print(f"  [LeafIK] {realized.numel()} leaves: rotation error mean {err_before.mean():.2f} -> "
              f"{err_after.mean():.2f} deg, max {err_before.max():.2f} -> {err_after.max():.2f} deg")
    if stats is not None:
        kk = kind[realized]
        stats.update({
            "leaf_rot_err_before_mean_deg": float(err_before.mean()), "leaf_rot_err_before_max_deg": float(err_before.max()),
            "leaf_rot_err_after_mean_deg": float(err_after.mean()), "leaf_rot_err_after_max_deg": float(err_after.max()),
            "n_leaves": int(realized.numel()),
        })
        for k, name in [(0, "single"), (1, "lateral"), (2, "terminal")]:
            m = kk == k
            if m.any():
                stats[f"leaf_rot_err_after_mean_deg_{name}"] = float(err_after[m].mean())
                stats[f"n_leaves_{name}"] = int(m.sum())
    return arr
