"""
Node-level rotation as a derived direction + a predicted roll, instead of an
independently predicted full 6D rotation.

Why: a node's 6D scaffold rotation carries 3 rotational DOF -- 2 for "which
direction is forward" and 1 for "roll about that axis" (which side a
petiole/leaflet points). The forward 2 DOF are exactly and fully determined
once topology is known (`normalize(this_node_pos - parent_node_pos)` -- the
same relation `phytomer_packets.assemble_packets(parent_pos=...)` already uses
for the internode itself), so predicting them independently is redundant with
position. The roll 1 DOF is NOT recoverable from positions at all (nothing
about two points tells you which way a leaf faces sideways), so it still needs
its own predicted output.

This module provides the two directions of that split:
  encode_roll(R, forward)  : full rotation -> (cos, sin) roll GIVEN a forward
                             axis (used to build training targets from GT,
                             where both R and a GT-derived forward exist).
  roll_to_matrix(forward, roll) : derived forward + predicted (cos, sin) ->
                             full rotation matrix (used at reconstruction
                             time, after phytomer_topology.chain_phytomers has
                             resolved parent links from POSITION alone --
                             measured to cost <=2.5 points of parent-recovery
                             accuracy without a rotation-based cue, see that
                             module's docstring -- so this never needs
                             rotation as an input to produce it as an output).

Both share one "zero-roll" reference frame construction (`_canonical_frame`)
so encode and decode agree exactly (round-trip exact by construction, see
tests/test_phytomer_roll.py).
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def _least_parallel_axis(forward: torch.Tensor) -> torch.Tensor:
    """World axis (x, y, or z) least parallel to each row of `forward`.

    Cross-product-based frame construction is unstable/discontinuous right
    where the reference axis is parallel to `forward`; picking whichever of
    the three world axes has the smallest |dot| with `forward` keeps that
    dot product bounded away from +-1 everywhere (worst case dot = 1/sqrt(3)
    when `forward` is equidistant from all three axes), unlike a single fixed
    reference (e.g. always world Z), which is exactly degenerate for a
    perfectly vertical forward axis -- the shoot-base fallback case this
    exists for.
    """
    eye = torch.eye(3, device=forward.device, dtype=forward.dtype)  # (3, 3)
    dots = (forward.unsqueeze(-2) * eye).sum(-1).abs()              # (..., 3)
    which = dots.argmin(dim=-1)                                     # (...,)
    return eye[which]


def _canonical_frame(forward: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic zero-roll (x0, z0) orthonormal to a unit `forward` (..., 3).

    Column convention matches rot6d_to_matrix/matrix_to_rot6d: a full rotation
    is stack([x, forward, z], -1) (forward is column 1, the tube axis).
    """
    ref = _least_parallel_axis(forward)
    x0 = F.normalize(torch.cross(ref, forward, dim=-1), dim=-1)
    z0 = torch.cross(x0, forward, dim=-1)  # already unit: x0, forward orthonormal
    return x0, z0


def encode_roll(R: torch.Tensor, forward: torch.Tensor) -> torch.Tensor:
    """Full rotation matrix (..., 3, 3) + a (possibly different) unit forward
    axis (..., 3) -> (..., 2) (cos, sin) roll of R's own x-column about that
    forward axis, measured from the canonical zero-roll frame.

    `forward` is passed separately (not read from R) because at training time
    it comes from GT node positions (parent -> child), which need not be
    bit-identical to R's own column-1 after the packet's various frame
    conventions -- projecting R's x-column onto the plane orthogonal to the
    supplied forward keeps the target well-defined even then.
    """
    x0, z0 = _canonical_frame(forward)
    x = R[..., :, 0]
    # Project out any component along `forward` (should be ~0 already if R's
    # own column 1 matches `forward`; the projection makes this well-defined
    # even if they differ slightly) and read off the angle in the (x0, z0) basis.
    x_perp = F.normalize(x - (x * forward).sum(-1, keepdim=True) * forward, dim=-1)
    cos_r = (x_perp * x0).sum(-1)
    sin_r = (x_perp * z0).sum(-1)
    return torch.stack([cos_r, sin_r], dim=-1)


def roll_to_matrix(forward: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
    """Derived unit forward axis (..., 3) + predicted (cos, sin) roll (..., 2)
    -> full rotation matrix (..., 3, 3), columns [x, forward, z].

    `roll` need not be exactly unit norm (a raw network output isn't); it is
    normalized here, matching how rot6d_to_matrix tolerates non-orthonormal
    raw input.
    """
    x0, z0 = _canonical_frame(forward)
    r = F.normalize(roll, dim=-1)
    cos_r, sin_r = r[..., 0:1], r[..., 1:2]
    x = cos_r * x0 + sin_r * z0
    z = -sin_r * x0 + cos_r * z0
    return torch.stack([x, forward, z], dim=-1)


def derive_forward(
    pos: torch.Tensor,
    parent_idx: torch.Tensor,
    fallback: torch.Tensor = None,
) -> torch.Tensor:
    """Per-node forward axis from resolved topology: (child - parent), unit.

    Args:
        pos: (P, 3) node positions.
        parent_idx: (P,) int64 from phytomer_topology.chain_phytomers; -1 for
            shoot bases (no parent to derive a direction from).
        fallback: (3,) unit vector used for parent_idx == -1 rows. Defaults to
            world +Z (shoot bases are, by construction, the start of a shoot
            growing upward from the plant's own base or a branch point).

    Returns:
        (P, 3) unit forward vectors.
    """
    if fallback is None:
        fallback = torch.tensor([0.0, 0.0, 1.0], device=pos.device, dtype=pos.dtype)
    has_parent = parent_idx >= 0
    safe_parent = parent_idx.clamp(min=0)
    d = pos - pos[safe_parent]
    norm = d.norm(dim=-1, keepdim=True)
    unit = torch.where(norm > 1e-6, d / norm.clamp(min=1e-6), fallback.expand_as(d))
    return torch.where(has_parent.unsqueeze(-1), unit, fallback.expand_as(d))
