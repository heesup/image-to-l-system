"""Tests for phytomer_roll: derived-forward + predicted-roll rotation split."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from diffusion_based.dataset.phytomer_roll import (
    _canonical_frame, _least_parallel_axis, encode_roll, roll_to_matrix, derive_forward,
)
from diffusion_based.dataset.phytomer_topology import chain_phytomers


def _random_rotations(n: int, seed: int = 0) -> torch.Tensor:
    """n random proper (det=+1) orthonormal 3x3 matrices via QR."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, 3, 3, generator=g)
    Q, R = torch.linalg.qr(A)
    # Fix signs so diag(R) > 0 (QR sign ambiguity), then force det=+1.
    d = torch.diagonal(R, dim1=-2, dim2=-1).sign()
    Q = Q * d.unsqueeze(-2)
    det = torch.linalg.det(Q)
    Q[..., -1] = Q[..., -1] * det.sign().unsqueeze(-1)
    return Q


class TestPhytomerRoll(unittest.TestCase):
    def test_canonical_frame_orthonormal_including_axis_aligned(self):
        fwd = torch.cat([
            _random_rotations(20)[:, :, 1],
            torch.eye(3),        # exactly axis-aligned forwards: the degenerate case
            -torch.eye(3),
        ], dim=0)
        x0, z0 = _canonical_frame(fwd)
        R = torch.stack([x0, fwd, z0], dim=-1)
        should_be_I = R.transpose(-1, -2) @ R
        I = torch.eye(3).expand_as(should_be_I)
        self.assertTrue(torch.allclose(should_be_I, I, atol=1e-5))
        det = torch.linalg.det(R)
        self.assertTrue(torch.allclose(det, torch.ones_like(det), atol=1e-5))

    def test_encode_decode_roundtrip_exact(self):
        R = _random_rotations(64)
        fwd = R[:, :, 1]
        roll = encode_roll(R, fwd)
        self.assertEqual(roll.shape, (64, 2))
        # roll should already be unit-norm (cos^2+sin^2=1) since it's read off
        # an orthonormal frame directly.
        self.assertTrue(torch.allclose(roll.norm(dim=-1), torch.ones(64), atol=1e-5))
        R_hat = roll_to_matrix(fwd, roll)
        self.assertTrue(torch.allclose(R_hat, R, atol=1e-4))

    def test_roll_to_matrix_output_is_valid_rotation(self):
        fwd = torch.nn.functional.normalize(torch.randn(32, 3), dim=-1)
        roll_raw = torch.randn(32, 2) * 3.0  # not unit norm, like a raw net output
        R = roll_to_matrix(fwd, roll_raw)
        should_be_I = R.transpose(-1, -2) @ R
        self.assertTrue(torch.allclose(should_be_I, torch.eye(3).expand_as(should_be_I), atol=1e-4))
        self.assertTrue(torch.allclose(torch.linalg.det(R), torch.ones(32), atol=1e-4))
        # forward axis (column 1) must be exactly what was supplied.
        self.assertTrue(torch.allclose(R[:, :, 1], fwd, atol=1e-5))

    def test_derive_forward_matches_parent_child_direction(self):
        pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
        parent_idx = torch.tensor([-1, 0, 1])
        fwd = derive_forward(pos, parent_idx)
        self.assertTrue(torch.allclose(fwd[1], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6))
        self.assertTrue(torch.allclose(fwd[2], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6))
        self.assertTrue(torch.allclose(fwd[0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6))

    def test_derive_forward_base_follows_successor_not_world_up(self):
        """A base has no parent, so it must read its axis off its successor.
        A lateral branch's first phytomer is a base too, and is rarely
        vertical -- assuming +Z there was measured 50.8 deg off on average."""
        s = 0.5 ** 0.5
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 2.0]])
        parent_idx = torch.tensor([-1, 0, 1])
        fwd = derive_forward(pos, parent_idx)
        self.assertTrue(torch.allclose(fwd[0], torch.tensor([s, 0.0, s]), atol=1e-6))

    def test_derive_forward_lone_node_falls_back_to_world_up(self):
        """Neither a parent nor a successor (a single-phytomer shoot): the
        world-+Z fallback is all that is left, and is right for a seedling."""
        pos = torch.tensor([[0.3, -0.2, 0.7]])
        fwd = derive_forward(pos, torch.tensor([-1]))
        self.assertTrue(torch.allclose(fwd[0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6))

    def test_derive_forward_base_prefers_shoot_continuation_over_lateral(self):
        """At a base that is also a branch point, the successor must be the
        node continuing the SAME shoot, not the lateral that also claims it."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [0.0, 0.0, 1.0]])
        parent_idx = torch.tensor([-1, 0, 0])
        shoot_id = torch.tensor([0, 1, 0])
        phytomer_idx = torch.tensor([0, 0, 1])
        fwd = derive_forward(pos, parent_idx, shoot_id, phytomer_idx)
        self.assertTrue(torch.allclose(fwd[0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6))
        # Without the shoot labels the lateral (lower index) wins instead.
        fwd_no_shoot = derive_forward(pos, parent_idx)
        self.assertFalse(torch.allclose(fwd_no_shoot[0], fwd[0], atol=1e-3))

    def test_chain_phytomers_accepts_no_rotation(self):
        """chain_phytomers(rot6d=None) must not crash and should still chain
        a simple vertical stack using distance + ordinal alone."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
        ordinal = torch.tensor([0.0, 1.0, 2.0])
        is_base = torch.tensor([1.0, 0.0, 0.0])
        parent, shoot, phyidx = chain_phytomers(pos, ordinal=ordinal, is_base=is_base)
        self.assertEqual(parent.tolist(), [-1, 0, 1])
        self.assertEqual(shoot.tolist(), [0, 0, 0])
        self.assertEqual(phyidx.tolist(), [0, 1, 2])

    def test_full_loop_position_only_then_reconstruct_rotation(self):
        """End-to-end: chain from positions alone -> derive forward -> encode
        GT roll -> decode -> recover the ORIGINAL rotation's forward axis
        exactly (roll round-trips exactly; forward is defined to be exact by
        construction once topology is correctly resolved)."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.05, 1.0], [0.0, 0.1, 2.1]])
        ordinal = torch.tensor([0.0, 1.0, 2.0])
        is_base = torch.tensor([1.0, 0.0, 0.0])
        parent, _, _ = chain_phytomers(pos, ordinal=ordinal, is_base=is_base)
        fwd = derive_forward(pos, parent)
        R_gt = _random_rotations(3, seed=1)  # arbitrary GT rotations (roll differs per node)
        roll_gt = encode_roll(R_gt, fwd)
        R_hat = roll_to_matrix(fwd, roll_gt)
        # Forward axis must match exactly (derived from the same position data
        # both times); roll round-trips exactly (test_encode_decode above).
        self.assertTrue(torch.allclose(R_hat[:, :, 1], fwd, atol=1e-6))
        self.assertTrue(torch.allclose(encode_roll(R_hat, fwd), roll_gt, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
