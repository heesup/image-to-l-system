"""Tests for canonical phytomer packet builder (shared util for PhytomerVAE)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from diffusion_based.dataset.phytomer_packets import (
    build_phytomer_packets,
    cluster_organs,
    decode_packet,
)
from diffusion_based.dataset.part_array_dataset import (
    FM_OT_END,
    FM_BASE_START,
    FM_NODE_DIM,
)


def _make_organ(otype: int, base, dtype=torch.float32):
    row = torch.zeros(FM_NODE_DIM, dtype=dtype)
    row[otype] = 1.0
    row[FM_BASE_START:FM_BASE_START + 3] = torch.tensor(base, dtype=dtype) * 20.0
    row[22:25] = 0.02 * 50.0  # nonzero scale
    row[16:22] = torch.tensor([1, 0, 0, 0, 1, 0], dtype=dtype)
    return row


class TestPhytomerPackets(unittest.TestCase):
    def test_single_canonical_phytomer(self):
        """One internode + petiole + 3 leaflets + peduncle + 2 repro -> full packet."""
        rows = [
            _make_organ(3, [0.0, 0.0, 0.10]),   # internode
            _make_organ(4, [0.0, 0.0, 0.12]),   # petiole (cluster center)
            _make_organ(5, [0.05, 0.0, 0.20]),  # leaflets
            _make_organ(5, [-0.05, 0.0, 0.22]),
            _make_organ(5, [0.0, 0.05, 0.21]),
            _make_organ(6, [0.0, 0.0, 0.25]),   # peduncle
            _make_organ(9, [0.01, 0.0, 0.27]),  # flowers
            _make_organ(10, [-0.01, 0.0, 0.28]),
        ]
        nodes = torch.stack(rows)
        packets, presence, centers, refs = build_phytomer_packets(nodes)
        self.assertEqual(packets.shape, (1, 8, 26))
        self.assertTrue(presence.all())
        # Slot roles: 0->internode(3), 1->petiole(4), 2-4->leaf(5), 5->peduncle(6), 6-7->repro
        got = packets[0, :, :FM_OT_END].argmax(-1).tolist()
        self.assertEqual(got, [3, 4, 5, 5, 5, 6, 9, 10])
        # Center is the petiole base
        self.assertTrue(torch.allclose(centers[0], torch.tensor([0.0, 0.0, 0.12]), atol=1e-6))
        # Relative base of slot 1 (petiole) must be ~zero
        self.assertTrue(packets[0, 1, FM_BASE_START:FM_BASE_START + 3].abs().max() < 1e-5)
        # Slot 0 (internode) rotation is relativized to itself -> ~identity
        self.assertTrue(torch.allclose(packets[0, 0, 16:22],
                                       torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32),
                                       atol=1e-5))
        # Roundtrip: decode_packet restores absolute coords up to canonical
        # within-role reordering (leaflets sorted bottom -> top by design).
        back = decode_packet(packets[0], centers[0], presence[0], refs[0])
        self.assertTrue(torch.allclose(back[0], nodes[0], atol=1e-5))   # stem
        self.assertTrue(torch.allclose(back[1], nodes[1], atol=1e-5))   # petiole
        self.assertTrue(torch.allclose(back[5], nodes[5], atol=1e-5))   # peduncle
        leaf_z_in = sorted([0.20, 0.22, 0.21])
        leaf_z_out = sorted((back[2:5, FM_BASE_START + 2] / 20.0).tolist())
        for got, want in zip(leaf_z_out, leaf_z_in):
            self.assertAlmostEqual(got, want, places=5)

    def test_sparse_phytomer_keeps_empty_convention(self):
        """Young phytomer (internode + petiole only): absent slots stay NONE+zero."""
        rows = [
            _make_organ(3, [0.0, 0.0, 0.05]),
            _make_organ(4, [0.0, 0.0, 0.06]),
        ]
        nodes = torch.stack(rows)
        packets, presence, centers, refs = build_phytomer_packets(nodes)
        self.assertEqual(packets.shape, (1, 8, 26))
        self.assertEqual(presence[0].tolist(), [True, True, False, False, False, False, False, False])
        # Absent slots: NONE one-hot + EXACT zero geometry (no center leakage)
        for s in range(2, 8):
            self.assertEqual(packets[0, s, :FM_OT_END].argmax().item(), 0)
            self.assertTrue((packets[0, s, FM_BASE_START:] == 0).all())
        back = decode_packet(packets[0], centers[0], presence[0], refs[0])
        self.assertTrue(torch.allclose(back[:2], nodes[:2], atol=1e-5))
        # Absent slots untouched by re-anchoring
        self.assertTrue((back[2:, FM_BASE_START:] == 0).all())

    def test_overflow_drops_by_canonical_order(self):
        """4th leaflet is dropped; first 3 by (z, azimuth) kept. Deterministic."""
        rows = [_make_organ(4, [0.0, 0.0, 0.10])]
        leaf_z = [0.30, 0.20, 0.25, 0.22]
        for z in leaf_z:
            rows.append(_make_organ(5, [0.05, 0.0, z]))
        nodes = torch.stack(rows)
        stats = {}
        packets, presence, centers, refs = build_phytomer_packets(nodes, drop_stats=stats)
        self.assertEqual(stats.get("dropped_organs", 0), 1)
        kept_z = sorted((packets[0, 2:5, FM_BASE_START + 2] / 20.0 + centers[0][2]).tolist())
        # canonical order keeps lowest-z first: 0.20, 0.22, 0.25 (0.30 dropped)
        for got, want in zip(kept_z, [0.20, 0.22, 0.25]):
            self.assertAlmostEqual(got, want, places=5)

    def test_empty_input(self):
        nodes = torch.zeros((10, FM_NODE_DIM))
        packets, presence, centers, refs = build_phytomer_packets(nodes)
        self.assertEqual(packets.shape, (0, 8, 26))
        self.assertEqual(presence.shape, (0, 8))
        self.assertEqual(centers.shape, (0, 3))
        self.assertEqual(refs.shape, (0, 6))

    def test_cluster_organs_matches_matcher_rule(self):
        """Two petiole groups 0.5m apart -> 2 clusters; far internode splits off."""
        pos = torch.tensor([
            [0.0, 0.0, 0.10], [0.0, 0.0, 0.10],   # petiole A x?? (use distinct)
            [0.5, 0.0, 0.10],
        ])
        # distinct petiole positions
        pos = torch.tensor([[0.0, 0.0, 0.10], [0.5, 0.0, 0.10], [0.0, 0.0, 0.30]])
        lab = torch.tensor([4, 4, 3])
        centers, assign = cluster_organs(pos, lab)
        # internode 0.2 above petiole A... dist to A = 0.2 > 0.04 -> standalone
        self.assertEqual(centers.shape[0], 3)
        self.assertEqual(sorted(assign.tolist()), [0, 1, 2])

    def test_relative_rotation_roundtrip_exact(self):
        """Differences from the reference survive build+decode exactly (rot 6D)."""
        # Internode tilted 45 deg about y (reference), petiole/leaflet rotated
        # additionally so their ABSOLUTE rots are non-trivially off from the ref.
        def rot6d(R):
            return torch.cat([R[:, 0], R[:, 1]])
        def Rx(a):
            c, s = torch.cos(torch.tensor(a)), torch.sin(torch.tensor(a))
            return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]])
        def Ry(a):
            c, s = torch.cos(torch.tensor(a)), torch.sin(torch.tensor(a))
            return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        ref = rot6d(Ry(0.6) @ Rx(0.3))
        org_rot = rot6d(Ry(0.6) @ Rx(0.1) @ Ry(-1.1))  # differs from ref in two axes
        rows = [
            _make_organ(3, [0.0, 0.0, 0.10]),
            _make_organ(4, [0.0, 0.0, 0.12]),
        ]
        nodes = torch.stack(rows).clone()
        nodes[0, 16:22] = ref
        nodes[1, 16:22] = org_rot
        packets, presence, centers, refs = build_phytomer_packets(nodes)
        back = decode_packet(packets[0], centers[0], presence[0], refs[0])
        # Decoded rotations must match the originals (relative transform is exact).
        self.assertTrue(torch.allclose(back[0, 16:22], ref, atol=1e-5))
        self.assertTrue(torch.allclose(back[1, 16:22], org_rot, atol=1e-5))

    def test_option1_explicit_anchor_reference(self):
        """Option 1: an externally-provided anchor frame relativizes ALL slots.

        Pass reference_rot = a fixed anchor rotation; every present organ
        (including slot-0 internode) is expressed relative to that frame, and
        decode_packet with the same reference recovers the original ABSOLUTE rots.
        """
        def rot6d(R):
            return torch.cat([R[:, 0], R[:, 1]])
        def Rx(a):
            c, s = torch.cos(torch.tensor(a)), torch.sin(torch.tensor(a))
            return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]])
        def Ry(a):
            c, s = torch.cos(torch.tensor(a)), torch.sin(torch.tensor(a))
            return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        anchor_rot = rot6d(Ry(0.4) @ Rx(0.2))          # the anchor/node frame
        org_rot = rot6d(Ry(1.2) @ Rx(0.5))             # a leaflet far from anchor
        rows = [
            _make_organ(3, [0.0, 0.0, 0.10]),
            _make_organ(5, [0.05, 0.0, 0.20]),
        ]
        nodes = torch.stack(rows).clone()
        nodes[0, 16:22] = anchor_rot                    # internode == anchor frame
        nodes[1, 16:22] = org_rot
        packets, presence, centers, refs = build_phytomer_packets(
            nodes, reference_rot=anchor_rot)
        # reference_rots output must be the provided anchor frame (option 1)
        self.assertTrue(torch.allclose(refs[0], anchor_rot, atol=1e-5))
        # The leaflet (type 5) occupies slot 2 (leaflet role range), not slot 1.
        self.assertTrue(torch.allclose(packets[0, 0, 16:22],
                                       torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32),
                                       atol=1e-5))
        # decode with the anchor reference recovers absolute rots
        back = decode_packet(packets[0], centers[0], presence[0], refs[0])
        self.assertTrue(torch.allclose(back[0, 16:22], anchor_rot, atol=1e-5))
        self.assertTrue(torch.allclose(back[2, 16:22], org_rot, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
