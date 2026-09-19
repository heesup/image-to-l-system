"""The hybrid (absolute) Stage 3 state layout, and that it does not disturb the parent-relative one.

Absolute layout: [ pos*BASE_SCALE (3) | rot6d (6) | scale (3) | latent (D) ] = 12 + D
Parent-relative: [ dpos*BASE_SCALE (3) | roll   (2) | scale (3) | latent (D) ] =  8 + D

The rotation width is the whole difference. Parent-relative carries only roll because the forward axis
comes from the resolved chain; absolute must carry the full rotation because no chain is resolved while
sampling. A node's absolute position must therefore depend on NOTHING but its own slice of the state --
that independence is what removes the inference-time compounding and the render-pass gradient
accumulation up the chain, and it is what this file guards.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from plant_recon.models.hierarchical_part_flow_matching import (
    BASE_SCALE, STAGE3_GEOM_DIM, STAGE3_GEOM_DIM_ABS,
    geometry_from_flow, geometry_from_flow_absolute, split_flow_state, stage3_geom_dim,
)


class TestStage3AbsoluteLayout(unittest.TestCase):
    D = 16
    B, K = 2, 7

    def test_widths(self):
        self.assertEqual(stage3_geom_dim(False), STAGE3_GEOM_DIM)
        self.assertEqual(stage3_geom_dim(True), STAGE3_GEOM_DIM_ABS)
        self.assertEqual(STAGE3_GEOM_DIM_ABS - STAGE3_GEOM_DIM, 4)  # roll(2) -> rot6d(6)

    def test_split_handles_both_layouts_without_being_told(self):
        for width in (STAGE3_GEOM_DIM, STAGE3_GEOM_DIM_ABS):
            x = torch.randn(self.B, self.K, width + self.D)
            geom, lat = split_flow_state(x, self.D)
            self.assertEqual(geom.shape[-1], width)
            self.assertEqual(lat.shape[-1], self.D)
            torch.testing.assert_close(lat, x[..., -self.D:])

    def test_latent_only_state_still_returns_no_geometry(self):
        """organ granularity flows the latent alone; that path must be untouched."""
        geom, lat = split_flow_state(torch.randn(self.B, self.K, self.D), self.D)
        self.assertIsNone(geom)
        self.assertEqual(lat.shape[-1], self.D)

    def test_absolute_position_is_independent_of_every_other_node(self):
        """The property the hybrid exists for: perturbing one node must not move another."""
        g = torch.randn(self.B, self.K, STAGE3_GEOM_DIM_ABS)
        pos_a, _, _ = geometry_from_flow_absolute(g)
        g2 = g.clone()
        g2[:, 0, :3] += 5.0                      # displace node 0 hard
        pos_b, _, _ = geometry_from_flow_absolute(g2)
        moved = (pos_a - pos_b).norm(dim=-1) > 1e-6
        self.assertTrue(bool(moved[:, 0].all()), "node 0 should have moved")
        self.assertFalse(bool(moved[:, 1:].any()), "no other node may move with it")

    def test_parent_relative_position_DOES_depend_on_the_parent(self):
        """The contrast, stated so the difference is not silently lost later."""
        g = torch.randn(self.B, self.K, STAGE3_GEOM_DIM)
        parent = torch.randn(self.B, self.K, 3)
        pos_a, _, _ = geometry_from_flow(g, parent)
        pos_b, _, _ = geometry_from_flow(g, parent + 1.0)
        self.assertTrue(bool(((pos_a - pos_b).norm(dim=-1) > 1e-6).all()))

    def test_absolute_scale_matches_the_position_convention(self):
        g = torch.zeros(self.B, self.K, STAGE3_GEOM_DIM_ABS)
        g[..., :3] = torch.tensor([1.0, 2.0, 3.0]) * BASE_SCALE
        g[..., 9:12] = 0.25
        pos, rot6d, scale = geometry_from_flow_absolute(g)
        torch.testing.assert_close(pos[0, 0], torch.tensor([1.0, 2.0, 3.0]))
        torch.testing.assert_close(scale[0, 0], torch.full((3,), 0.25))
        self.assertEqual(rot6d.shape[-1], 6)


if __name__ == "__main__":
    unittest.main()
