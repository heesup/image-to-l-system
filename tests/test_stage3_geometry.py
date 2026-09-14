"""Stage 3 geometry block: state split / reassembly, and widening a latent-only checkpoint."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from diffusion_based.models.hierarchical_part_flow_matching import (
    split_flow_state, geometry_from_flow, STAGE3_GEOM_DIM, PhytomerFlowMatchingDecoder, BASE_SCALE)
from diffusion_based.training.train_hierarchical_flow_matching import _widen_optimizer_state


class TestStage3Geometry(unittest.TestCase):
    def test_split_and_reassemble(self):
        D = 16
        x = torch.randn(2, 5, STAGE3_GEOM_DIM + D)
        geom, lat = split_flow_state(x, D)
        self.assertEqual(tuple(geom.shape), (2, 5, STAGE3_GEOM_DIM)); self.assertEqual(tuple(lat.shape), (2, 5, D))
        self.assertTrue(torch.equal(lat, x[..., -D:]))
        g2, l2 = split_flow_state(torch.randn(2, 5, D), D)
        self.assertIsNone(g2); self.assertEqual(tuple(l2.shape), (2, 5, D))
        parent = torch.randn(2, 5, 3); parent[0, 0] = float("nan")          # a root: NaN parent -> origin
        pos, roll, scale = geometry_from_flow(geom, parent)
        self.assertTrue(torch.allclose(pos[1], parent[1] + geom[1, :, :3] / BASE_SCALE))
        self.assertTrue(torch.allclose(pos[0, 0], geom[0, 0, :3] / BASE_SCALE))
        self.assertTrue(torch.allclose(roll.norm(dim=-1), torch.ones(2, 5), atol=1e-3))
        self.assertTrue(torch.equal(scale, geom[..., 5:8]))

    def test_decoder_widths(self):
        d0 = PhytomerFlowMatchingDecoder(latent_dim=16, base_dim=0, rot_dim=0, scale_dim=0, embed_dim=32, num_heads=4, num_layers=1)
        d1 = PhytomerFlowMatchingDecoder(latent_dim=16, base_dim=3, rot_dim=2, scale_dim=3, embed_dim=32, num_heads=4, num_layers=1)
        self.assertEqual(d0.node_flow_dim, 16); self.assertEqual(d1.node_flow_dim, 16 + STAGE3_GEOM_DIM)

    def test_widen_optimizer_state_keeps_latent_block_last(self):
        p_old = torch.nn.Parameter(torch.zeros(4, 16)); p_new = torch.nn.Parameter(torch.zeros(4, 24))
        opt = torch.optim.AdamW([p_new])
        m = torch.arange(64.0).reshape(4, 16)
        opt.state[p_new] = {"step": torch.tensor(3.0), "exp_avg": m.clone(), "exp_avg_sq": m.clone() ** 2}
        n = _widen_optimizer_state(opt)
        self.assertEqual(n, 2)
        self.assertEqual(tuple(opt.state[p_new]["exp_avg"].shape), (4, 24))
        self.assertTrue(torch.equal(opt.state[p_new]["exp_avg"][:, -16:], m))
        self.assertTrue(torch.all(opt.state[p_new]["exp_avg"][:, :8] == 0))


if __name__ == "__main__":
    unittest.main()
