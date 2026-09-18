"""--render_feedback: the learned correction from the render residual.

An untrained corrector must be the identity (its output layer is zero-initialised), so turning the flag
on cannot change a checkpoint's predictions until it has been trained; and once its weights are not zero
it must actually move the nodes it is given.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from plant_recon.models.hierarchical_part_flow_matching import (HierarchicalPartFlowMatchingModel,
                                                                 RenderFeedbackCorrector)


class TestRenderFeedback(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B, self.K, self.D, self.S = 2, 5, 32, 64
        self.c = RenderFeedbackCorrector(embed_dim=96, latent_dim=self.D)
        self.pos = torch.randn(self.B, self.K, 3) * 0.1
        self.scale = torch.rand(self.B, self.K, 3)
        self.lat = torch.randn(self.B, self.K, self.D) * 0.2
        self.res = torch.randn(self.B, self.S, self.S)
        self.dep = torch.rand(self.B, self.S, self.S)

    def test_zero_init_is_identity(self):
        dpos, dscale, dlat = self.c(self.res, self.dep, self.pos, self.scale, self.lat)
        for t in (dpos, dscale, dlat):
            self.assertTrue(torch.allclose(t, torch.zeros_like(t)), "untrained corrector must not move anything")

    def test_trained_weights_move_nodes(self):
        torch.nn.init.normal_(self.c.head[-1].weight, std=0.05)
        dpos, dscale, dlat = self.c(self.res, self.dep, self.pos, self.scale, self.lat)
        self.assertGreater(dpos.abs().mean().item(), 0.0)
        self.assertEqual(dpos.shape, self.pos.shape)
        self.assertEqual(dlat.shape, self.lat.shape)

    def test_residual_changes_the_correction(self):
        torch.nn.init.normal_(self.c.head[-1].weight, std=0.05)
        a = self.c(self.res, self.dep, self.pos, self.scale, self.lat)[0]
        b = self.c(self.res * -1.0, self.dep, self.pos, self.scale, self.lat)[0]
        self.assertFalse(torch.allclose(a, b), "the correction must depend on the residual")

    def test_model_flag_builds_and_defaults_off(self):
        off = HierarchicalPartFlowMatchingModel(max_phytomers=8, slots_per_phytomer=10, node_dim=16,
                                                embed_dim=96, coarse_layers=1, fine_layers=1,
                                                flow_granularity="phytomer", phytomer_latent_dim=32)
        self.assertIsNone(off.feedback)
        on = HierarchicalPartFlowMatchingModel(max_phytomers=8, slots_per_phytomer=10, node_dim=16,
                                               embed_dim=96, coarse_layers=1, fine_layers=1,
                                               flow_granularity="phytomer", phytomer_latent_dim=32,
                                               render_feedback=True)
        self.assertIsNotNone(on.feedback)


if __name__ == "__main__":
    unittest.main()
