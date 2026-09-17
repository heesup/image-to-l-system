"""--stage3_gt_nodes: the Stage 3 conditioning override leaves Stage 2 untouched, changes
Stage 3's velocity, and sample_ode reports the override as the plant's node geometry."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from plant_recon.models.hierarchical_part_flow_matching import HierarchicalPartFlowMatchingModel


class TestStage3TeacherForcing(unittest.TestCase):
    def test_cond_override(self):
        torch.manual_seed(0)
        model = HierarchicalPartFlowMatchingModel(
            max_phytomers=16, slots_per_phytomer=10, node_dim=16, embed_dim=96, coarse_layers=1, fine_layers=1,
            flow_granularity="phytomer", phytomer_latent_dim=32).eval()
        B, D = 2, 32
        img = torch.randn(B, 16, 128, 128); daps = torch.tensor([15.0, 25.0])
        with torch.no_grad():
            so0 = model.sample_ode(images=img, daps=daps, num_steps=2)
            K = so0["phytomer_pos"].shape[1]
            ov = {"pos": torch.randn(B, K, 3), "roll": F.normalize(torch.randn(B, K, 2), dim=-1), "scale": torch.rand(B, K, 3)}
            x = torch.randn(B, K, D); t = torch.rand(B)
            out0 = model(noisy_fine_nodes=x, timesteps=t, images=img, daps=daps)
            out1 = model(noisy_fine_nodes=x, timesteps=t, images=img, daps=daps, stage3_cond_override=ov)
            so1 = model.sample_ode(images=img, daps=daps, num_steps=2, cond_override=ov)
        # Stage 2's own outputs are not affected by the override ...
        self.assertTrue(torch.equal(out0["pred_phytomer_pos"], out1["pred_phytomer_pos"]))
        self.assertTrue(torch.equal(out0["pred_phytomer_roll"], out1["pred_phytomer_roll"]))
        # ... Stage 3's velocity is (it is conditioned on the overridden geometry)
        self.assertFalse(torch.allclose(out0["pred_velocity"], out1["pred_velocity"]))
        # sample_ode reports the override as the node geometry (latent-only Stage 3)
        self.assertTrue(torch.equal(so1["phytomer_pos"], ov["pos"]))
        self.assertTrue(torch.equal(so1["phytomer_roll"], ov["roll"]))
        self.assertTrue(torch.equal(so1["phytomer_scale"], ov["scale"]))
        self.assertFalse(torch.equal(so0["phytomer_pos"], so1["phytomer_pos"]))


if __name__ == "__main__":
    unittest.main()
