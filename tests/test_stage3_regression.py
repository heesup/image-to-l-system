"""--stage3_regression: Stage 3 reads its flow state out in one deterministic evaluation at (x = 0, t = 0).
The sample is therefore independent of the ODE step count and of the noise RNG, while the flow-matching
model's sample depends on both."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from plant_recon.models.hierarchical_part_flow_matching import HierarchicalPartFlowMatchingModel


def _model(regression, geometry=False):
    torch.manual_seed(0)
    return HierarchicalPartFlowMatchingModel(
        max_phytomers=16, slots_per_phytomer=10, node_dim=16, embed_dim=96, coarse_layers=1, fine_layers=1,
        flow_granularity="phytomer", phytomer_latent_dim=32, stage3_geometry=geometry,
        stage3_regression=regression).eval()


class TestStage3Regression(unittest.TestCase):
    def _samples(self, model):
        img = torch.randn(2, 16, 128, 128); daps = torch.tensor([15.0, 25.0])
        with torch.no_grad():
            torch.manual_seed(1); a = model.sample_ode(images=img, daps=daps, num_steps=2)
            torch.manual_seed(2); b = model.sample_ode(images=img, daps=daps, num_steps=5)
        return a, b

    def test_regression_is_deterministic_and_step_free(self):
        for geometry in (False, True):
            a, b = self._samples(_model(True, geometry))
            self.assertEqual(a["pred_latent"].shape[-1], 32)
            self.assertTrue(torch.allclose(a["pred_latent"], b["pred_latent"]), f"geometry={geometry}")
            self.assertTrue(torch.allclose(a["phytomer_pos"], b["phytomer_pos"]), f"geometry={geometry}")

    def test_flow_matching_still_samples(self):
        a, b = self._samples(_model(False))
        self.assertFalse(torch.allclose(a["pred_latent"], b["pred_latent"]))


if __name__ == "__main__":
    unittest.main()
