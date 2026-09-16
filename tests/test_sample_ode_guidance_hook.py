"""sample_ode's guidance_fn hook (2026-09-16): identity by default; when supplied, is called once per
step with the documented kwargs and its return value replaces v_1 for that step."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from diffusion_based.models.hierarchical_part_flow_matching import HierarchicalPartFlowMatchingModel


class TestSampleOdeGuidanceHook(unittest.TestCase):
    def _model(self):
        torch.manual_seed(0)
        return HierarchicalPartFlowMatchingModel(
            max_phytomers=16, slots_per_phytomer=10, node_dim=16, embed_dim=96, coarse_layers=1, fine_layers=1,
            flow_granularity="phytomer", phytomer_latent_dim=32).eval()

    def test_none_is_identical_to_no_hook(self):
        model = self._model()
        img = torch.randn(2, 16, 128, 128); daps = torch.tensor([15.0, 25.0])
        with torch.no_grad():
            torch.manual_seed(1); a = model.sample_ode(images=img, daps=daps, num_steps=3)
            torch.manual_seed(1); b = model.sample_ode(images=img, daps=daps, num_steps=3, guidance_fn=None)
        self.assertTrue(torch.equal(a["pred_latent"], b["pred_latent"]))
        self.assertTrue(torch.equal(a["phytomer_pos"], b["phytomer_pos"]))

    def test_hook_called_every_step_and_its_return_value_is_used(self):
        model = self._model()
        img = torch.randn(2, 16, 128, 128); daps = torch.tensor([15.0, 25.0])
        calls = []

        def hook(model, x, v1, t, dt, step, forward_kwargs, chain_parent_pos):
            calls.append((step, float(t)))
            self.assertIs(model, self._last_model)
            self.assertEqual(x.shape, v1.shape)
            self.assertIn("phytomer_pos", forward_kwargs)
            self.assertEqual(chain_parent_pos.shape, x.shape[:2] + (3,))
            return torch.zeros_like(v1)  # zero velocity -> x should never move past x_0

        self._last_model = model
        num_steps = 4
        with torch.no_grad():
            torch.manual_seed(2); baseline = model.sample_ode(images=img, daps=daps, num_steps=num_steps)
            torch.manual_seed(2)
            guided = model.sample_ode(images=img, daps=daps, num_steps=num_steps, guidance_fn=hook)
        self.assertEqual([c[0] for c in calls], list(range(num_steps)))
        # zero velocity every step -> the flow state never left its t=0 sample -> different result from baseline
        self.assertFalse(torch.equal(baseline["pred_latent"], guided["pred_latent"]))

    def test_hook_ignored_for_non_phytomer_granularity(self):
        torch.manual_seed(0)
        model = HierarchicalPartFlowMatchingModel(
            max_phytomers=16, slots_per_phytomer=10, node_dim=16, embed_dim=96, coarse_layers=1, fine_layers=1,
            flow_granularity="organ").eval()
        img = torch.randn(2, 16, 128, 128); daps = torch.tensor([15.0, 25.0])
        called = {"n": 0}

        def hook(**kw):
            called["n"] += 1
            return kw["v1"]

        with torch.no_grad():
            model.sample_ode(images=img, daps=daps, num_steps=2, guidance_fn=hook)
        self.assertEqual(called["n"], 0)


if __name__ == "__main__":
    unittest.main()
