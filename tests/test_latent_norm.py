"""--latent_norm: the flow-state latent block is the standardized VAE latent; every consumer gets the raw latent back."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from diffusion_based.models.hierarchical_part_flow_matching import HierarchicalPartFlowMatchingModel, split_flow_state


class TestLatentNorm(unittest.TestCase):
    def test_round_trip_and_buffers(self):
        torch.manual_seed(0)
        model = HierarchicalPartFlowMatchingModel(
            max_phytomers=16, slots_per_phytomer=10, node_dim=16, embed_dim=96, coarse_layers=1, fine_layers=1,
            flow_granularity="phytomer", phytomer_latent_dim=32).eval()
        self.assertFalse(model.latent_norm_active())
        z = torch.randn(5, 32) * 0.4 + 0.1
        self.assertTrue(torch.equal(model.normalize_latent(z), z))          # identity by default
        mu, sigma = torch.randn(32) * 0.1, torch.rand(32) * 0.5 + 0.05
        model.set_latent_norm(mu, sigma)
        self.assertTrue(model.latent_norm_active())
        x = model.normalize_latent(z)
        self.assertTrue(torch.allclose(model.denormalize_latent(x), z, atol=1e-6))
        self.assertTrue(torch.allclose(x, (z - mu) / sigma))
        # the statistics travel with the state dict
        sd = model.state_dict()
        self.assertIn("latent_mu", sd); self.assertTrue(torch.equal(sd["latent_sigma"], sigma))
        # sample_ode hands back the raw (denormalized) latent: with sigma -> 0 the returned latent is mu
        model.set_latent_norm(mu, torch.full((32,), 1e-3))
        img = torch.randn(2, 16, 128, 128); daps = torch.tensor([15.0, 25.0])
        with torch.no_grad():
            so = model.sample_ode(images=img, daps=daps, num_steps=2)
        self.assertTrue(torch.allclose(so["phytomer_latent"], mu.expand_as(so["phytomer_latent"]), atol=0.2))


if __name__ == "__main__":
    unittest.main()
