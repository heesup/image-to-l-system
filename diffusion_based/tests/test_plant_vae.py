"""
Unit Tests for PlantOrganVAE & PlantTransformerVAE.
"""

import unittest
import torch
from diffusion_based.models.plant_organ_array import PlantOrganArray, NUM_FEATURES_TYPED
from diffusion_based.models.plant_vae import (
    PlantOrganVAE,
    PlantTransformerVAE,
    compute_organ_vae_loss,
)


class TestPlantVAE(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_organ_vae_shapes_and_loss(self):
        model = PlantOrganVAE(latent_dim=16).to(self.device)
        B, N = 4, 32
        x = torch.randn(B, N, NUM_FEATURES_TYPED, device=self.device)
        x[..., 11] = torch.randint(0, 8, (B, N), device=self.device).float()
        x[..., 12] = torch.randint(0, 2, (B, N), device=self.device).float()
        x[..., 32] = torch.randint(0, 3, (B, N), device=self.device).float()
        x[..., 34] = torch.randint(0, 2, (B, N), device=self.device).float()
        x[..., 39] = 1.0  # existence

        loss_dict = compute_organ_vae_loss(model, x, beta=1e-3)
        self.assertIn("loss", loss_dict)
        self.assertTrue(torch.isfinite(loss_dict["loss"]))

        # Check backward
        loss_dict["loss"].backward()
        for p in model.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())

        # Test decode
        mu, logvar = model.encode(x)
        self.assertEqual(mu.shape, (B, N, 16))
        z = model.reparameterize(mu, logvar)
        recon_hard = model.decode(z, hard_categoricals=True)
        self.assertEqual(recon_hard.shape, (B, N, NUM_FEATURES_TYPED))

    def test_transformer_vae_shapes(self):
        model = PlantTransformerVAE(latent_dim=64, d_model=64, nhead=2, num_encoder_layers=2, num_decoder_layers=2).to(self.device)
        B, N = 2, 20
        x = torch.randn(B, N, NUM_FEATURES_TYPED, device=self.device)
        x[..., 11] = torch.randint(0, 8, (B, N), device=self.device).float()
        x[..., 12] = torch.randint(0, 2, (B, N), device=self.device).float()
        x[..., 32] = torch.randint(0, 3, (B, N), device=self.device).float()
        x[..., 34] = torch.randint(0, 2, (B, N), device=self.device).float()
        x[..., 39] = 1.0

        recon_x, mu, logvar = model(x)
        self.assertEqual(recon_x.shape, (B, N, NUM_FEATURES_TYPED))
        self.assertEqual(mu.shape, (B, 64))
        self.assertEqual(logvar.shape, (B, 64))


if __name__ == "__main__":
    unittest.main()
