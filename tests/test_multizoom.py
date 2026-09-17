"""--multizoom: pyramid tokens through the frozen backbone, level-aware node token, old checkpoints still load."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from plant_recon.models.hierarchical_part_flow_matching import HierarchicalPartFlowMatchingModel, PhytomerVisualProjector


class TestMultizoom(unittest.TestCase):
    def test_projector_picks_finest_level(self):
        torch.manual_seed(0)
        proj = PhytomerVisualProjector(embed_dim=8, num_levels=4)
        B, L, N, C = 1, 4, 16, 8
        # level l patches carry the constant value l so the sampled feature reveals the chosen level
        patches = torch.arange(L, dtype=torch.float32).view(1, L, 1, 1).expand(B, L, N, C).reshape(B, L * N, C)
        tokens = torch.cat([torch.zeros(B, 1, C), patches], dim=1)
        pos = torch.tensor([[[0.0, 0.0, 0.1], [0.5, 0.0, 0.1], [0.12, 0.0, 0.1]]])   # origin / far / mid
        with torch.no_grad():
            # bypass the zero-initialised fusion: read the sampled level through pos_to_uv geometry
            proj.fusion[-1].weight.zero_(); proj.fusion[-1].bias.zero_()
            proj.fusion = torch.nn.Identity()
            out = proj(tokens, pos)   # Identity fusion -> cat([sampled, pos]) (B, K, C + 3)
        lvl = out[0, :, 0]
        self.assertEqual(int(lvl[0].item()), 3)       # at the origin every level contains the node -> finest (8x)
        self.assertEqual(int(lvl[1].item()), 0)       # 0.5 m out: only the 1x view
        self.assertTrue(0 < lvl[2].item() < 3)        # in between

    def test_model_forward_and_sample_multizoom(self):
        torch.manual_seed(0)
        model = HierarchicalPartFlowMatchingModel(
            max_phytomers=16, slots_per_phytomer=10, node_dim=16, embed_dim=96, coarse_layers=1, fine_layers=1,
            flow_granularity="phytomer", phytomer_latent_dim=32, multizoom=True).eval()
        self.assertEqual(model.num_levels, 4)
        model.fine_stage.phytomer_projector.window = 3   # windowed node token must run too
        img = torch.randn(2, 16, 128, 128); daps = torch.tensor([15.0, 25.0])
        with torch.no_grad():
            tok = model.image_encoder(img)
            self.assertEqual(tok.shape[1], 1 + 4 * model.image_encoder.num_patches)
            so = model.sample_ode(images=img, daps=daps, num_steps=2)
            K = so["phytomer_pos"].shape[1]
            out = model(noisy_fine_nodes=torch.randn(2, K, 32), timesteps=torch.rand(2), images=img, daps=daps)
        self.assertEqual(out["pred_velocity"].shape, (2, K, 32))
        # a checkpoint without multizoom parameters loads (level embeddings stay at their zero init)
        plain = HierarchicalPartFlowMatchingModel(
            max_phytomers=16, slots_per_phytomer=10, node_dim=16, embed_dim=96, coarse_layers=1, fine_layers=1,
            flow_granularity="phytomer", phytomer_latent_dim=32)
        missing, unexpected = model.load_state_dict(plain.state_dict(), strict=False)
        self.assertTrue(all("level_embed" in k for k in missing), missing)
        self.assertEqual(unexpected, [])


if __name__ == "__main__":
    unittest.main()
