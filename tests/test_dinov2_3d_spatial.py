"""
Unit tests for DINOv2 3D Spatial Vision and Hierarchical Flow Matching.
"""

import unittest
import torch

from diffusion_based.models.dinov2_ray_encoder import DINOv2RayEncoder
from diffusion_based.models.hierarchical_part_flow_matching import (
    CoarseSkeletalTransformer,
    HierarchicalPartFlowMatchingModel,
)


class TestDINOv23DSpatial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_dinov2_ray_encoder_forward(self):
        """Verify DINOv2RayEncoder extracts tokens and injects 3D ray embeddings."""
        encoder = DINOv2RayEncoder(pretrained=True, freeze_backbone=True).to(self.device)
        dummy_rgb = torch.randn(2, 3, 224, 224, device=self.device)
        tokens = encoder(dummy_rgb)
        self.assertEqual(tokens.shape, (2, 257, 384))

        # Also test 16-channel multi-zoom input with auto-resize
        dummy_16ch = torch.randn(2, 16, 128, 128, device=self.device)
        tokens_16 = encoder(dummy_16ch)
        self.assertEqual(tokens_16.shape, (2, 257, 384))

    def test_coarse_skeletal_3d_reference_points(self):
        """Verify DETR3D-style 3D reference points and delta pos head."""
        coarse = CoarseSkeletalTransformer(max_anchors=32, embed_dim=384, num_heads=4, num_layers=2).to(self.device)
        self.assertTrue(hasattr(coarse, "ref_points"))
        self.assertEqual(coarse.ref_points.shape, (32, 3))

        dummy_tokens = torch.randn(2, 257, 384, device=self.device)
        out = coarse(dummy_tokens, active_k=16)

        self.assertEqual(out["anchor_pos"].shape, (2, 16, 3))
        self.assertEqual(out["anchor_rot"].shape, (2, 16, 6))
        self.assertEqual(out["anchor_logits"].shape, (2, 16, 1))
        self.assertEqual(out["anchor_features"].shape, (2, 16, 384))

    def test_end_to_end_model_forward_backward(self):
        """Verify full 4-in-1 model forward and backward pass."""
        model = HierarchicalPartFlowMatchingModel(
            max_anchors=16,
            slots_per_anchor=8,
            node_dim=16,
            embed_dim=384,
            coarse_layers=2,
            fine_layers=2,
        ).to(self.device)

        B = 2
        K = 16
        M = 8
        N_fine = K * M
        noisy_nodes = torch.randn(B, N_fine, 16, device=self.device)
        timesteps = torch.rand(B, device=self.device)
        images = torch.randn(B, 16, 128, 128, device=self.device)
        daps = torch.tensor([25.0, 45.0], device=self.device)

        out = model(noisy_fine_nodes=noisy_nodes, timesteps=timesteps, images=images, daps=daps)

        self.assertIn("pred_anchor_pos", out)
        self.assertIn("pred_anchor_logits", out)
        self.assertIn("pred_velocity", out)
        self.assertIn("pred_fine_exist_logits", out)

        # Check gradient backprop through the entire pipeline
        loss = out["pred_anchor_pos"].sum() + out["pred_velocity"].sum()
        loss.backward()

        # Check that ray_mlp and decoder received gradients
        self.assertIsNotNone(model.image_encoder.ray_mlp[0].weight.grad)
        self.assertIsNotNone(model.coarse_stage.ref_points.grad)
        self.assertIsNotNone(model.fine_stage.velocity_head[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
