"""
Unit tests for Hierarchical Matryoshka Botanical Flow Matching System.
"""

import unittest
import torch

from diffusion_based.models.hierarchical_part_flow_matching import (
    HierarchicalPartFlowMatchingModel,
    compute_matryoshka_slice,
)
from diffusion_based.training.hierarchical_hungarian_matcher import (
    HierarchicalBotanicalMatcher,
)


class TestHierarchicalFlowMatching(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.B = 2
        self.max_anchors = 64  # Compact scale for quick test
        self.slots_per_anchor = 8
        self.image_size = 64
        self.node_dim = 13
        self.num_classes = 13

        self.model = HierarchicalPartFlowMatchingModel(
            max_anchors=self.max_anchors,
            slots_per_anchor=self.slots_per_anchor,
            node_dim=self.node_dim,
            num_classes=self.num_classes,
            image_size=self.image_size,
            patch_size=8,
            embed_dim=128,
            vit_layers=2,
            vit_heads=4,
            coarse_layers=2,
            fine_layers=2,
        ).to(self.device)

    def test_matryoshka_slicing(self):
        # Test exponential doubling schedule
        dap_young = torch.tensor([3.0])
        dap_mid = torch.tensor([25.0])
        dap_old = torch.tensor([85.0])

        k_young = compute_matryoshka_slice(dap_young, max_anchors=512)
        k_mid = compute_matryoshka_slice(dap_mid, max_anchors=512)
        k_old = compute_matryoshka_slice(dap_old, max_anchors=512)

        self.assertEqual(k_young, 16)
        self.assertGreaterEqual(k_mid, 64)
        self.assertEqual(k_old, 512)

    def test_forward_pass(self):
        images = torch.randn(self.B, 4, self.image_size, self.image_size, device=self.device)
        daps = torch.tensor([10.0, 40.0], device=self.device)
        timesteps = torch.rand(self.B, device=self.device)

        # Slice fine slots
        active_k = compute_matryoshka_slice(daps, max_anchors=self.max_anchors)
        N_fine = active_k * self.slots_per_anchor
        noisy_nodes = torch.randn(self.B, N_fine, self.node_dim, device=self.device)

        out = self.model(
            noisy_fine_nodes=noisy_nodes,
            timesteps=timesteps,
            images=images,
            daps=daps,
        )

        self.assertIn("pred_anchor_pos", out)
        self.assertIn("pred_velocity", out)
        self.assertIn("pred_fine_exist_logits", out)
        self.assertIn("pred_dap", out)

        self.assertEqual(out["pred_anchor_pos"].shape, (self.B, active_k, 3))
        self.assertEqual(out["pred_velocity"].shape, (self.B, N_fine, self.node_dim))
        self.assertEqual(out["pred_fine_exist_logits"].shape, (self.B, N_fine, 1))

        # Check backward pass
        loss = out["pred_velocity"].sum() + out["pred_fine_exist_logits"].sum() + out["pred_anchor_pos"].sum()
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")

    def test_matcher(self):
        matcher = HierarchicalBotanicalMatcher(slots_per_anchor=self.slots_per_anchor)

        pred_anchor_pos = torch.randn(self.B, 32, 3, device=self.device)
        pred_anchor_logits = torch.randn(self.B, 32, 1, device=self.device)
        pred_fine_geom = torch.randn(self.B, 32 * self.slots_per_anchor, self.node_dim, device=self.device)
        pred_fine_logits = torch.randn(self.B, 32 * self.slots_per_anchor, self.num_classes, device=self.device)

        # Create synthetic GT organs
        tgt_geoms = [
            torch.randn(25, self.node_dim, device=self.device),
            torch.randn(45, self.node_dim, device=self.device),
        ]
        tgt_labels = [
            torch.randint(1, self.num_classes, (25,), device=self.device),
            torch.randint(1, self.num_classes, (45,), device=self.device),
        ]

        matches = matcher(
            pred_anchor_pos=pred_anchor_pos,
            pred_anchor_logits=pred_anchor_logits,
            pred_fine_geom=pred_fine_geom,
            pred_fine_logits=pred_fine_logits,
            tgt_geoms=tgt_geoms,
            tgt_labels=tgt_labels,
        )

        self.assertEqual(len(matches), self.B)
        for b in range(self.B):
            self.assertIn("anchor_src_idx", matches[b])
            self.assertIn("fine_src_idx", matches[b])
            self.assertIn("fine_tgt_idx", matches[b])

    def test_ode_sampling(self):
        images = torch.randn(self.B, 4, self.image_size, self.image_size, device=self.device)
        daps = torch.tensor([5.0, 15.0], device=self.device)

        sample_out = self.model.sample_ode(images=images, daps=daps, num_steps=3)
        self.assertIn("pred_geometry", sample_out)
        self.assertIn("pred_cls", sample_out)
        self.assertIn("slot_active", sample_out)
        self.assertIn("anchor_pos", sample_out)

    def test_forward_backward_step(self):
        from diffusion_based.training.train_hierarchical_flow_matching import forward_backward_step
        from diffusion_based.training.flow_matching import FlowMatchingScheduler

        scheduler = FlowMatchingScheduler()
        matcher = HierarchicalBotanicalMatcher(slots_per_anchor=self.slots_per_anchor)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        # Construct synthetic batch
        N_max = self.max_anchors * self.slots_per_anchor
        nodes = torch.zeros(self.B, N_max, 26, device=self.device)
        # Set some active organs
        nodes[:, :20, 1] = 1.0  # class 1
        nodes[:, :20, 13:] = torch.randn(self.B, 20, 13, device=self.device)
        existence_mask = torch.zeros(self.B, N_max, device=self.device)
        existence_mask[:, :20] = 1.0

        batch = {
            "image": torch.randn(self.B, 4, self.image_size, self.image_size, device=self.device),
            "nodes": nodes,
            "existence_mask": existence_mask,
            "dap": torch.tensor([15.0, 30.0], device=self.device),
        }

        metrics = forward_backward_step(
            model=self.model,
            batch=batch,
            optimizer=optimizer,
            scheduler=scheduler,
            matcher=matcher,
            device=self.device,
            slots_per_anchor=self.slots_per_anchor,
            renderer=None,
        )

        self.assertIsNotNone(metrics)
        self.assertIn("loss", metrics)
        self.assertIn("fine_vel_loss", metrics)
        self.assertIn("fine_exist_loss", metrics)
        self.assertIn("anchor_pos_loss", metrics)
        self.assertIn("anchor_exist_loss", metrics)
        self.assertIn("dense_depth_loss", metrics)
        self.assertIn("cos_color_loss", metrics)
        self.assertIn("cls_acc", metrics)


if __name__ == "__main__":
    unittest.main()
