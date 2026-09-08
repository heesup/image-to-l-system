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

    def test_role_partitioned_matcher(self):
        """Verify that GT organs match strictly to their canonical phytomer slots:
        Slot 0 = Stem, Slot 1 = Petiole, Slots 2..4 = Leaves, Slot 5 = Peduncle, Slots 6..7 = Flower/Fruit.
        Also verifies that unifoliate (2 leaves) leaves slot 4 unmatched without error.
        """
        matcher = HierarchicalBotanicalMatcher(slots_per_anchor=8)

        pred_anchor_pos = torch.zeros(1, 1, 3, device=self.device)  # 1 anchor at origin
        pred_anchor_logits = torch.tensor([[[5.0]]], device=self.device)  # Confident anchor
        pred_fine_geom = torch.randn(1, 8, 16, device=self.device)
        pred_fine_logits = torch.ones(1, 8, 1, device=self.device) * 2.0  # Confident existence

        # Case 1: Complete Reproductive Phytomer (1 stem, 1 petiole, 3 leaves, 1 peduncle, 2 flowers = 8 organs)
        tgt_labels_full = torch.tensor([3, 4, 5, 5, 5, 6, 10, 10], device=self.device)
        tgt_geoms_full = torch.randn(8, 16, device=self.device)
        tgt_pos_full = torch.zeros(8, 3, device=self.device)

        matches = matcher(
            pred_anchor_pos=pred_anchor_pos,
            pred_anchor_logits=pred_anchor_logits,
            pred_fine_geom=pred_fine_geom,
            pred_fine_logits=pred_fine_logits,
            tgt_geoms=[tgt_geoms_full],
            tgt_labels=[tgt_labels_full],
            tgt_positions=[tgt_pos_full],
        )

        m = matches[0]
        fine_src = m["fine_src_idx"]
        fine_tgt = m["fine_tgt_idx"]

        # Map slot to matched target type
        slot_to_tgt_type = {}
        for s, t in zip(fine_src.tolist(), fine_tgt.tolist()):
            slot_to_tgt_type[s] = tgt_labels_full[t].item()

        # Slot 0 MUST be Stem (type 3)
        self.assertEqual(slot_to_tgt_type[0], 3, "Slot 0 must match Stem")
        # Slot 1 MUST be Petiole (type 4)
        self.assertEqual(slot_to_tgt_type[1], 4, "Slot 1 must match Petiole")
        # Slots 2, 3, 4 MUST be Leaflets (type 5)
        for s in [2, 3, 4]:
            self.assertEqual(slot_to_tgt_type[s], 5, f"Slot {s} must match Leaflet")
        # Slot 5 MUST be Peduncle (type 6)
        self.assertEqual(slot_to_tgt_type[5], 6, "Slot 5 must match Peduncle")
        # Slots 6, 7 MUST be Reproductive (type 10)
        for s in [6, 7]:
            self.assertEqual(slot_to_tgt_type[s], 10, f"Slot {s} must match Flower")

        # Case 2: Unifoliate Seedling Node (1 stem, 2 leaves = 3 organs)
        tgt_labels_unifoliate = torch.tensor([3, 5, 5], device=self.device)
        tgt_geoms_unifoliate = torch.randn(3, 16, device=self.device)
        tgt_pos_unifoliate = torch.zeros(3, 3, device=self.device)

        matches_uni = matcher(
            pred_anchor_pos=pred_anchor_pos,
            pred_anchor_logits=pred_anchor_logits,
            pred_fine_geom=pred_fine_geom,
            pred_fine_logits=pred_fine_logits,
            tgt_geoms=[tgt_geoms_unifoliate],
            tgt_labels=[tgt_labels_unifoliate],
            tgt_positions=[tgt_pos_unifoliate],
        )

        m_uni = matches_uni[0]
        src_uni = m_uni["fine_src_idx"].tolist()
        tgt_uni = m_uni["fine_tgt_idx"].tolist()

        slot_to_uni_type = {s: tgt_labels_unifoliate[t].item() for s, t in zip(src_uni, tgt_uni)}
        self.assertEqual(slot_to_uni_type[0], 3, "Slot 0 must match Stem in unifoliate")
        # The 2 leaves MUST match within leaf slots {2, 3, 4}
        matched_leaf_slots = [s for s in src_uni if slot_to_uni_type[s] == 5]
        self.assertEqual(len(matched_leaf_slots), 2, "Exactly 2 leaf slots must be matched")
        for s in matched_leaf_slots:
            self.assertIn(s, [2, 3, 4], f"Matched leaf slot {s} must be in [2, 3, 4]")
        # Petiole (Slot 1), Peduncle (Slot 5), Repro (Slots 6, 7) must NOT be matched
        for s in [1, 5, 6, 7]:
            self.assertNotIn(s, src_uni, f"Slot {s} must remain unmatched for unifoliate node")


if __name__ == "__main__":
    unittest.main()
