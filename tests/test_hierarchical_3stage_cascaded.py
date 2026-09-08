"""
Unit tests for 3-Stage Cascaded Hierarchical Botanical Flow Matching Architecture:
  Stage 1: Macro Biological Prior (DAP & Phytomer Count with Soft Tapering Margin)
  Stage 2: 3D Node Point Cloud Scaffold (DETR3D-style continuous 3D anchor placement)
  Stage 3: Intra-Phytomer Canonical Flow Matching (Kinematic conditioning on 3D scaffold)
"""

import math
import unittest
import torch
import torch.nn.functional as F

from diffusion_based.models.hierarchical_part_flow_matching import (
    compute_matryoshka_slice,
    MacroBiologicalHead,
    CoarseSkeletalTransformer,
    FineBotanicalFlowMatchingDecoder,
    HierarchicalPartFlowMatchingModel,
)
from diffusion_based.training.hierarchical_hungarian_matcher import HierarchicalBotanicalMatcher


class TestHierarchical3StageCascaded(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_matryoshka_slicing_with_phytomer_count(self):
        """Verify dynamic Matryoshka slicing based on phytomer count and DAP."""
        # 1. Low phytomer count seedling: 3 phytomers -> ceil(3 + 4) = 7 -> tier 8
        phy_low = torch.tensor([3.0])
        self.assertEqual(compute_matryoshka_slice(num_phytomers=phy_low, max_anchors=512), 8)

        # 2. Mid plant: 10 phytomers -> ceil(10 + 4) = 14 -> tier 16
        phy_mid = torch.tensor([10.0])
        self.assertEqual(compute_matryoshka_slice(num_phytomers=phy_mid, max_anchors=512), 16)

        # 3. Mature bushy canopy: 45 phytomers -> ceil(45 + 4) = 49 -> tier 64
        phy_high = torch.tensor([45.0])
        self.assertEqual(compute_matryoshka_slice(num_phytomers=phy_high, max_anchors=512), 64)

        # 4. Fallback DAP slicing
        dap = torch.tensor([10.0])
        k_dap = compute_matryoshka_slice(dap=dap, max_anchors=512)
        self.assertIn(k_dap, [8, 16, 32, 64, 128, 256, 512])

    def test_stage1_macro_biological_head(self):
        """Verify Stage 1 MacroBiologicalHead predictions, soft margin schedule, and gradient highway."""
        head = MacroBiologicalHead(embed_dim=384, default_margin=1.0, default_tau=0.8).to(self.device)
        B, K = 4, 32
        dummy_cls = torch.randn(B, 384, device=self.device, requires_grad=True)

        out = head(dummy_cls, max_k=K)
        pred_dap = out["pred_dap"]
        pred_phy = out["pred_num_phytomers"]
        soft_weights = out["soft_margin_weights"]
        init_logits = out["init_logits"]

        self.assertEqual(pred_dap.shape, (B, 1))
        self.assertEqual(pred_phy.shape, (B, 1))
        self.assertEqual(soft_weights.shape, (B, K))
        self.assertEqual(init_logits.shape, (B, K, 1))

        # Check positivity
        self.assertTrue((pred_dap >= 0.0).all())
        self.assertTrue((pred_phy >= 0.0).all())
        self.assertTrue((soft_weights >= 0.0).all() and (soft_weights <= 1.0).all())

        # Verify soft tapering property: weights monotonically decrease along K
        for b in range(B):
            w = soft_weights[b].detach().cpu()
            # w[0] should be >= w[-1]
            self.assertGreaterEqual(float(w[0]), float(w[-1]))

        # Test Differentiable Gradient Flow from init_logits back to dummy_cls
        target_logits = torch.ones_like(init_logits)
        loss = F.mse_loss(init_logits, target_logits)
        loss.backward()
        self.assertIsNotNone(dummy_cls.grad)
        self.assertGreater(dummy_cls.grad.abs().sum().item(), 0.0)

    def test_stage2_coarse_skeletal_node_scaffold(self):
        """Verify Stage 2 CoarseSkeletalTransformer generates 3D nodes with soft margin logit bias."""
        coarse = CoarseSkeletalTransformer(max_anchors=32, embed_dim=384, num_heads=4, num_layers=2).to(self.device)
        B, K = 2, 16
        dummy_tokens = torch.randn(B, 64, 384, device=self.device)

        out = coarse(dummy_tokens, active_k=K)

        self.assertIn("anchor_pos", out)
        self.assertIn("anchor_rot", out)
        self.assertIn("anchor_logits", out)
        self.assertIn("pred_num_phytomers", out)
        self.assertIn("soft_margin_weights", out)

        self.assertEqual(out["anchor_pos"].shape, (B, K, 3))
        self.assertEqual(out["anchor_rot"].shape, (B, K, 6))
        self.assertEqual(out["anchor_logits"].shape, (B, K, 1))
        self.assertEqual(out["soft_margin_weights"].shape, (B, K))

    def test_stage3_fine_decoder_with_3d_scaffold_conditioning(self):
        """Verify Stage 3 FineBotanicalFlowMatchingDecoder conditions on 3D node coordinates."""
        decoder = FineBotanicalFlowMatchingDecoder(
            slots_per_anchor=8,
            node_dim=16,
            embed_dim=384,
            num_heads=4,
            num_layers=2,
        ).to(self.device)

        B, K, M = 2, 8, 8
        N_fine = K * M
        noisy_nodes = torch.randn(B, N_fine, 16, device=self.device)
        timesteps = torch.rand(B, device=self.device)
        anchor_features = torch.randn(B, K, 384, device=self.device)
        image_tokens = torch.randn(B, 32, 384, device=self.device)
        anchor_pos = torch.randn(B, K, 3, device=self.device)
        anchor_rot = torch.randn(B, K, 6, device=self.device)

        out = decoder(
            noisy_fine_nodes=noisy_nodes,
            timesteps=timesteps,
            anchor_features=anchor_features,
            image_tokens=image_tokens,
            anchor_pos=anchor_pos,
            anchor_rot=anchor_rot,
        )

        self.assertEqual(out["pred_velocity"].shape, (B, N_fine, 16))
        self.assertEqual(out["pred_exist_logits"].shape, (B, N_fine, 1))

    def test_hierarchical_matcher_with_soft_margin_and_gt_clusters(self):
        """Verify Hungarian matcher returns num_gt_phytomers and accepts soft_margin_weights."""
        matcher = HierarchicalBotanicalMatcher(slots_per_anchor=8)
        B, K, M = 2, 8, 8
        N_fine = K * M

        pred_pos = torch.randn(B, K, 3, device=self.device)
        pred_logits = torch.randn(B, K, 1, device=self.device)
        pred_geom = torch.randn(B, N_fine, 16, device=self.device)
        pred_fine_logits = torch.randn(B, N_fine, 1, device=self.device)
        soft_margin_w = torch.rand(B, K, device=self.device)

        # Mock GT organs: 2 petioles (type 4), 1 internode (type 3), 3 leaflets (type 5)
        tgt_geoms = [torch.randn(6, 16, device=self.device), torch.randn(10, 16, device=self.device)]
        tgt_labels = [
            torch.tensor([3, 4, 4, 5, 5, 5], dtype=torch.long, device=self.device),
            torch.tensor([3, 3, 4, 4, 4, 5, 5, 5, 5, 5], dtype=torch.long, device=self.device),
        ]
        tgt_pos = [torch.randn(6, 3, device=self.device), torch.randn(10, 3, device=self.device)]

        matches = matcher(
            pred_anchor_pos=pred_pos,
            pred_anchor_logits=pred_logits,
            pred_fine_geom=pred_geom,
            pred_fine_logits=pred_fine_logits,
            tgt_geoms=tgt_geoms,
            tgt_labels=tgt_labels,
            tgt_positions=tgt_pos,
            soft_margin_weights=soft_margin_w,
        )

        self.assertEqual(len(matches), B)
        for m in matches:
            self.assertIn("num_gt_phytomers", m)
            self.assertIn("gt_node_centers", m)
            self.assertGreater(m["num_gt_phytomers"], 0)
            self.assertEqual(m["gt_node_centers"].shape[1], 3)

    def test_end_to_end_3stage_forward_and_ode_sampling(self):
        """Verify full 3-Stage Cascaded model forward pass and ODE generation."""
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
        dummy_img = torch.randn(B, 16, 128, 128, device=self.device)
        noisy_nodes = torch.randn(B, N_fine, 16, device=self.device)
        timesteps = torch.rand(B, device=self.device)
        daps = torch.tensor([15.0, 25.0], device=self.device)
        num_phy = torch.tensor([4.0, 8.0], device=self.device)

        # 1. Forward Pass
        out = model(
            noisy_fine_nodes=noisy_nodes,
            timesteps=timesteps,
            images=dummy_img,
            daps=daps,
            num_phytomers=num_phy,
        )

        self.assertIn("pred_num_phytomers", out)
        self.assertIn("soft_margin_weights", out)
        self.assertIn("pred_anchor_pos", out)
        self.assertIn("pred_velocity", out)
        self.assertEqual(out["pred_num_phytomers"].shape, (B, 1))
        self.assertEqual(out["soft_margin_weights"].shape, (B, out["active_k"]))

        # 2. ODE Sampling
        sample_out = model.sample_ode(
            images=dummy_img,
            daps=daps,
            num_steps=3,
        )

        self.assertIn("pred_latent", sample_out)
        self.assertIn("slot_active", sample_out)
        self.assertIn("pred_num_phytomers", sample_out)
        self.assertIn("anchor_pos", sample_out)
        self.assertEqual(sample_out["pred_num_phytomers"].shape, (B, 1))


if __name__ == "__main__":
    unittest.main()
