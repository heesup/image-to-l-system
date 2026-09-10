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
    PhytomerFlowMatchingDecoder,
    build_phytomer_flow_target,
    split_phytomer_flow_target,
)
from diffusion_based.training.hierarchical_hungarian_matcher import HierarchicalBotanicalMatcher


class TestHierarchical3StageCascaded(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_matryoshka_slicing_with_phytomer_count(self):
        """Verify dynamic Matryoshka slicing based on phytomer count and DAP."""
        # Current semantics (2026-09-08 recalibration): continuous capacity
        # K = ceil(val * ANCHOR_MARGIN + ANCHOR_MARGIN_FLAT), clamped to [8, 512].
        # 1. Low phytomer count seedling: 3 phytomers -> ceil(3*1.1 + 6) = 10
        phy_low = torch.tensor([3.0])
        self.assertEqual(compute_matryoshka_slice(num_phytomers=phy_low, max_anchors=512), 10)

        # 2. Mid plant: 10 phytomers -> ceil(10*1.1 + 6) = 17
        phy_mid = torch.tensor([10.0])
        self.assertEqual(compute_matryoshka_slice(num_phytomers=phy_mid, max_anchors=512), 17)

        # 3. Mature bushy canopy: 45 phytomers -> ceil(45*1.1 + 6) = 56
        phy_high = torch.tensor([45.0])
        self.assertEqual(compute_matryoshka_slice(num_phytomers=phy_high, max_anchors=512), 56)

        # 4. Fallback DAP slicing (continuous logistic curve, clamped [8, 512])
        dap = torch.tensor([10.0])
        k_dap = compute_matryoshka_slice(dap=dap, max_anchors=512)
        self.assertGreaterEqual(k_dap, 8)
        self.assertLessEqual(k_dap, 512)

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

    def test_anchor_locality_radius_parity_and_swap_suppression(self):
        """Radius=None reproduces legacy assignments; radius set suppresses far swaps.

        Setup: 1 GT cluster at origin; anchor #0 at 2cm with very low existence,
        anchor #1 at 30cm with very high existence. Legacy cost prefers the FAR
        anchor (0.9 - 0.993 = -0.093 < 0.06 - 0.007 = 0.053). With R=0.12m the
        quadratic penalty (50 * 0.18^2 = 1.62) flips the assignment to the near
        anchor — the correct behavior when node RMSE is ~2cm.
        """
        torch.manual_seed(0)
        matcher_plain = HierarchicalBotanicalMatcher(slots_per_anchor=8)
        matcher_local = HierarchicalBotanicalMatcher(
            slots_per_anchor=8, anchor_locality_radius=0.12, anchor_locality_weight=50.0
        )
        B, K, M = 1, 2, 8
        N_fine = K * M
        tgt_pos = [torch.tensor([
            [0.00, 0.00, 0.10],  # internode (cluster center fallback)
            [0.00, 0.00, 0.10],  # petiole (cluster center)
        ], dtype=torch.float32, device=self.device)]
        tgt_labels = [torch.tensor([3, 4], dtype=torch.long, device=self.device)]
        tgt_geoms = [torch.randn(2, 16, device=self.device)]
        pred_pos = torch.tensor([[
            [0.02, 0.00, 0.10],   # anchor 0: 2cm away, low existence
            [0.30, 0.00, 0.10],   # anchor 1: 30cm away, high existence
        ]], dtype=torch.float32, device=self.device)
        pred_logits = torch.tensor([[[ -5.0], [5.0]]], dtype=torch.float32, device=self.device)
        pred_geom = torch.randn(B, N_fine, 16, device=self.device)
        pred_fine_logits = torch.zeros(B, N_fine, 1, device=self.device)

        def run(matcher):
            with torch.no_grad():
                return matcher(
                    pred_anchor_pos=pred_pos,
                    pred_anchor_logits=pred_logits,
                    pred_fine_geom=pred_geom,
                    pred_fine_logits=pred_fine_logits,
                    tgt_geoms=tgt_geoms,
                    tgt_labels=tgt_labels,
                    tgt_positions=tgt_pos,
                )[0]

        m_plain = run(matcher_plain)
        m_local = run(matcher_local)
        # Legacy: far anchor wins on existence bias.
        self.assertEqual(m_plain["anchor_src_idx"].tolist(), [1])
        # Locality: near anchor wins once the 30cm swap is penalized.
        self.assertEqual(m_local["anchor_src_idx"].tolist(), [0])
        # Both still report the single GT cluster.
        self.assertEqual(m_plain["num_gt_phytomers"], 1)
        self.assertEqual(m_local["num_gt_phytomers"], 1)

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

    def test_end_to_end_phytomer_mode_forward_and_ode(self):
        """Phytomer granularity: forward + sample_ode on the (B, K, 12+D) path."""
        model = HierarchicalPartFlowMatchingModel(
            max_anchors=16,
            slots_per_anchor=8,
            node_dim=16,
            embed_dim=96,
            coarse_layers=1,
            fine_layers=1,
            flow_granularity="phytomer",
            phytomer_latent_dim=32,
        ).to(self.device)

        B, D = 2, 12 + 32
        dummy_img = torch.randn(B, 16, 128, 128, device=self.device)
        daps = torch.tensor([15.0, 25.0], device=self.device)

        # 1. Forward (flow x is (B, K, 12+D))
        K = 16
        noisy_flow = torch.randn(B, K, D, device=self.device)
        timesteps = torch.rand(B, device=self.device)
        out = model(
            noisy_fine_nodes=noisy_flow,
            timesteps=timesteps,
            images=dummy_img,
            daps=daps,
        )
        self.assertEqual(out["flow_granularity"], "phytomer")
        self.assertEqual(out["pred_velocity"].shape, (B, max(K, 8), D))
        self.assertEqual(out["pred_fine_exist_logits"].shape, (B, max(K, 8), 8))

        # 2. Sample ODE
        sample_out = model.sample_ode(images=dummy_img, daps=daps, num_steps=2)
        self.assertEqual(sample_out["flow_granularity"], "phytomer")
        self.assertIn("refined_anchor_pos", sample_out)
        self.assertIn("refined_anchor_rot", sample_out)
        self.assertIn("phytomer_latent", sample_out)
        self.assertIn("pred_slot_exist_logits", sample_out)
        self.assertEqual(sample_out["phytomer_latent"].shape[-1], 32)

    def test_phytomer_flow_decoder_shapes_and_gradients(self):
        """Phytomer mode: flow-matches [base(3) + rot(6) + scale(3) + latent(D)] per anchor."""
        latent_dim = 64
        D = 12 + latent_dim
        decoder = PhytomerFlowMatchingDecoder(
            latent_dim=latent_dim, embed_dim=128, num_heads=4, num_layers=2
        ).to(self.device)
        B, K, T = 2, 8, 16
        noisy = torch.randn(B, K, D, device=self.device, requires_grad=True)
        t = torch.rand(B, device=self.device)
        anchor_feat = torch.randn(B, K, 128, device=self.device)
        img_tokens = torch.randn(B, T, 128, device=self.device)
        anchor_pos = torch.randn(B, K, 3, device=self.device)
        anchor_rot = torch.randn(B, K, 6, device=self.device)
        out = decoder(
            noisy_flow=noisy, timesteps=t, anchor_features=anchor_feat,
            image_tokens=img_tokens, anchor_pos=anchor_pos, anchor_rot=anchor_rot,
        )
        self.assertEqual(out["pred_velocity"].shape, (B, K, D))
        self.assertEqual(out["pred_slot_exist_logits"].shape, (B, K, 10))
        loss = out["pred_velocity"].pow(2).mean() + out["pred_slot_exist_logits"].pow(2).mean()
        loss.backward()
        self.assertIsNotNone(noisy.grad)
        for name, p in decoder.named_parameters():
            self.assertIsNotNone(p.grad, f"missing grad: {name}")

    def test_phytomer_flow_target_roundtrip(self):
        """build/split the 12+D flow vector and recover the parts exactly."""
        latent_dim = 64
        B, K = 2, 8
        pos = torch.randn(B, K, 3)
        rot = torch.randn(B, K, 6)
        scl = torch.randn(B, K, 3).abs() + 0.1
        lat = torch.randn(B, K, latent_dim)
        z = build_phytomer_flow_target(pos, rot, scl, lat)
        self.assertEqual(z.shape, (B, K, 12 + latent_dim))
        p2, r2, s2, l2 = split_phytomer_flow_target(z, latent_dim)
        self.assertTrue(torch.allclose(p2, pos))
        self.assertTrue(torch.allclose(r2, rot))
        self.assertTrue(torch.allclose(s2, scl))
        self.assertTrue(torch.allclose(l2, lat))


if __name__ == "__main__":
    unittest.main()
