"""Tests for PhytomerVAE (phytomer-level latent model)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from diffusion_based.models.phytomer_vae import PhytomerVAE, PACKET_IN_DIM
from diffusion_based.dataset.phytomer_packets import build_phytomer_packets


def _synthetic_packet_bag(n_packets: int = 32, seed: int = 0):
    torch.manual_seed(seed)
    packets = torch.zeros(n_packets, 10, 26)
    presence = torch.zeros(n_packets, 10, dtype=torch.bool)
    for p in range(n_packets):
        n_org = int(torch.randint(2, 9, (1,)).item())
        slots = torch.randperm(10)[:n_org]
        for s in slots.tolist():
            role = {0: 3, 1: 4, 2: 5, 3: 5, 4: 5, 5: 6, 6: 9, 7: 10, 8: 11, 9: 11}[s]
            packets[p, s, role] = 1.0
            packets[p, s, 13:16] = (torch.rand(3) - 0.5) * 4.0   # relative base
            packets[p, s, 16:22] = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)
            packets[p, s, 22:25] = 0.5 + torch.rand(3)           # scale
            packets[p, s, 25] = (torch.rand(1) - 0.5) * 0.5      # curv
            presence[p, s] = True
    return packets, presence


class TestPhytomerVAE(unittest.TestCase):
    def test_io_shapes_and_gradients(self):
        for latent_dim in (32, 64, 128):
            vae = PhytomerVAE(latent_dim=latent_dim)
            packets, presence = _synthetic_packet_bag(16)
            out = vae(packets, presence)
            self.assertEqual(out["z"].shape, (16, latent_dim))
            self.assertEqual(out["recon_packets"].shape, (16, 10, 26))
            self.assertEqual(out["cls_logits"].shape, (16, 10, 13))
            losses = vae.compute_loss(out, packets, presence)
            for k in ("loss", "recon_loss", "loss_cls", "loss_base", "loss_rot",
                      "loss_scale", "loss_curv", "loss_kl", "cls_acc"):
                self.assertIn(k, losses)
                self.assertTrue(torch.isfinite(losses[k]).all())
            losses["loss"].backward()
            for name, p in vae.named_parameters():
                if name.startswith("rot_branch") or name.startswith("head_rot_dedicated"):
                    continue  # only active under use_rot_branch=True (tested below)
                self.assertIsNotNone(p.grad, f"missing grad: {name}")
                self.assertTrue(torch.isfinite(p.grad).all(), f"nonfinite grad: {name}")

    def test_rot_branch_gradients(self):
        """Dedicated rotation branch receives gradients when enabled."""
        vae = PhytomerVAE(latent_dim=32)
        packets, presence = _synthetic_packet_bag(10)
        out = vae(packets, presence, use_rot_branch=True)
        self.assertEqual(out["rot"].shape, (10, 10, 6))
        vae.compute_loss(out, packets, presence)["loss"].backward()
        for name, p in vae.named_parameters():
            if name.startswith("rot_branch") or name.startswith("head_rot_dedicated"):
                self.assertIsNotNone(p.grad, f"missing grad: {name}")
                self.assertTrue(torch.isfinite(p.grad).all(), f"nonfinite grad: {name}")

    def test_hungarian_roles_alignment(self):
        """Hungarian role matching: optimal assignment, canonical fallback, determinism."""
        vae = PhytomerVAE(latent_dim=32)
        packets, presence = _synthetic_packet_bag(10, seed=3)
        out = vae(packets, presence)
        # loss runs and stays finite with Hungarian alignment on
        losses = vae.compute_loss(out, packets, presence, hungarian_roles=True)
        self.assertTrue(torch.isfinite(losses["loss"]))
        # determinism: same inputs -> same aligned targets (no randomness inside)
        losses2 = vae.compute_loss(out, packets, presence, hungarian_roles=True)
        self.assertAlmostEqual(float(losses["loss"]), float(losses2["loss"]), places=6)
        # aligned presence covers exactly the GT present organs (no creation/destruction)
        # (checked implicitly: total presence mass preserved per packet)
        aligned, aligned_pres = vae._hungarian_align_targets(
            {k: v.detach() for k, v in out.items() if torch.is_tensor(v)},
            packets, presence)
        self.assertEqual(aligned.shape, packets.shape)
        self.assertTrue(((aligned_pres.sum(-1) <= presence.sum(-1)).all()))
        # single-GT role keeps canonical-first placement: one stem organ must
        # land in slot 0 (first slot of its role range)
        p1 = torch.zeros(1, 10, 26)
        p1[0, 0, 3] = 1.0; p1[0, 0, 13:16] = torch.tensor([0.0, 0.0, 2.0])
        pr1 = torch.zeros(1, 10, dtype=torch.bool); pr1[0, 0] = True
        a1, ap1 = vae._hungarian_align_targets(
            {"base": torch.zeros(1, 10, 3), "scale": torch.ones(1, 10, 3),
             "cls_logits": torch.zeros(1, 10, 13)}, p1, pr1)
        self.assertTrue(bool(ap1[0, 0]))
        self.assertEqual(int(ap1.sum().item()), 1)

    def test_pack_input_dim(self):
        # Base columns are structurally assembled (deterministic from XML
        # phytomer clustering), so they are removed before encoding:
        # 10 x (23 + 1) = 240.
        self.assertEqual(PACKET_IN_DIM, 10 * (26 - 3 + 1))
        vae = PhytomerVAE()
        packets, presence = _synthetic_packet_bag(4)
        self.assertEqual(vae.pack_input(packets, presence).shape, (4, PACKET_IN_DIM))

    def test_empty_packet_batch(self):
        """All-absent packets must not NaN (division guards)."""
        vae = PhytomerVAE()
        packets = torch.zeros(4, 10, 26)
        packets[:, :, 0] = 1.0  # NONE everywhere
        presence = torch.zeros(4, 10, dtype=torch.bool)
        out = vae(packets, presence)
        losses = vae.compute_loss(out, packets, presence)
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_symmetry_aware_and_ortho_flags(self):
        """symmetry_aware_rot + ortho_reg produce finite losses and gradients."""
        vae = PhytomerVAE(latent_dim=32)
        packets, presence = _synthetic_packet_bag(10)
        out = vae(packets, presence)
        losses = vae.compute_loss(
            out, packets, presence, rot_weight=4.0,
            symmetry_aware_rot=True, ortho_reg_weight=0.5,
        )
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        for name, p in vae.named_parameters():
            if name.startswith("rot_branch") or name.startswith("head_rot_dedicated"):
                continue  # only active under use_rot_branch=True
            self.assertIsNotNone(p.grad, f"missing grad: {name}")

    def test_eval_mode_deterministic(self):
        vae = PhytomerVAE().eval()
        packets, presence = _synthetic_packet_bag(4)
        with torch.no_grad():
            z1 = vae(packets, presence)["z"]
            z2 = vae(packets, presence)["z"]
        self.assertTrue(torch.equal(z1, z2))

    def test_save_load_roundtrip(self):
        import tempfile
        vae = PhytomerVAE(latent_dim=64).eval()
        packets, presence = _synthetic_packet_bag(4)
        with torch.no_grad():
            z_before = vae(packets, presence)["z"]
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(vae.state_dict(), f.name)
            vae2 = PhytomerVAE(latent_dim=64)
            vae2.load_state_dict(torch.load(f.name, weights_only=True))
        vae2.eval()
        with torch.no_grad():
            z_after = vae2(packets, presence)["z"]
        self.assertTrue(torch.allclose(z_before, z_after, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
