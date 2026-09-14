"""The per-plant depth / dice losses of the training render block, vectorised over the rendered
plants (2026-09-14), must equal the per-plant loop they replaced."""
import unittest

import torch
import torch.nn.functional as F


def loop_losses(pred_depth, gt_depth):
    """The loop the render block ran per plant until 2026-09-14 (one pyramid level)."""
    ld, lc = 0.0, 0.0
    for b in range(pred_depth.shape[0]):
        p, g = pred_depth[b], gt_depth[b]
        canopy_mask = (g > 0.005) | (p > 0.005)
        if canopy_mask.sum() > 0:
            loss_d = F.smooth_l1_loss(p[canopy_mask], g[canopy_mask], beta=0.02)
        else:
            loss_d = F.smooth_l1_loss(p, g, beta=0.02)
        pm = torch.sigmoid((p - 0.005) * 100.0); gm = (g > 0.005).float()
        inter = (pm * gm).sum(); denom = pm.sum() + gm.sum()
        ld = ld + loss_d; lc = lc + (1.0 - (2.0 * inter + 1e-4) / (denom + 1e-4))
    return ld, lc


def vectorized_losses(pred_depth, gt_depth):
    canopy = (gt_depth > 0.005) | (pred_depth > 0.005)
    l1 = F.smooth_l1_loss(pred_depth, gt_depth, beta=0.02, reduction="none")
    cnt = canopy.sum(dim=(1, 2))
    loss_d = torch.where(cnt > 0, (l1 * canopy).sum(dim=(1, 2)) / cnt.clamp(min=1), l1.mean(dim=(1, 2)))
    pm = torch.sigmoid((pred_depth - 0.005) * 100.0); gm = (gt_depth > 0.005).float()
    inter = (pm * gm).sum(dim=(1, 2)); denom = pm.sum(dim=(1, 2)) + gm.sum(dim=(1, 2))
    return loss_d.sum(), (1.0 - (2.0 * inter + 1e-4) / (denom + 1e-4)).sum()


class TestRenderLossVectorized(unittest.TestCase):
    def test_matches_loop_including_empty_masks(self):
        torch.manual_seed(0)
        n, H = 6, 32
        gt = (torch.rand(n, H, H) * 0.3) * (torch.rand(n, H, H) > 0.6)
        pred = (torch.rand(n, H, H) * 0.3) * (torch.rand(n, H, H) > 0.5)
        gt[0] = 0.0; pred[0] = 0.0                     # a plant with no canopy pixels: plain mean
        pred.requires_grad_(True)
        ld_l, lc_l = loop_losses(pred, gt); ld_v, lc_v = vectorized_losses(pred, gt)
        self.assertLess(float((ld_l - ld_v).abs()), 1e-6); self.assertLess(float((lc_l - lc_v).abs()), 1e-6)
        g_l, = torch.autograd.grad(ld_l + lc_l, pred, retain_graph=True)
        g_v, = torch.autograd.grad(ld_v + lc_v, pred)
        self.assertTrue(torch.allclose(g_l, g_v, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
