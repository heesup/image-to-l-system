"""--use_depth: the CHM/depth channel reaches the backbone via a trainable, plant-normalised adaptor."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from plant_recon.models.hierarchical_part_flow_matching import HierarchicalPartFlowMatchingModel


class TestUseDepth(unittest.TestCase):
    def _model(self, **kw):
        return HierarchicalPartFlowMatchingModel(
            max_phytomers=16, slots_per_phytomer=10, node_dim=16, embed_dim=96,
            coarse_layers=1, fine_layers=1, flow_granularity="phytomer",
            phytomer_latent_dim=32, **kw)

    def test_depth_path_reaches_tokens(self):
        m = self._model(use_depth=True).eval()
        img = torch.randn(2, 4, 128, 128)   # RGB + depth, single level
        with torch.no_grad():
            tok = m.image_encoder(img)
        self.assertEqual(tok.shape, (2, 1 + m.image_encoder.num_patches, 96))

    def test_depth_adaptor_trainable_under_frozen_backbone(self):
        m = self._model(use_depth=True, freeze_backbone=True)
        trainable = [n for n, p in m.named_parameters() if p.requires_grad]
        self.assertTrue(any("depth_adapt" in n for n in trainable), trainable)
        # the pretrained backbone itself stays frozen
        self.assertFalse(any(p.requires_grad for p in m.image_encoder.backbone.parameters()))

    def test_no_depth_model_has_no_adaptor(self):
        m = self._model(use_depth=False)
        self.assertIsNone(m.image_encoder.depth_adapt)

    def test_multizoom_depth(self):
        m = self._model(use_depth=True, multizoom=True).eval()
        img = torch.randn(2, 16, 128, 128)  # 4 levels x (RGB + depth)
        with torch.no_grad():
            tok = m.image_encoder(img)
        self.assertEqual(tok.shape[1], 1 + 4 * m.image_encoder.num_patches)

    def test_backward_compat_load_without_depth(self):
        # a use_depth model loads a plain (no-depth) state dict with only depth_adapt missing
        plain = self._model(use_depth=False)
        deep = self._model(use_depth=True)
        missing, unexpected = deep.load_state_dict(plain.state_dict(), strict=False)
        self.assertTrue(all("depth_adapt" in k for k in missing), missing)
        self.assertEqual(unexpected, [])


if __name__ == "__main__":
    unittest.main()
