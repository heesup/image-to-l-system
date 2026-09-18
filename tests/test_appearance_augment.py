"""Appearance augmentation: the RGB planes change, the depth planes do not, the range stays valid,
and only the wrapped (training) stream is affected."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from plant_recon.dataset.appearance_augment import (AppearanceAugmentedDataset, AugmentConfig,
                                                     augment_image16)


def _fake_image(S=64):
    """(16, S, S): a plant blob with height, on a uniform background, in the cache's conventions."""
    torch.manual_seed(0)
    img = torch.empty(16, S, S)
    yy, xx = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
    for l in range(4):
        r = S // (3 + l)
        blob = ((yy - S // 2) ** 2 + (xx - S // 2) ** 2) < r * r
        rgb = torch.full((3, S, S), 0.44)                      # uniform tan, in [-1, 1]
        rgb[1][blob] = 0.0; rgb[0][blob] = -0.5; rgb[2][blob] = -0.5
        img[4 * l:4 * l + 3] = rgb
        img[4 * l + 3] = blob.float() * 0.3                     # CHM: 0 off-plant, 0.3 m on
    return img


class TestAppearanceAugment(unittest.TestCase):
    def test_depth_untouched_rgb_changed_range_valid(self):
        img = _fake_image()
        out = augment_image16(img, None, AugmentConfig(), torch.Generator().manual_seed(1))
        for l in range(4):
            self.assertTrue(torch.equal(out[4 * l + 3], img[4 * l + 3]), f"depth plane {l} was modified")
        self.assertGreater((out[:3] - img[:3]).abs().mean().item(), 1e-3)
        self.assertGreaterEqual(out.min().item(), -1.0 - 1e-5)
        self.assertLessEqual(out.max().item(), 1.0 + 1e-5)

    def test_seeded_and_varied(self):
        img = _fake_image()
        a = augment_image16(img, None, AugmentConfig(), torch.Generator().manual_seed(7))
        b = augment_image16(img, None, AugmentConfig(), torch.Generator().manual_seed(7))
        c = augment_image16(img, None, AugmentConfig(), torch.Generator().manual_seed(8))
        self.assertTrue(torch.equal(a, b), "same seed must give the same augmentation")
        self.assertFalse(torch.equal(a, c), "different seeds must differ")

    def test_p_apply_zero_is_identity(self):
        img = _fake_image()
        out = augment_image16(img, None, AugmentConfig(p_apply=0.0), torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(out, img))

    def test_wrapper_only_touches_the_image(self):
        img = _fake_image()
        base = [{"image": img.clone(), "dap": torch.tensor(30.0), "prefix": "p0", "nodes": torch.zeros(4, 26)}]
        ds = AppearanceAugmentedDataset(base, None, AugmentConfig(), seed=5)
        self.assertEqual(len(ds), 1)
        d = ds[0]
        self.assertEqual(d["prefix"], "p0")
        self.assertTrue(torch.equal(d["nodes"], torch.zeros(4, 26)))
        self.assertTrue(torch.equal(d["image"][3], img[3]))            # depth passes through
        self.assertFalse(torch.equal(d["image"][:3], img[:3]))         # RGB does not


if __name__ == "__main__":
    unittest.main()
