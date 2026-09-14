"""render_batched (one range-mode nvdiffrast pass for N plants) must match forward() per plant:
same images, same gradients."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer

_XMLS = [os.path.join(os.path.dirname(__file__), "..", "dataset", "helios_data", "cowpea",
                      f"cowpea_dap{d:03d}_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml") for d in (15, 40, 75)]


@unittest.skipUnless(torch.cuda.is_available() and all(os.path.exists(p) for p in _XMLS), "needs a GPU and the cowpea XMLs")
class TestRenderBatched(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev = torch.device("cuda:0")
        cls.geo = HeliosPlantGeometryBuilder()
        cls.ren = HeliosPyTorchRenderer(image_size=128).to(cls.dev)
        cls.parts = [PlantOrganArray.from_xml_file(p).to_part_tensor().float().to(cls.dev) for p in _XMLS]

    def _meshes(self, parts):
        return [self.geo.build_mesh_from_part_tensor(p, device=self.dev) for p in parts]

    def test_images_match_per_plant_forward(self):
        for scale in (1.0, 2.0):
            meshes = self._meshes(self.parts)
            got = self.ren.render_batched(meshes, differentiable=True, image_size=128, zoom_factor=scale, reference_window_size=1.2)
            for i, m in enumerate(meshes):
                ref = self.ren.forward(m, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, background="ground",
                                       differentiable=True, focus_plant=False, include_depth=True, image_size=128,
                                       zoom_factor=scale, reference_window_size=1.2)
                self.assertEqual(got[i].shape, ref.shape)
                self.assertLess(float((got[i] - ref).abs().max()), 1e-4, f"plant {i} scale {scale}")

    def test_gradients_match_per_plant_forward(self):
        parts_a = [p.clone().requires_grad_(True) for p in self.parts]
        parts_b = [p.clone().requires_grad_(True) for p in self.parts]
        w = None
        got = self.ren.render_batched(self._meshes(parts_a), differentiable=True, image_size=128, zoom_factor=2.0)
        w = torch.randn_like(got)
        (got * w).sum().backward()
        for i, (m, pb) in enumerate(zip(self._meshes(parts_b), parts_b)):
            ref = self.ren.forward(m, differentiable=True, focus_plant=False, include_depth=True, image_size=128,
                                   zoom_factor=2.0, reference_window_size=1.2)
            (ref * w[i]).sum().backward()
            ga, gb = parts_a[i].grad, pb.grad
            self.assertTrue(torch.allclose(ga, gb, atol=1e-3, rtol=1e-3), f"plant {i}: grad max diff {(ga-gb).abs().max()}")

    def test_empty_mesh_renders_background(self):
        empty = {k: v[:0] for k, v in self._meshes(self.parts[:1])[0].items()}
        meshes = self._meshes(self.parts[:1]) + [empty]
        got = self.ren.render_batched(meshes, differentiable=True, image_size=64)
        self.assertEqual(tuple(got.shape), (2, 4, 64, 64))
        self.assertTrue(torch.all(got[1, 3] == 0))


if __name__ == "__main__":
    unittest.main()
