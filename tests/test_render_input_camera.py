"""render_batched(centers=[c]) must reproduce forward(focus_plant=True) — the camera generate_cache used for the
input CHM (centred on the GT plant's mesh bbox) — for a plant that does not sit at the origin, and differ from the
origin-window render the training loss used before --render_input_camera."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from plant_recon.models.plant_organ_array import PlantOrganArray
from plant_recon.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder
from plant_recon.models.helios_pytorch_renderer import HeliosPyTorchRenderer

_XMLS = [os.path.join(os.path.dirname(__file__), "..", "dataset", "helios_data", "cowpea",
                      f"cowpea_dap{d:03d}_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml") for d in (40, 75)]


@unittest.skipUnless(torch.cuda.is_available() and all(os.path.exists(p) for p in _XMLS), "needs a GPU and the cowpea XMLs")
class TestRenderInputCamera(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev = torch.device("cuda:0")
        cls.geo = HeliosPlantGeometryBuilder()
        cls.ren = HeliosPyTorchRenderer(image_size=128).to(cls.dev)
        cls.parts = [PlantOrganArray.from_xml_file(p).to_part_tensor().float().to(cls.dev) for p in _XMLS]

    def _shifted_mesh(self, parts, dx, dy):
        mesh = self.geo.build_mesh_from_part_tensor(parts, device=self.dev)
        mesh = dict(mesh)
        mesh["vertices"] = mesh["vertices"] + torch.tensor([dx, dy, 0.0], device=self.dev)
        return mesh

    def test_centers_match_focus_plant_camera(self):
        for parts in self.parts:
            mesh = self._shifted_mesh(parts, 0.25, -0.15)
            v = mesh["vertices"]
            c = 0.5 * (v.min(0).values + v.max(0).values)
            for zoom in (1.0, 2.0):
                ref = self.ren.forward(mesh, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, background="ground",
                                       focus_plant=True, include_depth=True, differentiable=True, image_size=128,
                                       zoom_factor=zoom, reference_window_size=1.2)
                got = self.ren.render_batched([mesh], azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
                                              differentiable=True, image_size=128, zoom_factor=zoom,
                                              reference_window_size=1.2, centers=[c])[0]
                origin = self.ren.render_batched([mesh], azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
                                                 differentiable=True, image_size=128, zoom_factor=zoom,
                                                 reference_window_size=1.2)[0]
                ref_m, got_m, org_m = (ref[3] > 0.005), (got[3] > 0.005), (origin[3] > 0.005)
                iou = (ref_m & got_m).sum().item() / max((ref_m | got_m).sum().item(), 1)
                iou_origin = (ref_m & org_m).sum().item() / max((ref_m | org_m).sum().item(), 1)
                self.assertGreater(iou, 0.98, f"zoom {zoom}: centred render should reproduce the focus_plant camera (IoU {iou:.3f})")
                self.assertLess(iou_origin, 0.9, f"zoom {zoom}: origin-window render should differ for an off-centre plant (IoU {iou_origin:.3f})")
                self.assertLess((ref[3] - got[3]).abs().max().item(), 1e-3)

    def test_none_center_is_origin_window(self):
        mesh = self._shifted_mesh(self.parts[0], 0.25, -0.15)
        a = self.ren.render_batched([mesh], azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, differentiable=True,
                                    image_size=128, zoom_factor=1.0, reference_window_size=1.2, centers=[None])[0]
        b = self.ren.render_batched([mesh], azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, differentiable=True,
                                    image_size=128, zoom_factor=1.0, reference_window_size=1.2)[0]
        self.assertTrue(torch.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
