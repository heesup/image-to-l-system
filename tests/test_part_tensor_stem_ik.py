"""Stem inverse kinematics: Helios's FK must land on the 14D stem after the solve."""
import glob
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from diffusion_based.models.plant_organ_array import (
    ORGAN_INTERNODE, PlantOrganArray, T_COL_CURV_PERT_0, T_COL_EXISTENCE, T_COL_ORGAN_TYPE,
    T_COL_PHYLLOTACTIC_ANGLE, T_COL_YAW_PERT_0,
)
from diffusion_based.models.part_tensor_to_40d import PartTensorTo40DConverter, rotation_6d_to_matrix
from diffusion_based.models.part_tensor_stem_ik import refine_stem_to_part_tensor
from diffusion_based.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder

_XMLS = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "dataset", "helios_data", "cowpea",
                                      "cowpea_dap01[0-2]_*_0000_plant_0000.xml")))


def _tip_error_cm(arr_40d, part_14d):
    poses = HeliosPlantGeometryBuilder().extract_part_tensor(PlantOrganArray(arr_40d.clone()), return_node_poses=True)[1]
    ot = arr_40d[:, T_COL_ORGAN_TYPE].round().long()
    ino = torch.nonzero((ot == ORGAN_INTERNODE) & (arr_40d[:, T_COL_EXISTENCE] > 0.5)).flatten()
    R = rotation_6d_to_matrix(part_14d[:, 4:10]).transpose(1, 2)
    fwd = R[:, :, 1]; fwd = fwd / fwd.norm(dim=-1, keepdim=True)
    tip = part_14d[:, 1:4] + fwd * part_14d[:, 10:11].abs()
    return (poses["tip"][ino] - tip[ino]).norm(dim=-1) * 100.0


@unittest.skipUnless(_XMLS, "no cowpea seedling XML available")
class TestStemIK(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.part = PlantOrganArray.from_xml_file(_XMLS[0]).to_part_tensor().float()
        cls.arr = PartTensorTo40DConverter().convert(cls.part, plant_id=0)

    def test_solve_reduces_tip_error_on_ground_truth(self):
        before = _tip_error_cm(self.arr, self.part)
        out = refine_stem_to_part_tensor(self.arr, self.part, n_iter=2)
        after = _tip_error_cm(out, self.part)
        self.assertLessEqual(float(after.mean()), float(before.mean()) + 1e-6)
        self.assertLess(float(after.max()), 0.3)

    def test_solve_recovers_from_perturbed_stem_parameters(self):
        g = torch.Generator().manual_seed(0)
        noisy = self.arr.clone()
        ot = noisy[:, T_COL_ORGAN_TYPE].round().long()
        ino = (ot == ORGAN_INTERNODE) & (noisy[:, T_COL_EXISTENCE] > 0.5)
        n = int(ino.sum())
        noisy[ino, T_COL_CURV_PERT_0] += torch.randn(n, generator=g) * 3.0      # degrees
        noisy[ino, T_COL_YAW_PERT_0] += torch.randn(n, generator=g) * 3.0
        noisy[ino, T_COL_PHYLLOTACTIC_ANGLE] += torch.randn(n, generator=g) * 5.0
        before = _tip_error_cm(noisy, self.part)
        out = refine_stem_to_part_tensor(noisy, self.part, n_iter=2)
        after = _tip_error_cm(out, self.part)
        self.assertGreater(float(before.max()), 0.3)   # the perturbation did something
        self.assertLess(float(after.mean()), 0.5 * float(before.mean()))
        self.assertLess(float(after.max()), 0.3)


if __name__ == "__main__":
    unittest.main()
