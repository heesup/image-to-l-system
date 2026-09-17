"""Leaf orientation inverse: after the solve, the FK reproduces each leaf's 14D rotation."""
import glob
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from plant_recon.models.plant_organ_array import (
    ORGAN_LEAF, PlantOrganArray, T_COL_EXISTENCE, T_COL_ORGAN_TYPE, T_COL_PITCH, T_COL_ROLL, T_COL_YAW,
)
from plant_recon.models.part_tensor_to_40d import PartTensorTo40DConverter, rotation_6d_to_matrix
from plant_recon.models.part_tensor_stem_ik import refine_stem_to_part_tensor
from plant_recon.models.part_tensor_leaf_ik import refine_leaf_orientation, leaf_rotation_error_deg
from plant_recon.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder

_XMLS = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "dataset", "helios_data", "cowpea",
                                      "cowpea_dap01[5-7]_*_0000_plant_0000.xml")))


def _leaf_error_deg(arr_40d, part_14d):
    out, poses = HeliosPlantGeometryBuilder().extract_part_tensor(PlantOrganArray(arr_40d.clone()), return_node_poses=True)
    ot = arr_40d[:, T_COL_ORGAN_TYPE].round().long()
    leaf = torch.nonzero((ot == ORGAN_LEAF) & (arr_40d[:, T_COL_EXISTENCE] > 0.5) & (poses["leaf_kind"] >= 0)).flatten()
    R_t = rotation_6d_to_matrix(part_14d[:, 4:10]).transpose(1, 2)
    R_r = rotation_6d_to_matrix(out[:, 4:10]).transpose(1, 2)
    return leaf_rotation_error_deg(R_t[leaf], R_r[leaf]), poses["leaf_kind"][leaf]


@unittest.skipUnless(_XMLS, "no cowpea DAP 15-17 XML available")
class TestLeafIK(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.part = PlantOrganArray.from_xml_file(_XMLS[0]).to_part_tensor().float()
        arr = PartTensorTo40DConverter().convert(cls.part, plant_id=0)
        cls.arr_stem = refine_stem_to_part_tensor(arr, cls.part)

    def test_solve_reproduces_leaf_rotations(self):
        before, _ = _leaf_error_deg(self.arr_stem, self.part)
        stats = {}
        arr = refine_leaf_orientation(self.arr_stem, self.part, stats=stats)
        after, kind = _leaf_error_deg(arr, self.part)
        self.assertGreater(float(before.mean()), 5.0, "the analytic converter should be measurably off on a dataset plant")
        # Terminal leaflets and single leaves keep whatever the petiole tip frame
        # leaves (the stem IK lands petiole axes to ~1-5 deg); laterals are exact.
        self.assertLess(float(after.mean()), 3.0)
        self.assertLess(float(after.mean()), float(before.mean()) / 3.0)
        self.assertLess(float(after[kind == 1].max()), 0.1, "lateral leaflets have 3 free angles and must be exact")
        self.assertEqual(stats["n_leaves"], int(after.numel()))

    def test_only_leaf_angles_change(self):
        arr = refine_leaf_orientation(self.arr_stem, self.part)
        diff = (arr != self.arr_stem)
        changed_cols = torch.nonzero(diff.any(dim=0)).flatten().tolist()
        self.assertTrue(set(changed_cols) <= {T_COL_PITCH, T_COL_YAW, T_COL_ROLL}, changed_cols)
        ot = self.arr_stem[:, T_COL_ORGAN_TYPE].round().long()
        self.assertTrue(bool((ot[diff.any(dim=1)] == ORGAN_LEAF).all()))

    def test_idempotent(self):
        once = refine_leaf_orientation(self.arr_stem, self.part)
        twice = refine_leaf_orientation(once, self.part)
        self.assertTrue(torch.allclose(once, twice, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
