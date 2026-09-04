"""
Test 14D Part Tensor extraction and 40D reconstruction with Curvature.
"""

import os
import math
import torch
import numpy as np

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    ORGAN_INTERNODE,
    ORGAN_PETIOLE,
    ORGAN_PEDUNCLE,
    T_COL_CURVATURE,
    T_COL_CURV_PERT_0,
    T_COL_ORGAN_TYPE,
)
from diffusion_based.models.part_tensor_to_40d import PartTensorTo40DConverter

def main():
    xml_path = "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap010_0000_plant_0000.xml"
    arr = PlantOrganArray.from_xml_file(xml_path)
    gt_40d = arr.tensor.clone()
    N = gt_40d.shape[0]

    # Current 13D part tensor
    part_13d = arr.to_part_tensor()

    # Create 14D part tensor by appending curvature from gt_40d
    curv_col = torch.zeros((N, 1), dtype=torch.float32)
    for i in range(N):
        ot = int(gt_40d[i, T_COL_ORGAN_TYPE].item())
        if ot == ORGAN_PETIOLE:
            curv_col[i, 0] = gt_40d[i, T_COL_CURVATURE]
        elif ot == ORGAN_PEDUNCLE:
            curv_col[i, 0] = gt_40d[i, T_COL_CURVATURE]
        elif ot == ORGAN_INTERNODE:
            curv_col[i, 0] = gt_40d[i, T_COL_CURV_PERT_0]

    part_14d = torch.cat([part_13d, curv_col], dim=1)

    converter = PartTensorTo40DConverter()
    recon_40d_13d = converter.convert(part_13d)
    
    # Temporarily test 14D conversion by updating recon_40d with curvature
    recon_40d_14d = converter.convert(part_13d)
    for i in range(N):
        ot = int(gt_40d[i, T_COL_ORGAN_TYPE].item())
        if ot in (ORGAN_PETIOLE, ORGAN_PEDUNCLE):
            recon_40d_14d[i, T_COL_CURVATURE] = part_14d[i, 13]
        elif ot == ORGAN_INTERNODE:
            recon_40d_14d[i, T_COL_CURV_PERT_0] = part_14d[i, 13]

    mae_curv_13d = torch.mean(torch.abs(gt_40d[:, T_COL_CURVATURE] - recon_40d_13d[:, T_COL_CURVATURE])).item()
    mae_curv_14d = torch.mean(torch.abs(gt_40d[:, T_COL_CURVATURE] - recon_40d_14d[:, T_COL_CURVATURE])).item()

    print(f"Curvature MAE with 13D: {mae_curv_13d:.4f}")
    print(f"Curvature MAE with 14D: {mae_curv_14d:.4f}")

    # Now let's serialize both to temporary XML and test Helios rendering!
    out_dir = "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders"
    tmp_xml_13d = os.path.join(out_dir, "_tmp_recon_13d.xml")
    tmp_xml_14d = os.path.join(out_dir, "_tmp_recon_14d.xml")

    arr_13d = PlantOrganArray(recon_40d_13d)
    arr_14d = PlantOrganArray(recon_40d_14d)

    with open(tmp_xml_13d, "w") as f:
        f.write(arr_13d.to_xml_string())
    with open(tmp_xml_14d, "w") as f:
        f.write(arr_14d.to_xml_string())

    print(f"Wrote {tmp_xml_13d} and {tmp_xml_14d}")

if __name__ == "__main__":
    main()
