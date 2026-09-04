"""
Render Ground Truth XML, 13D Recon XML, and 14D Recon XML via Helios C++ and compute IoUs.
"""

import os
import sys
import numpy as np
import torch
from PIL import Image

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    ORGAN_INTERNODE,
    ORGAN_PETIOLE,
    ORGAN_PEDUNCLE,
    T_COL_ORGAN_TYPE,
    T_COL_CURVATURE,
    T_COL_CURV_PERT_0,
)
from diffusion_based.eval.eval_13d_xml_organ_masks import (
    render_helios_full,
    compute_iou_per_class,
    ORGAN_CLASSES,
)

def test_plant(dap_str="050"):
    gt_xml = f"Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap{dap_str}_0000_plant_0000.xml"
    out_dir = "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders"
    xml_13d = os.path.join(out_dir, f"_tmp_recon_13d_dap{dap_str}.xml")
    xml_14d = os.path.join(out_dir, f"_tmp_recon_14d_dap{dap_str}.xml")

    arr = PlantOrganArray.from_xml_file(gt_xml)
    gt_40d = arr.tensor.clone()
    N = gt_40d.shape[0]

    part_13d = arr.to_part_tensor()

    curv_col = torch.zeros((N, 1), dtype=torch.float32)
    for i in range(N):
        ot = int(gt_40d[i, T_COL_ORGAN_TYPE].item())
        if ot in (ORGAN_PETIOLE, ORGAN_PEDUNCLE):
            curv_col[i, 0] = gt_40d[i, T_COL_CURVATURE]
        elif ot == ORGAN_INTERNODE:
            curv_col[i, 0] = gt_40d[i, T_COL_CURV_PERT_0]

    part_14d = torch.cat([part_13d, curv_col], dim=1)

    from diffusion_based.models.part_tensor_to_40d import PartTensorTo40DConverter
    converter = PartTensorTo40DConverter()
    recon_40d_13d = converter.convert(part_13d)
    
    recon_40d_14d = converter.convert(part_14d)

    arr_13d = PlantOrganArray(recon_40d_13d)
    arr_14d = PlantOrganArray(recon_40d_14d)

    with open(xml_13d, "w") as f:
        f.write(arr_13d.to_xml_string())
    with open(xml_14d, "w") as f:
        f.write(arr_14d.to_xml_string())

    print(f"\n--- Testing DAP {dap_str} ---")
    print("Rendering GT XML...")
    gt_res = render_helios_full(gt_xml, f"test_dap{dap_str}_gt")
    print("Rendering 13D XML...")
    res_13d = render_helios_full(xml_13d, f"test_dap{dap_str}_13d")
    print("Rendering 14D XML...")
    res_14d = render_helios_full(xml_14d, f"test_dap{dap_str}_14d")

    # Compute IoUs
    metrics_13d = compute_iou_per_class(gt_res["mask_map"], res_13d["mask_map"])
    metrics_14d = compute_iou_per_class(gt_res["mask_map"], res_14d["mask_map"])

    # Compute overall foreground IoU
    fg_gt = gt_res["mask_map"] >= 0
    fg_13 = res_13d["mask_map"] >= 0
    fg_14 = res_14d["mask_map"] >= 0
    iou_fg_13 = np.logical_and(fg_gt, fg_13).sum() / max(1, np.logical_or(fg_gt, fg_13).sum())
    iou_fg_14 = np.logical_and(fg_gt, fg_14).sum() / max(1, np.logical_or(fg_gt, fg_14).sum())

    print("\n--- DAP 10 COMPARISON: 13D vs 14D ---")
    print(f"{'Metric':<20} | {'13D Recon':<12} | {'14D Recon':<12}")
    print("-" * 50)
    print(f"{'Overall Mask IoU':<20} | {iou_fg_13*100:.2f}%      | {iou_fg_14*100:.2f}%")
    for cls_name in ORGAN_CLASSES:
        iou_13 = metrics_13d.get(cls_name, 0.0) * 100
        iou_14 = metrics_14d.get(cls_name, 0.0) * 100
        print(f"{cls_name:<20} | {iou_13:.2f}%      | {iou_14:.2f}%")

if __name__ == "__main__":
    test_plant("050")
