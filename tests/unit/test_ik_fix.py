import os
import math
import numpy as np
import torch

from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.part_tensor_to_40d import PartTensorTo40DConverter
from diffusion_based.eval.eval_13d_xml_organ_masks import render_helios_full, compute_iou_per_class

def test_eval(xml_rel, tag):
    orig_xml = os.path.abspath(xml_rel)
    arr = PlantOrganArray.from_xml_file(orig_xml)
    pt = arr.to_part_tensor()
    conv = PartTensorTo40DConverter()
    t40 = conv.convert(pt)
    recon_xml_str = PlantOrganArray(t40).to_xml_string()
    recon_xml_path = f"/tmp/helios_organ_mask_eval/recon_{tag}.xml"
    os.makedirs(os.path.dirname(recon_xml_path), exist_ok=True)
    with open(recon_xml_path, "w") as f:
        f.write(recon_xml_str)
    
    print(f"\n--- Testing {tag} ---")
    gt = render_helios_full(orig_xml, f"gt_{tag}")
    recon = render_helios_full(recon_xml_path, f"recon_{tag}")

    fg_gt = (gt["mask_map"] >= 0)
    fg_recon = (recon["mask_map"] >= 0)
    fg_iou = np.logical_and(fg_gt, fg_recon).sum() / max(1, np.logical_or(fg_gt, fg_recon).sum())
    print(f"[{tag}] Foreground IoU: {fg_iou*100:.2f}%")
    ious = compute_iou_per_class(gt["mask_map"], recon["mask_map"])
    for k, v in ious.items():
        if not math.isnan(v):
            print(f"  * {k}: {v*100:.2f}%")

if __name__ == "__main__":
    test_eval("Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap050_0000_plant_0000.xml", "dap050")
    test_eval("Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap090_0000_plant_0000.xml", "dap090")
