import os, sys
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.eval.test_helios_coco_mask_comparison import decode_helios_coco_leaf_mask

base = "Digital-Crops/projects/syntheticdata_generation/build/output_rad_dap30"
prefix = "seed0_0000"
xml_path = os.path.join(base, f"{prefix}_plant_0000.xml")
json_path = os.path.join(base, f"{prefix}_masks.json")
rad_path = os.path.join(base, f"{prefix}_rad.jpeg")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
organ_array = PlantOrganArray.from_xml_file(xml_path)
helios_gt_mask = decode_helios_coco_leaf_mask(json_path)
H, W = helios_gt_mask.shape

builder = HeliosPlantGeometryBuilder(use_generic_leaves=False, leaf_scale_factor=1.0, tube_radial_subdivisions=6)
renderer = HeliosPyTorchRenderer(image_size=W)
renderer.geo_builder = builder

mesh_dict = renderer.geo_builder.build_mesh_from_organ_array(organ_array, device=device)
pytorch_render_t = renderer.forward(mesh_dict, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, background="ground", focus_plant=True)
pytorch_rgb = pytorch_render_t.permute(1, 2, 0).cpu().numpy()
organ_type_buffer = renderer.render_organ_type_buffer(mesh_dict, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, focus_plant=True)
pytorch_leaf_mask = (organ_type_buffer == 2).cpu().numpy()

inter = (pytorch_leaf_mask & helios_gt_mask).sum()
union = (pytorch_leaf_mask | helios_gt_mask).sum()
iou = inter / union

rad_rgb = np.array(Image.open(rad_path).convert("RGB"), dtype=np.float32) / 255.0

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
axes[0].imshow(rad_rgb); axes[0].set_title("Helios GT RGB"); axes[0].axis("off")
axes[1].imshow(pytorch_rgb); axes[1].set_title("PyTorch RGB"); axes[1].axis("off")
axes[2].imshow(helios_gt_mask, cmap="gray"); axes[2].set_title("GT leaf mask"); axes[2].axis("off")
axes[3].imshow(pytorch_leaf_mask, cmap="gray"); axes[3].set_title("PyTorch leaf mask"); axes[3].axis("off")
mc = np.zeros((H, W, 3)); mc[helios_gt_mask, 0] = 1.0; mc[pytorch_leaf_mask, 1] = 1.0
axes[4].imshow(mc); axes[4].set_title(f"Overlay IoU={iou:.3f}"); axes[4].axis("off")
plt.tight_layout()
out = "/tmp/opencode/dbg_new.png"
plt.savefig(out, dpi=100)
print("saved", out, "iou", iou)
def stats(mask, name):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        print(name, "empty"); return
    print(f"{name}: count={len(ys)} centroid=({xs.mean():.1f},{ys.mean():.1f}) bbox=({xs.min()},{ys.min()})-({xs.max()},{ys.max()})")

stats(helios_gt_mask, "GT ")
stats(pytorch_leaf_mask, "PT ")
stem_mask = (organ_type_buffer == 1).cpu().numpy()
stats(stem_mask, "PTstem")

# Search best translation of PT mask to maximize IoU (test if pure translation)
from scipy import ndimage
best = (0, 0, 0.0)
for dx in range(-120, 121, 10):
    for dy in range(-120, 121, 10):
        shifted = ndimage.shift(pytorch_leaf_mask.astype(float), (dy, dx), order=0) > 0.5
        inter = (shifted & helios_gt_mask).sum()
        union = (shifted | helios_gt_mask).sum()
        if union > 0 and inter / union > best[2]:
            best = (dx, dy, inter / union)
print("best shift", best)
