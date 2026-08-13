"""3x3 renderer comparison: Helios Visualizer vs Radiation vs 15D Torch.

Rows = days-after-planting (10 / 50 / 90), Columns = renderer.
Each cell shows the rendered image and the rendering wall-clock time.
"""

import os
import sys
import time
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

repo_root = "/home/lion397/codes/image-to-l-system"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.helios_xml_parser import HeliosXMLParser
from diffusion_based.models.legacy.helios_geometry_track_a import nodes_to_geometry_torch, build_helios_geometry_from_xml
from diffusion_based.models.legacy.helios_rasterizer_3d_track_a import HeliosGeometryRasterizer
from diffusion_based.models.legacy.differentiable_pipeline_track_a import DifferentiableHeliosRenderer

# ---------------------------------------------------------------------------
# Timing results collected from the C++ main binary (--renderer vis / radiation)
# and from the XML-reconstruction / torch renderers below. Wall-clock includes
# plant growth for C++. XML-reconstruction and torch times are render-only, so
# they are naturally much faster than the C++ wall-clock timings.
# ---------------------------------------------------------------------------
DAP_CONFIG = {
    "dap10": {
        "xml": f"{repo_root}/Digital-Crops/projects/syntheticdata_generation/build/dapcmp/d10/d10_all_0000_plant_0000.xml",
        "vis": f"{repo_root}/Digital-Crops/projects/syntheticdata_generation/build/dapcmp/d10/d10_all_0000_vis.jpeg",
        "rad": f"{repo_root}/Digital-Crops/projects/syntheticdata_generation/build/dapcmp/d10/d10_all_0000_rad.jpeg",
        "cpp_vis_ms": 6034,
        "cpp_rad_ms": 8439,
    },
    "dap50": {
        "xml": f"{repo_root}/Digital-Crops/projects/syntheticdata_generation/build/dapcmp/d50/d50_all_0000_plant_0000.xml",
        "vis": f"{repo_root}/Digital-Crops/projects/syntheticdata_generation/build/dapcmp/d50/d50_all_0000_vis.jpeg",
        "rad": f"{repo_root}/Digital-Crops/projects/syntheticdata_generation/build/dapcmp/d50/d50_all_0000_rad.jpeg",
        "cpp_vis_ms": 9497,
        "cpp_rad_ms": 13450,
    },
    "dap90": {
        "xml": f"{repo_root}/Digital-Crops/projects/syntheticdata_generation/build/dapcmp/d90/d90_all_0000_plant_0000.xml",
        "vis": f"{repo_root}/Digital-Crops/projects/syntheticdata_generation/build/dapcmp/d90/d90_all_0000_vis.jpeg",
        "rad": f"{repo_root}/Digital-Crops/projects/syntheticdata_generation/build/dapcmp/d90/d90_all_0000_rad.jpeg",
        "cpp_vis_ms": 18162,
        "cpp_rad_ms": 25608,
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rasterizer = HeliosGeometryRasterizer(image_size=256).to(device)
renderer = DifferentiableHeliosRenderer(rasterizer).to(device)

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
COLS = ["Helios Visualizer", "Helios Radiation", "XML Reconstruct", "15D Torch (diff)"]

for row, (label, cfg) in enumerate(DAP_CONFIG.items()):
    # --- Helios Visualizer ---
    vis_np = np.array(Image.open(cfg["vis"]).convert("RGB"), dtype=np.float32) / 255.0
    axes[row, 0].imshow(np.clip(vis_np, 0, 1))
    axes[row, 0].set_title(
        f"{COLS[0]}\n{cfg['cpp_vis_ms']/1000:.1f}s", fontsize=11
    )
    axes[row, 0].axis("off")

    # --- Helios Radiation ---
    rad_np = np.array(Image.open(cfg["rad"]).convert("RGB"), dtype=np.float32) / 255.0
    axes[row, 1].imshow(np.clip(rad_np, 0, 1))
    axes[row, 1].set_title(
        f"{COLS[1]}\n{cfg['cpp_rad_ms']/1000:.1f}s", fontsize=11
    )
    axes[row, 1].axis("off")

    # --- XML Reconstruct (numpy) ---
    parser = HeliosXMLParser(cfg["xml"])
    parser.parse()
    geom = build_helios_geometry_from_xml(cfg["xml"])
    t0 = time.time()
    xml_img = rasterizer.render_numpy_geometry(
        geom.tubes, geom.leaflets, geom.ellipsoids,
        camera_height=1.0, distance_from_center=0.0, azimuth_deg=0.0,
        focus_plant=True, background=None,
    )
    t_xml = time.time() - t0
    axes[row, 2].imshow(np.clip(xml_img, 0, 1))
    axes[row, 2].set_title(
        f"{COLS[2]}\n{t_xml:.2f}s  ({len(geom.tubes)}t {len(geom.leaflets)}l {len(geom.ellipsoids)}e)",
        fontsize=11,
    )
    axes[row, 2].axis("off")

    # --- 15D Torch (differentiable) ---
    nodes = parser.get_all_organ_nodes()
    arr = np.array([n.to_16d() for n in nodes], dtype=np.float32)
    nt = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)
    parents = torch.tensor([n.parent_idx for n in nodes], dtype=torch.long).unsqueeze(0).to(device)

    t0 = time.time()
    with torch.no_grad():
        img = renderer(
            nt, parents,
            camera_height=1.0, distance_from_center=0.0, azimuth_deg=0.0,
            focus_plant=True, background=None,
        )
    t_render = time.time() - t0
    img_np = img[0].permute(1, 2, 0).detach().cpu().numpy()
    img_np = img_np[..., :3]  # drop alpha

    axes[row, 3].imshow(np.clip(img_np, 0, 1))
    axes[row, 3].set_title(
        f"{COLS[3]}\n{t_render:.2f}s  ({len(nodes)} nodes)", fontsize=11
    )
    axes[row, 3].axis("off")

    # Row label
    axes[row, 0].set_ylabel(f"DAP {label[-2:]}", fontsize=13, fontweight="bold")

plt.suptitle(
    "Renderer Comparison  (C++ times include plant growth; XML-reconstruct and torch are render-only)",
    fontsize=14, y=0.995,
)
plt.tight_layout()
out = os.path.join(repo_root, "notebooks", "renderer_timing_comparison.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n=== Renderer timing summary (ms) ===")
print(f"{'dap':<6}{'XML_recon':>12}{'15D Torch':>12}{'Torch/N':>10}")
for label, cfg in DAP_CONFIG.items():
    parser = HeliosXMLParser(cfg["xml"]); parser.parse()
    geom = build_helios_geometry_from_xml(cfg["xml"])
    t0 = time.time()
    rasterizer.render_numpy_geometry(
        geom.tubes, geom.leaflets, geom.ellipsoids,
        camera_height=1.0, distance_from_center=0.0, azimuth_deg=0.0,
        focus_plant=True, background=None,
    )
    t_xml = (time.time() - t0) * 1000

    nodes = parser.get_all_organ_nodes()
    arr = np.array([n.to_16d() for n in nodes], dtype=np.float32)
    nt = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)
    parents = torch.tensor([n.parent_idx for n in nodes], dtype=torch.long).unsqueeze(0).to(device)
    t0 = time.time()
    with torch.no_grad():
        renderer(nt, parents, camera_height=1.0, distance_from_center=0.0,
                 azimuth_deg=0.0, focus_plant=True, background=None)
    t_ms = (time.time() - t0) * 1000
    per_node = t_ms / max(len(nodes), 1)
    print(f"{label:<6}{t_xml:>12.0f}{t_ms:>12.0f}{per_node:>10.3f}")
