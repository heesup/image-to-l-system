"""
Generate Figure comparing the 3 Leaf Modes in PyTorch Differentiable Renderer:
1. Generic Parametric (generate_generic_leaf_mesh_torch, 49 verts, 72 faces)
2. Lowpoly Alpha-Cutout (get_generic_leaf_mesh, 81 verts, 62 faces)
3. High-Res OBJ Prototype (CowpeaLeaf_tip_highres.obj, 1,458 verts, 1,939 faces)

Follows the visual aesthetic of docs/results/assets/fig8_multimodal_depth_mask.png:
Dark theme (#12131C), 6 columns:
Col 1: Leaf Prototype (Close-up Shaded + Projected Wireframe)
Col 2: Drone Top-Down RGB (Elevation 90°)
Col 3: Perspective 3D RGB (Elevation 45°, Azimuth 30°)
Col 4: Canopy Height (CHM / Depth, Magma Colormap)
Col 5: Foreground Silhouette Mask & IoU vs OBJ
Col 6: Organ-Type Semantic Map & Performance Stats
"""

import os
import sys
import time
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
from PIL import Image

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.helios_pytorch_renderer import (
    HeliosPyTorchRenderer,
    compute_focus_plant_camera,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512
ASSETS_DIR = os.path.join(repo_root, "docs", "results", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

XML_PATH = os.path.join(
    repo_root,
    "Digital-Crops", "projects", "syntheticdata_generation", "build", "output", "exact_gt_renders",
    "rad_dap050_0000_plant_0000.xml"
)

ORGAN_META = {
    0: ("#8B4513", "Stem/Internode"),
    1: ("#6B8E23", "Petiole"),
    2: ("#228B22", "Leaf"),
    3: ("#BDB76B", "Peduncle"),
    4: ("#FFD700", "Flower"),
    5: ("#DAA520", "Pod/Fruit"),
}


def depth_to_canopy_height_cm(depth_tensor):
    d = depth_tensor.detach().cpu().numpy()
    fg_mask = (d > 1e-4)
    if not fg_mask.any():
        return np.ma.masked_all_like(d), 0.0, 0.0

    d_cm = d * 100.0
    min_h_cm = float(d_cm[fg_mask].min())
    max_h_cm = float(d_cm[fg_mask].max())
    masked_d_cm = np.ma.masked_where(~fg_mask, d_cm)
    return masked_d_cm, min_h_cm, max_h_cm


def organ_buffer_to_rgb(type_buf: torch.Tensor, H: int, W: int) -> np.ndarray:
    buf_np = type_buf.detach().cpu().numpy()
    rgb = np.ones((H, W, 3), dtype=np.float32)
    for ot_id, (hex_col, _) in ORGAN_META.items():
        r = int(hex_col[1:3], 16) / 255.0
        g = int(hex_col[3:5], 16) / 255.0
        b = int(hex_col[5:7], 16) / 255.0
        mask = (buf_np == ot_id)
        rgb[mask, 0] = r
        rgb[mask, 1] = g
        rgb[mask, 2] = b
    return rgb


def compute_wireframe_segments(verts_3d, faces_np, view_mat, proj_mat, W, H, device):
    v_hom = torch.cat([verts_3d, torch.ones((verts_3d.shape[0], 1), device=device)], dim=-1)
    v_cam = (view_mat @ v_hom.T).T
    v_ndc = (proj_mat @ v_cam.T).T
    pts_ndc = v_ndc[:, :3] / v_ndc[:, 3:4].clamp(min=1e-5)
    xs = (pts_ndc[:, 0].detach().cpu().numpy() + 1.0) * 0.5 * (W - 1)
    ys = (1.0 - pts_ndc[:, 1].detach().cpu().numpy()) * 0.5 * (H - 1)
    pts_2d = np.stack([xs, ys], axis=-1)

    edges = set()
    for f in faces_np:
        edges.add((min(f[0], f[1]), max(f[0], f[1])))
        edges.add((min(f[1], f[2]), max(f[1], f[2])))
        edges.add((min(f[2], f[0]), max(f[2], f[0])))

    segments = [[pts_2d[e0], pts_2d[e1]] for e0, e1 in edges]
    return segments


def main():
    print("Loading DAP 50 Plant Organ Array...")
    arr = PlantOrganArray.from_xml_file(XML_PATH)
    part = arr.to_part_tensor(device=DEVICE)

    renderer = HeliosPyTorchRenderer(image_size=IMG_SIZE, device=DEVICE).to(DEVICE)

    # Prototype leaf tensor: 1 single canonical leaflet
    pt_leaf = torch.zeros(1, 14, device=DEVICE)
    pt_leaf[0, 0] = 5.0  # ORGAN_LEAF
    pt_leaf[0, 4] = 1.0
    pt_leaf[0, 8] = 1.0  # identity rotation
    pt_leaf[0, 10:13] = torch.tensor([0.15, 0.0, 0.15], device=DEVICE)

    modes = [
        ("Mode 1: Generic Parametric\n(generate_generic_leaf_mesh_torch)", "generic", "#70d6ff"),
        ("Mode 2: Lowpoly Cutout\n(get_generic_leaf_mesh)", "lowpoly", "#88d8c0"),
        ("Mode 3: High-Res OBJ\n(CowpeaLeaf_tip_highres.obj)", "obj", "#ff9999"),
    ]

    results = []
    ref_mask = None

    print("Rendering each leaf mode across 6 modalities...")
    for label, mode_str, color in modes:
        t0 = time.time()
        mesh = renderer.geo_builder.build_mesh_from_part_tensor(part, device=DEVICE, leaf_mode=mode_str)
        t_build = (time.time() - t0) * 1000.0

        n_verts = mesh["vertices"].shape[0]
        n_faces = mesh["faces"].shape[0]

        # Top-down render (Elevation 90)
        t0 = time.time()
        rgbd_top = renderer.forward(
            mesh, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
            background="ground", image_size=IMG_SIZE, include_depth=True, focus_plant=True
        )
        t_render = (time.time() - t0) * 1000.0

        rgb_top = rgbd_top[:3].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        depth_top = rgbd_top[3]
        chm_cm, min_h, max_h = depth_to_canopy_height_cm(depth_top)
        fg_mask = (depth_top > 1e-4).detach().cpu().numpy()
        fg_ratio = float(fg_mask.mean()) * 100.0

        if mode_str == "obj":
            ref_mask = fg_mask

        # Perspective render (Elevation 45, Azimuth 30)
        rgbd_persp = renderer.forward(
            mesh, azimuth_deg=30.0, elevation_deg=45.0, camera_height=5.0,
            background="ground", image_size=IMG_SIZE, include_depth=False, focus_plant=True
        )
        rgb_persp = rgbd_persp[:3].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()

        # Organ type buffer
        type_buf = renderer.render_organ_type_buffer(
            mesh, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
            focus_plant=True, image_size=IMG_SIZE
        )
        organ_rgb = organ_buffer_to_rgb(type_buf, IMG_SIZE, IMG_SIZE)

        # Single leaf prototype render + wireframe
        p_mesh = renderer.geo_builder.build_mesh_from_part_tensor(pt_leaf, device=DEVICE, leaf_mode=mode_str)
        p_verts = p_mesh["vertices"].shape[0]
        p_faces = p_mesh["faces"].shape[0]

        rgb_proto = renderer.forward(
            p_mesh, azimuth_deg=25.0, elevation_deg=65.0, camera_height=1.5,
            background="black", image_size=IMG_SIZE, include_depth=False, focus_plant=True
        )
        rgb_proto_np = rgb_proto[:3].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()

        # Wireframe projection for prototype
        view_mat, proj_mat, _ = compute_focus_plant_camera(
            p_mesh['vertices'], p_mesh['organ_types'],
            azimuth_deg=25.0, elevation_deg=65.0, camera_height=1.5, aspect_ratio=1.0, focus_plant=True
        )
        segments = compute_wireframe_segments(
            p_mesh['vertices'], p_mesh['faces'].detach().cpu().numpy(),
            view_mat, proj_mat, IMG_SIZE, IMG_SIZE, DEVICE
        )

        results.append({
            "label": label,
            "mode": mode_str,
            "color": color,
            "n_verts": n_verts,
            "n_faces": n_faces,
            "p_verts": p_verts,
            "p_faces": p_faces,
            "t_build": t_build,
            "t_render": t_render,
            "rgb_proto": rgb_proto_np,
            "segments": segments,
            "rgb_top": rgb_top,
            "rgb_persp": rgb_persp,
            "chm_cm": chm_cm,
            "min_h": min_h,
            "max_h": max_h,
            "fg_mask": fg_mask,
            "fg_ratio": fg_ratio,
            "organ_rgb": organ_rgb,
        })

    # Compute Silhouette IoU vs OBJ
    for r in results:
        if ref_mask is not None:
            intersection = np.logical_and(r["fg_mask"], ref_mask).sum()
            union = np.logical_or(r["fg_mask"], ref_mask).sum()
            iou = (intersection / max(union, 1)) * 100.0
        else:
            iou = 100.0
        r["iou"] = iou

    # -----------------------------------------------------------------------
    # Plotting: Match fig8_multimodal_depth_mask.png style
    # -----------------------------------------------------------------------
    print("Composing high-resolution comparison figure...")
    n_rows = 3
    n_cols = 6

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(26, 4.4 * n_rows))
    fig.patch.set_facecolor("#12131C")
    plt.subplots_adjust(wspace=0.07, hspace=0.22, left=0.08, right=0.98, top=0.86, bottom=0.04)

    fig.suptitle(
        "Figure: Comparison of 3 Leaf Modes in PyTorch Differentiable Renderer\n"
        "(1. Generic Parametric vs 2. Lowpoly Alpha-Cutout vs 3. High-Res OBJ Prototype on DAP 50 Canopy)",
        fontsize=16, fontweight="bold", color="#ffffff", y=0.96
    )

    col_titles = [
        "1. Leaf Prototype Mesh\n(Shaded + Wireframe)",
        "2. Drone Top-Down RGB\n(Elevation 90° Training View)",
        "3. Perspective 3D RGB\n(Elevation 45°, Azimuth 30°)",
        "4. Canopy Height (CHM)\n(Depth: taller = brighter)",
        "5. Foreground Silhouette\n(Mask & IoU vs OBJ)",
        "6. Organ-Type Semantic Map\n(Semantics & Complexity)",
    ]
    col_colors = ["#ff9999", "#70d6ff", "#ffd166", "#c3a6e0", "#88d8c0", "#f4a261"]

    for col, (title, color) in enumerate(zip(col_titles, col_colors)):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold", color=color, pad=10)

    for row_idx, r in enumerate(results):
        ax_row = axes[row_idx]

        # Row label on the left
        ax_row[0].set_ylabel(
            r["label"], fontsize=11, color=r["color"], rotation=0,
            labelpad=95, va="center", fontweight="bold"
        )

        # Col 1: Leaf Prototype (Close-up Shaded + Wireframe)
        ax = ax_row[0]
        ax.imshow(r["rgb_proto"])
        # Overlay wireframe
        lw = 0.9 if r["p_verts"] < 100 else 0.25
        lc = LineCollection(r["segments"], colors="cyan", linewidths=lw, alpha=0.55)
        ax.add_collection(lc)
        ax.set_facecolor("#0d0d1a")
        ax.axis("off")
        ax.text(
            0.04, 0.05, f"Proto: {r['p_verts']:,} V | {r['p_faces']:,} F",
            color="#ffffff", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#000000", alpha=0.75, edgecolor="none"),
            transform=ax.transAxes
        )

        # Col 2: Drone Top-Down RGB
        ax = ax_row[1]
        ax.imshow(r["rgb_top"])
        ax.set_facecolor("#0d0d1a")
        ax.axis("off")
        ax.text(
            0.04, 0.05, f"Plant: {r['n_verts']:,} V | {r['n_faces']:,} F",
            color="#ffffff", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#000000", alpha=0.75, edgecolor="none"),
            transform=ax.transAxes
        )

        # Col 3: Perspective RGB
        ax = ax_row[2]
        ax.imshow(r["rgb_persp"])
        ax.set_facecolor("#0d0d1a")
        ax.axis("off")
        ax.text(
            0.04, 0.05, f"Build: {r['t_build']:.1f}ms | Render: {r['t_render']:.1f}ms",
            color="#ffffff", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#000000", alpha=0.75, edgecolor="none"),
            transform=ax.transAxes
        )

        # Col 4: Canopy Height (CHM Magma)
        ax = ax_row[3]
        im = ax.imshow(r["chm_cm"], cmap="magma", vmin=0, vmax=55.0)
        ax.set_facecolor("#0d0d1a")
        ax.axis("off")
        ax.text(
            0.04, 0.05, f"Height: {r['min_h']:.1f}-{r['max_h']:.1f} cm",
            color="#ffffff", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#000000", alpha=0.75, edgecolor="none"),
            transform=ax.transAxes
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=8, colors="#c0c0c0")
        cb.set_label("Height (cm)", color="#c0c0c0", fontsize=9)

        # Col 5: Foreground Silhouette Mask & IoU vs OBJ
        ax = ax_row[4]
        mask_rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        # Mint green foreground (#88d8c0 -> [0.533, 0.847, 0.753])
        mask_rgb[r["fg_mask"]] = [0.533, 0.847, 0.753]
        ax.imshow(mask_rgb)
        ax.set_facecolor("#000000")
        ax.axis("off")
        iou_str = "Reference" if r["mode"] == "obj" else f"IoU vs OBJ: {r['iou']:.1f}%"
        ax.text(
            0.04, 0.05, f"FG: {r['fg_ratio']:.1f}% | {iou_str}",
            color="#88d8c0", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#000000", alpha=0.75, edgecolor="none"),
            transform=ax.transAxes
        )

        # Col 6: Organ-Type Semantic Map
        ax = ax_row[5]
        ax.imshow(r["organ_rgb"])
        ax.set_facecolor("#ffffff")
        ax.axis("off")
        # Add legend only to bottom row
        if row_idx == n_rows - 1:
            patches = [
                mpatches.Patch(color=hex_c, label=lbl)
                for _, (hex_c, lbl) in ORGAN_META.items() if lbl in ("Stem/Internode", "Petiole", "Leaf")
            ]
            ax.legend(
                handles=patches, loc="lower right", fontsize=8,
                facecolor="#1e1e28", edgecolor="#444", labelcolor="#e0e0e0",
                framealpha=0.9
            )

    out_path = os.path.join(ASSETS_DIR, "fig_leaf_modes_comparison.png")
    plt.savefig(out_path, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Successfully saved figure: {out_path}")


if __name__ == "__main__":
    main()
