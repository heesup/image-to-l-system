"""Multi-DAP benchmark comparing C++ GT, Track A (XML-native), and Track B (22D)."""
import os
import sys
import time
import json
import subprocess
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.helios_xml_parser import HeliosXMLParser
from diffusion_based.models.legacy.differentiable_pipeline_track_a import DifferentiableHeliosRenderer
from diffusion_based.models.legacy.helios_geometry_legacy import build_helios_geometry_from_xml, DifferentiableHeliosXMLRenderer
from diffusion_based.models.legacy.helios_rasterizer_3d_track_a import HeliosGeometryRasterizer
from notebooks.run_differentiable_renderer_stability_test import setup_display_env, compute_ssim_numpy, compute_silhouette_iou


def render_bbox_overlay(img_pil: Image.Image, masks_path: str, boxes_path: str) -> Image.Image:
    """Draw organ bounding box overlay on GT image."""
    W, H = img_pil.size
    overlay = img_pil.copy()
    draw = ImageDraw.Draw(overlay)
    colors = {
        0: (0, 150, 255),    # plant (blue)
        1: (255, 230, 0),    # flower (yellow)
        2: (0, 255, 120),    # pod (cyan-green)
    }
    if os.path.exists(masks_path):
        with open(masks_path) as f:
            coco = json.load(f)
        orig_w = coco.get("images", [{}])[0].get("width", W)
        orig_h = coco.get("images", [{}])[0].get("height", H)
        scale_x = W / float(orig_w) if orig_w else 1.0
        scale_y = H / float(orig_h) if orig_h else 1.0

        for ann in coco.get("annotations", []):
            cat_id = ann.get("category_id", 0)
            color = colors.get(cat_id, (255, 255, 255))
            x, y, w, h = ann["bbox"]
            x1, y1 = int(x * scale_x), int(y * scale_y)
            x2, y2 = int((x + w) * scale_x), int((y + h) * scale_y)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    elif os.path.exists(boxes_path):
        with open(boxes_path) as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cat_id = int(parts[0])
                cx, cy, w, h = [float(p) for p in parts[1:5]]
                x1 = int((cx - w / 2.0) * W)
                y1 = int((cy - h / 2.0) * H)
                x2 = int((cx + w / 2.0) * W)
                y2 = int((cy + h / 2.0) * H)
                color = colors.get(cat_id, (255, 255, 255))
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    return overlay


def render_segmentation_mask(masks_path: str, image_size: tuple = (256, 256)) -> np.ndarray:
    """Render semantic segmentation mask from COCO masks.json."""
    W, H = image_size
    mask_canvas = np.zeros((H, W, 3), dtype=np.uint8)
    colors = {
        0: (34, 139, 34),    # plant (forest green)
        1: (255, 230, 0),    # flower (yellow)
        2: (0, 255, 120),    # pod (cyan-green)
    }
    if os.path.exists(masks_path):
        with open(masks_path) as f:
            coco = json.load(f)
        orig_w = coco.get("images", [{}])[0].get("width", W)
        orig_h = coco.get("images", [{}])[0].get("height", H)
        scale_x = W / float(orig_w) if orig_w else 1.0
        scale_y = H / float(orig_h) if orig_h else 1.0

        for ann in coco.get("annotations", []):
            cat_id = ann.get("category_id", 0)
            color = colors.get(cat_id, (255, 255, 255))
            seg = ann.get("segmentation", [])
            for poly in seg:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                pts[:, 0] *= scale_x
                pts[:, 1] *= scale_y
                pts_int = pts.astype(np.int32)
                cv2.fillPoly(mask_canvas, [pts_int], color)
    return mask_canvas


def generate_gt_sample(output_dir: str, dap: int, seed: int = 42) -> dict:
    """Generate C++ Helios GT image and XML for given DAP using --renderer radiation."""
    main_binary = os.path.join(
        repo_root, "Digital-Crops", "projects", "syntheticdata_generation", "build", "main"
    )
    base_params_file = os.path.join(
        repo_root, "Digital-Crops", "projects", "syntheticdata_generation", "params.json"
    )

    assert os.path.exists(main_binary), f"Main C++ binary not found at {main_binary}"
    assert os.path.exists(base_params_file), f"Params JSON not found at {base_params_file}"

    with open(base_params_file, "r") as f:
        params = json.load(f)

    params.setdefault("camera", {}).setdefault("positioning", {})["azimuth_angle"] = 0.0
    params["camera"]["positioning"]["camera_height"] = 5.0
    params["camera"]["positioning"]["distance_from_center"] = 2.0
    params["camera"]["positioning"]["focusing_plants"] = True
    params.setdefault("environment", {}).setdefault("soil", {})["use_obj_ground"] = False
    params.setdefault("metadata", {})["dap"] = int(dap)
    params["metadata"].pop("DAP", None)
    params["seed"] = int(seed)

    prefix = f"dap{dap}_gt"
    params_file = os.path.join(output_dir, f"{prefix}_params.json")
    with open(params_file, "w") as f:
        json.dump(params, f, indent=2)

    env = setup_display_env()
    cmd = [
        main_binary,
        "--renderer", "radiation",
        "--save-xml",
        "--focus-plant",
        "--dap", str(dap),
        "-o", output_dir,
        "-f", params_file,
        "-n", prefix,
    ]

    print(f"\n[C++ HELIOS] Generating DAP {dap} (Seed={seed}) [--renderer radiation]...")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=os.path.dirname(main_binary), env=env, capture_output=True, text=True)
    cpp_time_ms = (time.time() - t0) * 1000.0

    if res.returncode != 0:
        print(f"Error running C++ binary for DAP {dap}: {res.stderr}")
        raise RuntimeError(res.stderr)

    output_build_dir = os.path.join(os.path.dirname(main_binary), "output")
    xml_path = os.path.join(output_build_dir, f"{prefix}_0000_plant_0000.xml")
    if not os.path.exists(xml_path):
        xml_path = os.path.join(output_dir, f"{prefix}_0000_plant_0000.xml")

    img_path = os.path.join(output_build_dir, f"{prefix}_0000_rad.jpeg")
    if not os.path.exists(img_path):
        img_path = os.path.join(output_dir, f"{prefix}_0000_rad.jpeg")
    if not os.path.exists(img_path):
        img_path = os.path.join(output_dir, f"{prefix}_0000_vis.jpeg")

    masks_path = os.path.join(output_build_dir, f"{prefix}_0000_masks.json")
    if not os.path.exists(masks_path):
        masks_path = os.path.join(output_dir, f"{prefix}_0000_masks.json")

    boxes_path = os.path.join(output_build_dir, f"{prefix}_0000_boxes.txt")
    if not os.path.exists(boxes_path):
        boxes_path = os.path.join(output_dir, f"{prefix}_0000_boxes.txt")

    return {
        "dap": dap,
        "xml_path": xml_path,
        "img_path": img_path,
        "masks_path": masks_path,
        "boxes_path": boxes_path,
        "cpp_time_ms": cpp_time_ms,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-DAP Helios vs PyTorch Renderer Benchmark (Track A + Track B)")
    parser.add_argument("--daps", type=int, nargs="+", default=[10, 50, 90],
                        help="List of DAP values to benchmark (e.g. --daps 10 50 90)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: notebooks/output_dap_benchmark)")
    parser.add_argument("--skip-cpp", action="store_true",
                        help="Skip C++ GT generation and reuse existing XML/images")
    args = parser.parse_args()

    output_dir = args.out or os.path.join(repo_root, "notebooks", "output_dap_benchmark")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dap_str = ", ".join(str(d) for d in args.daps)
    print(f"Running Multi-DAP ({dap_str}) Renderer Benchmark on device: {device}")

    rasterizer = HeliosGeometryRasterizer(image_size=256).to(device)
    xml_renderer = DifferentiableHeliosXMLRenderer(rasterizer).to(device)
    node_renderer = DifferentiableHeliosRenderer(rasterizer).to(device)

    daps = args.daps
    seed = args.seed
    results = []

    for dap in daps:
        if args.skip_cpp:
            xml_path = os.path.join(output_dir, f"dap{dap}_gt_0000_plant_0000.xml")
            img_path = os.path.join(output_dir, f"dap{dap}_gt_0000_rad.jpeg")
            masks_path = os.path.join(output_dir, f"dap{dap}_gt_0000_masks.json")
            boxes_path = os.path.join(output_dir, f"dap{dap}_gt_0000_boxes.txt")
            gt_info = {
                "dap": dap,
                "xml_path": xml_path,
                "img_path": img_path,
                "masks_path": masks_path,
                "boxes_path": boxes_path,
                "cpp_time_ms": 0.0,
            }
        else:
            gt_info = generate_gt_sample(output_dir, dap=dap, seed=seed)

        # Load C++ Helios image
        cpp_pil = Image.open(gt_info["img_path"]).convert("RGB").resize((256, 256), Image.LANCZOS)
        cpp_np = np.array(cpp_pil, dtype=np.float32) / 255.0

        # Bounding box overlay (Col 2)
        bbox_pil = render_bbox_overlay(cpp_pil, gt_info["masks_path"], gt_info["boxes_path"])
        bbox_np = np.array(bbox_pil, dtype=np.float32) / 255.0

        # Segmentation mask (Col 3)
        seg_mask_np = render_segmentation_mask(gt_info["masks_path"], image_size=(256, 256)) / 255.0

        # Parse 22D organ nodes from XML
        t_parse_start = time.time()
        parser_xml = HeliosXMLParser(gt_info["xml_path"])
        parser_xml.parse()
        organ_nodes = parser_xml.get_all_organ_nodes()
        nodes_np = np.stack([n.to_vec() for n in organ_nodes], axis=0)  # (N, 22)
        nodes_t = torch.tensor(nodes_np, dtype=torch.float32, device=device).unsqueeze(0)
        t_parse_ms = (time.time() - t_parse_start) * 1000.0

        # Track A: XML-native renderer
        geom_xml = build_helios_geometry_from_xml(gt_info["xml_path"])
        t_a_start = time.time()
        with torch.no_grad():
            track_a_rgba = xml_renderer(
                geom_xml,
                camera_height=5.0,
                distance_from_center=2.0,
                azimuth_deg=0.0,
                focus_plant=True,
                background="black",
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        track_a_time_ms = (time.time() - t_a_start) * 1000.0
        track_a_rgb = track_a_rgba[0, :3].permute(1, 2, 0).cpu().numpy().clip(0, 1)

        # Track B: 22D node-array renderer
        # Warmup
        with torch.no_grad():
            _ = node_renderer(nodes_t, camera_height=5.0, distance_from_center=2.0, azimuth_deg=0.0, focus_plant=True, background="black")
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_b_start = time.time()
        with torch.no_grad():
            track_b_rgba = node_renderer(nodes_t, camera_height=5.0, distance_from_center=2.0, azimuth_deg=0.0, focus_plant=True, background="black")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        track_b_time_ms = (time.time() - t_b_start) * 1000.0
        track_b_rgb = track_b_rgba[0, :3].permute(1, 2, 0).cpu().numpy().clip(0, 1)

        # Compute Metrics
        ssim_a = compute_ssim_numpy(cpp_np, track_a_rgb)
        ssim_b = compute_ssim_numpy(cpp_np, track_b_rgb)

        mask_cpp = np.linalg.norm(cpp_np - np.array([0.61, 0.58, 0.55]), axis=-1) > 0.1
        mask_a = track_a_rgba[0, 3].cpu().numpy() > 0.05
        mask_b = track_b_rgba[0, 3].cpu().numpy() > 0.05
        iou_a = float(np.logical_and(mask_cpp, mask_a).sum() / max(np.logical_or(mask_cpp, mask_a).sum(), 1))
        iou_b = float(np.logical_and(mask_cpp, mask_b).sum() / max(np.logical_or(mask_cpp, mask_b).sum(), 1))

        mae_a = float(np.abs(cpp_np - track_a_rgb).mean())
        mae_b = float(np.abs(cpp_np - track_b_rgb).mean())
        mae_ab = float(np.abs(track_a_rgb - track_b_rgb).mean())

        n_tubes = int(nodes_np[:, 11:13].sum())
        n_leaflets = int(nodes_np[:, 13].sum())

        results.append({
            "dap": dap,
            "cpp_np": cpp_np,
            "bbox_np": bbox_np,
            "seg_mask_np": seg_mask_np,
            "track_a_rgb": track_a_rgb,
            "track_b_rgb": track_b_rgb,
            "diff_ab": np.abs(track_a_rgb - track_b_rgb),
            "cpp_time_ms": gt_info["cpp_time_ms"],
            "track_a_time_ms": track_a_time_ms,
            "track_b_time_ms": track_b_time_ms,
            "parse_ms": t_parse_ms,
            "ssim_a": ssim_a,
            "ssim_b": ssim_b,
            "iou_a": iou_a,
            "iou_b": iou_b,
            "mae_a": mae_a,
            "mae_b": mae_b,
            "mae_ab": mae_ab,
            "n_tubes": n_tubes,
            "n_leaflets": n_leaflets,
        })

        print(f"DAP {dap:02d} | C++: {gt_info['cpp_time_ms']:.1f}ms")
        print(f"       | Track A XML : {track_a_time_ms:.2f}ms | SSIM={ssim_a:.4f} | IoU={iou_a:.4f} | MAE={mae_a:.5f}")
        print(f"       | Track B 22D : {track_b_time_ms:.2f}ms | SSIM={ssim_b:.4f} | IoU={iou_b:.4f} | MAE={mae_b:.5f}")
        print(f"       | Track A-B   : MAE={mae_ab:.5f}")

    # Create N-Row x 6-Column Comparison Grid Figure
    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 6, figsize=(30, 5.5 * n_rows), facecolor="black")
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    for row in axes:
        for ax in row:
            ax.set_facecolor("black")

    for row_idx, res in enumerate(results):
        dap = res["dap"]

        # Col 0: C++ Helios Reference GT
        axes[row_idx, 0].imshow(res["cpp_np"])
        axes[row_idx, 0].set_title(f"DAP {dap} - C++ Helios GT\n(Time: {res['cpp_time_ms']:.0f} ms)", color="white", fontsize=12, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # Col 1: Bounding Box Overlay
        axes[row_idx, 1].imshow(res["bbox_np"])
        axes[row_idx, 1].set_title(f"DAP {dap} - Bounding Box Overlay", color="yellow", fontsize=12, fontweight="bold")
        axes[row_idx, 1].axis("off")

        # Col 2: Organ Segmentation Mask
        axes[row_idx, 2].imshow(res["seg_mask_np"])
        axes[row_idx, 2].set_title(f"DAP {dap} - Organ Mask\n(Yellow: Flower | Cyan: Pod | Green: Plant)", color="cyan", fontsize=11, fontweight="bold")
        axes[row_idx, 2].axis("off")

        # Col 3: Track A XML-native Renderer
        axes[row_idx, 3].imshow(res["track_a_rgb"])
        axes[row_idx, 3].set_title(f"DAP {dap} - Track A XML\n(Render: {res['track_a_time_ms']:.2f} ms | MAE={res['mae_a']:.5f})", color="springgreen", fontsize=11, fontweight="bold")
        axes[row_idx, 3].axis("off")

        # Col 4: Track B 22D Node-Array Renderer
        axes[row_idx, 4].imshow(res["track_b_rgb"])
        axes[row_idx, 4].set_title(f"DAP {dap} - Track B 22D\n(Render: {res['track_b_time_ms']:.2f} ms | MAE={res['mae_b']:.5f})", color="springgreen", fontsize=11, fontweight="bold")
        axes[row_idx, 4].axis("off")

        # Col 5: Difference Map & Summary Box
        axes[row_idx, 5].imshow(res["diff_ab"])
        axes[row_idx, 5].set_title(f"DAP {dap} - |Track A - Track B|\n(MAE={res['mae_ab']:.5f})", color="white", fontsize=11, fontweight="bold")
        axes[row_idx, 5].axis("off")

    plt.tight_layout()
    dap_tag = "_".join(str(d) for d in daps)
    bench_fig_path = os.path.join(output_dir, f"dap_{dap_tag}_track_a_vs_b_benchmark.png")
    plt.savefig(bench_fig_path, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"\nSaved Multi-DAP Track A vs B Benchmark Figure to: {bench_fig_path}")


if __name__ == "__main__":
    main()
