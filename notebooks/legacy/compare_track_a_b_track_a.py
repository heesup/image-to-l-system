"""Quick comparison of Track A (XML-native) vs Track B (22D node array)."""
import os
import sys
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.helios_xml_parser import HeliosXMLParser
from diffusion_based.models.legacy.differentiable_pipeline_track_a import DifferentiableHeliosRenderer
from diffusion_based.models.legacy.helios_geometry_legacy import build_helios_geometry_from_xml, DifferentiableHeliosXMLRenderer
from diffusion_based.models.legacy.helios_rasterizer_3d_track_a import HeliosGeometryRasterizer


def render_track_a(xml_path, rasterizer, device):
    geom = build_helios_geometry_from_xml(xml_path)
    renderer = DifferentiableHeliosXMLRenderer(rasterizer).to(device)
    with torch.no_grad():
        rgba = renderer(geom, camera_height=5.0, distance_from_center=2.0, azimuth_deg=0.0, focus_plant=True, background="black")
    img = rgba[0, :3].permute(1, 2, 0).cpu().numpy().clip(0, 1)
    return img


def render_track_b(xml_path, rasterizer, device):
    parser = HeliosXMLParser(xml_path)
    parser.parse()
    nodes = parser.get_all_organ_nodes()
    nodes_np = np.stack([n.to_vec() for n in nodes], axis=0)
    nodes_t = torch.tensor(nodes_np, dtype=torch.float32, device=device).unsqueeze(0)
    renderer = DifferentiableHeliosRenderer(rasterizer).to(device)
    with torch.no_grad():
        rgba = renderer(nodes_t, camera_height=5.0, distance_from_center=2.0, azimuth_deg=0.0, focus_plant=True, background="black")
    img = rgba[0, :3].permute(1, 2, 0).cpu().numpy().clip(0, 1)
    return img


def main():
    output_dir = os.path.join(repo_root, "notebooks", "output_dap_benchmark")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rasterizer = HeliosGeometryRasterizer(image_size=256).to(device)
    daps = [10, 50, 90]
    fig, axes = plt.subplots(len(daps), 4, figsize=(20, 5 * len(daps)), facecolor="black")
    if len(daps) == 1:
        axes = np.expand_dims(axes, 0)
    for ax in axes.flatten():
        ax.set_facecolor("black")

    for i, dap in enumerate(daps):
        xml_path = os.path.join(output_dir, f"dap{dap}_gt_0000_plant_0000.xml")
        if not os.path.exists(xml_path):
            print(f"Missing {xml_path}")
            continue
        print(f"Processing DAP {dap}...")
        gt_img = np.array(Image.open(xml_path.replace("plant_0000.xml", "rad.jpeg")).convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
        track_a = render_track_a(xml_path, rasterizer, device)
        track_b = render_track_b(xml_path, rasterizer, device)
        diff = np.abs(track_a - track_b)
        axes[i, 0].imshow(gt_img)
        axes[i, 0].set_title(f"DAP {dap} C++ GT", color="white")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(track_a)
        axes[i, 1].set_title(f"DAP {dap} Track A XML", color="white")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(track_b)
        axes[i, 2].set_title(f"DAP {dap} Track B 22D", color="white")
        axes[i, 2].axis("off")
        axes[i, 3].imshow(diff)
        axes[i, 3].set_title(f"DAP {dap} |Track A - Track B|", color="white")
        axes[i, 3].axis("off")
        mae_a = np.abs(gt_img - track_a).mean()
        mae_b = np.abs(gt_img - track_b).mean()
        mae_ab = diff.mean()
        print(f"  DAP {dap}: Track A MAE vs GT={mae_a:.5f}, Track B MAE vs GT={mae_b:.5f}, Track A-B MAE={mae_ab:.5f}")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "track_a_vs_b_preview.png")
    plt.savefig(save_path, dpi=150, facecolor="black")
    plt.close()
    print(f"Saved preview to {save_path}")


if __name__ == "__main__":
    main()
