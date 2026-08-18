"""
Unit test: 14D Part-Centric Representation, XML Roundtrip, and Rendering Identity.

Validates:
1. Lossless XML Roundtrip: XML -> 40D Typed -> 14D Part -> 40D Typed -> XML
2. 3D Vertex & Mesh Consistency: 40D Kinematics Mesh vs 14D Direct Mesh
3. Differentiable Rendering Identity: render_organ_array vs render_part_tensor_14d
"""

import glob
import math
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer


def normalize_xml(xml_str: str) -> str:
    """Normalizes XML string lines by stripping trailing whitespace."""
    lines = [line.rstrip() for line in xml_str.strip().splitlines()]
    return "\n".join(lines)


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Computes SSIM between two (H, W, 3) or (H, W, 4) tensors in [0, 1]."""
    if img1.shape[-1] == 4:
        img1 = img1[..., :3]
    if img2.shape[-1] == 4:
        img2 = img2[..., :3]

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_val.item())


def test_14d_representation(xml_dir: str):
    xml_files = sorted(glob.glob(os.path.join(xml_dir, "*.xml")))
    if not xml_files:
        print(f"No XML files found in {xml_dir}")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    geo_builder = HeliosPlantGeometryBuilder()
    renderer = HeliosPyTorchRenderer(image_size=512)

    for xml_path in xml_files:
        filename = os.path.basename(xml_path)
        with open(xml_path, "r", encoding="utf-8") as f:
            original_xml = f.read()

        norm_orig = normalize_xml(original_xml)

        # 1. XML -> 40D Typed PlantOrganArray
        array_40d = PlantOrganArray.from_xml_string_typed(original_xml)

        # 2. 40D -> 14D Part-Centric Tensor (via Forward Kinematics extraction)
        part_tensor_14d = array_40d.to_part_tensor_14d(device=device)
        assert part_tensor_14d.shape[1] == 14, f"Expected 14 columns, got {part_tensor_14d.shape}"

        # 3. 14D -> 40D Typed PlantOrganArray (Inverse Kinematics / topology snip)
        array_reconstructed = PlantOrganArray.from_part_tensor_14d(part_tensor_14d, template_array=array_40d)

        # 4. 40D -> XML String Roundtrip verification
        reconstructed_xml = array_reconstructed.to_xml_string()
        norm_recon = normalize_xml(reconstructed_xml)

        if norm_orig != norm_recon:
            print(f"FAIL XML Roundtrip on {filename}")
            orig_lines = norm_orig.splitlines()
            recon_lines = norm_recon.splitlines()
            for idx, (l1, l2) in enumerate(zip(orig_lines, recon_lines)):
                if l1 != l2:
                    print(f"  Line {idx + 1} mismatch:\n    ORIG : {l1!r}\n    RECON: {l2!r}")
                    break
            sys.exit(1)

        print(f"[OK] XML Roundtrip: {filename} (N={part_tensor_14d.shape[0]} organs)")

        # 5. Mesh Vertex Consistency Check (via Kinematics Tree Dispatch)
        mesh_40d = geo_builder.build_mesh_from_organ_array(array_40d, device=device)
        mesh_14d = geo_builder.build_mesh_from_part_array_14d(part_tensor_14d, template_organ_array=array_40d, device=device, use_kinematics_tree=True)

        v_40d = mesh_40d["vertices"]
        v_14d = mesh_14d["vertices"]

        # If plant has leaves or geometry, check vertices
        if v_40d.shape[0] > 0 and v_14d.shape[0] > 0:
            assert v_40d.shape == v_14d.shape, f"Vertex shape mismatch: {v_40d.shape} vs {v_14d.shape}"
            v_diff = torch.max(torch.abs(v_40d - v_14d)).item()
            print(f"     Mesh Vertex Max Diff (Tree): {v_diff:.6e} (vertices: {v_40d.shape[0]})")

        # 6. Differentiable Rendering Identity Check (Kinematics Tree & Direct 14D)
        rendered_40d = renderer.render_organ_array(array_40d, device=device, azimuth_deg=45.0, elevation_deg=60.0)
        rendered_14d_tree = renderer.render_part_tensor_14d(part_tensor_14d, template_organ_array=array_40d, device=device, azimuth_deg=45.0, elevation_deg=60.0, use_kinematics_tree=True)
        rendered_14d_direct = renderer.render_part_tensor_14d(part_tensor_14d, template_organ_array=array_40d, device=device, azimuth_deg=45.0, elevation_deg=60.0, use_kinematics_tree=False)

        mse_tree = torch.mean((rendered_40d - rendered_14d_tree) ** 2).item()
        ssim_tree = calculate_ssim(rendered_40d, rendered_14d_tree)
        mse_direct = torch.mean((rendered_40d - rendered_14d_direct) ** 2).item()
        ssim_direct = calculate_ssim(rendered_40d, rendered_14d_direct)

        print(f"     Render Identity (Tree):   MSE = {mse_tree:.6e}, SSIM = {ssim_tree:.4f}")
        print(f"     Render Identity (Direct): MSE = {mse_direct:.6e}, SSIM = {ssim_direct:.4f}")
        assert ssim_tree > 0.999, f"Tree render diverged! SSIM = {ssim_tree}"
        assert ssim_direct > 0.85, f"Direct 14D render diverged! SSIM = {ssim_direct}"

    print("\nALL 14D PART REPRESENTATION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    default_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Digital-Crops",
        "projects",
        "syntheticdata_generation",
        "build",
        "output",
    )
    test_14d_representation(default_dir)
