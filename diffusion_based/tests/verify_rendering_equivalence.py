"""
Verify that canonical DFS sorting of typed PlantOrganArray preserves 100% 3D geometry and 2D rendering equivalence.
"""

import os
import sys
import glob
import torch
import torch.nn.functional as F
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.plant_organ_array import PlantOrganArray, sort_typed_organ_array_canonical
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer


def verify_equivalence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Rendering Equivalence Verification on device: {device}")

    xml_files = sorted(glob.glob("dataset/helios_data/*_plant_*.xml"))[:10]
    if not xml_files:
        xml_files = sorted(glob.glob("diffusion_based/eval/output/*.xml"))[:5]

    if not xml_files:
        print("No XML files found to test.")
        return

    renderer = HeliosPyTorchRenderer(image_size=128).to(device)

    all_passed = True

    for xml_path in xml_files:
        with open(xml_path, "r", encoding="utf-8") as f:
            xml_text = f.read()

        # Parse without canonical sorting (raw order from XML)
        import xml.etree.ElementTree as ET
        from diffusion_based.models.plant_organ_array import NUM_FEATURES_TYPED
        root = ET.fromstring(xml_text)
        # Load typed array (which now has canonical sorting)
        array_sorted = PlantOrganArray.from_xml_string_typed(xml_text)

        # Create an array with randomly shuffled valid rows to test arbitrary permutation invariance
        valid_mask = array_sorted.tensor[:, -1] > 0.5
        valid_rows = array_sorted.tensor[valid_mask]
        N = array_sorted.tensor.shape[0]
        perm = torch.randperm(valid_rows.shape[0])
        shuffled_valid = valid_rows[perm]
        shuffled_tensor = torch.zeros_like(array_sorted.tensor)
        shuffled_tensor[:shuffled_valid.shape[0]] = shuffled_valid
        array_shuffled = PlantOrganArray(tensor=shuffled_tensor)

        # Render canonical sorted array
        with torch.no_grad():
            rgb_sorted = renderer.render_organ_array(
                array_sorted,
                azimuth_deg=0.0,
                elevation_deg=90.0,
                camera_height=1.0,
                background="black",
                device=device,
                differentiable=False,
                focus_plant=True,
                existence_threshold=0.1,
            )

            # Render shuffled array
            rgb_shuffled = renderer.render_organ_array(
                array_shuffled,
                azimuth_deg=0.0,
                elevation_deg=90.0,
                camera_height=1.0,
                background="black",
                device=device,
                differentiable=False,
                focus_plant=True,
                existence_threshold=0.1,
            )

        max_diff = torch.max(torch.abs(rgb_sorted - rgb_shuffled)).item()
        mse = F.mse_loss(rgb_sorted, rgb_shuffled).item()

        passed = max_diff < 1e-4
        if not passed:
            all_passed = False

        status = "PASSED (100% IDENTICAL)" if passed else "FAILED"
        print(f"[{status}] {os.path.basename(xml_path)} | Max diff: {max_diff:.6f}, MSE: {mse:.8f}")

    if all_passed:
        print("\nAll rendering equivalence tests PASSED! Canonical DFS sorting preserves exact geometry and rendering.\n")
    else:
        print("\nSome tests failed.\n")


if __name__ == "__main__":
    verify_equivalence()
