"""
Unit test for Plant Organ Array Tensor (N, 78) XML round-trip text identity verification.
"""

import os
import glob
import sys
from diffusion_based.models.plant_organ_array import PlantOrganArray


def normalize_xml(xml_str: str) -> str:
    """Normalizes XML string lines by stripping trailing whitespace."""
    lines = [line.rstrip() for line in xml_str.strip().splitlines()]
    return "\n".join(lines)


def run_organ_array_roundtrip_test():
    search_dir = "/home/lion397/codes/image-to-l-system/Digital-Crops/projects/syntheticdata_generation/build/output"
    xml_files = glob.glob(os.path.join(search_dir, "*.xml"))

    if not xml_files:
        print(f"No XML files found in {search_dir}")
        sys.exit(1)

    print(f"Testing Plant Organ Array Tensor (N, 78) XML Round-Trip Text Identity on {len(xml_files)} files...")

    passed = 0
    for xml_path in xml_files:
        filename = os.path.basename(xml_path)
        with open(xml_path, "r", encoding="utf-8") as f:
            original_str = f.read()

        organ_array = PlantOrganArray.from_xml_string(original_str)
        reconstructed_str = organ_array.to_xml_string()

        norm_orig = normalize_xml(original_str)
        norm_recon = normalize_xml(reconstructed_str)

        if norm_orig != norm_recon:
            print(f"FAIL: Organ Array Roundtrip mismatch for {filename}")
            orig_lines = norm_orig.splitlines()
            recon_lines = norm_recon.splitlines()
            for idx, (l1, l2) in enumerate(zip(orig_lines, recon_lines)):
                if l1 != l2:
                    print(f"Line {idx+1} mismatch:\n  ORIG : {l1!r}\n  RECON: {l2!r}")
                    break
            sys.exit(1)
        else:
            print(f"  ✓ {filename} (Organ Array Tensor shape: {organ_array.tensor.shape} - Identity verified)")
            passed += 1

    print(f"\nSUCCESS: All {passed} XML files passed Plant Organ Array Tensor round-trip text identity verification!")


if __name__ == "__main__":
    run_organ_array_roundtrip_test()
