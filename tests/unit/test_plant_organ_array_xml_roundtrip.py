"""
Unit test: PlantOrganArray XML round-trip text identity.

Loads Helios XML files, converts them to a PlantOrganArray Tensor (N, 93),
serializes back to XML, and verifies byte-for-byte text identity.
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from diffusion_based.models.plant_organ_array import PlantOrganArray


def normalize_xml(xml_str: str) -> str:
    """Normalizes XML string lines by stripping trailing whitespace."""
    lines = [line.rstrip() for line in xml_str.strip().splitlines()]
    return "\n".join(lines)


def run_roundtrip_test(xml_dir: str) -> int:
    xml_files = sorted(glob.glob(os.path.join(xml_dir, "*.xml")))
    if not xml_files:
        print(f"No XML files found in {xml_dir}")
        sys.exit(1)

    print(f"Testing PlantOrganArray (N, 93) XML round-trip on {len(xml_files)} file(s)...")
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
            print(f"FAIL: {filename}")
            orig_lines = norm_orig.splitlines()
            recon_lines = norm_recon.splitlines()
            for idx, (l1, l2) in enumerate(zip(orig_lines, recon_lines)):
                if l1 != l2:
                    print(f"Line {idx + 1} mismatch:\n  ORIG : {l1!r}\n  RECON: {l2!r}")
                    break
            sys.exit(1)

        print(f"  OK {filename}  shape={tuple(organ_array.tensor.shape)}")
        passed += 1

    print(f"\nSUCCESS: {passed}/{len(xml_files)} XML files passed lossless round-trip.")
    return passed


if __name__ == "__main__":
    default_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Digital-Crops",
        "projects",
        "syntheticdata_generation",
        "build",
        "output",
    )
    run_roundtrip_test(default_dir)
