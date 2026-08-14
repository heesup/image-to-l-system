"""
Unit test: PlantOrganArray XML round-trip text identity.

Loads Helios XML files, converts them to both legacy (N, 94) and typed (N, 40)
PlantOrganArray tensors, serializes back to XML, and verifies byte-for-byte
text identity for both paths.
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

    print(f"Testing PlantOrganArray XML round-trip on {len(xml_files)} file(s)...")
    passed_legacy = 0
    passed_typed = 0
    for xml_path in xml_files:
        filename = os.path.basename(xml_path)
        with open(xml_path, "r", encoding="utf-8") as f:
            original_str = f.read()

        # Legacy (N, 94) path
        organ_array_legacy = PlantOrganArray.from_xml_string(original_str)
        reconstructed_legacy = organ_array_legacy.to_xml_string()
        norm_orig = normalize_xml(original_str)
        norm_recon_legacy = normalize_xml(reconstructed_legacy)

        if norm_orig != norm_recon_legacy:
            print(f"FAIL legacy: {filename}")
            orig_lines = norm_orig.splitlines()
            recon_lines = norm_recon_legacy.splitlines()
            for idx, (l1, l2) in enumerate(zip(orig_lines, recon_lines)):
                if l1 != l2:
                    print(f"Line {idx + 1} mismatch:\n  ORIG : {l1!r}\n  RECON: {l2!r}")
                    break
            sys.exit(1)

        print(f"  OK legacy {filename}  shape={tuple(organ_array_legacy.tensor.shape)}")
        passed_legacy += 1

        # Typed (N, 40) path
        organ_array_typed = PlantOrganArray.from_xml_string_typed(original_str)
        reconstructed_typed = organ_array_typed.to_xml_string()
        norm_recon_typed = normalize_xml(reconstructed_typed)

        if norm_orig != norm_recon_typed:
            print(f"FAIL typed: {filename}")
            orig_lines = norm_orig.splitlines()
            recon_lines = norm_recon_typed.splitlines()
            for idx, (l1, l2) in enumerate(zip(orig_lines, recon_lines)):
                if l1 != l2:
                    print(f"Line {idx + 1} mismatch:\n  ORIG : {l1!r}\n  RECON: {l2!r}")
                    break
            sys.exit(1)

        print(f"  OK typed  {filename}  shape={tuple(organ_array_typed.tensor.shape)}")
        passed_typed += 1

    print(f"\nSUCCESS: {passed_legacy}/{len(xml_files)} passed legacy (N, 94) round-trip.")
    print(f"SUCCESS: {passed_typed}/{len(xml_files)} passed typed (N, 40) round-trip.")
    return passed_legacy + passed_typed


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
