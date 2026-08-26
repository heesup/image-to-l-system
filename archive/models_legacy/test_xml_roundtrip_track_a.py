"""Round-trip test: XML -> 15D -> XML -> 15D -> render compare."""

import os
import sys
import tempfile
import numpy as np
from PIL import Image

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from diffusion_based.models.helios_xml_parser import HeliosXMLParser
from diffusion_based.models.legacy.helios_xml_writer_track_a import write_organ_nodes_to_xml
from diffusion_based.models.legacy.helios_geometry_track_a import build_helios_geometry_from_nodes
from diffusion_based.models.legacy.helios_rasterizer_3d_track_a import HeliosGeometryRasterizer


def render_nodes(nodes, image_size=256):
    """Render 15D nodes using the differentiable rasterizer."""
    geom = build_helios_geometry_from_nodes(nodes)
    raster = HeliosGeometryRasterizer(
        image_size=image_size,
        fov_deg=50.0,
    )
    pts, _, _ = geom.to_point_cloud()
    if pts.shape[0] == 0:
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)
    center = (pts.min(0) + pts.max(0)) / 2.0
    img_np = raster.render_numpy_geometry(
        geom.tubes,
        geom.leaflets,
        geom.ellipsoids,
        camera_height=1.0,
        distance_from_center=1.2,
        azimuth_deg=225.0,
        target_center=center,
        hfov_deg=50.0,
    )
    return (img_np * 255).clip(0, 255).astype(np.uint8)


def compute_image_similarity(img_a, img_b):
    """Return intersection over union (IoU) of binary silhouettes."""
    if img_a.shape != img_b.shape:
        raise ValueError(f"Shape mismatch: {img_a.shape} vs {img_b.shape}")
    mask_a = img_a.sum(axis=-1) > 0
    mask_b = img_b.sum(axis=-1) > 0
    inter = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return float(inter) / float(union) if union > 0 else 1.0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("xml", help="Input Helios plant XML file")
    parser.add_argument("--out-dir", default=None, help="Directory to save debug outputs")
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    out_dir = args.out_dir or tempfile.mkdtemp(prefix="xml_roundtrip_")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Stage 1: XML -> 15D
    parser_a = HeliosXMLParser(args.xml)
    nodes_a = parser_a.get_all_organ_nodes()
    print(f"XML -> 15D: {len(nodes_a)} nodes")

    # Stage 2: 15D -> XML
    xml_b_path = os.path.join(out_dir, "roundtrip_reconstructed.xml")
    write_organ_nodes_to_xml(nodes_a, xml_b_path)
    print(f"15D -> XML: {xml_b_path}")

    # Stage 3: XML -> 15D again
    parser_b = HeliosXMLParser(xml_b_path)
    nodes_b = parser_b.get_all_organ_nodes()
    print(f"XML -> 15D again: {len(nodes_b)} nodes")

    # Compare node arrays
    if len(nodes_a) != len(nodes_b):
        print(f"WARNING: node count mismatch: {len(nodes_a)} vs {len(nodes_b)}")

    pose_diffs = []
    attr_diffs = []
    for i, (na, nb) in enumerate(zip(nodes_a, nodes_b)):
        pose_diffs.append(np.linalg.norm(na.position - nb.position))
        attr_diffs.append(
            np.linalg.norm(
                np.array([
                    na.length - nb.length,
                    na.radius - nb.radius,
                    na.pitch - nb.pitch,
                    na.yaw - nb.yaw,
                    na.roll - nb.roll,
                ])
            )
        )

    print(f"Position MAE: {float(np.mean(pose_diffs)):.6e}")
    print(f"Attribute MAE: {float(np.mean(attr_diffs)):.6e}")

    # Stage 4: Render both 15D graphs
    img_a = render_nodes(nodes_a, args.image_size)
    img_b = render_nodes(nodes_b, args.image_size)

    Image.fromarray(img_a).save(os.path.join(out_dir, "render_original.png"))
    Image.fromarray(img_b).save(os.path.join(out_dir, "render_reconstructed.png"))

    iou = compute_image_similarity(img_a, img_b)
    print(f"Render silhouette IoU: {iou:.4f}")

    if iou > 0.90 and float(np.mean(pose_diffs)) < 1e-3:
        print("ROUND-TRIP OK")
    else:
        print("ROUND-TRIP MISMATCH")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
