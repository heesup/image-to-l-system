"""Compare Helios-generated 3D geometry with XML-derived 3D geometry.

This script loads:
  1. A Helios PLY (ground-truth mesh vertices, with organ labels).
  2. A Helios plant XML file.

It reconstructs the XML into a 3D point cloud using plant_geometry_3d.py and
computes the symmetric Chamfer distance against the Helios point cloud after
simple centroid/bbox normalization.

Usage:
    python diffusion_based/eval/compare_xml_helios_3d.py \
        --helios-ply output/plant_0000_helios.ply \
        --xml output/plant_0000.xml \
        --out-ply-xml /tmp/xml_pointcloud.ply \
        --visualize
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Allow imports from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from diffusion_based.models.legacy.helios_geometry_track_a import build_helios_geometry_from_xml
from diffusion_based.models.legacy.pointcloud_loss_3d_track_a import (
    chamfer_distance_numpy,
    normalize_point_clouds,
    write_ply,
)

def xml_to_point_cloud(xml_path, n_circ=8, n_axis=4, n_leaf_u=6, n_leaf_v=10, n_ellipsoid_theta=8, n_ellipsoid_phi=6):
    """Backward-compatible wrapper: Helios XML -> point cloud."""
    geom = build_helios_geometry_from_xml(xml_path)
    return geom.to_point_cloud(
        n_circ=n_circ,
        n_axis_per_seg=n_axis,
        leaf_subdiv_u=n_leaf_u,
        leaf_subdiv_v=n_leaf_v,
        ellipsoid_theta=n_ellipsoid_theta,
        ellipsoid_phi=n_ellipsoid_phi,
    )


def load_ply(path: str) -> tuple:
    """Load a PLY file with x,y,z,r,g,b,organ layout. Returns xyz, colors, organs."""
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.strip() == b"end_header":
                break
        header_str = header.decode()
        n_verts = 0
        has_color = False
        has_organ = False
        for line in header_str.split("\n"):
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("property uchar red"):
                has_color = True
            elif line.startswith("property uchar organ"):
                has_organ = True

        xyz = np.zeros((n_verts, 3), dtype=np.float32)
        colors = None
        organs = None
        if has_color:
            colors = np.zeros((n_verts, 3), dtype=np.uint8)
        if has_organ:
            organs = np.zeros(n_verts, dtype=np.uint8)

        for i in range(n_verts):
            xyz[i] = np.fromfile(f, dtype=np.float32, count=3)
            if has_color:
                colors[i] = np.fromfile(f, dtype=np.uint8, count=3)
            if has_organ:
                organs[i] = np.fromfile(f, dtype=np.uint8, count=1)[0]

        return xyz, colors, organs


def compute_organ_wise_chamfer(pred: np.ndarray, target: np.ndarray, pred_organs: np.ndarray, target_organs: np.ndarray) -> dict:
    """Compute Chamfer distance per organ class.

    Organ mapping: 0=internode, 1=petiole, 2=leaf, 3=flower, 4=fruit, 9=unknown
    """
    from scipy.spatial.distance import cdist

    organ_names = {
        0: "internode",
        1: "petiole",
        2: "leaf",
        3: "flower",
        4: "fruit",
        9: "unknown",
    }
    results = {}
    for organ in sorted(set(np.unique(pred_organs).tolist() + np.unique(target_organs).tolist())):
        pred_mask = pred_organs == organ
        tgt_mask = target_organs == organ
        if pred_mask.sum() == 0 or tgt_mask.sum() == 0:
            results[organ_names.get(organ, f"organ_{organ}")] = float("nan")
            continue
        d = cdist(pred[pred_mask], target[tgt_mask], metric="euclidean")
        cd = d.min(axis=1).mean() + d.min(axis=0).mean()
        results[organ_names.get(organ, f"organ_{organ}")] = float(cd)
    return results


def visualize_side_by_side(pred: np.ndarray, target: np.ndarray, pred_organs, target_organs, save_path: str):
    """Create a side-by-side 3D scatter plot."""
    organ_colors = {
        0: [0.55, 0.27, 0.07],  # internode brown
        1: [0.68, 0.85, 0.38],  # petiole yellow-green
        2: [0.13, 0.55, 0.13],  # leaf green
        3: [0.85, 0.85, 0.20],  # flower yellow
        4: [0.82, 0.41, 0.12],  # fruit orange
        9: [0.5, 0.5, 0.5],     # unknown gray
    }

    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    for ax, pts, organs, title in [(ax1, pred, pred_organs, "XML-derived point cloud"),
                                    (ax2, target, target_organs, "Helios ground-truth")]:
        for organ in np.unique(organs):
            mask = organs == organ
            if mask.sum() == 0:
                continue
            c = organ_colors.get(int(organ), [0.5, 0.5, 0.5])
            ax.scatter(pts[mask, 0], pts[mask, 1], pts[mask, 2], c=[c], s=1, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved side-by-side visualization to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--helios-ply", required=True, help="Path to Helios-generated PLY")
    parser.add_argument("--xml", required=True, help="Path to Helios plant XML")
    parser.add_argument("--out-ply-xml", default="/tmp/xml_pointcloud.ply",
                        help="Path to write XML-derived PLY for inspection")
    parser.add_argument("--visualize", action="store_true", help="Generate side-by-side plot")
    parser.add_argument("--visualize-path", default="/tmp/compare_3d.png")
    parser.add_argument("--subsample", type=int, default=0,
                        help="If >0, randomly subsample both clouds to N points for speed")
    args = parser.parse_args()

    print(f"Loading Helios PLY: {args.helios_ply}")
    helio_xyz, helio_colors, helio_organs = load_ply(args.helios_ply)
    print(f"  Helios points: {helio_xyz.shape[0]}")

    print(f"Loading XML and generating point cloud: {args.xml}")
    xml_xyz, xml_colors, xml_organs = xml_to_point_cloud(args.xml)
    print(f"  XML points: {xml_xyz.shape[0]}")

    if xml_xyz.shape[0] == 0:
        print("ERROR: XML-derived point cloud is empty.")
        return

    if args.out_ply_xml:
        write_ply(args.out_ply_xml, xml_xyz, xml_colors, xml_organs)
        print(f"  Wrote XML-derived PLY: {args.out_ply_xml}")

    # Normalize both clouds to the target (Helios) bounding box
    xml_norm, helio_norm = normalize_point_clouds(xml_xyz, helio_xyz)

    if args.subsample > 0:
        rng = np.random.default_rng(42)
        if xml_norm.shape[0] > args.subsample:
            idx = rng.choice(xml_norm.shape[0], args.subsample, replace=False)
            xml_norm = xml_norm[idx]
            xml_organs = xml_organs[idx]
        if helio_norm.shape[0] > args.subsample:
            idx = rng.choice(helio_norm.shape[0], args.subsample, replace=False)
            helio_norm = helio_norm[idx]
            helio_organs = helio_organs[idx]

    print("Computing Chamfer distance...")
    cd = chamfer_distance_numpy(xml_norm, helio_norm)
    print(f"  Overall Chamfer distance (target-normalized): {cd:.6f}")

    print("Computing organ-wise Chamfer distance...")
    organ_cd = compute_organ_wise_chamfer(xml_norm, helio_norm, xml_organs, helio_organs)
    for name, val in organ_cd.items():
        print(f"    {name}: {val:.6f}")

    if args.visualize:
        visualize_side_by_side(xml_norm, helio_norm, xml_organs, helio_organs, args.visualize_path)


if __name__ == "__main__":
    main()
