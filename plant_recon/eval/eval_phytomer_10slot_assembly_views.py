"""
fig12: the 10-slot ordered packet assembly seen from nadir AND from a 45-degree
perspective, next to the ground-truth mesh under the SAME two cameras, plus the
Helios round-trips (packet path only, and packet path + PhytomerVAE) on
dataset plants (DAP 15 / 40 / 75, with internode curvature, phyllotaxy and
yaw perturbations -- unlike the exact_gt_renders trio of fig14).

Replaces the 2026-09-11 panel of the same name, which was drawn by an
uncommitted script, showed the assembly's 45-degree view with no ground-truth
45-degree view to compare against, and whose Helios column predates the
shoot-partition fix and the stem IK export.

Columns:
  0  Helios GT raytrace (nadir, --focus-plant)
  1  GT 14D -> PyTorch mesh, nadir
  2  GT 14D -> PyTorch mesh, 45-degree perspective
  3  packets -> (VAE = identity) -> 14D -> PyTorch mesh, nadir      [RGB MAE vs col 1]
  4  same, 45-degree perspective                                    [RGB MAE vs col 2]
  5  that 14D -> XML (stem IK) -> Helios                            [FG IoU vs col 0]
  6  packets -> PhytomerVAE -> 14D -> XML (stem IK) -> Helios       [FG IoU vs col 0]

Columns 1-4 share one camera per row (bounds taken from the GT mesh), so a
shape difference between 2 and 4 is geometry, not framing.

Output: docs/results/assets/fig12_phytomer_10slot_helios_roundtrip.png
"""
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

from plant_recon.models.plant_organ_array import (
    PlantOrganArray, ORGAN_NONE, ORGAN_INTERNODE, P_COL_ORGAN_TYPE)
from plant_recon.models.part_tensor_to_40d import assemble_part_tensor_to_xml
from plant_recon.models.phytomer_vae import PhytomerVAE
from plant_recon.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder
from plant_recon.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from plant_recon.dataset.part_array_dataset import encode_fm, BASE_SCALE
from plant_recon.dataset.phytomer_packets import (
    build_phytomer_packets, decode_packets, assemble_packets, phytomer_scale,
    normalize_packet_scales, denormalize_packet_scales, strip_base,
    emit_part_tensor_with_shoot_meta, FM_BASE_START, FM_BASE_END)
from plant_recon.dataset.phytomer_topology import chain_phytomers, gt_parent_links
from plant_recon.dataset.generate_cache import extract_phytomer_ids
from plant_recon.eval.eval_13d_xml_organ_masks import (
    render_helios_full, IMG_SIZE, OUTPUT_DIR)
from plant_recon.eval.eval_phytomer_vae_helios_roundtrip import (
    build_vae_roundtrip_xml, DEFAULT_VAE_CHECKPOINT)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SCRATCH_DIR = "/tmp/helios_10slot_views_eval"
_DS = "dataset/helios_data/cowpea/cowpea_dap{dap:03d}_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml"
PLANTS = [
    ("DAP 15 (Seedling)", _DS.format(dap=15), "dap015"),
    ("DAP 40 (Vegetative)", _DS.format(dap=40), "dap040"),
    ("DAP 75 (Reproductive)", _DS.format(dap=75), "dap075"),
]


def organ_rows(t14: torch.Tensor) -> torch.Tensor:
    return t14[t14[:, P_COL_ORGAN_TYPE] >= ORGAN_INTERNODE]


def packet_identity_14d(arr: PlantOrganArray, gt14: torch.Tensor) -> torch.Tensor:
    """The packet path with the VAE replaced by identity: GT 14D -> 10-slot packets
    -> normalise/strip base -> denormalise -> assemble -> decode -> emit 14D."""
    nodes_fm = encode_fm(gt14)
    exist = (gt14[:, P_COL_ORGAN_TYPE] > ORGAN_NONE).float()
    pids = extract_phytomer_ids(arr, gt14.shape[0])
    packets, presence, centers, refs, keys = build_phytomer_packets(
        nodes_fm, existence_mask=exist, phytomer_ids=pids, return_keys=True)
    packets, presence = packets.to(DEVICE).float(), presence.to(DEVICE)
    centers, refs = centers.to(DEVICE), refs.to(DEVICE)
    s_a = phytomer_scale(packets)
    ibase = decode_packets(packets, centers, presence.bool(), refs)[:, 0, FM_BASE_START:FM_BASE_END] / BASE_SCALE
    _, _, depth = gt_parent_links(centers, keys.to(DEVICE), internode_base=ibase)   # depth-from-root ordinal, as Stage 2 predicts it
    parent_idx, shoot_id, phy_idx = chain_phytomers(
        centers, refs, ordinal=depth.float(), is_base=(depth == 0).float(), root_own_shoot=True)
    # A shoot's first internode starts at its own decoded base, not at the
    # branch node's centre (0.2-0.9 cm apart on dataset plants); drawing it
    # from the centre leaves a 0.35 cm floor under the stem IK on every shoot.
    first = phy_idx == 0
    parent_pos = torch.where(((parent_idx >= 0) & ~first).unsqueeze(-1),
                             centers[parent_idx.clamp(min=0)],
                             torch.full_like(centers, float("nan")))
    ident = strip_base(normalize_packet_scales(packets, s_a))
    den = denormalize_packet_scales(ident, s_a)
    asm = assemble_packets(den, refs, parent_pos=parent_pos, centers=centers)
    absd = decode_packets(asm, centers, presence.bool(), refs)
    return emit_part_tensor_with_shoot_meta(absd, presence.bool(), centers, refs, shoot_id, phy_idx).cpu()


def render_torch(renderer, mesh, elev, azim, bounds):
    rgb = renderer.forward(mesh, azimuth_deg=azim, elevation_deg=elev, camera_height=5.0,
                           background="ground", focus_plant=True, image_size=IMG_SIZE,
                           fixed_camera_bounds=bounds)
    return rgb[:3].permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy()


def fg_iou(gt, rec):
    a, b = gt["mask_map"] >= 0, rec["mask_map"] >= 0
    return float(np.logical_and(a, b).sum() / max(1, np.logical_or(a, b).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae_checkpoint", default=DEFAULT_VAE_CHECKPOINT)
    ap.add_argument("--elev", type=float, default=45.0, help="perspective elevation (deg)")
    ap.add_argument("--azim", type=float, default=0.0, help="perspective azimuth (deg)")
    ap.add_argument("--out", default=os.path.join(OUTPUT_DIR, "fig12_phytomer_10slot_helios_roundtrip.png"))
    args = ap.parse_args()
    os.environ.setdefault("PHYTOMER_TERMINAL_LAST",
                          "1" if "_tl" in os.path.basename(os.path.dirname(args.vae_checkpoint)) else "0")
    os.makedirs(SCRATCH_DIR, exist_ok=True)

    vae = PhytomerVAE(latent_dim=128, hidden_dim=256).to(DEVICE)
    vae.load_state_dict(torch.load(args.vae_checkpoint, map_location=DEVICE, weights_only=True))
    vae.eval()
    geo = HeliosPlantGeometryBuilder()
    renderer = HeliosPyTorchRenderer(image_size=IMG_SIZE).to(DEVICE)

    titles = [
        "Helios GT\nraytrace (nadir)",
        "GT 14D -> PyTorch mesh\nnadir",
        f"GT 14D -> PyTorch mesh\n{args.elev:.0f}° perspective",
        "10-slot packets -> 14D\n(VAE = identity), nadir",
        f"10-slot packets -> 14D\n(VAE = identity), {args.elev:.0f}°",
        "10-slot packets -> 14D -> XML\n(stem IK) -> Helios",
        "packets -> PhytomerVAE -> 14D\n-> XML (stem IK) -> Helios",
    ]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                        "font.size": 9, "axes.edgecolor": "#444444", "axes.linewidth": 0.6, "text.color": "#111111"})
    fig, axes = plt.subplots(len(PLANTS), 7, figsize=(31.5, 4.8 * len(PLANTS)), facecolor="white")
    plt.subplots_adjust(wspace=0.03, hspace=0.10, left=0.05, right=0.99, top=0.92, bottom=0.02)
    for c, t in enumerate(titles):
        axes[0, c].set_title(t, fontsize=10, color="#111111", pad=10)

    rows = []
    for r, (label, rel, tag) in enumerate(PLANTS):
        path = os.path.join(REPO_ROOT, rel)
        print(f"\n--- {label}: {rel}")
        arr = PlantOrganArray.from_xml_file(path)
        gt14 = arr.to_part_tensor().float()
        asm14 = packet_identity_14d(arr, gt14)
        n_gt, n_asm = len(organ_rows(gt14)), len(organ_rows(asm14))

        mesh_gt = geo.build_mesh_from_part_tensor(gt14.to(DEVICE), device=DEVICE)
        mesh_asm = geo.build_mesh_from_part_tensor(asm14.to(DEVICE), device=DEVICE)
        v = mesh_gt["vertices"]
        bounds = {"min": v.min(0).values.tolist(), "max": v.max(0).values.tolist()}
        gt_nadir = render_torch(renderer, mesh_gt, 90.0, 0.0, bounds)
        gt_persp = render_torch(renderer, mesh_gt, args.elev, args.azim, bounds)
        asm_nadir = render_torch(renderer, mesh_asm, 90.0, 0.0, bounds)
        asm_persp = render_torch(renderer, mesh_asm, args.elev, args.azim, bounds)
        mae_nadir = float(np.abs(asm_nadir - gt_nadir).mean())
        mae_persp = float(np.abs(asm_persp - gt_persp).mean())

        ik_xml = os.path.join(SCRATCH_DIR, f"assembled_{tag}.xml")
        with open(ik_xml, "w", encoding="utf-8") as f:
            f.write(assemble_part_tensor_to_xml(asm14))
        vae_xml_str, _, n_vae = build_vae_roundtrip_xml(arr, vae)
        vae_xml = os.path.join(SCRATCH_DIR, f"vae_{tag}.xml")
        with open(vae_xml, "w", encoding="utf-8") as f:
            f.write(vae_xml_str)
        print("  Helios: GT / assembled / VAE round-trip ...")
        h_gt = render_helios_full(path, f"gt_{tag}")
        h_asm = render_helios_full(ik_xml, f"assembled_{tag}")
        h_vae = render_helios_full(vae_xml, f"vae_{tag}")
        iou_asm, iou_vae = fg_iou(h_gt, h_asm), fg_iou(h_gt, h_vae)
        rows.append((label, n_gt, n_asm, n_vae, mae_nadir, mae_persp, iou_asm, iou_vae))
        print(f"  organs GT {n_gt} / assembled {n_asm} / VAE {n_vae}; RGB MAE nadir {mae_nadir:.4f}, "
              f"{args.elev:.0f}° {mae_persp:.4f}; Helios FG IoU assembled {iou_asm*100:.1f}%, VAE {iou_vae*100:.1f}%")

        imgs = [h_gt["rgb"], gt_nadir, gt_persp, asm_nadir, asm_persp, h_asm["rgb"], h_vae["rgb"]]
        notes = [f"{n_gt} organs", "", "", f"MAE vs col 2: {mae_nadir:.4f}\n{n_asm} organs",
                 f"MAE vs col 3: {mae_persp:.4f}", f"FG IoU: {iou_asm*100:.1f}%",
                 f"FG IoU: {iou_vae*100:.1f}%\norgans {n_vae}/{n_gt}"]
        for c, (im, note) in enumerate(zip(imgs, notes)):
            ax = axes[r, c]
            ax.imshow(im); ax.axis("off")
            if note:
                ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=9, color="#111111",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#888888", linewidth=0.5, alpha=0.9))
        axes[r, 0].text(-0.02, 0.5, label, transform=axes[r, 0].transAxes, fontsize=12, fontweight="bold",
                        color="#111111", rotation=90, ha="right", va="center")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"\nsaved {args.out}")
    print(f"\n{'plant':<24}{'GT':>5}{'asm':>5}{'VAE':>5}{'MAE nadir':>11}{'MAE persp':>11}{'IoU asm':>9}{'IoU VAE':>9}")
    for label, n_gt, n_asm, n_vae, m1, m2, i1, i2 in rows:
        print(f"{label:<24}{n_gt:>5}{n_asm:>5}{n_vae:>5}{m1:>11.4f}{m2:>11.4f}{i1*100:>8.1f}%{i2*100:>8.1f}%")


if __name__ == "__main__":
    main()
