"""
Full-chain roundtrip check: Helios C++ GT -> XML -> 14D Part Tensor -> 10-slot
Phytomer Packets -> PhytomerVAE encode/decode -> 14D -> Helios XML -> Helios
C++ raytrace, compared against both the Helios GT and the VAE-free "IK-only"
roundtrip (diffusion_based/eval/eval_13d_xml_organ_masks.py) so the VAE's own
marginal contribution to the roundtrip error can be isolated from the known
IK/FK assembly bug (see docs/ongoing/AGENT_TAKEOVER_GUIDE.md section 8.11 /
section 4 of the 2026-09-11 takeover doc).

This complements the existing checks, none of which puts the actual
PhytomerVAE in the loop against a real Helios raytrace:
  - eval_13d_xml_organ_masks.py:      XML -> 14D -> XML -> Helios   (no VAE)
  - benchmark_organ_vae_roundtrip.py: VAE roundtrip -> PyTorch differentiable
                                       render only (old OrganLatentVAE, no XML)
  - tools/phytomer_vae_visualizer.py: PhytomerVAE roundtrip -> PyTorch
                                       differentiable render only (no XML)

Produces:
1. Per-class IoU (Internode, Petiole, Leaf, Peduncle, Flower, Fruit) for both
   IK-only and IK+VAE roundtrips vs Helios GT.
2. Comparison figure: docs/results/assets/fig14_phytomer_vae_helios_roundtrip.png
"""

import os
import sys
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

from diffusion_based.models.plant_organ_array import PlantOrganArray, ORGAN_NONE, P_COL_ORGAN_TYPE
from diffusion_based.models.part_tensor_to_40d import assemble_part_tensor_to_xml
from diffusion_based.models.phytomer_vae import PhytomerVAE
from diffusion_based.dataset.part_array_dataset import encode_fm, decode_fm, FM_NODE_DIM
from diffusion_based.dataset.phytomer_packets import (
    build_phytomer_packets, decode_packets, assemble_packets,
    phytomer_scale, denormalize_packet_scales,
)
from diffusion_based.dataset.generate_cache import extract_phytomer_ids

from diffusion_based.eval.eval_13d_xml_organ_masks import (
    render_helios_full, compute_iou_per_class, rasterize_semantic_color,
    ORGAN_CLASSES, ORGAN_COLORS, TEST_PLANTS, IMG_SIZE, SCRATCH_DIR, OUTPUT_DIR,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEFAULT_VAE_CHECKPOINT = os.path.join(
    REPO_ROOT, "diffusion_based/checkpoints/phytomer_vae_v7/phytomer_vae_64d_best.pt")


def build_ik_only_xml(part_13d: torch.Tensor) -> str:
    """Direct 14D -> XML roundtrip, no VAE (isolates the known IK/FK bug)."""
    return assemble_part_tensor_to_xml(part_13d.cpu())


def build_vae_roundtrip_xml(arr: PlantOrganArray, vae: PhytomerVAE) -> Tuple[str, int, int]:
    """Helios XML -> 14D -> 10-slot packets -> PhytomerVAE encode/decode -> 14D -> XML.

    Returns (xml_str, n_gt_organs, n_recon_organs). Uses the GROUND-TRUTH
    phytomer scale s_a and packet reference frames (centers/refs) — this isolates
    the VAE's own 64D-latent reconstruction fidelity, matching how VAE training
    loss and tools/phytomer_vae_visualizer.py evaluate reconstruction (the
    phytomer pose/scale are exogenous conditioning in the real 3-stage model,
    not something the VAE itself predicts).
    """
    part_13d = arr.to_part_tensor()  # CPU (N, 14)
    num_organs = part_13d.shape[0]
    nodes_fm = encode_fm(part_13d)  # (N, 26)
    existence = (part_13d[:, P_COL_ORGAN_TYPE] > ORGAN_NONE).float()
    phytomer_ids = extract_phytomer_ids(arr, num_organs)

    packets, presence, centers, refs = build_phytomer_packets(
        nodes_fm, existence_mask=existence, phytomer_ids=phytomer_ids)
    if packets.shape[0] == 0:
        raise RuntimeError("No phytomer packets could be built (empty plant?)")

    packets = packets.to(DEVICE).float()
    presence = presence.to(DEVICE)
    centers = centers.to(DEVICE)
    refs = refs.to(DEVICE)

    s_a = phytomer_scale(packets)
    x = vae.pack_input(packets, presence)
    with torch.no_grad():
        mu, _ = vae.encode(x)
        out = vae.decode(mu)

    recon_norm = out["recon_packets"]                   # (P, 10, 26), normalized scale, zero base
    recon_presence = out["cls_logits"].argmax(dim=-1) > 0  # (P, 10) predicted existence

    recon_denorm = denormalize_packet_scales(recon_norm, s_a)
    recon_assembled = assemble_packets(recon_denorm, refs)
    recon_abs = decode_packets(recon_assembled, centers, recon_presence, refs)

    flat_fm = recon_abs.reshape(-1, FM_NODE_DIM)
    flat_presence = recon_presence.reshape(-1)
    active_fm = flat_fm[flat_presence]
    if active_fm.shape[0] == 0:
        raise RuntimeError("VAE roundtrip predicted zero organs")

    part14_recon = decode_fm(active_fm).cpu()
    xml_str = assemble_part_tensor_to_xml(part14_recon)
    return xml_str, int(existence.sum().item()), int(active_fm.shape[0])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_checkpoint", type=str, default=DEFAULT_VAE_CHECKPOINT)
    args = parser.parse_args()

    print("=" * 80)
    print("PHYTOMER VAE + HELIOS ROUNDTRIP CHECK")
    print(f"Helios GT -> XML -> 14D -> Phytomer Packets -> PhytomerVAE -> 14D -> XML -> Helios")
    print(f"VAE checkpoint: {args.vae_checkpoint}")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SCRATCH_DIR, exist_ok=True)

    vae = PhytomerVAE(latent_dim=64, hidden_dim=256).to(DEVICE)
    vae.load_state_dict(torch.load(args.vae_checkpoint, map_location=DEVICE, weights_only=True))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    num_rows = len(TEST_PLANTS)
    fig, axes = plt.subplots(num_rows, 6, figsize=(28, 4.8 * num_rows), facecolor="#0a0a14")
    plt.subplots_adjust(wspace=0.03, hspace=0.08, left=0.06, right=0.98, top=0.93, bottom=0.06)
    col_titles = [
        "Helios GT\nRaytrace RGB",
        "Helios GT\nOrgan Mask (COCO)",
        "IK-only Recon\nRaytrace RGB (14D XML, no VAE)",
        "IK-only Recon\nOrgan Mask",
        "VAE Roundtrip Recon\nRaytrace RGB (14D->Pkt->VAE->14D->XML)",
        "VAE Roundtrip Recon\nOrgan Mask",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold", color="#7ee8fa", pad=12)

    summary_rows = []

    for row_idx, (label, orig_xml_rel, tag) in enumerate(TEST_PLANTS):
        orig_xml_path = os.path.join(REPO_ROOT, orig_xml_rel)
        if not os.path.exists(orig_xml_path):
            print(f"Skipping {label} (missing: {orig_xml_path})")
            continue

        print(f"\n--- Processing [{label}] ---")
        arr = PlantOrganArray.from_xml_file(orig_xml_path)
        part_13d = arr.to_part_tensor()

        # IK-only roundtrip (no VAE)
        ik_xml_str = build_ik_only_xml(part_13d)
        ik_xml_path = os.path.join(SCRATCH_DIR, f"ikonly_{tag}.xml")
        with open(ik_xml_path, "w", encoding="utf-8") as f:
            f.write(ik_xml_str)

        # Full VAE roundtrip
        vae_xml_str, n_gt, n_recon = build_vae_roundtrip_xml(arr, vae)
        vae_xml_path = os.path.join(SCRATCH_DIR, f"vaeroundtrip_{tag}.xml")
        with open(vae_xml_path, "w", encoding="utf-8") as f:
            f.write(vae_xml_str)
        print(f"  VAE roundtrip organ count: GT {n_gt} -> Recon {n_recon} (delta {n_recon - n_gt:+d})")

        print("  Rendering Helios GT...")
        helios_gt = render_helios_full(orig_xml_path, f"gt_{tag}")
        print("  Rendering Helios IK-only recon...")
        helios_ik = render_helios_full(ik_xml_path, f"ikonly_{tag}")
        print("  Rendering Helios VAE roundtrip recon...")
        helios_vae = render_helios_full(vae_xml_path, f"vaeroundtrip_{tag}")

        def _metrics(helios_recon):
            ious = compute_iou_per_class(helios_gt["mask_map"], helios_recon["mask_map"])
            valid = [v for v in ious.values() if not math.isnan(v)]
            m_iou = float(np.mean(valid)) if valid else 0.0
            fg_gt = helios_gt["mask_map"] >= 0
            fg_r = helios_recon["mask_map"] >= 0
            fg_iou = np.logical_and(fg_gt, fg_r).sum() / max(1, np.logical_or(fg_gt, fg_r).sum())
            d_mask = np.logical_and(fg_gt, fg_r)
            depth_mse = float(np.mean((helios_gt["depth"][d_mask] - helios_recon["depth"][d_mask]) ** 2)) if d_mask.any() else 0.0
            depth_psnr = -10.0 * math.log10(max(depth_mse, 1e-8))
            return ious, m_iou, float(fg_iou), depth_psnr

        ik_ious, ik_miou, ik_fg, ik_psnr = _metrics(helios_ik)
        vae_ious, vae_miou, vae_fg, vae_psnr = _metrics(helios_vae)

        print(f"  {'Class':<10s} {'IK-only IoU':>12s} {'VAE-RT IoU':>12s}")
        for cname in ORGAN_CLASSES:
            iv, vv = ik_ious[cname], vae_ious[cname]
            iv_s = f"{iv*100:.1f}%" if not math.isnan(iv) else "n/a"
            vv_s = f"{vv*100:.1f}%" if not math.isnan(vv) else "n/a"
            print(f"  {cname:<10s} {iv_s:>12s} {vv_s:>12s}")
        print(f"  {'Foreground':<10s} {ik_fg*100:>11.1f}% {vae_fg*100:>11.1f}%")
        print(f"  {'Mean organ':<10s} {ik_miou*100:>11.1f}% {vae_miou*100:>11.1f}%")
        print(f"  {'Depth PSNR':<10s} {ik_psnr:>10.2f}dB {vae_psnr:>10.2f}dB")

        summary_rows.append((label, ik_fg, vae_fg, ik_miou, vae_miou, ik_psnr, vae_psnr, n_gt, n_recon))

        ax_row = axes[row_idx]
        ax_row[0].imshow(helios_gt["rgb"]); ax_row[0].axis("off"); ax_row[0].set_facecolor("#0a0a14")
        ax_row[0].set_ylabel(label, fontsize=12, fontweight="bold", color="#f0f0f0", rotation=0, labelpad=70, va="center")

        ax_row[1].imshow(rasterize_semantic_color(helios_gt["mask_map"])); ax_row[1].axis("off"); ax_row[1].set_facecolor("#0a0a14")

        ax_row[2].imshow(helios_ik["rgb"]); ax_row[2].axis("off"); ax_row[2].set_facecolor("#0a0a14")
        ax_row[2].text(0.03, 0.03, f"FG IoU: {ik_fg*100:.1f}%", transform=ax_row[2].transAxes, fontsize=9, color="#7ee8fa",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

        ax_row[3].imshow(rasterize_semantic_color(helios_ik["mask_map"])); ax_row[3].axis("off"); ax_row[3].set_facecolor("#0a0a14")
        ax_row[3].text(0.03, 0.03, f"mIoU: {ik_miou*100:.1f}%", transform=ax_row[3].transAxes, fontsize=9, color="#7ee8fa",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

        ax_row[4].imshow(helios_vae["rgb"]); ax_row[4].axis("off"); ax_row[4].set_facecolor("#0a0a14")
        ax_row[4].text(0.03, 0.03, f"FG IoU: {vae_fg*100:.1f}%\nOrgans: {n_recon}/{n_gt}", transform=ax_row[4].transAxes, fontsize=9, color="#ffd166",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

        ax_row[5].imshow(rasterize_semantic_color(helios_vae["mask_map"])); ax_row[5].axis("off"); ax_row[5].set_facecolor("#0a0a14")
        ax_row[5].text(0.03, 0.03, f"mIoU: {vae_miou*100:.1f}%", transform=ax_row[5].transAxes, fontsize=9, color="#ffd166",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

    patches = [mpatches.Patch(color=ORGAN_COLORS[i] / 255.0, label=ORGAN_CLASSES[i]) for i in range(len(ORGAN_CLASSES))]
    fig.legend(handles=patches, loc="lower center", ncol=len(ORGAN_CLASSES), fontsize=12,
               facecolor="#151525", edgecolor="#444466", labelcolor="white", bbox_to_anchor=(0.52, 0.005))

    save_path = os.path.join(OUTPUT_DIR, "fig14_phytomer_vae_helios_roundtrip.png")
    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"\nSaved comparison figure -> {save_path}")

    print("\n" + "=" * 80)
    print("SUMMARY: IK-only vs IK+VAE roundtrip (Foreground IoU / Mean Organ IoU / Depth PSNR)")
    print("=" * 80)
    print(f"{'Stage':<24s} {'IK-only FG':>11s} {'VAE-RT FG':>11s} {'IK-only mIoU':>13s} {'VAE-RT mIoU':>12s} {'IK PSNR':>9s} {'VAE PSNR':>9s} {'Organs':>10s}")
    for label, ik_fg, vae_fg, ik_miou, vae_miou, ik_psnr, vae_psnr, n_gt, n_recon in summary_rows:
        print(f"{label:<24s} {ik_fg*100:>10.1f}% {vae_fg*100:>10.1f}% {ik_miou*100:>12.1f}% {vae_miou*100:>11.1f}% {ik_psnr:>8.2f} {vae_psnr:>8.2f} {n_recon:>4d}/{n_gt:<4d}")


if __name__ == "__main__":
    main()
