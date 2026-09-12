"""
Packet-level (26D) reconstruction fidelity check — NO Helios, NO XML, GPU only.

Answers: does the PhytomerVAE reconstruct each phytomer's slots (class,
rotation, scale, curvature) close to ground truth, and does the deterministic
assemble_packets() curve-integration reproduce the true leaflet/flower/fruit
BASE positions? This isolates the VAE's own regression fidelity from the
downstream IK/XML/Helios pipeline, using the SAME anchor-relative packet frame
the VAE was trained on (build_phytomer_packets output) so no extra conversion
error is introduced.

If per-field errors here are small but the full Helios roundtrip
(eval_phytomer_vae_helios_roundtrip.py) is catastrophic, the bug is in how
that reconstruction propagates to XML/Helios (or a compounding effect through
the deterministic curve integration), not in the VAE itself.
"""

import os
import sys
import math
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

from diffusion_based.models.plant_organ_array import PlantOrganArray, ORGAN_NONE, P_COL_ORGAN_TYPE
from diffusion_based.models.phytomer_vae import PhytomerVAE
from diffusion_based.dataset.part_array_dataset import encode_fm, FM_NODE_DIM
from diffusion_based.dataset.phytomer_packets import (
    build_phytomer_packets, assemble_packets, rot6d_to_matrix,
    anchor_scale, denormalize_packet_scales,
    FM_BASE_START, FM_BASE_END, FM_ROT_START, FM_ROT_END,
    FM_SCALE_START, FM_SCALE_END, FM_CURV, FM_OT_END,
    ROLE_SLOT_RANGES,
)
from diffusion_based.dataset.generate_cache import extract_phytomer_ids
from diffusion_based.eval.eval_13d_xml_organ_masks import TEST_PLANTS

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEFAULT_VAE_CHECKPOINT = os.path.join(
    REPO_ROOT, "diffusion_based/checkpoints/phytomer_vae_v6/phytomer_vae_64d_best.pt")
SLOT_NAMES = ["Internode", "Petiole", "Leaflet2", "Leaflet3", "Leaflet4",
              "Peduncle", "Repro6", "Repro7", "Repro8", "Repro9"]


def rot_geodesic_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (degrees) between rotation matrices, batched over leading dims."""
    R_rel = R_pred.transpose(-1, -2) @ R_gt
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos_theta))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_checkpoint", type=str, default=DEFAULT_VAE_CHECKPOINT)
    args = parser.parse_args()

    print("=" * 90)
    print("PACKET-LEVEL (26D) VAE RECONSTRUCTION FIDELITY — no XML, no Helios")
    print(f"VAE checkpoint: {args.vae_checkpoint}")
    print("=" * 90)

    vae = PhytomerVAE(latent_dim=64, hidden_dim=256).to(DEVICE)
    vae.load_state_dict(torch.load(args.vae_checkpoint, map_location=DEVICE, weights_only=True))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    for label, orig_xml_rel, tag in TEST_PLANTS:
        orig_xml_path = os.path.join(REPO_ROOT, orig_xml_rel)
        if not os.path.exists(orig_xml_path):
            continue
        print(f"\n--- {label} ---")

        arr = PlantOrganArray.from_xml_file(orig_xml_path)
        part_13d = arr.to_part_tensor()
        num_organs = part_13d.shape[0]
        nodes_fm = encode_fm(part_13d)
        existence = (part_13d[:, P_COL_ORGAN_TYPE] > ORGAN_NONE).float()
        phytomer_ids = extract_phytomer_ids(arr, num_organs)

        packets, presence, centers, refs = build_phytomer_packets(
            nodes_fm, existence_mask=existence, phytomer_ids=phytomer_ids)
        packets = packets.to(DEVICE).float()
        presence = presence.to(DEVICE)
        centers = centers.to(DEVICE)
        refs = refs.to(DEVICE)
        P = packets.shape[0]

        s_a = anchor_scale(packets)
        x = vae.pack_input(packets, presence)
        with torch.no_grad():
            mu, _ = vae.encode(x)
            out = vae.decode(mu)

        recon_norm = out["recon_packets"]
        recon_presence = out["cls_logits"].argmax(dim=-1) > 0
        recon_denorm = denormalize_packet_scales(recon_norm, s_a)

        # ---- 1. Presence / class agreement (only where GT says a slot exists) ----
        pres_agree = (recon_presence == presence).float().mean().item()
        gt_cls = packets[:, :, :FM_OT_END].argmax(dim=-1)
        pred_cls = out["cls_logits"].argmax(dim=-1)
        cls_acc = ((pred_cls == gt_cls).float() * presence.float()).sum() / presence.float().sum().clamp(min=1)
        print(f"  Packets (phytomers): {P} | Slot presence agreement: {pres_agree*100:.1f}% | "
              f"Class acc (GT-present slots): {cls_acc.item()*100:.1f}%")

        # ---- 2. Rotation error (degrees), present GT slots only ----
        R_pred = rot6d_to_matrix(recon_denorm[:, :, FM_ROT_START:FM_ROT_END])
        R_gt = rot6d_to_matrix(packets[:, :, FM_ROT_START:FM_ROT_END])
        rot_err_deg = rot_geodesic_deg(R_pred, R_gt)  # (P, 10)

        # ---- 3. Scale relative error (%), present GT slots only ----
        pred_scale = recon_denorm[:, :, FM_SCALE_START:FM_SCALE_END]
        gt_scale = packets[:, :, FM_SCALE_START:FM_SCALE_END]
        scale_rel_err = ((pred_scale - gt_scale).abs() / gt_scale.abs().clamp(min=1e-3)).mean(dim=-1)  # (P,10)

        # ---- 4. Curvature error (deg/m), present GT slots only ----
        curv_err = (recon_denorm[:, :, FM_CURV] - packets[:, :, FM_CURV]).abs()  # (P,10)

        # ---- 5. Base position error AFTER assemble_packets (cm), the key test:
        #         does curve-integration reproduce true leaflet/repro offsets? ----
        recon_assembled = assemble_packets(recon_denorm, refs)
        base_err_cm = (recon_assembled[:, :, FM_BASE_START:FM_BASE_END]
                       - packets[:, :, FM_BASE_START:FM_BASE_END]).norm(dim=-1) / 20.0 * 100.0  # FM base is /BASE_SCALE=20 -> meters -> cm

        def _role_stat(err: torch.Tensor, unit: str):
            for role, (lo, hi) in ROLE_SLOT_RANGES.items():
                mask = presence[:, lo:hi]
                vals = err[:, lo:hi][mask]
                if vals.numel() == 0:
                    continue
                role_name = ["Stem", "Petiole", "Leaflets", "Peduncle", "Repro"][role]
                print(f"      {role_name:<10s} n={vals.numel():5d}  mean={vals.mean().item():7.3f}{unit}  "
                      f"p50={vals.median().item():7.3f}{unit}  p90={vals.quantile(0.9).item():7.3f}{unit}")

        print("  Rotation error (deg), by role:")
        _role_stat(rot_err_deg, "deg")
        print("  Scale relative error (frac), by role:")
        _role_stat(scale_rel_err, "")
        print("  Curvature error (deg/m), by role:")
        _role_stat(curv_err, "")
        print("  Base position error AFTER assemble_packets (cm), by role:")
        _role_stat(base_err_cm, "cm")


if __name__ == "__main__":
    main()
