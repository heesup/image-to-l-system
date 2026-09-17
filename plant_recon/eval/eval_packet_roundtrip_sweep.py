"""
Does the phytomer packet representation survive a round trip, across the whole
growth cycle and all the way out to the physical renderer?

Two arms per plant, both starting from the ground-truth 14D part tensor:

  packets      XML -> 14D -> packets -> chain -> assemble -> 14D
  packets+VAE  the same, with PhytomerVAE encode/decode in the middle

The first says whether the representation itself is lossless — whether grouping
organs into 10-slot phytomers, recovering the stem chain, and reapplying the
assembly rules gets the same plant back. The second adds the 64D bottleneck.

Numeric comparison against the original 14D array is cheap and runs over the
whole DAP range. `--helios` additionally raytraces both and reports mask IoU,
which is slow (a C++ render per plant per arm) and is meant for a small sample.

Usage:
    python plant_recon/eval/eval_packet_roundtrip_sweep.py
    python plant_recon/eval/eval_packet_roundtrip_sweep.py --helios --per-dap 1 \
        --daps 5,20,40,60,80,100
"""

import os
import re
import sys
import glob
import math
import random
import argparse
from collections import Counter

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

from plant_recon.models.plant_organ_array import (
    PlantOrganArray, ORGAN_NONE, P_COL_ORGAN_TYPE, T_COL_SHOOT_ID,
    ORGAN_ROOT_META, ORGAN_SHOOT_META)
from plant_recon.models.part_tensor_to_40d import assemble_part_tensor_to_xml
from plant_recon.models.phytomer_vae import PhytomerVAE
from plant_recon.dataset.part_array_dataset import encode_fm
from plant_recon.dataset.phytomer_packets import (
    build_phytomer_packets, assemble_packets, decode_packets,
    phytomer_scale, denormalize_packet_scales, rot6d_to_matrix,
    FM_ROT_START, FM_ROT_END, FM_BASE_START, FM_BASE_END,
    emit_part_tensor_with_shoot_meta)
from plant_recon.dataset.phytomer_topology import chain_phytomers
from plant_recon.dataset.generate_cache import extract_phytomer_ids

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEFAULT_CKPT = os.path.join(
    REPO_ROOT, "outputs/checkpoints/phytomer_vae_v8/phytomer_vae_128d_best.pt")
XML_DIR = os.path.join(REPO_ROOT, "dataset/helios_data/cowpea")
_DAP_RE = re.compile(r"dap(\d+)")


def slot_errors(gt_abs: torch.Tensor, rec_abs: torch.Tensor, presence: torch.Tensor):
    """Position / rotation error compared slot by slot.

    Matching organs by nearest position cannot work here: LEAFLET_ATTACH_FRAC
    puts slots 2 and 3 at the SAME arc fraction along the petiole, so the two
    lateral leaflets are reconstructed at one point and a position match pairs
    them arbitrarily — which showed up as 27% of leaves apparently rotated
    150-175 deg. Both tensors come from the same packet decomposition, so slot
    identity is exact and unambiguous.
    """
    m = presence.reshape(-1)
    g = gt_abs.reshape(-1, gt_abs.shape[-1])[m]
    r = rec_abs.reshape(-1, rec_abs.shape[-1])[m]
    pos = (r[:, FM_BASE_START:FM_BASE_END] - g[:, FM_BASE_START:FM_BASE_END]
           ).norm(dim=-1) / 20.0 * 100.0                                    # cm
    Rr = rot6d_to_matrix(r[:, FM_ROT_START:FM_ROT_END])
    Rg = rot6d_to_matrix(g[:, FM_ROT_START:FM_ROT_END])
    rel = Rr.transpose(-1, -2) @ Rg
    rot = torch.rad2deg(torch.acos(
        (((rel.diagonal(dim1=-2, dim2=-1).sum(-1)) - 1) / 2).clamp(-1, 1)))
    return pos, rot


def rebuild(arr: PlantOrganArray, vae=None):
    """GT -> packets -> (optional VAE) -> assembled 14D, plus chain diagnostics."""
    gt = arr.to_part_tensor()
    nodes = encode_fm(gt)
    ex = (gt[:, P_COL_ORGAN_TYPE] > ORGAN_NONE).float()
    ids = extract_phytomer_ids(arr, gt.shape[0])
    pk, pres, ctr, refs, keys = build_phytomer_packets(
        nodes, existence_mask=ex, phytomer_ids=ids, return_keys=True)
    if pk.shape[0] == 0:
        return None

    parent, shoot, ordi = chain_phytomers(
        ctr, refs, ordinal=keys[:, 1].float(), is_base=(keys[:, 1] == 0).float())
    ppos = torch.where((parent >= 0).unsqueeze(-1),
                       ctr[parent.clamp(min=0)], torch.full_like(ctr, float("nan")))

    packets = pk
    presence = pres
    if vae is not None:
        s_a = phytomer_scale(pk.to(DEVICE))
        with torch.no_grad():
            mu, _ = vae.encode(vae.pack_input(pk.to(DEVICE), pres.to(DEVICE)))
            out = vae.decode(mu)
        packets = denormalize_packet_scales(out["recon_packets"], s_a).cpu()
        presence = (out["cls_logits"].argmax(-1) > 0).cpu()

    absn = decode_packets(
        assemble_packets(packets, refs, parent_pos=ppos, centers=ctr),
        ctr, presence, refs)
    recon = emit_part_tensor_with_shoot_meta(absn, presence, ctr, refs, shoot, ordi)
    gt_abs = decode_packets(pk, ctr, pres, refs)      # GT packets in absolute frame
    return gt, recon, shoot, keys, ctr, gt_abs, absn, (pres & presence)


def shoot_accuracy(arr, ctr, shoot, gt):
    from scipy.optimize import linear_sum_assignment
    sid = arr.tensor[:, T_COL_SHOOT_ID].long()
    gts = torch.tensor([int(sid[int((gt[:, 1:4] - ctr[c]).norm(dim=-1).argmin())])
                        for c in range(ctr.shape[0])])
    A, Bv = sorted(set(shoot.tolist())), sorted(set(gts.tolist()))
    M = np.zeros((len(A), len(Bv)))
    for i, a in enumerate(A):
        for j, b in enumerate(Bv):
            M[i, j] = -((shoot == a) & (gts == b)).sum()
    r, c = linear_sum_assignment(M)
    return int(-M[r, c].sum()) / max(len(gts), 1) * 100.0, len(A), len(Bv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae_checkpoint", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--daps", type=str,
                    default="1,5,10,15,20,30,40,50,60,70,80,90,100")
    ap.add_argument("--per-dap", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--helios", action="store_true",
                    help="also raytrace both arms and report mask IoU (slow)")
    args = ap.parse_args()

    vae = PhytomerVAE(latent_dim=128, hidden_dim=256).to(DEVICE)
    vae.load_state_dict(torch.load(args.vae_checkpoint, map_location=DEVICE, weights_only=True))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    rng = random.Random(args.seed)
    daps = [int(d) for d in args.daps.split(",")]
    all_xml = glob.glob(os.path.join(XML_DIR, "*_plant_*.xml"))
    by_dap = {}
    for f in all_xml:
        m = _DAP_RE.search(os.path.basename(f))
        if m:
            by_dap.setdefault(int(m.group(1)), []).append(f)

    print("=" * 104)
    print("PACKET ROUND TRIP ACROSS THE GROWTH CYCLE")
    print(f"checkpoint: {args.vae_checkpoint}")
    print("=" * 104)
    print(f"{'DAP':>4} {'plants':>6} {'organs':>13} | {'packets only':^25} | {'packets + VAE':^25} | {'shoots':>12}")
    print(f"{'':>4} {'':>6} {'GT -> rebuilt':>13} | {'pos cm':>8} {'rot deg':>8} {'p99 cm':>7} | "
          f"{'pos cm':>8} {'rot deg':>8} {'p99 cm':>7} | {'acc / n':>12}")
    print("-" * 104)

    helios_rows = []
    for dap in daps:
        files = by_dap.get(dap, [])
        if not files:
            continue
        picks = rng.sample(files, min(args.per_dap, len(files)))
        agg = {"n": 0, "gt": 0, "rb": 0,
               "p0": [], "r0": [], "p1": [], "r1": [], "sa": [], "sn": 0, "sg": 0}
        for f in picks:
            arr = PlantOrganArray.from_xml_file(f)
            a = rebuild(arr, vae=None)
            b = rebuild(arr, vae=vae)
            if a is None or b is None:
                continue
            gt, rec0, shoot, keys, ctr, gt_abs0, abs0, m0 = a
            _, rec1, _, _, _, _, abs1, m1 = b
            p0, r0 = slot_errors(gt_abs0, abs0, m0)
            p1, r1 = slot_errors(gt_abs0, abs1, m1)
            ng = int(((gt[:, 0] != ORGAN_ROOT_META) & (gt[:, 0] != ORGAN_SHOOT_META)
                      & (gt[:, 0] > 0)).sum())
            nr0 = int(m0.sum())
            sacc, sn, sg = shoot_accuracy(arr, ctr, shoot, gt)
            agg["n"] += 1
            agg["gt"] += ng
            agg["rb"] += nr0
            agg["p0"].append(p0); agg["r0"].append(r0)
            agg["p1"].append(p1); agg["r1"].append(r1)
            agg["sa"].append(sacc); agg["sn"] += sn; agg["sg"] += sg
            if args.helios:
                helios_rows.append((dap, f, gt, rec0, rec1))
        if agg["n"] == 0:
            continue
        p0 = torch.cat(agg["p0"]); r0 = torch.cat(agg["r0"])
        p1 = torch.cat(agg["p1"]); r1 = torch.cat(agg["r1"])
        print(f"{dap:>4} {agg['n']:>6} {agg['gt']:>6}->{agg['rb']:<6} | "
              f"{p0.mean():8.3f} {r0.mean():8.2f} {p0.quantile(0.99):7.2f} | "
              f"{p1.mean():8.3f} {r1.mean():8.2f} {p1.quantile(0.99):7.2f} | "
              f"{np.mean(agg['sa']):6.1f}% {agg['sn']:>2}/{agg['sg']:<2}")

    if args.helios:
        from plant_recon.eval.eval_13d_xml_organ_masks import (
            render_helios_full, SCRATCH_DIR)
        print()
        print("=" * 104)
        print("HELIOS FULL-CYCLE ROUND TRIP (raytraced foreground IoU vs the ground-truth render)")
        print("=" * 104)
        print(f"{'DAP':>4} | {'packets only':>13} | {'packets + VAE':>14}")
        print("-" * 40)
        for dap, f, gt, rec0, rec1 in helios_rows:
            tag = f"sweep{dap:03d}"
            g = render_helios_full(f, f"gt_{tag}")
            ious = []
            for name, rec in (("pkt", rec0), ("vae", rec1)):
                x = os.path.join(SCRATCH_DIR, f"{name}_{tag}.xml")
                with open(x, "w") as fh:
                    fh.write(assemble_part_tensor_to_xml(rec))
                r = render_helios_full(x, f"{name}_{tag}")
                a, b = g["mask_map"] >= 0, r["mask_map"] >= 0
                ious.append(np.logical_and(a, b).sum() / max(1, np.logical_or(a, b).sum()) * 100)
            print(f"{dap:>4} | {ious[0]:12.1f}% | {ious[1]:13.1f}%")


if __name__ == "__main__":
    main()
