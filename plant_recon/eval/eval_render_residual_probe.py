"""Is the render residual informative about the correction the model needs?

Test-time refinement beats the trained model by ~33 points even though training ALREADY carries a
render loss (Stage 4, `--render_fraction`). The difference is not that refinement has a better
objective -- it is that refinement can look at `render(current hypothesis)` next to the input and
act on the difference, many times, while a forward pass never sees its own render at all.

The proposal is to close that loop: feed the residual back as CONDITIONING rather than only as a
loss. Before building it (residual encoder + second conditioning path + second forward), this probe
asks the cheap question the project asked of Stage 3's conditioning in 2026-09-18: **is the signal
even present?**

Construction, which deliberately avoids the matcher:

  1. take a GT plant and render it -- this stands in for the input observation
  2. displace every organ's base position by a KNOWN delta
  3. render the displaced plant -- this stands in for the model's hypothesis
  4. residual = normalise(hypothesis depth) - normalise(input depth), per plant
  5. for each organ, cut a local patch of that residual at the organ's projected position
  6. ridge-regress the organ's known delta from its patch, scored held-out against the training mean

If a linear probe cannot recover a delta it was TOLD about, a learned closed loop will not recover
one it has to discover. A positive R^2 is an estimate of the headroom the idea has.

The control is the same probe run on the input render alone. The delta is drawn independently of the
plant, so input-only features carry no information about it by construction and must score ~0; that
control exists to confirm the pipeline is not leaking the answer through some other route.

Depth is normalised per plant before differencing, never per node -- per-node standardisation
destroyed the signal in the 2026-09-18 depth probe, and the cart's depth is inaccurate in absolute
terms anyway.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plant_recon.dataset.part_array_dataset import PartArrayDataset, FM_OT_END, FM_BASE_START
from plant_recon.eval.ckpt_compat import fix_ckpt_args
from plant_recon.eval.eval_gt_substitution_ablation import decode_predictions_to_part_tensor
from plant_recon.eval.eval_latent_conditioning_ceiling import (
    CACHE_CAMERA_HEIGHT, CACHE_WINDOW_M, project, ridge_r2,
)
from plant_recon.models.helios_pytorch_renderer import HeliosPyTorchRenderer, compute_focus_plant_camera


def norm_depth(d, fg):
    """Per-plant standardisation over foreground pixels; zero elsewhere."""
    if fg.sum() < 16:
        return torch.zeros_like(d)
    m, s = d[fg].mean(), d[fg].std().clamp(min=1e-6)
    return torch.where(fg, (d - m) / s, torch.zeros_like(d))


def patches(img, uv, half):
    """(S,S) map, (N,2) centres -> (N, (2*half+1)^2), zero-padded past the border."""
    S = img.shape[-1]
    k = 2 * half + 1
    pad = torch.nn.functional.pad(img[None, None], (half,) * 4)[0, 0]
    out = torch.zeros(uv.shape[0], k * k)
    for i in range(uv.shape[0]):
        cx, cy = int(round(float(uv[i, 0]))), int(round(float(uv[i, 1])))
        if cx < -half or cy < -half or cx > S + half or cy > S + half:
            continue
        x0, y0 = min(max(cx, 0), S - 1), min(max(cy, 0), S - 1)
        out[i] = pad[y0:y0 + k, x0:x0 + k].flatten().cpu()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="supplies data_dir / cache_dir only; no model is run")
    ap.add_argument("--n_plants", type=int, default=120)
    ap.add_argument("--test_frac", type=float, default=0.25)
    ap.add_argument("--render_px", type=int, default=512)
    ap.add_argument("--sigma_cm", type=float, default=2.0,
                    help="std of the known per-organ displacement; 2 cm is the scaffold's node RMSE")
    ap.add_argument("--patch_half", type=int, default=5, help="residual patch is (2*half+1)^2")
    ap.add_argument("--max_nodes_per_plant", type=int, default=150)
    ap.add_argument("--perturb_frac", type=float, default=1.0,
                    help="fraction of organs displaced. At 1.0 every organ moves, so each local patch is a "
                         "superposition of many displacements and the probe must separate them; a low value "
                         "isolates the signal. Real refinement moves coherent phytomer groups, which sits "
                         "between the two.")
    ap.add_argument("--visible_only", action="store_true",
                    help="keep only organs that actually paint pixels. 53.4%% of organs contribute none "
                         "(2026-09-18 visibility probe), and an invisible organ's displacement CANNOT be in "
                         "the residual -- leaving them in guarantees a null no matter what the signal is.")
    ap.add_argument("--min_visible_px", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/logs/20260919/render_residual_probe.json")
    a = ap.parse_args()

    dev = torch.device("cuda:0")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    args = fix_ckpt_args(ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"]))
    renderer = HeliosPyTorchRenderer(image_size=a.render_px).to(dev)
    renderer.collect_part_visibility = bool(a.visible_only)
    ds = PartArrayDataset(data_root=args["data_dir"], max_nodes=args["max_phytomers"] * args["slots_per_phytomer"],
                          cache_dir=args["cache_dir"], species="cowpea", image_size=128)
    g = torch.Generator().manual_seed(a.seed)
    picks = torch.randperm(len(ds), generator=g)[: a.n_plants].tolist()
    torch.manual_seed(a.seed)

    RES, INP, DLT, PID, DAP = [], [], [], [], []
    for pi, idx in enumerate(picks):
        it = ds[idx]
        nodes, ex = it["nodes"].to(dev), it["existence_mask"].to(dev)
        types = nodes[:, :FM_OT_END].argmax(dim=-1)
        parts = decode_predictions_to_part_tensor(nodes[:, FM_BASE_START:], types, ex, device=dev)
        if parts.shape[0] < 4:
            continue
        parts = parts[: a.max_nodes_per_plant]

        def shoot(p, centre=None):
            mesh = renderer.geo_builder.build_mesh_from_part_tensor(p, device=dev)
            v = mesh["vertices"]
            c = centre if centre is not None else 0.5 * (v.min(0).values + v.max(0).values)
            with torch.no_grad():
                rgbd = renderer.forward(mesh, azimuth_deg=0.0, elevation_deg=90.0,
                                        camera_height=CACHE_CAMERA_HEIGHT, background="ground",
                                        focus_plant=False, include_depth=True, image_size=a.render_px,
                                        zoom_factor=1.0, reference_window_size=CACHE_WINDOW_M,
                                        center_override=c)
            view, proj, _ = compute_focus_plant_camera(v, None, azimuth_deg=0.0, elevation_deg=90.0,
                                                       camera_height=CACHE_CAMERA_HEIGHT, focus_plant=False,
                                                       reference_window_size=CACHE_WINDOW_M, zoom_factor=1.0,
                                                       center_override=c)
            return rgbd, view, proj, c

        rgbd_gt, view, proj, centre = shoot(parts)
        if a.visible_only:
            vis = renderer.last_part_visibility
            n = parts.shape[0]
            if vis.numel() < n:
                vis = torch.cat([vis, vis.new_zeros(n - vis.numel())])
            vis = vis[:n].cpu()
        else:
            vis = torch.full((parts.shape[0],), 1e9)

        # Known displacement. Both renders keep the GT camera so the residual is caused by the plant
        # moving, not by the frame moving with it.
        delta = torch.randn(parts.shape[0], 3, device=dev) * (a.sigma_cm / 100.0)
        if a.perturb_frac < 1.0:
            keep = torch.rand(parts.shape[0], device=dev) < a.perturb_frac
            delta = delta * keep.unsqueeze(-1).float()
        p2 = parts.clone()
        p2[:, 1:4] = p2[:, 1:4] + delta
        rgbd_hy, _, _, _ = shoot(p2, centre=centre)

        d_gt, d_hy = rgbd_gt[3], rgbd_hy[3]
        fg_gt, fg_hy = d_gt > 1e-6, d_hy > 1e-6
        if fg_gt.sum() < 64:
            continue
        n_gt, n_hy = norm_depth(d_gt, fg_gt), norm_depth(d_hy, fg_hy)
        resid = n_hy - n_gt

        uv = project(p2[:, 1:4], view, proj, a.render_px)     # the hypothesis' own positions
        keepm = vis >= a.min_visible_px if a.visible_only else torch.ones(parts.shape[0], dtype=torch.bool)
        if a.perturb_frac < 1.0:
            keepm = keepm & (delta.abs().sum(-1) > 0).cpu()
        if int(keepm.sum()) < 2:
            continue
        RES.append(patches(resid, uv, a.patch_half)[keepm])
        INP.append(patches(n_gt, uv, a.patch_half)[keepm])
        DLT.append(delta.cpu()[keepm])
        PID.append(torch.full((int(keepm.sum()),), pi))
        DAP.append(torch.full((int(keepm.sum()),), int(it["dap"].item())))
        if (pi + 1) % 20 == 0:
            print(f"  {pi+1}/{len(picks)} plants, {sum(x.shape[0] for x in DLT)} organs", flush=True)

    RES, INP, DLT = torch.cat(RES), torch.cat(INP), torch.cat(DLT)
    PID, DAP = torch.cat(PID), torch.cat(DAP)
    plants = PID.unique()
    n_te = max(1, int(round(len(plants) * a.test_frac)))
    te_p = set(plants[torch.randperm(len(plants), generator=torch.Generator().manual_seed(1))[:n_te]].tolist())
    ts = torch.tensor([int(p) in te_p for p in PID])
    tr = ~ts
    print(f"\n{len(plants)} plants, {len(DLT)} organs; train {int(tr.sum())} / test {int(ts.sum())}"
          f"  (sigma {a.sigma_cm:.1f} cm, patch {2*a.patch_half+1}px)")

    rows = {}
    print(f"\n{'features':<22}{'target':<14}{'held-out R^2':>14}{'alpha':>9}")
    for fname, F in (("render residual", RES), ("input render only", INP), ("residual + input", torch.cat([RES, INP], 1))):
        for tname, Y in (("delta x,z", DLT[:, [0, 2]]), ("delta y (height)", DLT[:, [1]]), ("delta xyz", DLT)):
            r2, al = ridge_r2(F[tr], Y[tr], F[ts], Y[ts])
            rows[f"{fname}|{tname}"] = r2
            print(f"{fname:<22}{tname:<14}{r2:>14.4f}{al if al else 0:>9}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"sigma_cm": a.sigma_cm, "patch_half": a.patch_half, "render_px": a.render_px,
               "perturb_frac": a.perturb_frac, "visible_only": bool(a.visible_only),
               "n_plants": int(len(plants)), "n_organs": int(len(DLT)), "r2": rows}, open(a.out, "w"), indent=1)
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
