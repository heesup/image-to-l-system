"""A/B perturbation probe: which structure changes are visible from nadir at all, and recoverable?

Heesup's design (2026-09-17). Take a ground-truth plant A, perturb ONE node by a known amount to get B,
render both, and ask two questions of the render difference:

  visibility     how much does the image change at all?  (if nothing changes, nothing can recover it)
  recoverability can a probe read the perturbation back out of the difference, on held-out plants?

This is a better instrument than the earlier residual probe (eval_render_residual_probe.py), which used
test-time refinement's own correction as its target: that target is the output of an imperfect optimiser,
and it turned out to be dominated by the vertical axis (per-axis std 3.4 / 3.4 / 9.7 cm) -- exactly the
axis a top-down camera cannot see. Here the target is exact and each axis is perturbed separately, so
observability is measured per axis instead of being confounded.

What it decides: whether a learned render-feedback corrector is worth building, and if so, which degrees
of freedom it should be allowed to correct.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from plant_recon.dataset.part_array_dataset import PartArrayDataset, FM_BASE_START, FM_OT_END
from plant_recon.eval.ckpt_compat import fix_ckpt_args
from plant_recon.eval.eval_hierarchical_self_consistency import decode_predictions_to_part_tensor
from plant_recon.eval.eval_latent_conditioning_ceiling import ridge_r2
from plant_recon.models.helios_pytorch_renderer import HeliosPyTorchRenderer

WINDOW_M = 1.2


def render(renderer, parts, center, px, zoom):
    mesh = renderer.geo_builder.build_mesh_from_part_tensor(parts, device=parts.device)
    return renderer.render_batched([mesh], azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
                                   differentiable=False, image_size=px, zoom_factor=zoom,
                                   reference_window_size=WINDOW_M,
                                   centers=([center] if center is not None else None))[0, 3]


def patch(dmap, xy, half):
    S = dmap.shape[-1]
    u = float((xy[0] / (WINDOW_M * 0.5)).clamp(-1, 1)); v = float((-xy[1] / (WINDOW_M * 0.5)).clamp(-1, 1))
    px = int(round((u * 0.5 + 0.5) * (S - 1))); py = int(round((v * 0.5 + 0.5) * (S - 1)))
    px = min(max(px, half), S - 1 - half); py = min(max(py, half), S - 1 - half)
    return dmap[py - half:py + half + 1, px - half:px + half + 1].reshape(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval_set", default="outputs/checkpoints/hierarchical_fm_v9/eval_set.json")
    ap.add_argument("--n_plants", type=int, default=40)
    ap.add_argument("--per_plant", type=int, default=12, help="perturbations sampled per plant, per kind")
    ap.add_argument("--delta_cm", type=float, default=2.0)
    ap.add_argument("--px", type=int, default=128)
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--half", type=int, default=6)
    ap.add_argument("--out", default="outputs/logs/20260917/perturbation_observability.json")
    a = ap.parse_args()

    dev = torch.device("cuda:0")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    args = fix_ckpt_args(ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"]))
    M = args["slots_per_phytomer"]
    ds = PartArrayDataset(data_root=args["data_dir"], max_nodes=args["max_phytomers"] * M,
                          cache_dir=args["cache_dir"], pkt_cache_dir=args.get("pkt_cache_dir") or None,
                          species="cowpea", image_size=128)
    renderer = HeliosPyTorchRenderer(image_size=a.px).to(dev)
    g = torch.Generator().manual_seed(0)
    idxs = torch.randperm(len(ds), generator=g)[: a.n_plants].tolist()
    d_m = a.delta_cm / 100.0
    kinds = ["dx", "dy", "dz", "scale"]
    data = {k: {"patch": [], "y": [], "plant": [], "vis": []} for k in kinds}

    for n, i in enumerate(idxs):
        it = ds[i]
        nodes = it["nodes"].to(dev); ex = it["existence_mask"].to(dev)
        parts = decode_predictions_to_part_tensor(nodes[:, FM_BASE_START:], nodes[:, :FM_OT_END].argmax(-1), ex, device=dev)
        if parts.shape[0] < 8:
            continue
        v = renderer.geo_builder.build_mesh_from_part_tensor(parts, device=dev)["vertices"]
        centre = 0.5 * (v.min(0).values + v.max(0).values)
        with torch.no_grad():
            base = render(renderer, parts, centre, a.px, a.zoom)
        live = torch.nonzero(parts[:, 0] > 2).flatten()
        if live.numel() < 4:
            continue
        for kind in kinds:
            for _ in range(a.per_plant):
                j = int(live[torch.randint(live.numel(), (1,), generator=g).item()])
                delta = float(torch.empty(1).uniform_(-d_m, d_m, generator=g).item())
                p2 = parts.clone()
                if kind == "dx":   p2[j, 1] += delta
                elif kind == "dy": p2[j, 2] += delta
                elif kind == "dz": p2[j, 3] += delta
                else:              p2[j, 10:13] *= (1.0 + delta / max(d_m, 1e-6) * 0.25)   # +-25% size
                with torch.no_grad():
                    alt = render(renderer, p2, centre, a.px, a.zoom)
                diff = (alt - base)
                data[kind]["vis"].append(float(diff.abs().mean()))
                data[kind]["patch"].append(patch(diff, parts[j, 1:3], a.half).cpu())
                data[kind]["y"].append(torch.tensor([delta]))
                data[kind]["plant"].append(n)
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(idxs)} plants", flush=True)

    print(f"\nperturbation {a.delta_cm} cm (scale: +-25%), render {a.px} px at zoom {a.zoom:g}x "
          f"({WINDOW_M / a.zoom / a.px * 1000:.1f} mm/pixel)")
    print(f"{'kind':<7}{'image change (mean |Δdepth|, mm)':>34}{'held-out R² from the patch':>30}")
    res = {}
    for kind in kinds:
        d = data[kind]
        if len(d["y"]) < 20:
            continue
        X = torch.stack(d["patch"]); y = torch.stack(d["y"]); pl = torch.tensor(d["plant"])
        plants = pl.unique(); test = set(plants[-max(1, len(plants) // 4):].tolist())
        te = torch.tensor([int(p) in test for p in pl.tolist()])
        r2, alpha = ridge_r2(X[~te], y[~te], X[te], y[te])
        vis_mm = float(np.mean(d["vis"])) * 1000
        res[kind] = {"visibility_mm": round(vis_mm, 4), "r2": round(r2, 4), "n": int(len(y))}
        print(f"{kind:<7}{vis_mm:>34.4f}{r2:>30.4f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"checkpoint": a.checkpoint, "delta_cm": a.delta_cm, "px": a.px, "zoom": a.zoom,
               "results": res}, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
