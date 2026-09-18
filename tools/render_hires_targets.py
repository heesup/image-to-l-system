"""Re-render the evaluation plants' CHM pyramid at high resolution, to test whether the refinement
target's resolution is worth raising before any cache is regenerated.

Why this exists. The refinement loss compares a rendered depth against the cached input CHM, and the
cache holds 256 px, so raising only the render (--refine_px) sharpens the prediction against a blurry
target and cannot show what a genuinely finer target buys. Re-rendering the GT depth for a handful of
plants is minutes of work and answers that, where regenerating the 100k-sample cache is hours and
~250 GB. Note the network itself gains nothing from either: its backbone resizes every level to 224 px,
so 256 and 512 reach it identically (assessment §7 follow-up).

Only the CHM channel is written: the RGB planes exist for the backbone, which caps at 224, so the depth
is the only thing a higher resolution actually reaches.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from plant_recon.dataset.part_array_dataset import PartArrayDataset, FM_BASE_START, FM_OT_END  # noqa: E402
from plant_recon.eval.eval_hierarchical_self_consistency import decode_predictions_to_part_tensor  # noqa: E402
from plant_recon.models.helios_pytorch_renderer import HeliosPyTorchRenderer  # noqa: E402
from plant_recon.eval.ckpt_compat import fix_ckpt_args

ZOOMS = (1.0, 2.0, 4.0, 8.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval_set", default="outputs/checkpoints/hierarchical_fm_v9/eval_set.json")
    ap.add_argument("--px", type=int, default=512)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    dev = torch.device("cuda:0")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    args = fix_ckpt_args(ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"]))
    M = args["slots_per_phytomer"]
    ds = PartArrayDataset(data_root=args["data_dir"], max_nodes=args["max_phytomers"] * M, cache_dir=args["cache_dir"],
                          pkt_cache_dir=args.get("pkt_cache_dir") or None, species="cowpea", image_size=256)
    renderer = HeliosPyTorchRenderer(image_size=a.px).to(dev)
    os.makedirs(a.out_dir, exist_ok=True)
    idxs = json.load(open(a.eval_set))["indices"]
    for i in idxs:
        it = ds[i]
        nodes = it["nodes"].to(dev); ex = it["existence_mask"].to(dev)
        parts = decode_predictions_to_part_tensor(nodes[:, FM_BASE_START:], nodes[:, :FM_OT_END].argmax(-1), ex, device=dev)
        if parts.shape[0] == 0:
            continue
        mesh = renderer.geo_builder.build_mesh_from_part_tensor(parts, device=dev)
        v = mesh["vertices"]
        centre = 0.5 * (v.min(0).values + v.max(0).values)
        chm = []
        with torch.no_grad():
            for z in ZOOMS:
                # the cache's own camera: plant-bbox centred, fixed 1.2 m window at zoom 1x, 5 m above
                d = renderer.forward(mesh, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
                                     background="ground", focus_plant=False, include_depth=True,
                                     image_size=a.px, zoom_factor=float(z), reference_window_size=1.2,
                                     center_override=centre)[3]
                chm.append(d.clamp(min=0).cpu())
        torch.save(torch.stack(chm).half(), os.path.join(a.out_dir, f"{it['prefix']}_chm{a.px}.pt"))
        print(f"idx {i:6d} {it['prefix']}: CHM pyramid at {a.px} px, max {float(chm[0].max()):.3f} m", flush=True)
    print("done ->", a.out_dir)


if __name__ == "__main__":
    main()
