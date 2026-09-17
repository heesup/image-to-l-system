"""Helios raytraced RGB for the strict-protocol eval plants, framed exactly like the training cache.

Purpose (docs/ongoing/20260916_sim_to_real_assessment_and_plan.md §4): measure the appearance gap on
its own. The trained network reads only the RGB planes of its 16-channel input (DINORayEncoder.forward
takes channels 0:3 of every zoom level), and every training image is a flat-shaded PyTorch render on a
uniform tan ground. This script produces, for each eval plant, the same four zoom levels rendered by
Helios's radiation renderer (textured soil, shading, cast shadows), so that
eval_test_time_refinement.py --rgb_override_dir can score the same checkpoint on the same plants with
only the pixels changed.

Framing matches plant_recon/dataset/generate_cache.py exactly:
  - camera centre = the 3D bbox centre of the plant mesh built from the XML's part tensor (what
    compute_focus_plant_camera uses with focus_plant=True), camera 5 m above that centre, nadir;
  - zoom-1x window 1.2 m at the depth of that centre (hfov = 2 atan(0.6 / 5)), then 0.6 / 0.3 / 0.15 m
    for zooms 2 / 4 / 8. A narrower FOV from the same pinhole is exactly a centre crop, so one
    high-resolution render (--resolution, default 2048 px so the 8x crop is 256 px) yields all four.
Helios ignores a params entry's (x, y) for an XML-loaded plant (see use_cases/real_world/eval/run_multiplant_scene.py),
so the shift is baked into the XML's <base_position> instead: the plant is moved so its bbox centre sits
on the plot origin, where the non-focusing camera looks.

A framing check is written per plant: silhouette IoU between the Helios crop's green pixels and the
cache CHM > 0 mask at every zoom, plus the same IoU under the three flips, so a wrong orientation or
centre shows up before any P number is read.

Outputs under --out:
  rgb/<prefix>_helios_rgb.pt   (12, 256, 256) float32, level-major RGB in the cache's [-1, 1] convention
  panels/<prefix>.png          cache RGB | Helios RGB | mask overlay, one row per zoom
  helios/<prefix>/             the Helios run (render, camera.json, shifted XML)
  summary.json                 per-plant framing IoUs, render time, plant centre
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from plant_recon.dataset.part_array_dataset import PartArrayDataset  # noqa: E402
from plant_recon.models.helios_pytorch_renderer import HeliosPyTorchRenderer  # noqa: E402
from plant_recon.models.plant_organ_array import PlantOrganArray  # noqa: E402

HELIOS_ROOT = REPO / "submodules/Digital-Crops" / "projects" / "syntheticdata_generation"
BUILD_DIR = HELIOS_ROOT / "build"
MAIN_BIN = BUILD_DIR / "main"
PARAMS_TEMPLATE = HELIOS_ROOT / "configs" / "params_cowpea.json"
ZOOMS = (1.0, 2.0, 4.0, 8.0)
CACHE_CAMERA_HEIGHT = 5.0   # generate_cache.py: camera_height=5.0
CACHE_WINDOW_M = 1.2        # generate_cache.py: reference_window_size=1.2


def plant_center_from_xml(xml_path: str, renderer: HeliosPyTorchRenderer, dev: torch.device):
    """The cache's camera centre: bbox centre of the mesh generate_cache.py renders (XML -> part tensor -> mesh)."""
    arr = PlantOrganArray.from_xml_file(xml_path)
    parts = arr.to_part_tensor(device=dev)
    mesh = renderer.geo_builder.build_mesh_from_part_tensor(parts, device=dev)
    v = mesh["vertices"]
    c = 0.5 * (v.min(0).values + v.max(0).values)
    return [float(x) for x in c.tolist()], int(parts.shape[0])


def write_shifted_xml(src: str, dst: Path, cx: float, cy: float):
    """Copy the plant XML with its <base_position> moved by (-cx, -cy) so the bbox centre lands on the origin.
    base_position is the XML's only absolute coordinate (every other field is a length or an angle)."""
    txt = Path(src).read_text()
    pat = r"<base_position>\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s*</base_position>"

    def _sub(m):
        return (f"<base_position> {float(m.group(1)) - cx:.6f} {float(m.group(2)) - cy:.6f} "
                f"{float(m.group(3)):.6f} </base_position>")

    new, n = re.subn(pat, _sub, txt)
    if n != 1:
        raise RuntimeError(f"{src}: expected exactly one <base_position>, found {n}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(new)


def write_params(path: Path, dap: int, seed: int, xml_abs: Path, resolution: int, camera_height: float):
    cfg = json.load(open(PARAMS_TEMPLATE))
    cfg["seed"] = int(seed)
    cfg["metadata"]["dap"] = int(dap)
    cfg["field"]["layout"].update({"mode": "manual", "plot_size_x": 1.5, "plot_size_y": 1.5,
                                     "num_beds": 1, "num_rows": 1})
    cfg["field"]["plots"] = [{"bed": 1, "row": 1,
                               "plants": [{"x": 0.0, "y": 0.0, "z": 0.0, "xml": str(xml_abs)}]}]
    cam = cfg.setdefault("camera", {})
    cam.setdefault("sensor", {}).update({"resolution_x": int(resolution), "resolution_y": int(resolution)})
    cam.setdefault("positioning", {}).update({"focusing_plants": False, "camera_height": float(camera_height),
                                               "azimuth_angle": 0.0, "distance_from_center": 0.01})
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(path, "w"), indent=1)


def run_helios(params: Path, out_dir: Path, name: str, dap: int, seed: int, fov_deg: float,
               camera_height: float, renderer: str = "radiation", timeout: int = 1800):
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(MAIN_BIN), "--renderer", renderer, "--save-xml", "--ground-occlusion", "false",
           "--plant-type", "cowpea", "-n", name, "--dap", str(int(dap)), "-s", str(int(seed)),
           "--output", str(out_dir), "-f", str(params.resolve()),
           "-h", f"{camera_height:.5f}", "--fov", f"{fov_deg:.6f}"]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(BUILD_DIR), capture_output=True, text=True, timeout=timeout)
    renders = sorted((out_dir / "cowpea").glob(f"{name}_*_rad.jpeg")) if (out_dir / "cowpea").exists() else []
    if r.returncode != 0 or not renders:
        raise RuntimeError(f"Helios failed for {name} (rc {r.returncode})\n"
                           f"stdout tail: {r.stdout[-1500:]}\nstderr tail: {r.stderr[-800:]}")
    return renders[0], time.time() - t0


def pyramid_from_render(jpeg: Path, size: int = 256) -> torch.Tensor:
    """(4, 3, size, size) in [0, 1]: centre crops of the full render at 1/1, 1/2, 1/4, 1/8 of its width."""
    im = np.asarray(Image.open(jpeg).convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0)
    H, W = t.shape[-2:]
    if H != W:
        raise RuntimeError(f"{jpeg}: expected a square render, got {W}x{H}")
    out = []
    for z in ZOOMS:
        w = int(round(H / z))
        o = (H - w) // 2
        crop = t[..., o:o + w, o:o + w]
        if w != size:
            crop = F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        out.append(crop.squeeze(0).clamp(0.0, 1.0))
    return torch.stack(out, 0)


def green_mask(rgb01: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb01[0], rgb01[1], rgb01[2]
    return (g > r * 1.02) & (g > b * 1.02)


def iou(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = float((a & b).sum())
    uni = float((a | b).sum())
    return inter / uni if uni > 0 else float("nan")


def centroid_px(m: torch.Tensor):
    ys, xs = torch.nonzero(m, as_tuple=True)
    if ys.numel() == 0:
        return (float("nan"), float("nan"))
    return (float(xs.float().mean()), float(ys.float().mean()))


def framing_check(cache_img: torch.Tensor, pyr01: torch.Tensor):
    """cache_img (16, S, S) as stored; pyr01 (4, 3, S, S). Returns per-zoom IoUs and centroid offsets."""
    res = []
    for li, z in enumerate(ZOOMS):
        chm = cache_img[4 * li + 3] > 0.0
        hg = green_mask(pyr01[li])
        flips = {"identity": hg, "flip_ud": torch.flip(hg, [0]), "flip_lr": torch.flip(hg, [1]),
                 "flip_both": torch.flip(hg, [0, 1])}
        ious = {k: iou(chm, v) for k, v in flips.items()}
        cc, hc = centroid_px(chm), centroid_px(hg)
        res.append({"zoom": z, "iou": ious["identity"], "iou_flips": ious,
                    "chm_px": int(chm.sum()), "green_px": int(hg.sum()),
                    "centroid_offset_px": [hc[0] - cc[0], hc[1] - cc[1]]})
    return res


def save_panel(path: Path, prefix: str, dap: int, cache_img: torch.Tensor, pyr01: torch.Tensor, check):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.5,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, axes = plt.subplots(len(ZOOMS), 3, figsize=(7.0, 2.3 * len(ZOOMS)))
    for li, z in enumerate(ZOOMS):
        cache_rgb = (cache_img[4 * li:4 * li + 3] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
        hel_rgb = pyr01[li].permute(1, 2, 0).numpy()
        chm = (cache_img[4 * li + 3] > 0).numpy()
        hg = green_mask(pyr01[li]).numpy()
        overlay = np.ones(chm.shape + (3,), dtype=np.float32)
        overlay[chm] = [0.20, 0.35, 0.65]          # cache CHM > 0: dark blue
        overlay[hg & ~chm] = [0.90, 0.55, 0.15]    # Helios green only: orange
        overlay[hg & chm] = [0.25, 0.55, 0.30]     # both: green
        for ax, im, title in zip(axes[li], (cache_rgb, hel_rgb, overlay),
                                 (f"cache RGB, zoom {z:g}x", f"Helios RGB, zoom {z:g}x",
                                  f"masks, IoU {check[li]['iou'] * 100:.1f}%")):
            ax.imshow(im, interpolation="nearest")
            ax.set_title(title, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.5); s.set_color("0.4")
    fig.suptitle(f"{prefix}  (DAP {dap}): cache framing vs Helios re-render", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="any checkpoint of the lineage: supplies data_dir / cache_dir / eval-set geometry")
    ap.add_argument("--eval_set", default="outputs/checkpoints/hierarchical_fm_v9/eval_set.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma-separated eval indices (default: the whole eval set)")
    ap.add_argument("--resolution", type=int, default=2048, help="Helios render size in px; the 8x level is resolution/8 px before resizing to 256")
    ap.add_argument("--renderer", default="radiation", choices=["radiation", "vis"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no_panel", action="store_true")
    a = ap.parse_args()

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    args = ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"])
    M = args["slots_per_phytomer"]
    ds = PartArrayDataset(data_root=args["data_dir"], max_nodes=args["max_phytomers"] * M, cache_dir=args["cache_dir"],
                          pkt_cache_dir=args.get("pkt_cache_dir") or None, species="cowpea", image_size=256)
    idxs = json.load(open(a.eval_set))["indices"]
    if a.only:
        idxs = [int(x) for x in a.only.split(",")]
    renderer = HeliosPyTorchRenderer(image_size=256).to(dev)
    out = Path(a.out)
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    fov_deg = math.degrees(2.0 * math.atan((CACHE_WINDOW_M / 2.0) / CACHE_CAMERA_HEIGHT))
    print(f"zoom-1x window {CACHE_WINDOW_M} m at {CACHE_CAMERA_HEIGHT} m -> hfov {fov_deg:.4f} deg; {len(idxs)} plants", flush=True)

    summary_path = out / "summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    for i in idxs:
        it = ds[i]
        prefix = it["prefix"]
        xml = ds.samples[i]["xml"]
        dap = int(it["dap"].item())
        m = re.search(r"seed(\d+)", prefix)
        seed = int(m.group(1)) if m else 0
        rgb_path = out / "rgb" / f"{prefix}_helios_rgb.pt"
        if rgb_path.exists() and not a.overwrite:
            print(f"idx {i:6d} {prefix}: exists, skipping", flush=True)
            continue

        center, n_parts = plant_center_from_xml(xml, renderer, dev)
        cx, cy, cz = center
        run_dir = out / "helios" / prefix
        shifted_xml = run_dir / f"{prefix}_shifted.xml"
        write_shifted_xml(xml, shifted_xml, cx, cy)
        params = run_dir / "params.json"
        # the PyTorch camera sits CACHE_CAMERA_HEIGHT above the bbox CENTRE (cam_z = centre_z + dist), so the
        # 1.2 m window is exact at that depth; Helios's -h is height above the ground, hence + cz
        cam_h = CACHE_CAMERA_HEIGHT + cz
        write_params(params, dap, seed, shifted_xml.resolve(), a.resolution, cam_h)
        jpeg, secs = run_helios(params, run_dir, prefix, dap, seed, fov_deg, cam_h, renderer=a.renderer)

        pyr01 = pyramid_from_render(jpeg, size=256)
        cache_img = it["image"].float()
        check = framing_check(cache_img, pyr01)
        rgb = ((pyr01 - 0.5) / 0.5).reshape(len(ZOOMS) * 3, 256, 256).contiguous()
        torch.save(rgb, rgb_path)
        if not a.no_panel:
            save_panel(out / "panels" / f"{prefix}.png", prefix, dap, cache_img, pyr01, check)
        ious = " ".join(f"{c['zoom']:g}x {c['iou'] * 100:5.1f}" for c in check)
        best_flip = max(check[0]["iou_flips"], key=check[0]["iou_flips"].get)
        print(f"idx {i:6d} DAP {dap:3d} {prefix}: centre ({cx:+.3f}, {cy:+.3f}, {cz:+.3f}) parts {n_parts:4d} "
              f"render {secs:5.1f}s | IoU {ious} | best orientation at 1x: {best_flip}", flush=True)
        summary[str(i)] = {"index": i, "prefix": prefix, "dap": dap, "center": center, "n_parts": n_parts,
                           "render_s": secs, "jpeg": str(jpeg), "framing": check}
        json.dump(summary, open(summary_path, "w"), indent=1)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
