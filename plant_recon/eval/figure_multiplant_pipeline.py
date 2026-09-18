"""One figure for the whole whole-frame pipeline: a real multi-plant rover frame -> object detection ->
an age estimate -> Helios plants grown from that age alone -> the trained model's architecture ->
differentiable-renderer refinement, all rendered back as one plot.

Composed from a completed run of use_cases/real_world/eval/run_multiplant_scene.py (its results.json and
the Helios scene renders it wrote), so it re-renders nothing. White-background journal style.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dirs", required=True, help="comma list of run_multiplant_scene.py output folders; one row each")
    ap.add_argument("--detector_weights", default="outputs/logs/20260916/agml_plant_detector/weights/best.pt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.5,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    run_dirs = [d.strip() for d in a.run_dirs.split(",") if d.strip() and
                os.path.exists(os.path.join(d.strip(), "results.json"))]
    rows_all = [build_row(d, a) for d in run_dirs]
    rows_all = [r for r in rows_all if r]
    if not rows_all:
        raise SystemExit("no completed runs among: " + a.run_dirs)
    ncol = max(len(r) for r in rows_all)
    fig, axes = plt.subplots(len(rows_all), ncol, figsize=(2.25 * ncol, 2.95 * len(rows_all)), squeeze=False)
    for ri, panels in enumerate(rows_all):
        for ci in range(ncol):
            ax = axes[ri][ci]
            if ci >= len(panels):
                ax.axis("off"); continue
            im, title, sub, boxes = panels[ci]
            ax.imshow(np.asarray(im), interpolation="nearest")
            for (x1, y1, x2, y2) in boxes:
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.0, edgecolor="#eb6834"))
            if ri == 0:
                ax.set_title(title, fontsize=8.5, loc="left", pad=4)
            ax.set_xlabel(sub, fontsize=7, color="#52514e", linespacing=1.5)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5); sp.set_color("0.45")
    fig.suptitle("Whole-frame multi-plant reconstruction from a single rover image", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.03 / len(rows_all) * 2), w_pad=1.1, h_pad=1.4)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"{len(rows_all)} rows x {ncol} panels -> {a.out}")


_RENDERER = {}


def pytorch_scene(xml_paths, plot_w_m, cam_h, px=640):
    """Render a set of exported plant XMLs with the DIFFERENTIABLE renderer, in the plot's own camera.

    This is the renderer the refinement actually optimises against, so it shows what the optimiser
    converged to. The Helios panels show the same geometry after a second export path (14D -> XML ->
    Helios forward kinematics), which can differ by a few degrees -- the multiplant export runs with
    stem_ik/leaf_ik off, documented in the 2026-09-16 report.
    """
    import torch
    from plant_recon.models.helios_pytorch_renderer import HeliosPyTorchRenderer
    from plant_recon.models.plant_organ_array import PlantOrganArray
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if px not in _RENDERER:
        _RENDERER[px] = HeliosPyTorchRenderer(image_size=px).to(dev)
    renderer = _RENDERER[px]
    from plant_recon.models.plant_organ_array import ORGAN_ROOT_META
    parts = []
    for xp in sorted(xml_paths):
        try:
            pt = PlantOrganArray.from_xml_file(str(xp)).to_part_tensor(device=dev)
        except Exception:
            continue
        # The XML writer expresses organs RELATIVE to the plant and keeps the plant's placement in
        # <base_position>, which reads back as the ROOT_META row's base columns. Without adding it every
        # plant lands on the origin and the whole plot collapses into one clump.
        meta = pt[pt[:, 0].long() == ORGAN_ROOT_META]
        if len(meta):
            pt = pt.clone()
            pt[:, 1:4] = pt[:, 1:4] + meta[0, 1:4].view(1, 3)
        parts.append(pt)
    if not parts:
        return None
    allp = torch.cat(parts, 0)
    mesh = renderer.geo_builder.build_mesh_from_part_tensor(allp, device=dev)
    with torch.no_grad():
        rgbd = renderer.forward(mesh, azimuth_deg=0.0, elevation_deg=90.0, camera_height=cam_h,
                                background="ground", focus_plant=False, include_depth=True,
                                image_size=px, zoom_factor=1.0, reference_window_size=plot_w_m,
                                center_override=torch.zeros(3, device=dev))
    import numpy as _np
    from PIL import Image as _Image
    arr = (rgbd[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(_np.uint8)
    return _Image.fromarray(arr)


def saved_scene(run_dir, res, mode):
    """The PyTorch render the run itself wrote from the optimised part tensor, if it has one."""
    from PIL import Image
    p = (res.get("pytorch_scenes", {}) or {}).get(mode, "") or os.path.join(run_dir, f"pytorch_scene_{mode}.png")
    return Image.open(p).convert("RGB") if p and os.path.exists(p) else None


def build_row(run_dir, a):
    """One scene -> the panels of its row, or None if the run is incomplete."""
    import numpy as np
    from PIL import Image
    from use_cases.real_world.dataset.real_plant_crop_utils import crop_rover_margins
    res = json.load(open(os.path.join(run_dir, "results.json")))
    img_path = res["image"]
    if not os.path.exists(img_path):                       # path stored before the 2026-09-16 restructure
        img_path = img_path.replace("real_world/", "use_cases/real_world/", 1)
    frame = crop_rover_margins(Image.open(img_path).convert("RGB"))
    # Mark the plants the RUN actually reconstructed, recovered from its own record, rather than
    # re-detecting at figure time: a different detector (or a --max_plants cap) finds a different
    # number, and pairing that count with scenes built from the run's plants misstates the figure.
    W, H = frame.width, frame.height
    pw, ph = res["plot_width_m"], res["plot_height_m"]
    cam_h = float(res.get("camera_height_m", 1.5))
    boxes = []
    for (x_m, y_m) in res["plants_xy"]:
        cx = (x_m / pw + 0.5) * W
        cy = (0.5 - y_m / ph) * H
        r = 0.035 * W
        boxes.append((cx - r, cy - r, cx + r, cy + r))

    def scene(key):
        p = res["scene_renders"].get(key, "")
        if p and not os.path.exists(p):
            p = p.replace("/real_world/", "/use_cases/real_world/", 1)
        return Image.open(p).convert("RGB") if p and os.path.exists(p) else None

    rows = res["rows"]
    def stat(mode, field):
        v = [r[field] for r in rows if r["mode"] == mode]
        return float(np.mean(v)) if v else float("nan")
    def organs(mode, field):
        v = [r[field] for r in rows if r["mode"] == mode]
        return int(np.sum(v)) if v else 0

    n_det = int(res["num_detected"])
    # Logical order: what the camera saw -> what was found -> the prior in each renderer -> the optimised
    # result -> the same geometry after the Helios export. Columns 4 and 5 share a renderer, so the change
    # refinement makes is read directly; column 6 then shows what survives the export.
    panels_raw = [
        (frame, "(a) Real rover frame", f"nadir, {frame.width}×{frame.height} px\nplot width {pw} m"),
        (frame, "(b) Object detection", f"{n_det} plants reconstructed\nage estimate DAP {res['dap']}"),
        (scene("cold"), "(c) Helios prior",
         f"{organs('helios', 'organs_cold')} organs over {n_det} plants\ngrown from DAP, no image used"),
        (saved_scene(run_dir, res, "prior_helios"), "(d) PyTorch prior",
         "the same prior in the renderer\nthe optimiser works in"),
        (saved_scene(run_dir, res, "opt_helios"), "(e) PyTorch optimised",
         f"nodes moved {stat('helios', 'moved_cm'):.1f} cm, data loss {stat('helios', 'data_loss'):.2f}\n"
         f"the 14D tensor, before any XML"),
        (scene("helios") , "(f) Helios re-rendered",
         "the same result exported to XML\nand raytraced again"),
    ]
    out = []
    for i, (im, title, sub) in enumerate(panels_raw):
        if im is None:
            continue
        out.append((im, title, sub, boxes if i == 1 else []))
    return out


if __name__ == "__main__":
    main()
