"""Whole-frame, multi-plant reconstruction: one real field image in, one reconstructed 3D plot out.

Everything before this script worked on ONE detected plant at a time and compared it against its own
crop. Real rover/cart frames carry several plants at once (see
outputs/logs/20260916/agml_plant_detector/train_batch*.jpg), and the useful product is the
whole plot, so this script runs the full loop end to end:

  real frame
    -> YOLO detection (use_cases/real_world/dataset/real_plant_crop_utils.detect_plants): N plant boxes
    -> pixel centres mapped to plot metres, written as a Helios params.json `plants` list
    -> Helios grows one plant per detection at the estimated DAP (the same binary and config that
       produced this project's synthetic training set) -> one structure XML per plant
    -> each XML becomes the (pos, rot, scale, latent, exist, parent) state our model optimizes
       (use_cases/real_world/dataset/helios_cold_start.py), OR the trained network's own image-conditioned
       sample is used instead (--init network), so the two cold starts are directly comparable
    -> differentiable-renderer refinement against that plant's own real crop
       (use_cases/real_world/eval/run_approach2_refine.refine_one)
    -> refined 14D part tensor exported back to XML (assemble_part_tensor_to_xml)
    -> params.json rewritten with those XML paths, which Helios loads verbatim
       (readPlantStructureXML, main.cpp) and renders as one plot for comparison with the frame.

The params.json / per-plant-XML / `plants: [{x, y, z, xml}]` convention is the one already
established by the Image2PlantArchitecture VLM work (Yun et al. 2026), where a VLM writes that same
config from an image; here object detection supplies the plant count and positions instead.

Pixel -> metre caveat: the rover camera has no intrinsic calibration in this project (the one known
quantity is its 1.5 m height, see use_cases/real_world/dataset/depth_anything_calib.py), so the frame's real
width is an ASSUMPTION exposed as --plot_width_m. Its default (1.3 m) is the Davis cowpea plot width
used by the VLM pipeline for the same field; it sets the scene's absolute scale and nothing else.
"""
import argparse
import datetime
import glob
import json
import os
import shutil
import subprocess
import time
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from plant_recon.models.part_tensor_to_40d import assemble_part_tensor_to_xml
from plant_recon.dataset.phytomer_packets import part_tensor_with_shoots
from plant_recon.dataset.phytomer_topology import chain_phytomers
from use_cases.real_world.dataset.helios_cold_start import BUILD_DIR, MAIN_BIN, helios_state_from_xml
from use_cases.real_world.dataset.dap_from_timestamp import dap_from_filename
from use_cases.real_world.dataset.real_plant_crop_utils import (
    DEFAULT_ROVER_MARGINS, build_mask_pyramid, build_pyramid_16ch, crop_rover_margins, detect_plants)
from use_cases.real_world.dataset.depth_anything_calib import pseudo_chm_for_crop
from use_cases.real_world.eval.run_approach1_cold import load_pipeline, sample_cold
from use_cases.real_world.eval.run_approach2_refine import refine_one

PARAMS_TEMPLATE = REPO_ROOT / "submodules/Digital-Crops" / "projects" / "syntheticdata_generation" / "configs" / "params_cowpea.json"


def pixels_to_plot_metres(center_px, width_px, height_px, plot_w_m):
    """Detection centre (pixels, origin top-left) -> plot coordinates in metres, origin at the plot
    centre, +x right and +y up-image. The plot's height in metres follows the frame's aspect ratio,
    so a plant's position in the scene matches where it sits in the photo."""
    cx, cy = center_px
    plot_h_m = plot_w_m * (height_px / max(width_px, 1))
    return ((cx / width_px - 0.5) * plot_w_m, (0.5 - cy / height_px) * plot_h_m), plot_h_m


def write_params(out_path: Path, dap: int, plot_w_m: float, plot_h_m: float, plants_xy,
                  resolution, xml_paths=None, camera_height=1.5, seed=0):
    """Helios plot config for these detections. `xml_paths` given => each plant entry points at an
    existing structure XML, which Helios loads instead of growing a new plant."""
    cfg = json.load(open(PARAMS_TEMPLATE))
    cfg["seed"] = int(seed)
    cfg["metadata"]["dap"] = int(dap)
    cfg["field"]["layout"].update({"mode": "manual", "plot_size_x": float(plot_w_m),
                                     "plot_size_y": float(plot_h_m), "num_beds": 1, "num_rows": 1})
    plants = []
    for i, (x_m, y_m) in enumerate(plants_xy):
        entry = {"x": float(x_m), "y": float(y_m), "z": 0.0}
        if xml_paths is not None:
            # absolute: Helios runs with cwd=BUILD_DIR and resolves every path from there
            entry["xml"] = str(Path(xml_paths[i]).resolve())
        plants.append(entry)
    cfg["field"]["plots"] = [{"bed": 1, "row": 1, "plants": plants}]
    cam = cfg.setdefault("camera", {})
    cam.setdefault("sensor", {}).update({"resolution_x": int(resolution[0]), "resolution_y": int(resolution[1])})
    cam.setdefault("positioning", {}).update({"focusing_plants": False, "camera_height": float(camera_height),
                                               "azimuth_angle": 0.0, "distance_from_center": 0.01})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(out_path, "w"), indent=1)
    return out_path


def run_helios(params_path: Path, out_dir: Path, name: str, dap: int, renderer: str, seed: int = 0,
                timeout: int = 3600, fov_deg: float = 0.0, camera_height: float = 1.5):
    """Runs the Helios generator/renderer on a params.json; returns the per-plant XML paths it
    produced (or loaded) plus any rendered image, both discovered under <out_dir>/cowpea/.

    fov_deg is passed explicitly rather than left to the binary's auto-calculation: the scene has
    to frame exactly the plot the detections were mapped into, and auto-FOV derives its own extent
    (which framed roughly a quarter of the plot when this was first run)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # absolute paths: the binary runs with cwd=BUILD_DIR, so a path relative to the repo root
    # (which is how every caller naturally writes it) would not resolve.
    out_dir, params_path = out_dir.resolve(), Path(params_path).resolve()
    cmd = [str(MAIN_BIN), "--renderer", renderer, "--save-xml", "--ground-occlusion", "false",
           "--plant-type", "cowpea", "-n", name, "--dap", str(int(dap)), "-s", str(int(seed)),
           "--output", str(out_dir), "-f", str(params_path), "-h", str(float(camera_height))]
    if fov_deg > 0:
        cmd.extend(["--fov", str(float(fov_deg))])
    r = subprocess.run(cmd, cwd=str(BUILD_DIR), capture_output=True, text=True, timeout=timeout)
    species_dir = out_dir / "cowpea"
    xmls = sorted(species_dir.glob(f"{name}_*_plant_*.xml"))
    if not xmls:
        raise RuntimeError(f"Helios produced no plant XML for {params_path}\n"
                            f"stdout tail: {r.stdout[-1500:]}\nstderr tail: {r.stderr[-800:]}")
    renders = sorted(list(species_dir.glob(f"{name}_*.jpeg")) + list(species_dir.glob(f"{name}_*.png")))
    return xmls, (renders[0] if renders else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="one real field frame (multi-plant)")
    ap.add_argument("--checkpoint", default=str(REPO_ROOT / "outputs/checkpoints/hierarchical_fm_v10_cam/hierarchical_fm_epoch_085.pt"))
    ap.add_argument("--detector_weights", default=str(REPO_ROOT / "outputs/logs/20260916/agml_plant_detector/weights/best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max_plants", type=int, default=8, help="cap on detections reconstructed (Helios growth is ~20 s/plant)")
    ap.add_argument("--init", default="helios", choices=["helios", "network", "both"],
                    help="cold start for refinement: the DAP-only Helios plant, the trained network's "
                         "image-conditioned sample, or both (runs the comparison twice)")
    ap.add_argument("--camera_height", type=float, default=1.5,
                    help="nadir camera height in metres; 1.5 is the rover's documented height and the value "
                         "the rest of the real-image pipeline already assumes")
    ap.add_argument("--plot_width_m", type=float, default=1.3,
                    help="assumed real-world width of the margin-cropped frame; sets the scene's absolute scale")
    ap.add_argument("--planted_date", default="", help="YYYY-MM-DD; DAP from the filename timestamp instead of Stage 1's estimate")
    ap.add_argument("--dap", type=int, default=0, help="override the DAP estimate entirely (0 = estimate)")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--opt", default="pos,scale,latent")
    ap.add_argument("--no_refine", action="store_true", help="skip refinement (cold plot only)")
    ap.add_argument("--scene_renderer", default="radiation", choices=["radiation", "vis", "none"])
    ap.add_argument("--out", default=str(REPO_ROOT / "use_cases/real_world/eval/output/multiplant_scene"))
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, pvae, ovae, renderer, M = load_pipeline(a.checkpoint, dev)

    img_full = Image.open(a.image).convert("RGB")
    img = crop_rover_margins(img_full, DEFAULT_ROVER_MARGINS)
    W, H = img.size
    dets = [d for d in detect_plants(img, a.detector_weights, conf=a.conf)][: a.max_plants]
    if not dets:
        raise SystemExit(f"no plants detected in {a.image}")
    print(f"{len(dets)} plants detected in {Path(a.image).name} ({W}x{H} after margin crop)")

    if a.dap > 0:
        dap = a.dap
    elif a.planted_date:
        dap = dap_from_filename(a.image, datetime.date.fromisoformat(a.planted_date)) or 30
    else:
        tok = model.image_encoder(build_pyramid_16ch(img, dets[0], pseudo_chm_for_crop(img)).unsqueeze(0).to(dev))
        dap = int(round(float(model.probe_pred_dap(tok).item())))
    print(f"DAP used for the scene: {dap}")

    depth = pseudo_chm_for_crop(img)
    plants_xy, plot_h = [], None
    for d in dets:
        (x_m, y_m), plot_h = pixels_to_plot_metres(d.center, W, H, a.plot_width_m)
        plants_xy.append((x_m, y_m))

    # 1) grow one plant per detection at the estimated DAP -> per-plant XMLs
    cold_params = write_params(out / "params_cold.json", dap, a.plot_width_m, plot_h, plants_xy, (W, H),
                                camera_height=a.camera_height)
    # horizontal FOV that makes a nadir camera at this height see exactly the plot width
    fov_deg = 2.0 * float(np.degrees(np.arctan((a.plot_width_m / 2.0) / a.camera_height)))
    _t = {}
    _t0 = time.time()
    print(f"growing {len(dets)} Helios plants at DAP {dap} (~20 s each), scene FOV {fov_deg:.1f} deg ...", flush=True)
    cold_xmls, cold_render = run_helios(cold_params, out / "helios_cold", "scene_cold", dap, a.scene_renderer,
                                         fov_deg=fov_deg, camera_height=a.camera_height)
    _t["helios_growth_and_cold_render"] = time.time() - _t0; _t0 = time.time()
    print(f"  -> {len(cold_xmls)} plant XMLs, scene render: {cold_render}  [{_t['helios_growth_and_cold_render']:.0f} s]")

    # 2) per-plant: cold state -> refinement against that plant's own crop -> refined XML
    lrs = {"pos": 3e-3, "scale": 2e-2, "latent": 2e-2, "roll": 2e-2, "weight_decay": 1e-4}
    opt_set = set(a.opt.split(","))
    modes = ["helios", "network"] if a.init == "both" else [a.init]
    rows, refined_xml_by_mode = [], {}
    for mode in modes:
        refined_xmls = []
        scene_parts = []
        prior_parts = []
        for i, d in enumerate(dets):
            item = {"image": build_pyramid_16ch(img, d, depth), "mask_pyramid": build_mask_pyramid(d)}
            images = item["image"].unsqueeze(0).to(dev)
            if mode == "helios":
                xml_i = cold_xmls[min(i, len(cold_xmls) - 1)]
                pos0, rot, scale0, lat0, exist, par, parent_idx = helios_state_from_xml(xml_i, pvae, dev, M)
                from plant_recon.eval.eval_gt_substitution_ablation import plant_from_nodes
                with torch.no_grad():
                    parts0 = plant_from_nodes(pvae, pos0, rot, scale0, lat0, exist, par, M)
            else:
                parts0, _, st = sample_cold(model, pvae, ovae, images, dev, M,
                                             daps=torch.tensor([float(dap)], device=dev))
                pos0, rot, scale0, lat0, exist, par, parent_idx = st
            if parts0.shape[0] == 0:
                print(f"  [{mode} {i}] empty cold start, skipped"); continue
            if a.no_refine:
                parts1, moved, dl = parts0, 0.0, float("nan")
                state = (pos0, rot, scale0, lat0, exist, par)
            else:
                parts1, moved, dl, state = refine_one(model, pvae, renderer, item, pos0, rot, scale0, lat0, exist,
                                                par, parent_idx, M, dev, a.steps, opt_set, lrs, [1.0, 2.0, 4.0, 8.0],
                                                5.0, 0.5, 20.0, False, False, True, 1.5, 5.0, 0.3, return_state=True)
            xml_out = out / f"refined_{mode}" / f"plant_{i:04d}.xml"
            xml_out.parent.mkdir(parents=True, exist_ok=True)
            # Re-decode carrying shoot structure. plant_from_nodes (which produced parts1 for the
            # renderer) flattens the branches away, and the 40D converter rebuilds shoots by
            # splitting on ORGAN_SHOOT_META rows -- without them every branch exports chained into
            # one shoot.
            posf, rotf, scalef, latf, existf, parf = state
            with torch.no_grad():
                _, shoot_id, phyto_idx = chain_phytomers(posf, rotf, exist=(existf > 0.5).float(),
                                                          root_own_shoot=True)
                parts_x = part_tensor_with_shoots(pvae, posf, rotf, scalef, latf, existf, parf, M,
                                                   shoot_id, phyto_idx)
            # The plant's plot position is baked into the exported coordinates, because Helios does
            # NOT apply a params entry's per-plant (x, y) when it loads a plant from XML: main.cpp's
            # XML branch shifts the loaded plant by the PLOT origin only (`shift =
            # make_vec3(origin.x, origin.y, 0)`), while the grow-from-library branch uses
            # `origin + (X, Y)`. Without this every reloaded plant stacks at the plot centre.
            parts_world = parts_x.detach().cpu().clone()
            parts_world[:, 1] += float(plants_xy[i][0])
            parts_world[:, 2] += float(plants_xy[i][1])
            # stem_ik/leaf_ik off: both passes index the 40D array and the 14D tensor as one
            # row-aligned pair, and the converter is not always 1:1 on an assembled (packet-decoded)
            # plant -- a 454-row part tensor came back as 456 rows here, which the IK cannot align.
            # The plain analytical export is used instead, so the scene render can sit a few degrees
            # off the refined 14D geometry; the refinement numbers themselves are unaffected.
            assemble_part_tensor_to_xml(parts_world, plant_id=i, xml_filepath=str(xml_out),
                                          stem_ik=False, leaf_ik=False)
            refined_xmls.append(xml_out)
            # Keep the optimised 14D tensor itself, already placed in plot coordinates. Rendering THIS is
            # what the optimiser actually produced; everything downstream (XML export, Helios forward
            # kinematics) is a second conversion that can and does lose information -- the per-plant plot
            # offset among it (2026-09-17: only 1 of 3 plants kept its base_position through the export).
            scene_parts.append(parts_world.clone())
            prior_world = parts0.detach().cpu().clone()
            prior_world[:, 1] += float(plants_xy[i][0]); prior_world[:, 2] += float(plants_xy[i][1])
            prior_parts.append(prior_world)
            print(f"  [{mode} {i}] organs {parts0.shape[0]} -> {parts1.shape[0]}, moved {moved:.1f} cm, data_loss {dl:.3f}", flush=True)
            rows.append({"mode": mode, "plant": i, "organs_cold": int(parts0.shape[0]),
                          "organs_refined": int(parts1.shape[0]), "moved_cm": float(moved), "data_loss": float(dl)})
        refined_xml_by_mode[mode] = refined_xmls
        _t[f"refine_{mode}"] = time.time() - _t0; _t0 = time.time()
        print(f"  [{mode}] refinement of {len(refined_xmls)} plants took {_t[f'refine_{mode}']:.0f} s", flush=True)
        for tag, plist in (("prior", prior_parts), ("opt", scene_parts)):
            if not plist:
                continue
            _a = torch.cat(plist, 0).to(dev)
            _m = renderer.geo_builder.build_mesh_from_part_tensor(_a, device=dev)
            with torch.no_grad():
                _r = renderer.forward(_m, azimuth_deg=0.0, elevation_deg=90.0, camera_height=a.camera_height,
                                      background="ground", focus_plant=False, include_depth=True, image_size=640,
                                      zoom_factor=1.0, reference_window_size=a.plot_width_m,
                                      center_override=torch.zeros(3, device=dev))
            Image.fromarray((_r[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                            ).save(out / f"pytorch_{tag}_{mode}.png")
            torch.save(_a.cpu(), out / f"scene_parts_{tag}_{mode}.pt")
        if False:
            allp = torch.cat(scene_parts, 0).to(dev)
            mesh_s = renderer.geo_builder.build_mesh_from_part_tensor(allp, device=dev)
            with torch.no_grad():
                rgbd_s = renderer.forward(mesh_s, azimuth_deg=0.0, elevation_deg=90.0,
                                          camera_height=a.camera_height, background="ground",
                                          focus_plant=False, include_depth=True, image_size=640,
                                          zoom_factor=1.0, reference_window_size=a.plot_width_m,
                                          center_override=torch.zeros(3, device=dev))
            Image.fromarray((rgbd_s[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                            ).save(out / f"pytorch_scene_{mode}.png")
            torch.save(allp.cpu(), out / f"scene_parts_{mode}.pt")
            print(f"  [{mode}] optimiser's own render -> {out / f'pytorch_scene_{mode}.png'}", flush=True)

    # 3) rebuild the plot from the refined plants and render it as one scene
    scene_renders = {"cold": str(cold_render) if cold_render else ""}
    for mode, xmls in refined_xml_by_mode.items():
        if not xmls:
            continue
        p = write_params(out / f"params_refined_{mode}.json", dap, a.plot_width_m, plot_h,
                          plants_xy[: len(xmls)], (W, H), xml_paths=xmls, camera_height=a.camera_height)
        try:
            _, render = run_helios(p, out / f"helios_refined_{mode}", f"scene_{mode}", dap, a.scene_renderer,
                                    fov_deg=fov_deg, camera_height=a.camera_height)
            scene_renders[mode] = str(render) if render else ""
            _t[f"helios_scene_{mode}"] = time.time() - _t0; _t0 = time.time()
            print(f"refined scene ({mode}): {render}  [{_t[f'helios_scene_{mode}']:.0f} s]")
        except RuntimeError as e:
            print(f"refined scene ({mode}) failed: {str(e)[:300]}")

    img.save(out / "real_frame.png")
    json.dump({"image": a.image, "dap": dap, "num_detected": len(dets), "plot_width_m": a.plot_width_m,
                "camera_height_m": a.camera_height,
                "pytorch_scenes": {f"{t}_{m}": str(out / f"pytorch_{t}_{m}.png")
                                    for m in refined_xml_by_mode for t in ("prior", "opt")
                                    if (out / f"pytorch_{t}_{m}.png").exists()},
                "stage_seconds": {k: round(v, 1) for k, v in _t.items()},
                "plot_height_m": plot_h, "plants_xy": plants_xy, "scene_renders": scene_renders, "rows": rows},
               open(out / "results.json", "w"), indent=1)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
