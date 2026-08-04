"""Batch generate Helios synthetic plant image + XML pairs for model training.

Supports multi-view rendering with configurable camera angles and Davis, CA June
sun positions for diffusion / differentiable rendering research.

Example (quick test):
    python -m dataset.generate_helios_dataset --quick

Example (8 camera azimuths, Davis June sun randomization):
    python -m dataset.generate_helios_dataset --view-angles 8 --davis-june-sun
"""

import os
import sys
import argparse
import subprocess
import multiprocessing
import json
import tempfile
import math
import random
from datetime import datetime, timedelta
from typing import List, Tuple
from tqdm import tqdm

# Davis, CA coordinates
DAVIS_LAT = 38.5382   # degrees North
DAVIS_LON = -121.7617 # degrees West (not used in simple calculation but kept for reference)

# Ensure python environment bin directory (containing Xvfb) is in PATH
env_bin = os.path.dirname(sys.executable)
if env_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = env_bin + os.path.pathsep + os.environ.get("PATH", "")


def solar_declination(day_of_year: int) -> float:
    """Solar declination angle in degrees (approximate)."""
    return 23.45 * math.sin(math.radians((360.0 / 365.25) * (day_of_year - 81)))


def solar_position(day_of_year: int, hour: float, latitude: float = DAVIS_LAT) -> Tuple[float, float]:
    """Calculate solar elevation and azimuth for a given day/hour/latitude.
    
    Args:
        day_of_year: Day of year (1-365)
        hour: Hour in local solar time (0-24), 12 = solar noon
        latitude: Latitude in degrees
        
    Returns:
        (elevation_degrees, azimuth_degrees)
        Azimuth: 0=North, 90=East, 180=South, 270=West
    """
    lat_rad = math.radians(latitude)
    decl_rad = math.radians(solar_declination(day_of_year))
    
    # Hour angle: 15 degrees per hour from solar noon
    hour_angle_deg = 15.0 * (hour - 12.0)
    hour_angle_rad = math.radians(hour_angle_deg)
    
    # Elevation
    sin_elev = (math.sin(lat_rad) * math.sin(decl_rad) +
                math.cos(lat_rad) * math.cos(decl_rad) * math.cos(hour_angle_rad))
    # Clamp to valid range [-1, 1]
    sin_elev = max(-1.0, min(1.0, sin_elev))
    elev = math.degrees(math.asin(sin_elev))
    
    # Azimuth (0=North, 90=East, 180=South, 270=West)
    # Formula: atan2(sin(H), cos(H)*sin(lat) - tan(decl)*cos(lat))
    numerator = math.sin(hour_angle_rad)
    denominator = (math.cos(hour_angle_rad) * math.sin(lat_rad) -
                   math.tan(decl_rad) * math.cos(lat_rad))
    az_rad = math.atan2(numerator, denominator)
    az = (math.degrees(az_rad) + 180.0) % 360.0
    
    return elev, az


def randomize_davis_june_sun(seed: int, dap: int) -> Tuple[float, float]:
    """Generate a random sun position for Davis, CA in June (9am-6pm).
    
    Args:
        seed: Random seed
        dap: Days after planting. Used to map to a day in June.
            DAP 5 -> June 5, DAP 60 -> June 30 (clamped)
            
    Returns:
        (elevation_degrees, azimuth_degrees)
    """
    rng = random.Random(seed + dap * 10000)
    
    # Map DAP to June day of year.
    # June 1 = day 152, June 30 = day 181
    # For DAP values > 30, clamp to end of June
    day_of_year = min(152 + dap - 1, 181)
    
    # Random hour between 9:00 and 18:00 (9am - 6pm)
    hour = rng.uniform(9.0, 18.0)
    
    elev, az = solar_position(day_of_year, hour)
    
    # Ensure elevation is reasonable (daytime only)
    elev = max(elev, 5.0)
    
    return elev, az


def generate_one(args: Tuple) -> Tuple[int, int, int, float, float, float, bool, str]:
    """Generate a single Helios sample with specific camera and sun angles.

    Args:
        args: (main_binary, dap, seed, cam_az, cam_height, sun_elev, sun_az,
               base_params_file, output_dir, renderer, export_3d)
    Returns:
        (dap, seed, cam_az, cam_height, sun_elev, sun_az, success, message)
    """
    (main_binary, dap, seed, cam_az, cam_height, sun_elev, sun_az,
     base_params_file, output_dir, renderer, export_3d) = args

    name = (f"cowpea_dap{dap:03d}_seed{seed:02d}"
            f"_caz{cam_az:03d}_h{cam_height:.1f}"
            f"_se{int(sun_elev):03d}_saz{int(sun_az):03d}")
    build_dir = os.path.dirname(main_binary)

    # Create a temporary params.json with overridden camera/sun angles
    with open(base_params_file, 'r') as f:
        params = json.load(f)

    # Camera azimuth + height
    if "camera" not in params:
        params["camera"] = {}
    if "positioning" not in params["camera"]:
        params["camera"]["positioning"] = {}
    params["camera"]["positioning"]["azimuth_angle"] = float(cam_az)
    params["camera"]["positioning"]["camera_height"] = float(cam_height)

    # Sun position + shadow
    if "environment" not in params:
        params["environment"] = {}
    if "sun" not in params["environment"]:
        params["environment"]["sun"] = {}
    params["environment"]["sun"]["elevation_degrees"] = float(sun_elev)
    params["environment"]["sun"]["azimuth_degrees"] = float(sun_az)
    params["environment"]["sun"]["shadow"] = True

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
        tmp_params_path = tf.name
        json.dump(params, tf, indent=2)

    cmd = [
        main_binary,
        "--renderer", renderer,
        "--save-xml",
        "--focus-plant",
        "-n", name,
        "--days", str(dap),
        "-s", str(seed),
        "--output", output_dir,
        "-f", tmp_params_path,
    ]
    if export_3d == "ply":
        cmd.extend(["--export-3d", "ply"])

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as tf:
            log_path = tf.name
        with open(log_path, 'w') as log_fh:
            result = subprocess.run(
                cmd,
                cwd=build_dir,
                env=os.environ.copy(),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=1800,
            )
        with open(log_path, 'r', errors='replace') as log_fh:
            stdout_text = log_fh.read()
        os.unlink(log_path)
        os.unlink(tmp_params_path)

        # Determine expected output file based on renderer
        if renderer in ("vis", "all"):
            out_file = os.path.join(output_dir, f"{name}_0000_vis.jpeg")
        elif renderer == "radiation":
            out_file = os.path.join(output_dir, f"{name}_0000_rad.jpeg")
        else:
            out_file = None

        xml_file = os.path.join(output_dir, f"{name}_0000_plant_0000.xml")
        success = result.returncode == 0 and (out_file is None or os.path.exists(out_file))

        # Render differentiable comparison images (_diff_fixed.png and _diff_focus.png)
        if success and os.path.exists(xml_file):
            try:
                import torch
                from PIL import Image
                import numpy as np
                from diffusion_based.models.helios_xml_parser import HeliosXMLParser
                from diffusion_based.models.differentiable_renderer_3d import DifferentiablePlantRenderer3D

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                parser = HeliosXMLParser(xml_file)
                parser.parse()
                organ_nodes = parser.get_all_organ_nodes()

                if organ_nodes:
                    nodes_tensor = torch.stack([torch.tensor(n.to_15d(), dtype=torch.float32) for n in organ_nodes]).unsqueeze(0).to(device)
                    parents = torch.tensor([n.parent_idx for n in organ_nodes], dtype=torch.long).unsqueeze(0).to(device)

                    diff_renderer = DifferentiablePlantRenderer3D(image_size=256).to(device)

                    with torch.no_grad():
                        img_fixed = diff_renderer(nodes_tensor, parent_indices=parents, cam_azimuth_deg=float(cam_az), focus_plant=False)[0]
                        img_focus = diff_renderer(nodes_tensor, parent_indices=parents, cam_azimuth_deg=float(cam_az), focus_plant=True)[0]

                    img_fixed_np = (img_fixed.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
                    img_focus_np = (img_focus.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)

                    Image.fromarray(img_fixed_np).save(os.path.join(output_dir, f"{name}_0000_diff_fixed.png"))
                    Image.fromarray(img_focus_np).save(os.path.join(output_dir, f"{name}_0000_diff_focus.png"))
            except Exception as e_diff:
                stdout_text += f"\n(Warning: Differentiable renderer failed: {e_diff})"

        msg = stdout_text if success else f"{stdout_text}\n(Error: image file missing: {out_file})"
        return dap, seed, cam_az, cam_height, sun_elev, sun_az, success, msg
    except subprocess.TimeoutExpired:
        os.unlink(tmp_params_path)
        return dap, seed, cam_az, cam_height, sun_elev, sun_az, False, "Timeout"
    except Exception as e:
        try:
            os.unlink(tmp_params_path)
        except FileNotFoundError:
            pass
        return dap, seed, cam_az, cam_height, sun_elev, sun_az, False, str(e)


def build_job_list(dap_start: int, dap_end: int, dap_step: int,
                   seeds: int,
                   cam_azimuths: List[int], cam_heights: List[float],
                   sun_elevs: List[float], sun_azimuths: List[float],
                   main_binary: str, output_dir: str,
                   base_params_file: str, renderer: str = "vis",
                   export_3d: str = "none") -> List[Tuple]:
    jobs = []
    sun_idx = 0
    for dap in range(dap_start, dap_end + 1, dap_step):
        for seed in range(seeds):
            sun_elev = sun_elevs[sun_idx % len(sun_elevs)]
            sun_az = sun_azimuths[sun_idx % len(sun_azimuths)]
            sun_idx += 1
            for caz in cam_azimuths:
                ch = cam_heights[seed % len(cam_heights)]
                jobs.append((
                    main_binary, dap, seed, caz, ch, sun_elev, sun_az,
                    base_params_file, output_dir, renderer, export_3d
                ))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: DAP 5 x seed 0 x single view/sun")
    parser.add_argument("--dap-start", type=int, default=5)
    parser.add_argument("--dap-end", type=int, default=60)
    parser.add_argument("--dap-step", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of random seeds per DAP")
    parser.add_argument("--view-angles", type=int, default=1,
                        help="Number of evenly-spaced camera azimuth angles (e.g. 8 -> 0,45,90...). Default = 1")
    parser.add_argument("--camera-heights", type=float, nargs='+', default=[1.0],
                        help="Camera heights in meters. Default = [1.0]")
    parser.add_argument("--davis-june-sun", action="store_true",
                        help="Randomize sun position based on Davis, CA June 9am-6pm solar trajectory. Overrides --sun-elevations/--sun-azimuths.")
    parser.add_argument("--sun-elevations", type=float, nargs='+', default=[45.0],
                        help="Sun elevation angles in degrees (ignored if --davis-june-sun). Default = [45.0]")
    parser.add_argument("--sun-azimuths", type=float, nargs='+', default=[180.0],
                        help="Sun azimuth angles in degrees (ignored if --davis-june-sun). Default = [180.0]")
    parser.add_argument("--renderer", type=str, default="vis",
                        choices=["vis", "radiation", "all", "none"],
                        help="Helios renderer mode (vis/radiation/all/none)")
    parser.add_argument("--main-binary", type=str,
                        default="Digital-Crops/projects/syntheticdata_generation/build/main")
    parser.add_argument("--output-dir", type=str,
                        default="Digital-Crops/projects/syntheticdata_generation/build/output")
    parser.add_argument("--params-file", type=str,
                        default="Digital-Crops/projects/syntheticdata_generation/params.json",
                        help="Base params.json file path")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel processes; default = 4. On macOS with vis renderer, use 1 to avoid display races.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip existing outputs")
    parser.add_argument("--export-3d", type=str, default="none",
                        choices=["ply", "none"],
                        help="Export plant-only 3D PLY via Helios (none/ply)")
    parser.add_argument("--render-diff-comparison", action="store_true",
                        help="Render Python differentiable 2D comparison PNGs (slow; default off)")
    args = parser.parse_args()

    main_binary = os.path.abspath(args.main_binary)
    output_dir = os.path.abspath(args.output_dir)
    base_params_file = os.path.abspath(args.params_file)

    if not os.path.exists(main_binary):
        print(f"ERROR: main binary not found at {main_binary}")
        sys.exit(1)
    if not os.path.exists(base_params_file):
        print(f"ERROR: base params file not found at {base_params_file}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    if args.renderer in ("vis", "all") and sys.platform == "darwin":
        print("[INFO] Running Helios visualizer on macOS. Make sure XQuartz is running if offscreen fails.")

    if args.quick:
        dap_start, dap_end, dap_step = 5, 5, 5
        seeds = 1
        cam_azimuths = [0]
        cam_heights = [1.0]
        sun_elevs = [45.0]
        sun_azimuths = [180.0]
    else:
        dap_start, dap_end, dap_step = args.dap_start, args.dap_end, args.dap_step
        seeds = args.seeds
        if args.view_angles <= 1:
            cam_azimuths = [0]
        else:
            step = 360 // args.view_angles
            cam_azimuths = [i * step for i in range(args.view_angles)]
        cam_heights = list(args.camera_heights)
        
        if args.davis_june_sun:
            # Generate sun positions for each seed/DAP combination
            sun_elevs = []
            sun_azimuths = []
            for dap in range(dap_start, dap_end + 1, dap_step):
                for seed in range(seeds):
                    elev, az = randomize_davis_june_sun(seed, dap)
                    sun_elevs.append(elev)
                    sun_azimuths.append(az)
            print(f"Davis June sun positions: {len(sun_elevs)} total")
            print(f"  Elevation range: {min(sun_elevs):.1f} - {max(sun_elevs):.1f} deg")
            print(f"  Azimuth range:   {min(sun_azimuths):.1f} - {max(sun_azimuths):.1f} deg")
        else:
            sun_elevs = list(args.sun_elevations)
            sun_azimuths = list(args.sun_azimuths)

    jobs = build_job_list(
        dap_start, dap_end, dap_step, seeds,
        cam_azimuths, cam_heights, sun_elevs, sun_azimuths,
        main_binary, output_dir, base_params_file, args.renderer,
        export_3d=args.export_3d
    )

    if args.resume:
        if args.renderer == "radiation":
            out_suffix = "_0000_rad.jpeg"
        else:
            out_suffix = "_0000_vis.jpeg"
        existing_names = {
            f.replace(out_suffix, "")
            for f in os.listdir(output_dir)
            if f.endswith(out_suffix)
        }
        filtered = []
        for job in jobs:
            mb, dap, seed, caz, ch, selev, saz, pfile, out_dir, rend, export_3d = job
            name = (f"cowpea_dap{dap:03d}_seed{seed:02d}"
                    f"_caz{caz:03d}_h{ch:.1f}_se{int(selev):03d}_saz{int(saz):03d}")
            if name not in existing_names:
                filtered.append(job)
        jobs = filtered
        print(f"Resume mode: {len(jobs)} jobs remaining after skipping existing files")

    total = len(jobs)
    print(f"Generating {total} Helios samples in {output_dir}")
    print(f"DAP range: {dap_start}..{dap_end} step {dap_step}, seeds per DAP: {seeds}")
    print(f"Camera azimuths: {cam_azimuths}")
    print(f"Camera heights:    {cam_heights}")
    print(f"3D export:         {args.export_3d}")
    effective_workers = args.workers
    if sys.platform == "darwin" and args.renderer == "vis" and effective_workers != 1:
        print("[WARN] macOS + vis renderer + multiprocessing can race on the offscreen display. Forcing workers=1.")
        effective_workers = 1
    print(f"Parallel workers:  {effective_workers if effective_workers else 'auto (CPU count)'}")

    successes = 0
    failures = []

    with multiprocessing.Pool(processes=effective_workers if effective_workers else None) as pool:
        for dap, seed, cam_az, cam_h, sun_elev, sun_az, success, msg in tqdm(
            pool.imap_unordered(generate_one, jobs),
            total=total,
            desc="Generating Helios samples"
        ):
            if success:
                successes += 1
            else:
                failures.append((dap, seed, cam_az, cam_h, sun_elev, sun_az, msg[-200:]))

    print(f"\nDone. Successes: {successes}/{total}")
    if failures:
        print(f"Failures: {len(failures)}")
        for dap, seed, cam_az, cam_h, sun_elev, sun_az, msg in failures[:10]:
            print(f"  DAP {dap} seed {seed} caz={cam_az} h={cam_h} se={sun_elev:.1f} saz={sun_az:.1f}: {msg.strip()}")


if __name__ == "__main__":
    main()
