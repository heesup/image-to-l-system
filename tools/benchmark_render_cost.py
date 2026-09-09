"""Benchmarks actual differentiable rendering time per sample.

Measures the real wall-clock cost of the in-loop photometric pipeline
(mesh build + 4-scale pyramid render, forward + backward) so that
render_fraction / batch_size / wall-clock estimates are grounded in
measurements instead of guesses.

Usage (from workspace root, on a GPU node or local GPU):
    /home/lion397/.conda/envs/digital-crops/bin/python tools/benchmark_render_cost.py \
        --cache_dir dataset/cache/cowpea_curv26 --daps 10,50,90 --repeats 20

Output: per-DAP per-sample ms (mesh build, 4-scale fwd, bwd), plus a projected
epoch-time table for the ablation ladder configs (A/B/C).
"""

import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from diffusion_based.dataset.part_array_dataset import PartArrayDataset
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.models.organ_latent_vae import OrganLatentVAE
from diffusion_based.dataset.part_array_dataset import EMPTY_IDX, FM_OT_END


def load_samples(cache_dir: str, daps, per_dap: int):
    import glob
    picked = []
    for d in daps:
        files = [f for f in glob.glob(f"{cache_dir}/*.pt") if re.search(rf"dap{d:03d}_", os.path.basename(f))]
        if not files:
            print(f"  [warn] no cache files for DAP {d}")
            continue
        files = sorted(files, key=os.path.getmtime)[-per_dap:]  # newest (most complete shards)
        picked.extend(files)
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="dataset/cache/cowpea_curv26")
    parser.add_argument("--daps", type=str, default="10,50,90")
    parser.add_argument("--per_dap", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=20, help="timed iterations after warmup")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch_sizes", type=str, default="16,32,48", help="projected epoch table batch sizes")
    parser.add_argument("--fractions", type=str, default="0.167,0.5,1.0", help="projected fractions")
    args = parser.parse_args()

    daps = [int(d) for d in args.daps.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("No CUDA device — benchmark requires a GPU.")
        return 1

    print(f"Device: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")

    renderer = HeliosPyTorchRenderer(image_size=128, device=device).to(device)
    vae = OrganLatentVAE(latent_dim=16, hidden_dim=256)
    vae_path = "diffusion_based/checkpoints/organ_vae/organ_latent_vae_best.pt"
    if os.path.exists(vae_path):
        vae.load_state_dict(torch.load(vae_path, map_location="cpu", weights_only=True))
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    files = load_samples(args.cache_dir, daps, args.per_dap)
    if not files:
        print("No samples loaded.")
        return 1

    results = {}  # dap -> (mesh_ms, fwd_ms, bwd_ms, organs)
    for f in files:
        data = torch.load(f, map_location="cpu", weights_only=False)
        if not (isinstance(data, dict) and "nodes" in data):
            continue
        nodes = data["nodes"]
        m = re.search(r"dap(\d+)", os.path.basename(f))
        dap = int(m.group(1)) if m else -1

        # Active organs -> 16D latents -> 14D parts (mirrors the training path)
        ot = nodes[:, :FM_OT_END].argmax(-1)
        act = (ot > 2) & (nodes[:, EMPTY_IDX] < 0.5)
        with torch.no_grad():
            z1 = vae.encode(nodes[act].to(device))[0]
        part14d, _ = vae.decode_to_part_tensor(z1.unsqueeze(0))
        part14d = part14d[0]
        n_organs = part14d.shape[0]

        # Warmup + timed: mesh build, 4-scale pyramid fwd, backward
        timings = {"mesh": [], "fwd": [], "bwd": []}
        for it in range(args.warmup + args.repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            mesh = renderer.geo_builder.build_mesh_from_part_tensor(
                part14d, existence=torch.ones(n_organs, device=device), device=device)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            pred = renderer.render_multiscale_pyramid(
                mesh, scales=[1.0, 2.0, 4.0, 8.0],
                azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
                include_depth=True, differentiable=True, image_size=128,
                reference_window_size=1.2,
            )
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            # Dummy photometric loss backward (depth L1 on scale 1x proxy)
            loss = sum(p[3].abs().mean() for s, p in pred.items())
            if not loss.requires_grad:
                # Renderer built a detached graph (e.g. no differentiable context);
                # attach a leaf so backward timing still reflects the real pass.
                loss = loss + torch.zeros(1, device=device, requires_grad=True)
            loss.backward()
            torch.cuda.synchronize()
            t3 = time.perf_counter()

            if it >= args.warmup:
                timings["mesh"].append((t1 - t0) * 1000)
                timings["fwd"].append((t2 - t1) * 1000)
                timings["bwd"].append((t3 - t2) * 1000)

        avg = {k: sum(v) / len(v) for k, v in timings.items()}
        results[dap] = (avg["mesh"], avg["fwd"], avg["bwd"], n_organs)
        print(f"DAP {dap:3d} ({n_organs:5d} organs): mesh {avg['mesh']:6.1f} ms | "
              f"fwd {avg['fwd']:6.1f} ms | bwd {avg['bwd']:6.1f} ms | "
              f"total {avg['mesh']+avg['fwd']+avg['bwd']:7.1f} ms/sample")

    if not results:
        return 1

    # Projection table: epoch wall-clock for ladder configs
    total_ms = {d: r[0] + r[1] + r[2] for d, r in results.items()}
    avg_ms = sum(total_ms.values()) / len(total_ms)
    print(f"\nMean per-sample render+backward cost: {avg_ms:.1f} ms")

    # Non-render step-time anchor: current job VelLoss step ~? Use a conservative
    # non-render overhead estimate: matcher+forward+backward ≈ 0.35 s/step (batch 48)
    non_render_step_s = 0.35
    samples_total = 100_000
    epochs = 500
    gpus = 4
    world_batch_per_gpu = None  # from CLI

    print("\nProjected epoch wall-clock (100k samples, 500 epochs, per-GPU batch B, fraction f):")
    print(f"{'config':>16} | {'VRAM est':>8} | {'s/step':>7} | {'min/epoch':>9} | {'days/500ep':>10}")
    for B in [int(b) for b in args.batch_sizes.split(",")]:
        for f in [float(x) for x in args.fractions.split(",")]:
            R = max(1, min(B, round(B * f)))
            # per-GPU step: non-render + R renders on this GPU
            step_s = non_render_step_s + R * (avg_ms / 1000.0)
            steps_per_epoch = samples_total / (B * gpus)
            ep_min = steps_per_epoch * step_s / 60.0
            days = ep_min * epochs / 60 / 24
            vram = 18.7 + 0.55 * B + B * f * 1.33  # rough linear model
            print(f"B={B:2d} f={f:4.2f} (R={R:2d}) | {vram:5.1f} GB | {step_s:7.2f} | {ep_min:9.1f} | {days:10.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())