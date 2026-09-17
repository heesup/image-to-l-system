# image-to-l-system

**Single RGB-D Image → 3D Plant Architecture** via cascaded Flow Matching + differentiable rendering.

> **Active pipeline**: inverse 3D botanical reconstruction of cowpea seedlings — hierarchical
> phytomer-level Flow Matching (DINOv2 scaffold → VAE-latent decoder), a multi-scale
> differentiable renderer, and a Helios C++ OptiX XML round-trip for physical validation.
> See [`docs/handovers/agent-takeover-guide/agent-takeover-guide.md`](docs/handovers/agent-takeover-guide/agent-takeover-guide.md) for the
> full state, math, active jobs, and benchmark results — it is the single source of truth and
> is updated far more often than this README.

---

## What This Repo Does

- **Input**: a single monocular top-view RGB-D image (cowpea, 4×256×256 — RGB + Canopy Height Model depth)
- **Output**:
  1. Per-phytomer 14D organ tensors (`[organ_type, base_xyz(3), rot6d(6), scale_xyz(3), curvature]`),
     assembled in strict `[Internode | Petiole | Leaflets×3 | Peduncle | Repro×4]` slot order
  2. A native Helios C++ XML tree that compiles and raytraces in the physical simulation engine
- **Rendering**: fully-differentiable `nvdiffrast` rasterization (RGB + CHM depth + soft-existence
  alpha), with a multi-scale pyramid loss
- **Helios round-trip benchmarks** (foreground IoU against the Helios raytrace, 2026-09-14):
  exact_gt plants DAP 10/50/90 (`eval_phytomer_vae_helios_roundtrip.py`, fig14): IK-only
  99.9 / 99.6 / 97.7%, through the PhytomerVAE 93.7 / 98.3 / 95.8%; dataset plants DAP 15/40/75
  (`eval_phytomer_10slot_assembly_views.py`, fig12): packet path 98.3 / 98.2 / 95.6%, through the
  VAE 95.1 / 96.8 / 95.6%. The export runs a stem and a leaf inverse-kinematics pass so Helios's
  own forward kinematics lands on the predicted nodes and leaf rotations.

### Architecture: Hybrid-Decoupled 3-Stage Cascaded Flow Matching

```
RGB-D image (4×256×256)
   │
   ▼
[Stage 0] DINOv2 ViT-S/14 + PETR 3D ray PE  → 3D-aware image tokens (diffusion_based/models/dinov2_ray_encoder.py)
   │
   ▼
[Stage 1] MacroBiologicalHead (CLS token)   → phytomer count + soft-margin existence prior
   │
   ▼
[Stage 2] CoarseSkeletalTransformer          → per-node pos(3D)/rot(6D)/scale(3D)/existence
          (deterministic set transformer, "botanical scaffold")
   │
   ▼
[Stage 3] PhytomerFlowMatchingDecoder        → Rectified Flow over the 128D PhytomerVAE latent only
          (conditioned on the detached Stage-2 scaffold, the node's fixed parent, and image tokens; z_0 ~ N(0, I_128))
   │
   ▼
[Stage 4] PhytomerVAE v9 (frozen)            → 128D hybrid latent (48 coarse + 10×8 per-slot residual) → 10-slot canonical 14D organ packet
   │
   ▼
HeliosPyTorchRenderer                        → multi-scale CHM depth + soft-Dice silhouette loss
```

`diffusion_based/models/hierarchical_part_flow_matching.py` implements Stages 1–3;
`diffusion_based/models/phytomer_vae.py` implements Stage 4;
`diffusion_based/training/hierarchical_hungarian_matcher.py` does node↔GT bipartite matching.

A separate, earlier-generation **direct 26D-node Flow Matching** track
(`diffusion_based/models/part_flow_matching.py`, `training/train_part_flow_matching.py`) still
works and is useful as a simpler baseline, but the hierarchical/phytomer pipeline above is the
one under active development.

There is also an independent **VLM/language-model track** in [`lm_based/`](lm_based/README.md)
that predicts L-system grammar tokens directly from a rendered 2D image, trained with SFT + a
render-in-the-loop RL stage. It does not share code with the diffusion pipeline.

---

## Repository Structure

```
image-to-l-system/
├── diffusion_based/                      # ★ Active AI + differentiable rendering pipeline
│   ├── models/
│   │   ├── dinov2_ray_encoder.py         # Stage 0: DINOv2 + PETR 3D ray positional encoding
│   │   ├── hierarchical_part_flow_matching.py  # [CRITICAL] Stages 1-3: scaffold + phytomer-latent FM
│   │   ├── botanical_scaffold.py         # Coarse 3D skeletal transformer building blocks
│   │   ├── phytomer_vae.py               # Stage 4: PhytomerVAE (10-slot packets in / 128D hybrid latent)
│   │   ├── organ_latent_vae.py           # Earlier per-organ latent VAE (16D/organ), frozen bridge
│   │   ├── plant_organ_array.py          # 14D/40D constants, organ types, XML round-trip
│   │   ├── helios_pytorch_geometry.py    # 26D/40D → differentiable 3D mesh builder
│   │   ├── helios_pytorch_renderer.py    # nvdiffrast multi-scale pyramid rasterizer
│   │   ├── part_tensor_to_40d.py         # Analytical 14D → 40D converter + Helios XML assembler
│   │   ├── part_tensor_stem_ik.py        # Stem inverse kinematics: Helios's FK lands on the predicted nodes
│   │   ├── part_tensor_leaf_ik.py        # Leaf orientation inverse: FK reproduces each leaf's 14D rotation
│   │   ├── vit_image_encoder.py          # Conditioning encoder for the direct 26D-node FM track
│   │   └── part_flow_matching.py         # Direct 26D-node FM denoiser (earlier-gen baseline)
│   ├── training/
│   │   ├── train_hierarchical_flow_matching.py  # [CRITICAL] main loop for the 3-stage pipeline
│   │   ├── train_phytomer_vae.py         # PhytomerVAE trainer (canonical packets)
│   │   ├── hierarchical_hungarian_matcher.py  # coarse-node + local bipartite matching
│   │   ├── flow_matching.py              # Rectified Flow scheduler
│   │   └── train_part_flow_matching.py   # trainer for the direct 26D-node FM baseline
│   ├── dataset/
│   │   ├── part_array_dataset.py         # [CRITICAL] cache-first dataset, pkt-cache version gate
│   │   ├── phytomer_packets.py           # ordered 10-slot phytomer packet assembly + shoot-structured emitter
│   │   ├── phytomer_topology.py          # gt_parent_links (training targets) and chain_phytomers (node cloud → shoots)
│   │   ├── phytomer_roll.py              # forward axis from the chain, 1-DOF roll encoding
│   │   ├── generate_cache.py             # XML → GPU render → .pt cache + phytomer packets
│   │   └── dap_bucket_sampler.py         # DAP-stratified batch sampler
│   ├── eval/
│   │   ├── eval_hierarchical_self_consistency.py  # DAP-spread diagnostic panel generator
│   │   ├── eval_13d_xml_organ_masks.py   # per-organ COCO mask IoU + depth PSNR roundtrip
│   │   ├── eval_phytomer_vae_helios_roundtrip.py   # fig14: IK-only vs VAE round-trip in Helios (exact_gt plants)
│   │   ├── eval_phytomer_10slot_assembly_views.py  # fig12: GT vs assembly (nadir, 45°) + Helios round-trips (dataset plants)
│   │   ├── benchmark_helios_vs_torch_renderer.py
│   │   ├── benchmark_organ_vae_roundtrip.py
│   │   └── metrics.py                    # mSSIM, FG-IoU, Chamfer
│   └── checkpoints/                      # git-ignored; see "Key Checkpoints" below
├── lm_based/                              # Independent VLM/grammar-token track — see its own README
├── tools/                                 # Standalone utilities (phytomer VAE latent visualizer GUI, ...)
├── tests/                                 # pytest suite (hierarchical FM, phytomer VAE/packets, gradient flow)
│   └── unit/                              # lower-level roundtrip/geometry verification scripts
├── scratch/                               # Working/experimental scripts + outputs (git-ignored)
├── dataset/
│   ├── helios_data/cowpea/               # Raw Helios XMLs (100 seeds × 100 DAPs), git-ignored
│   └── cache/                            # Generated .pt tensor + phytomer-packet caches, git-ignored
├── scripts/generate_helios_dataset.py     # Helios XML synthesis entry point
├── slurm_scripts/                         # Cluster launchers (train_hierarchical_flow_matching.sh, ...)
├── archive/                               # Legacy/superseded code, kept for lineage — see archive/README.md
├── Digital-Crops/                         # Helios C++ OptiX simulation engine (git submodule)
└── docs/
    ├── _index.md                          # Map of Content (MOC) — topic-based directory index
    ├── handovers/current-status.md        # Live status dashboard (active jobs, next steps)
    ├── handovers/agent-takeover-guide/    # ★ Single source of truth — read this first
    ├── architecture/                      # System design, specs & camera geometry reference
    ├── experiments/                       # Training runs, benchmark reports & local assets
    ├── engineering/                       # Implementation sessions, refactors & PR records
    ├── planning/                          # Roadmaps & upcoming milestone plans
    └── archive/                           # Superseded documents & unreferenced asset backups
```

---

## Quick Start

### 1. Environment

```bash
git submodule update --init --recursive   # pulls in Digital-Crops (Helios C++/OptiX)
mamba env create -f environment.yml
mamba activate digital-crops
export PYTHONPATH=.   # REQUIRED — always run commands from the repo root
```

### 2. Generate / check the dataset

```bash
python scripts/generate_helios_dataset.py --help       # Helios XML synthesis
python diffusion_based/dataset/generate_cache.py --help  # XML -> .pt cache + phytomer packets
```

### 3. Train

```bash
# PhytomerVAE (train first — Stage 3 decodes into its latent space, frozen).
# Cached packets no longer store latents: the FM trainer encodes Stage-3 target
# latents on the fly with the VAE it loads (2026-09-14), so a new VAE does NOT
# need a packet-cache rebuild. Retrain the VAE only when the packet format changes.
python diffusion_based/training/train_phytomer_vae.py --help

# Stages 1-3: hierarchical scaffold + phytomer-latent flow matching.
python diffusion_based/training/train_hierarchical_flow_matching.py --help
sbatch slurm_scripts/train_hierarchical_flow_matching.sh
# One launcher trains the VAE first in the same allocation, then FM with it
# (the standalone train_phytomer_vae.sh is archived under archive/slurm_scripts/):
TRAIN_VAE=1 sbatch slurm_scripts/train_hierarchical_flow_matching.sh
```

### 4. Evaluate

```bash
python diffusion_based/eval/eval_hierarchical_self_consistency.py --help
python diffusion_based/eval/eval_13d_xml_organ_masks.py --help   # Helios raytrace + per-organ IoU
python diffusion_based/eval/eval_phytomer_vae_helios_roundtrip.py   # fig14 (exact_gt DAP 10/50/90)
python diffusion_based/eval/eval_phytomer_10slot_assembly_views.py  # fig12 (dataset DAP 15/40/75, GT + assembly at nadir and 45°)
```

### 5. Tests

```bash
pytest tests/
```

---

## Key Checkpoints

Checkpoints are git-ignored (`diffusion_based/checkpoints/`) and live on disk on the cluster.

| Directory | Model | Status |
|---|---|---|
| `diffusion_based/checkpoints/phytomer_vae_v9_tl_rw4_20k/` | PhytomerVAE 128D hybrid (48 + 10×8), terminal-last packets, 20k files | **Accepted default** (export VAE and the v9 FM run's VAE; `PHYTOMER_TERMINAL_LAST=1`) |
| `diffusion_based/checkpoints/phytomer_vae_v8/` | PhytomerVAE 128D hybrid, bottom-to-top packets | VAE of every FM run before v9 (`cowpea_curv26_pkt/`, pkt `6`, `PHYTOMER_TERMINAL_LAST=0`) |
| `diffusion_based/checkpoints/hierarchical_fm_v9_local2/` | 3-stage cascaded FM, v9 lineage (final-norm decoder, fp32 self-attention) | Active training — see `docs/handovers/current-status.md` |
| `diffusion_based/checkpoints/organ_vae/organ_latent_vae_best.pt` | Frozen per-organ latent VAE bridge (earlier design) | Kept for lineage |

Checkpoint naming/size is the fastest way to tell architectures apart — see §4 of
[`AGENT_TAKEOVER_GUIDE.md`](docs/handovers/agent-takeover-guide/agent-takeover-guide.md) before loading one, since
incompatible architectures have been saved under the same directory during migrations.

---

## Documentation

- [`docs/_index.md`](docs/_index.md) — Map of Content (MOC) indexing all topics, roadmaps, and experiments
- [`docs/handovers/agent-takeover-guide/agent-takeover-guide.md`](docs/handovers/agent-takeover-guide/agent-takeover-guide.md) — master handover:
  full system state, active SLURM jobs, failed-launch forensics, next steps, gotchas
- [`docs/handovers/current-status.md`](docs/handovers/current-status.md) — live status dashboard
- [`docs/handovers/20260912-stage2-stage3-boundary/20260912-stage2-stage3-boundary.md`](docs/handovers/20260912-stage2-stage3-boundary/20260912-stage2-stage3-boundary.md)
  — engineering record: status, Stage 2 gradient-burst root cause, Stage 2/3 boundary redesign, Helios round-trip and inverse kinematics passes
- [`docs/experiments/`](docs/experiments/) — milestone reports & benchmark figures with co-located image assets
- [`docs/engineering/`](docs/engineering/) — implementation session handoffs & PR records
- [`archive/README.md`](archive/README.md) — legacy module index and architectural evolution timeline
- [`lm_based/README.md`](lm_based/README.md) — the independent VLM/grammar-token track
