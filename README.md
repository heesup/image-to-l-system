# image-to-l-system

**Single RGB-D Image → 3D Plant Architecture** via conditional Flow Matching + differentiable rendering.

> **Active pipeline**: inverse 3D botanical reconstruction — 14D Part Tensor representation,
> concentric multi-scale differentiable renderer, Helios C++ OptiX XML round-trip.
> See [`docs/ongoing/AGENT_TAKEOVER_GUIDE.md`](docs/ongoing/AGENT_TAKEOVER_GUIDE.md) for the full
> state, math, and benchmark results.

---

## What This Repo Does

- **Input**: A single monocular top-view RGB-D image (cowpea seedling, 4×256×256)
- **Output**:
  1. A (N, 14) Canonical Part Tensor:
     `[organ_type(1), base_xyz(3), rot6d(6), scale_xyz(3), curvature(1)]`
  2. A native Helios C++ XML tree that compiles and raytraces in the physical engine
- **Rendering**: fully-differentiable nvdiffrast rasterization (RGB + CHM depth + soft-existence alpha)
- **Benchmarks (2026-09-04)**: direct render optimization **92% Mask IoU / 0.9mm Chamfer** (seedling),
  full-lifecycle Helios raytrace 95.1 / 92.9 / 86.5% IoU (DAP 10/50/90)

---

## Repository Structure (reorganized 2026-09-04 — only active code remains)

```
image-to-l-system/
├── scratch/                              # ★ Active experiment scripts + outputs (git-ignored)
│   ├── phase2_core.py                    # Shared anti-erasure loss core (used by exp2/4/5/7)
│   ├── make_target_unifoliate.py         # Canonical 7-row GT target generator
│   ├── exp1_per_organ_icp.py             # Method 1: Point Cloud ICP benchmark
│   ├── exp2_diff_render_opt.py           # Method 2: Differentiable renderer benchmark
│   ├── exp3b_flow_matching_fixed.py      # Method 3b: fixed Flow Matching (hybrid polish)
│   ├── exp4_overalloc_pruning.py         # Phase 2 Strategy B: existence pruning
│   ├── exp5_underalloc_spawn.py          # Phase 2 Strategy A: residual-driven spawning
│   ├── exp6_dimension_coverage.py        # 26D per-dimension differentiability audit
│   ├── exp7_dimension_recovery.py        # Group-wise perturb-recover test
│   └── eval_phase1_comparison.py         # Synthesis comparison (Figure 12)
├── tests/unit/                           # Reusable verification scripts
│   ├── test_14d_curvature.py             # 14D↔40D roundtrip w/ curvature
│   ├── test_multiscale_pyramid.py        # Pyramid renderer verification
│   └── ...                               # reproductive/render/soft-existence checks
├── diffusion_based/                      # ★ Active AI & differentiable rendering pipeline
│   ├── models/                           # 6 active modules:
│   │   ├── plant_organ_array.py          # 14D/40D constants + XML round-trip
│   │   ├── helios_pytorch_geometry.py    # 26D → 3D mesh (differentiable)
│   │   ├── helios_pytorch_renderer.py    # nvdiffrast multi-scale rasterizer
│   │   ├── part_tensor_to_40d.py         # Closed-form IK + XML assembler
│   │   ├── vit_image_encoder.py          # FM conditioning encoder
│   │   └── part_flow_matching.py         # FM denoiser model
│   ├── training/
│   │   ├── flow_matching.py              # Rectified Flow scheduler
│   │   └── train_part_flow_matching.py   # FM training loop
│   ├── dataset/
│   │   ├── part_array_dataset.py         # 25D FM encode/decode (BASE_SCALE=20)
│   │   ├── cowpea_shard_dataset.py       # Shard dataset loader
│   │   └── generate_tensor_shards.py     # XML → GPU render → .pt shards
│   ├── eval/
│   │   ├── eval_13d_xml_organ_masks.py   # Helios C++ raytrace benchmark (Figure 10)
│   │   └── metrics.py                    # mSSIM, FG-IoU, Chamfer
│   └── checkpoints/fm/part_flow_matching.pt  # Final FM checkpoint (only one kept)
├── dataset/helios_data/
│   ├── cowpea/                           # Raw XMLs (100 seeds × 100 DAPs)
│   └── cowpea_shard/                     # ~50K .pt tensor shards
├── archive/                              # Legacy code (git-tracked, NOT active)
│   ├── models_legacy/                    # Archived models (DIT, VAE, VLM scaffold, ...)
│   ├── training_legacy/  eval_scripts/   # Archived training/eval scripts
│   ├── dataset_scripts/                  # Archived dataset loaders
│   └── ... (dataset_legacy, notebooks_legacy, root_legacy, ...)
├── Digital-Crops/                        # Helios C++ OptiX simulation engine (submodule)
├── docs/
│   ├── ongoing/AGENT_TAKEOVER_GUIDE.md   # ★ Single source of truth — read this first
│   └── results/assets/                   # All benchmark figures
└── scripts/                              # Figure-generation & data utilities
```

---

## Quick Start

### 1. Environment

```bash
mamba env create -f environment.yml
mamba activate digital-crops
export PYTHONPATH=.   # REQUIRED from repo root
```

### 2. Run the benchmark suite

```bash
# Canonical target generation (7-row unifoliate seedling)
python scratch/make_target_unifoliate.py

# Method benchmarks
python scratch/exp1_per_organ_icp.py          # ICP
python scratch/exp2_diff_render_opt.py        # Diff renderer (SOTA: ~92% IoU)
python scratch/exp3b_flow_matching_fixed.py   # FM hybrid (~80% IoU)
python scratch/exp4_overalloc_pruning.py      # Topology: over-allocation pruning
python scratch/exp5_underalloc_spawn.py       # Topology: under-allocation spawning

# Audits
python scratch/exp6_dimension_coverage.py     # All 26 dims optimizable?
python scratch/exp7_dimension_recovery.py     # Per-group recovery test

# Synthesis figure + Helios raytrace benchmark
python scratch/eval_phase1_comparison.py
python diffusion_based/eval/eval_13d_xml_organ_masks.py
```

### 3. Train the Flow Matching model

```bash
python diffusion_based/training/train_part_flow_matching.py --help
```

---

## Key Checkpoints

| File | Model | Status |
|------|-------|--------|
| `diffusion_based/checkpoints/fm/part_flow_matching.pt` | Part Flow Matching (26D nodes) | Final |

---

## Documentation

- [`docs/ongoing/AGENT_TAKEOVER_GUIDE.md`](docs/ongoing/AGENT_TAKEOVER_GUIDE.md) — active handover doc: golden rules, gotchas, benchmark tables, roadmap
- [`docs/results/`](docs/results/) — milestone reports & benchmark assets
- [`archive/README.md`](archive/README.md) — legacy module index

## Cleanup Log (2026-09-04)

~93 GB freed; repository reduced to active code only:
- Archived: 9 model files, 6 training scripts, 22 eval scripts, 4 dataset scripts, 4 figure scripts
- Deleted: 60 intermediate FM epoch checkpoints + stale DIT/VLM/VAE checkpoints (73 GB), stale shard dataset (19 GB), `wandb/`, `logs/`, `training_logs/`, `agent_temp/`
- Moved: 9 `scratch/test_*` verification scripts → `tests/unit/`