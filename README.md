# image-to-l-system

**Single RGB Image → 3D Plant Architecture** via 26D Organ-Vector Flow Matching.

> **Active pipeline**: 232M DiT-Large Flow Matching model, 26D organ representation, 100K cowpea dataset (DAP 1–100).
> Training on 2×H100 via DDP on UC Davis Farm HPC.

---

## What This Repo Does

- **Input**: A single plant image (cowpea, DAP 1–100) + optional camera/sun metadata
- **Output**: A (N, 26) organ-typed plant array:
  ```
  [organ_type_one_hot (12), base_xyz (3), rotation_6D (6), scale_xyz (3), curvature (1), phyllotactic_angle (1)]
  ```
- **3D rendering**: Fully-differentiable GPU rasterization via nvdiffrast → RGB + Depth + Foreground Mask + Organ Map
- **XML round-trip**: Lossless conversion to/from Helios C++ XML (0.000000 mm vertex error)

---

## Repository Structure

```
image-to-l-system/
├── diffusion_based/                      # ★ Active AI & differentiable rendering pipeline
│   ├── models/
│   │   ├── canonical_cowpea_dit_large.py   # 232M DiT-Large (ViT-16L + Decoder-12L)
│   │   ├── plant_organ_array.py            # 40D Typed organ array & XML serialization bridge
│   │   ├── helios_pytorch_geometry.py      # 16D/26D → 3D mesh (vectorized GPU builder + FK bridge)
│   │   ├── helios_pytorch_renderer.py      # Multi-modal nvdiffrast GPU rasterizer
│   │   └── part_assembly_to_xml.py         # 16D Part assembly → Helios XML
│   ├── dataset/
│   │   ├── cowpea_shard_dataset.py         # Streaming shard loader + dynamic collation
│   │   ├── generate_tensor_shards.py       # XML → GPU render → 26D .pt shards
│   │   └── part_array_dataset.py           # 26D layout definition & normalization
│   ├── training/
│   │   ├── train_cowpea_dit_100k_ddp.py    # ★ 2×H100 DDP training (active)
│   │   └── train_cowpea_dit_100k.py        # Single-GPU fallback
│   ├── eval/
│   │   ├── test_render_part_tensor_quality.py # Direct 16D part renderer benchmark
│   │   ├── generate_part_report.py         # Multi-DAP quantitative eval & figures
│   │   ├── eval_cowpea_dit_100k.py         # Full lifespan benchmark
│   │   └── metrics.py                      # mSSIM, FG-IoU, Chamfer Distance
│   └── checkpoints/fm/                     # Model checkpoints
├── dataset/helios_data/
│   ├── cowpea/                             # 10K raw XMLs (100 seeds × 100 DAPs)
│   └── cowpea_shard/                       # 100K .pt tensor shards
├── slurm_scripts/
│   ├── train_cowpea_dit_h100_ddp.sh        # ★ DDP training launcher
│   └── generate_helios_dataset_jobs.sh     # Data generation pipeline
├── archive/                                # Consolidated legacy & historical modules (see archive/README.md)
│   ├── root_legacy/                        # Early renderer & 15D graph diffuser
│   ├── models_legacy/                      # 40D VAE & Track-A models
│   ├── training_legacy/                    # 40D FM, VAE & GRPO training scripts
│   ├── eval_legacy/                        # 40D & Track-A evaluation scripts
│   ├── eval_scripts/                       # One-off comparison scripts
│   ├── dataset_legacy/                     # Old dataset loaders
│   ├── notebooks_legacy/                   # Track-A benchmark notebooks
│   └── scratch/                            # Exploratory & debug snapshots
├── Digital-Crops/                          # Git submodule: Helios C++ simulation engine
├── docs/                                   # Project documentation (see docs/README.md)
└── environment.yml                         # Conda env spec
```

---

## Quick Start

### 1. Environment

```bash
conda env create -f environment.yml
conda activate digital-crops
```

### 2. Train 232M DiT-Large (2×H100 DDP)

```bash
sbatch slurm_scripts/train_cowpea_dit_h100_ddp.sh
```

### 3. Evaluate

```bash
python diffusion_based/eval/eval_cowpea_dit_100k.py
```

---

## Cluster Setup

| Item | Value |
|------|-------|
| **Cluster** | UC Davis Farm HPC (`farm.hpc.ucdavis.edu`) |
| **Training Node** | `gpu-10-58` (2× NVIDIA H100 SXM5) |
| **Python Env** | `/home/lion397/.conda/envs/digital-crops/bin/python` |
| **SLURM Account** | `lion397` / `publicgrp` / `geminigrp` |

---

## Key Checkpoints

| File | Model | Status |
|------|-------|--------|
| `fm/canonical_cowpea_dit_best.pt` | 73M DiT (60 epochs) | Baseline |
| `fm/cowpea_dit_large_2xh100_ddp.pt` | 232M DiT-Large | Training in progress |

---

## Minimal Direct Optimization Demo

A self-contained 3-organ example verifies that the differentiable PyTorch renderer
and geometry builder can directly optimize a continuous botanical parameter via
a 3D Chamfer loss:

```bash
python scripts/minimal_direct_opt_depth_chamfer_demo.py
```

- **Template**: one internode + one petiole + one leaf
- **Task**: optimize only `petiole_pitch` from 10° to the target 60°
- **Supervision**: 3D vertex Chamfer distance alone (no RGB/depth losses)
- **Result**: Chamfer distance drops from ~303 mm to ~0.07 mm in 200 steps
- **Output figure**: [`docs/results/assets/minimal_direct_opt_depth_chamfer_demo.png`](docs/results/assets/minimal_direct_opt_depth_chamfer_demo.png)

---

## Documentation

See [`docs/README.md`](docs/README.md) for detailed project documentation including:
- Active handoff documents
- Completed milestone records
- Benchmark results and figures

---

## Legacy Modules

Superseded code from earlier representations (5D, 14D, 15D, 40D) is preserved in `*/legacy/` subdirectories for reference. See `legacy/` and `diffusion_based/*/legacy/`.
