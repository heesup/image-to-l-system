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
├── diffusion_based/                      # ★ Active pipeline
│   ├── models/
│   │   ├── canonical_cowpea_dit_large.py   # 232M DiT-Large (ViT-16L + Decoder-12L)
│   │   ├── plant_organ_array.py            # 26D organ array core data structure
│   │   ├── helios_pytorch_geometry.py      # 26D → 3D mesh (kinematics, curvature)
│   │   ├── helios_pytorch_renderer.py      # nvdiffrast GPU rasterizer
│   │   ├── helios_xml_parser.py            # Helios XML → PlantOrganArray
│   │   └── part_assembly_to_xml.py         # PlantOrganArray → Helios XML
│   ├── dataset/
│   │   ├── cowpea_shard_dataset.py         # Streaming shard loader + dynamic collation
│   │   ├── generate_tensor_shards.py       # XML → GPU render → 26D .pt shards
│   │   └── part_array_dataset.py           # 26D layout definition & normalization
│   ├── training/
│   │   ├── train_cowpea_dit_100k_ddp.py    # ★ 2×H100 DDP training (active)
│   │   └── train_cowpea_dit_100k.py        # Single-GPU fallback
│   ├── eval/
│   │   ├── eval_cowpea_dit_100k.py          # Full lifespan benchmark
│   │   └── metrics.py                       # mSSIM, FG-IoU, Chamfer Distance
│   └── checkpoints/fm/                      # Model checkpoints
├── dataset/helios_data/
│   ├── cowpea/                              # 10K raw XMLs (100 seeds × 100 DAPs)
│   └── cowpea_shard/                        # 100K .pt tensor shards
├── slurm_scripts/
│   ├── train_cowpea_dit_h100_ddp.sh         # ★ DDP training launcher
│   └── generate_helios_dataset_jobs.sh      # Data generation pipeline
├── Digital-Crops/                           # Git submodule: Helios C++ simulation engine
├── docs/                                    # Project documentation (see docs/README.md)
├── legacy/                                  # Archived legacy modules
└── environment.yml                          # Conda env spec
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

## Documentation

See [`docs/README.md`](docs/README.md) for detailed project documentation including:
- Active handoff documents
- Completed milestone records
- Benchmark results and figures

---

## Legacy Modules

Superseded code from earlier representations (5D, 14D, 15D, 40D) is preserved in `*/legacy/` subdirectories for reference. See `legacy/` and `diffusion_based/*/legacy/`.
