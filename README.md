# image-to-l-system

3D plant architecture reconstruction from a single RGB image using **vision-conditioned graph diffusion**.

> **Active pipeline (15D organ-typed graph diffusion)** is in `diffusion_based/`. The legacy 2D L-system / VLM modules (`lm_based/`, top-level `dataset/`) are archived and not currently maintained.

---

## What this repo does

- **Input**: a single plant image (e.g., cowpea seedling) + optional camera/sun metadata.
- **Output**: a 15D organ-typed plant graph:
  - `[x, y, z, length, radius, pitch, yaw, roll, organ_type_one_hot (4), shoot_id, phytomer_idx, existence]`
- **3D proxy geometry + loss**: a fully-differentiable 3D point-cloud generator replicates Helios organ geometry from the 15D graph, and a Chamfer 3D loss compares it against a real Gaussian-splat PLY. The legacy 2D render loss is retained only for reference.

---

## Repository Structure

```
image-to-l-system/
├── dataset/                          # Helios XML / image dataset parser (15D)
│   ├── helios_dataset.py
│   └── helios_xml_parser.py
├── diffusion_based/
│   ├── models/
│   │   ├── graph_diffuser_3d.py         # 15D vision-conditioned graph diffuser (k-NN attention)
│   │   ├── differentiable_renderer_3d.py  # Helios-matching differentiable 2D renderer
│   │   ├── plant_geometry_3d.py         # Differentiable XML/15D → 3D point-cloud sampler
│   │   └── pointcloud_loss_3d.py        # Chamfer 3D point-cloud loss + PLY loader
│   ├── training/
│   │   └── train_diffusion_3d.py        # Training loop with optional 3D Chamfer or 2D render loss
│   └── eval/
│       ├── visualize_diffusion_3d.py
│       └── compare_xml_helios_3d.py     # Quantitative Helios-vs-XML geometry comparison
├── Digital-Crops/                    # Git submodule: Helios synthetic data generator
├── environment.yml                   # Conda env spec
└── README.md                         # This file
```

---

## Local Development on macOS (Apple Silicon / Intel)

The pure-PyTorch training/rendering code runs on CPU or MPS. Helios data generation itself is Linux/GPU-only, so for local Mac development you should **generate data on a Linux machine first** (or use a pre-generated dataset), then copy `Digital-Crops/projects/syntheticdata_generation/build/output/` into the same path on your Mac.

### 1. Create environment

```bash
conda env create -f environment.yml
conda activate l-system
```

The environment intentionally uses `pytorch` / `torchvision` from the `pytorch` channel. On macOS these install CPU/MPS builds automatically.

### 2. Quick render test (no Helios build needed)

```bash
python -c "
import torch
from diffusion_based.models.helios_xml_parser import HeliosXMLParser
from diffusion_based.models.differentiable_renderer_3d import DifferentiablePlantRenderer3D

xml = 'Digital-Crops/projects/syntheticdata_generation/build/output/dap10_gt_0000_plant_0000.xml'
nodes = HeliosXMLParser(xml).get_all_organ_nodes()
# ... see diffusion_based/models/differentiable_renderer_3d.py for full tensor construction
"
```

### 3. Train on a pre-generated Helios dataset

```bash
python diffusion_based/training/train_diffusion_3d.py \
    --data-dir Digital-Crops/projects/syntheticdata_generation/build/output \
    --target-ply data/gaussian_splat/2025-06-17-bed1tier2plant1.ply \
    --epochs 200 \
    --batch-size 2 \
    --pc-loss 1.0 \
    --pc-samples 2048 \
    --save-path diffusion_based/checkpoints/diffusion_3d_15d.pt
```

On a MacBook use `--batch-size 1` or CPU rendering; the renderer is chunked but still memory-intensive.

---

## Linux GPU Training / Data Generation

Helios data generation and large-batch training should be done on a CUDA machine.

```bash
# Inside Digital-Crops (see its README for build prerequisites)
cd Digital-Crops/projects/syntheticdata_generation
./build.sh              # or cmake + make
./syntheticdata_generation

# Training on the generated output
python diffusion_based/training/train_diffusion_3d.py \
    --data-dir Digital-Crops/projects/syntheticdata_generation/build/output \
    --epochs 500 \
    --batch-size 4 \
    --render-loss 1.0
```

---

## Current Status & Next Steps

- [x] 15D organ-typed graph diffuser (`graph_diffuser_3d.py`)
- [x] Helios-matching differentiable 2D renderer (`differentiable_renderer_3d.py`)
- [x] Helios plant-only 3D PLY export (`main.cpp` `--export-3d ply`)
- [x] Differentiable XML/15D → 3D point-cloud sampler (`plant_geometry_3d.py`)
- [x] Chamfer 3D point-cloud loss against real Gaussian-splat PLY (`pointcloud_loss_3d.py`)
- [x] End-to-end training with 3D Chamfer loss (`train_diffusion_3d.py`)
- [x] Helios-vs-XML geometry comparison (`compare_xml_helios_3d.py`)
- [ ] Improve leaf/trifoliate geometry proxy to drive Chamfer < 0.05 (currently ~0.065 on `cowpea_test_0000`)
- [ ] Run overfitting experiment on real PLY to verify gradient flow
- [ ] Evaluation / sampling from noise
- [ ] Scale to multi-DAP datasets

---

## Legacy Modules & Archived Files

The following directories and files are archived under **`legacy/`** for historical reference:

- **`legacy/graph_diffuser.py`**: Legacy 2D graph diffusion model (superseded by `graph_diffuser_3d.py`).
- **`legacy/differentiable_renderer.py`**: Early 2D line soft-rasterizer prototype (superseded by `helios_rasterizer_3d.py`).
- **`legacy/helios_geometry_legacy_15d.py`**: Legacy 15D node approximation renderer (superseded by `DifferentiableHeliosXMLRenderer` and `HeliosPlantGeometryTorch`).
- **`legacy/verify_*.py`**: One-off verification scripts for early 15D renderer prototypes.
- `lm_based/` & `l-systems-gnn/`: Legacy 2D text/L-System language model experiments.
