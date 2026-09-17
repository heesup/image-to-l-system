---
title: "3D Plant Geometry + Render Pipeline Progress"
date: 2026-08-14
tags: [engineering, implementation]
status: done
---

# 3D Plant Geometry + Render Pipeline Progress

> Last edited: 2026-08-04 (macOS)  
> Commit: `b9fd2cd` — single Helios-style geometry+render pipeline with focus-plant HFOV and shading

## 1. Completed Work

### 1.1 Establish Unified Geometry Pipeline
- Added `plant_recon/models/helios_geometry.py`
  - Helios XML → 3D tube/leaflet/ellipsoid geometry reconstruction
  - Fixed internode/petiole tube misclassification bug
  - Added leaf mesh sampling (triangle center/midpoints) → included leaf organs in point cloud
  - Differentiable point cloud sampler: 15D nodes → point cloud for Chamfer calculation
  - `nodes_to_geometry()`: 15D nodes → `HeliosTube`/`HeliosLeaflet`/`HeliosEllipsoid` (for rendering)
- Deleted legacy `plant_geometry_3d.py`, `differentiable_renderer_3d.py`

### 1.2 Establish Unified 2D Rasterizer
- Added `plant_recon/models/helios_rasterizer_3d.py`
  - Matches Helios `Context` camera model projection
  - Supports `--focus-plant`: `recompute_focus_plant_hfov()` — XY bbox + 5% margin, `2*atan(span/(2*h))`
  - Added **area normalization** to soft triangle rasterization → clean subpixel triangles without filling entire screen
  - Added simple diffuse shading to tubes/leaves (double-sided leaves)

### 1.3 3D Chamfer Loss
- Added `plant_recon/models/pointcloud_loss_3d.py`
  - `PlantPointCloudChamferLoss`: 15D nodes → point cloud → Chamfer distance
  - Organ-aware weighted Chamfer loss
  - Includes PLY load/write/Chamfer/normalize utilities

### 1.4 Helios Dataset Generator Improvements
- `plant_recon/dataset/generate_helios_dataset.py`
  - Removed macOS offscreen env override (XQuartz dependency)
  - Added `--export-3d ply` support

### 1.5 Training Script Integration
- `plant_recon/training/train_diffusion_3d.py`
  - Removed legacy renderer → uses `HeliosGeometryRasterizer`
  - Converts 15D predicted nodes via `nodes_to_geometry()` for batch rendering
  - Integrated `PlantPointCloudChamferLoss`

## 2. Verification Results

### 2.1 5-Sample Helios Chamfer (DAP 5, seed 0~4, single view)
| seed | Chamfer (target-normalized) |
|------|-----------------------------|
| 00   | 0.0389                      |
| 01   | 0.0439                      |
| 02   | 0.0384                      |
| 03   | 0.0399                      |
| 04   | 0.0392                      |
| Mean | **0.0401**                  |

### 2.2 Render Comparison
- Saved to `/tmp/helios_val5/cowpea_dap005_seed*_compare.png` (left: Helios `_vis.jpeg`, right: Python rasterizer)
- Organ structure and leaf/stem arrangement align closely with Helios imagery

## 3. Next Steps

### 3.1 GPU / Memory Optimization (High Priority)
- Currently on Mac MPS, `torch.cdist(pred, target)` requests 18 GB buffer → Out of Memory
  - Cause: point cloud has 124 points per 15D node × 256 nodes ≈ 31K points; batch size 2 requires large cdist buffer
  - Mitigation strategies:
    1. Lower `pc_loss` sampler resolution (`n_cylinder_circ=3, n_cylinder_axis=2, n_leaf_u=3, n_leaf_v=4`)
    2. Subsample target point cloud to smaller size (64–128)
    3. Mini-batch Chamfer evaluation (chunked `torch.cdist` over targets)
    4. Compute squared Euclidean distances after `F.normalize` to save memory

### 3.2 2D Render Loss Training Verification
- Run 1-epoch smoke training with `--render-loss > 0`
- Confirm predicted 15D nodes pass through `nodes_to_geometry()` to generate renders resembling Helios `_vis.jpeg`
- Note: `nodes_to_geometry()` currently generates quad leaves differing from Helios trifoliate shape → improve geometry later

### 3.3 Leaf Geometry Refinement (Long-term)
- Adopt OBJ prototypes and transformation chains matching Helios C++ `CowpeaLeafPrototype_trifoliate_OBJ`
- Resolve issue where only 1 leaf is written to XML (restore `leaves_per_petiole=3` in phytomer parameters)

### 3.4 Dataset Expansion
- 5 validation samples are smoke tests only; scale dataset to DAP 5–30 with varying viewpoints and solar angles
- Run `plant_recon/dataset/generate_helios_dataset.py` with `--workers 1 --renderer vis --export-3d ply`

### 3.5 Training Experiments
- Experiment with render-loss only, point-cloud loss only, and joint supervision
- Track convergence and visualizations
- Tune `render_loss_weight` and `pc_loss_weight` hyperparameters in `train_diffusion_3d.py`

## 4. Key Files & Commands

```bash
# Regenerate 5 validation samples
source /Users/lion397/homebrew/Caskroom/miniforge/base/bin/activate l-system
python plant_recon/dataset/generate_helios_dataset.py \
  --dap-start 5 --dap-end 5 --dap-step 5 --seeds 5 \
  --renderer vis --workers 1 --export-3d ply \
  --output-dir /tmp/helios_val5 \
  --main-binary submodules/Digital-Crops/projects/syntheticdata_generation/build/main

# Chamfer + render comparison
python plant_recon/eval/compare_xml_helios_3d.py \
  --helios-ply /tmp/helios_val5/..._helios.ply \
  --xml /tmp/helios_val5/..._plant_0000.xml \
  --visualize --visualize-path /tmp/compare.png

# Training (render loss)
python plant_recon/training/train_diffusion_3d.py \
  --data-dir /tmp/helios_val5 \
  --epochs 2 --batch-size 2 --render-loss 1.0

# Training (point-cloud loss, with memory optimizations)
python plant_recon/training/train_diffusion_3d.py \
  --data-dir /tmp/helios_val5 \
  --epochs 2 --batch-size 2 --pc-loss 1.0 \
  --target-ply /tmp/helios_val5/cowpea_dap005_seed00_..._helios.ply \
  --pc-samples 64
```

## 5. Known Issues

- Torchvision image extension/libjpeg dylib warning on macOS: falls back gracefully to PIL
- MPS point-cloud Chamfer memory pressure: test `pc_loss` on larger GPU/CPU instances first
- `nodes_to_geometry()` leaf quad shape differs visually from Helios trifoliate meshes
