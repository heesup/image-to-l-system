# Helios PyTorch Differentiable Renderer Handover & Architecture Specification

**Date**: 2026-08-11  
**Repository**: `/home/lion397/codes/image-to-l-system`  
**Primary Contact / Target Agent**: Next Agent taking over Helios PyTorch Differentiable Renderer & Diffusion Integration.

---

## 1. Executive Summary

This document provides a comprehensive handover of the **Helios PyTorch Differentiable Renderer**, the **25D Plant Organ Node Array Schema**, the **Dual-Track Renderer Architecture**, XML ↔ organ-array round-trip serialization, and exact actionable tasks for the next agent.

During recent benchmarking against C++ Helios Ground Truth (GT) across DAP 10, DAP 50, and DAP 90:
1. We expanded the Plant Organ Feature Vector layout from **22D to 25D**. The 22D layout stored a single `direction` vector plus `pitch`/`yaw`/`roll` angles, which was insufficient to reproduce the full C++ leaflet orientation chain. The new 25D layout stores a full **3×3 local-to-world orientation matrix** for every organ node.
2. We established a **Dual-Track Renderer Architecture** to bridge the gap between High-Fidelity GT Evaluation and Continuous Tensor 3D Graph Diffusion.
3. We implemented **lossless XML ↔ organ-node-array round-trip** via `organ_nodes_to_xml()` and `verify_xml_round_trip()`.

### Latest benchmark results (skip C++ GT, Track A XML-native vs Track B 25D node array)

| DAP | Track A vs C++ GT MAE | Track B vs C++ GT MAE | Track A–B MAE |
|-----|----------------------:|----------------------:|--------------:|
| 10  | 0.12079               | 0.10573               | 0.06976       |
| 50  | 0.15104               | 0.14692               | 0.07005       |
| 90  | 0.10056               | 0.10179               | 0.06158       |

---

## 2. Dual-Track Renderer Architecture

```
                                    ┌──► Track A: DifferentiableHeliosXMLRenderer (XML-Native)
                                    │    • Parses Helios XML Phytomer structures directly.
                                    │    • Reconstructs exact C++ 5-segment petiole curves & trifoliate rotation chains.
Helios Differentiable Pipeline ─────┤    • Use case: GT Benchmarks, Inverse Color/Scale Optimization, Visual Evaluation.
                                    │
                                    └──► Track B: DifferentiableHeliosRenderer (25D Node Array)
                                         • Takes continuous PyTorch Tensor inputs `nodes: (B, N, 25)`.
                                         • 100% compatible with 3D Graph Diffusion Model denoising (x_t -> x_t-1).
                                         • Fully differentiable via PyTorch Autograd.
                                         • Use case: Diffusion Prior Guidance, Generative Inverse Optimization.
```

### Track Details

* **Track A: `DifferentiableHeliosXMLRenderer`**
  - **Source File**: `diffusion_based/models/legacy/helios_geometry_legacy.py`
  - **Import Entrypoint**: `from diffusion_based.models.differentiable_pipeline import DifferentiableHeliosXMLRenderer, build_helios_geometry_from_xml`
  - **Functionality**: Reconstructs exact 3D geometry objects (`HeliosPlantGeometry`: `tubes`, `leaflets`, `ellipsoids`) directly from XML and passes them to `HeliosGeometryRasterizer.render_torch_geometry()`.
  - **Differentiability**: Fully differentiable with respect to organ colors, scales, visibility, and camera rasterization parameters.

* **Track B: `DifferentiableHeliosRenderer`**
  - **Source Files**: `diffusion_based/models/differentiable_pipeline.py` & `diffusion_based/models/helios_geometry.py`
  - **Functionality**: Accepts continuous PyTorch Tensors `nodes: (B, N, 25)` without disk XML. Executes `nodes_to_geometry_torch()` to build PyTorch mesh tensors dynamically.
  - **Differentiability**: Fully differentiable from output pixels all the way back to the input node vector array.

---

## 3. 25D Plant Organ Feature Vector Schema

`OrganNode3D.to_vec()` now returns a fixed-length **25D** vector:

| Index   | Name                | Type / Range            | Description                                                          |
|:--------|:--------------------|:------------------------|:---------------------------------------------------------------------|
| `0:3`   | `position`          | float (m)               | 3D World Position `[X, Y, Z]`                                        |
| `3`     | `length`            | float (m)               | Organ Length / Leaf Scale                                            |
| `4`     | `radius`            | float (m)               | Organ Radius / Leaf Width                                            |
| `5:14`  | `R_flat`            | float                   | **3×3 orientation matrix (row-major)**, local frame → world          |
| `14:20` | `organ_type`        | 6D one-hot              | `[INTERNODE, PETIOLE, LEAF, FLORAL_BUD, FLOWER, POD]`                |
| `20`    | `shoot_id`          | float (int)             | Shoot Identifier                                                     |
| `21`    | `phytomer_idx`      | float (int)             | Phytomer Index along shoot                                           |
| `22`    | `existence`         | float `[0.0, 1.0]`      | Soft Existence / Visibility Mask                                     |
| `23`    | `flower_head_radius`| float (m)               | Flower / Fruit Head Radius                                           |
| `24`    | `parent_idx`        | float (int)             | Parent Node Index in array (`-1` if root)                            |

### Orientation matrix semantics

* For **LEAF** nodes, channels `5:14` hold the exact local-to-world rotation matrix used by `_leaflet_local_mesh()`. The first row is the midrib direction, the second row is the width axis, and the third row is the normal axis.
* For **non-LEAF** organs, only the first column is used as the organ axis/direction; the remaining columns are zero-padded.

### Backward compatibility

`OrganNode3D.from_vec()` still accepts 22D, 19D, 18D, and legacy 16D layouts, so existing saved tensors can be loaded and rendered.

---

## 4. Key Implementation Details

### 4.1 Exact Leaflet Orientation Matrix in the Parser

In `diffusion_based/models/helios_xml_parser.py`, the parser now computes the full C++-style rotation chain for each leaflet and stores it as `leaf['R_matrix']`:

1. **roll** about local x-axis
2. **-pitch** about local y-axis
3. **yaw** about local z-axis (lateral leaflets only)
4. **azimuth + compound rotation** about world z-axis
5. **blade-up correction** for single leaves

This matrix is copied into `OrganNode3D.R_matrix` when leaf nodes are created.

### 4.2 25D-Aware `nodes_to_geometry_torch`

`diffusion_based/models/helios_geometry.py` detects `D >= 25`, reshapes channels `5:14` into a `(B, N, 3, 3)` rotation matrix, and uses it directly for leaf rendering instead of reconstructing a frame from `direction + roll`. Non-leaf organs still use the first column as the direction vector.

### 4.3 GraphDiffuser3D now outputs 25D

`diffusion_based/models/graph_diffuser_3d.py`:

* Default `node_dim` changed from `22` to `25`.
* The prediction head now outputs 25 channels.
* After prediction, the 9 orientation channels are orthonormalized via Gram–Schmidt so the predicted matrix stays a valid rotation matrix during diffusion denoising.

### 4.4 XML ↔ Organ-Node Round-Trip

`diffusion_based/models/helios_xml_parser.py` now provides:

* `organ_nodes_to_xml(nodes, base_position, plant_age, plant_id)` — serializes a flat `OrganNode3D` list back to byte-identical Helios XML.
* `verify_xml_round_trip(xml_path)` — parses, serializes, and checks both **text_equal** and **semantic_equal**.
* `extract_xml_tag_coverage(xml_path)` — audits consumed vs ignored XML tags.

All XML fixtures under `notebooks/output_dap*/` pass round-trip verification with `text_equal=True` and `semantic_equal=True`.

---

## 5. Relevant Code Files & Entry Points

1. `diffusion_based/models/differentiable_pipeline.py`
   - `DifferentiableHeliosRenderer` (Track B) and re-exports for Track A.
2. `diffusion_based/models/helios_geometry.py`
   - `nodes_to_geometry_torch()`, `_leaflet_local_mesh_torch()`, `_leaflet_local_mesh()`, `HeliosPlantGeometry.get_geometry_tensors()`.
3. `diffusion_based/models/helios_xml_parser.py`
   - `HeliosXMLParser`, `OrganNode3D.to_vec()` / `from_vec()`, `organ_nodes_to_xml()`, `verify_xml_round_trip()`, `extract_xml_tag_coverage()`.
4. `diffusion_based/models/legacy/helios_geometry_legacy.py`
   - XML-native geometry construction (`build_helios_geometry_from_xml`).
5. `diffusion_based/models/helios_rasterizer_3d.py`
   - PyTorch 3D Geometry Rasterizer.
6. `diffusion_based/models/graph_diffuser_3d.py`
   - 25D 3D graph diffusion model.
7. `notebooks/compare_track_a_b.py`
   - Quick Track A vs Track B comparison script.
8. `notebooks/run_dap_multi_benchmark_with_track_a.py`
   - Multi-DAP Track A / Track B benchmark.

---

## 6. Verification Commands

```bash
source /cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/mamba.sh
mamba activate digital-crops

# Track A vs Track B quick comparison
python notebooks/compare_track_a_b.py

# Multi-DAP benchmark (skip slow C++ rendering)
python notebooks/run_dap_multi_benchmark_with_track_a.py --daps 10 50 90 --skip-cpp

# XML round-trip verification
python - <<'PY'
from diffusion_based.models.helios_xml_parser import verify_xml_round_trip
print(verify_xml_round_trip('notebooks/output_dap_benchmark/dap50_gt_0000_plant_0000.xml'))
PY

# 25D GraphDiffuser3D integration test
python - <<'PY'
from diffusion_based.models.graph_diffuser_3d import PlantGraphDiffuser3D
from diffusion_based.models.differentiable_pipeline import DifferentiableHeliosRenderer
from diffusion_based.models.helios_rasterizer_3d import HeliosGeometryRasterizer
import torch
model = PlantGraphDiffuser3D(node_dim=25).cuda()
rast = HeliosGeometryRasterizer(image_size=256).cuda()
renderer = DifferentiableHeliosRenderer(rast).cuda()
x = torch.randn(1, 128, 25).cuda()
exist = torch.rand(1, 128, 1).cuda()
t = torch.randint(0, 100, (1,)).cuda()
img = torch.rand(1, 3, 64, 64).cuda()
out = model(x, noisy_existence=exist, timesteps=t, images=img)
rgba = renderer(out['pred_x0'])
print(rgba.shape)  # (1, 4, 256, 256)
PY
```

---

## 7. Training Pipeline Update (Completed 2026-08-11)

The 3D graph diffusion training script (`diffusion_based/training/train_diffusion_3d.py`) and the dataset loader (`dataset/helios_dataset.py`) now support the 25D organ-node representation end-to-end.

### Key changes

* **`HeliosPlantDataset` gains `node_dim` argument** (`15` or `25`, default `15` for backward compatibility). When `node_dim == 25`, the dataset uses `HeliosXMLParser.get_all_organ_nodes()` and `OrganNode3D.to_vec()` to produce full 25D tensors. Positions, lengths, and radii are normalized to `[0, 1]` using the per-plant bounding box; metric bounds are returned as `xyz_min` / `xyz_scale` so the renderer can denormalize.
* **`train_diffusion_3d.py` defaults to `node_dim=25` and `max_nodes=2048`**. It accepts `--node-dim {15,25}` and `--max-nodes N`.
* **Loss function updated for 25D**:
  * Existence-aware masked MSE for positions, full node vector, and predicted noise.
  * Added `loss_rot` on the 9 rotation-matrix channels.
  * Removed the old pitch/yaw snap heuristic; the snap loss now uses the predicted R matrix midrib axis for all organ types (falls back to pitch/yaw for 15D).
  * Organ-type classification uses the 6-channel one-hot positions `[14:20]` for 25D or `[8:12]` for 15D.
* **Render-in-the-loop switched to `DifferentiableHeliosRenderer`** (Track B) with automatic metric denormalization from `xyz_min`/`xyz_scale`. Use `--render-fast-mode` (default enabled) to lower leaf mesh subdivisions and keep full-plant renders tractable at 2048 nodes.

### Verified smoke tests

```bash
# 25D non-render forward/backward
python - <<'PY'
from diffusion_based.training.train_diffusion_3d import train_3d_diffusion
train_3d_diffusion(
    data_dir='/home/lion397/codes/image-to-l-system/notebooks/output_dap_benchmark',
    num_epochs=1, batch_size=1, save_path='/tmp/opencode/smoke_25d.pt',
    node_dim=25, max_nodes=256, render_loss_weight=0.0)
PY

# 25D + differentiable render loss
python - <<'PY'
from diffusion_based.training.train_diffusion_3d import train_3d_diffusion
train_3d_diffusion(
    data_dir='/home/lion397/codes/image-to-l-system/notebooks/output_dap_benchmark',
    num_epochs=1, batch_size=1, save_path='/tmp/opencode/smoke_25d_render.pt',
    node_dim=25, max_nodes=256, render_loss_weight=0.1, render_fast_mode=True)
PY
```

Both tests complete successfully with non-zero gradient norms.

## 8. Actionable Next Steps for the Incoming Agent

1. **Reduce Track A–B Render MAE Further (optional)**
   - The current A–B MAE is ~0.06–0.07, dominated by subtle leaflet orientation differences for trifoliate side leaflets.
   - To close the gap completely, consider storing the **exact world-space leaflet base row center** and a perfect orthogonal matrix derived from the parser (or, at train time, from the Track A mesh via orthogonal Procrustes).
   - This may require adding 3 extra channels for the local mesh base offset, or switching the node position to the actual leaflet base center for leaf organs.

2. **Run a Real Multi-Epoch 25D Diffusion Training**
   - `train_diffusion_3d.py` is ready; start with a modest dataset and `max_nodes=1024` if 2048 OOMs.
   - Consider curriculum training: train without render loss for the first N epochs, then enable `--render-loss 0.1 --render-fast-mode` for refinement.

3. **Validate End-to-End Inverse Optimization**
   - Use `notebooks/run_single_image_optimization_demo.py` with 25D node arrays to verify backpropagation from pixels to organ nodes still works.
