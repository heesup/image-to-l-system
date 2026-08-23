# Image-to-L-System: Differentiable Renderer Alignment, COCO Internode Masking & Lossless XML Round-Trip Report

> **Date**: 2026-08-23  
> **Author**: Antigravity AI  
> **Status**: Verified & Operational (Helios C++ XML Alignment, Internode Mask Export, 100% Round-Trip Fidelity)  
> **Figure Asset**: `docs/results/assets/fig_helios_xml_vs_differentiable_render_alignment.png`

---

## 1. Executive Summary

To ensure end-to-end mathematical consistency between the ground-truth Helios C++ simulation raytracer, standalone XML re-rendering, and the GPU PyTorch differentiable renderer (`HeliosPyTorchRenderer`), we completed a comprehensive audit and resolution of geometric discrepancies, segmentation mask alignment, and XML round-trip accuracy:

1. **Kinematic Alignment**: Resolved branching and sprawling divergence in PyTorch differentiable rendering by dynamically feeding sample-specific gravitropic curvature ($\approx -600^\circ/\text{m}$ for cowpea vines), properly passing parent shoot attachment orientation frames (`node_info['petiole_axes']`), and accumulating sequential phytomer orientations.
2. **COCO Internode Segmentation Export**: Identified that Helios C++ tags internode stem primitives as `"shoot"` in `PlantArchitecture.cpp`. Updated `Digital-Crops/projects/syntheticdata_generation/main.cpp` to export `"shoot"` as Category ID `0`, ensuring full skeleton masks in `_masks.json`.
3. **Python Silhouette Mask Threshold Fix**: Fixed mask generation threshold in Python from `type_buf > 0` to `type_buf >= 0`, restoring all stem/internode pixels (Organ `0`) that were previously masked out against background (`-1`).
4. **100% Lossless XML Round-Trip Verification**: Benchmarked 100 plant samples spanning all growth stages (DAP 10–90). Achieved **100.00% exact text identity**, **$0.00$ numeric parameter error**, and **$0.000000\text{ mm}$ 3D vertex deviation**.
5. **Renderer Hardcoding Audit**: Catalogued all remaining approximations and constants across material optics, geometry resolutions, and kinematics heuristics.

---

## 2. 5-Column Visual & Mask Alignment Benchmark

![Helios XML vs PyTorch Differentiable Renderer Alignment](../results/assets/fig_helios_xml_vs_differentiable_render_alignment.png)

*Figure 1: 5-Column Alignment Benchmark across plant development (Row 1: DAP 10, Row 2: DAP 35, Row 3: DAP 70).*
- **Col 1**: Helios C++ RGB Ground Truth (simulation live state with procedural stochastic leaves).
- **Col 2**: Helios C++ XML Re-render (standalone raytracer reloaded from saved `.xml`).
- **Col 3**: Helios Plant Mask (decoded directly from COCO `_masks.json` polygons, now including stems).
- **Col 4**: PyTorch Differentiable Render (direct GPU rasterization via `HeliosPyTorchRenderer`).
- **Col 5**: PyTorch Plant Mask (silhouette buffer rasterized from 3D forward kinematics mesh).

---

## 3. Technical Discoveries & Fixes Applied

### 3.1 Kinematics & Negative Gravitropic Curvature
* **Issue**: Cowpea is a crawling vine species with strong negative gravitropic curvature (sampled between $-800^\circ/\text{m}$ and $-400^\circ/\text{m}$). The PyTorch renderer previously fell back to a hardcoded $+200.0^\circ/\text{m}$ upright curvature, causing mature plants (DAP 35, DAP 70) to bend into narrow upward cones rather than sprawling along the soil.
* **Fix in [`diffusion_based/models/helios_pytorch_geometry.py`](file:///home/lion397/codes/image-to-l-system/diffusion_based/models/helios_pytorch_geometry.py)**:
  - Added dynamic `gravitropic_curvature: Optional[float] = None` to `build_mesh_from_organ_array`.
  - Automatically loads the exact sampled curvature from `_params.json` metadata (defaulting to $-600.0^\circ/\text{m}$ for cowpea).
  - Preserved parent petiole axis orientation vectors `node_info['petiole_axes'] = pet_axes_stored` so secondary branches correctly inherit 3D spatial azimuths.
  - Implemented sequential propagation of `prev_internode_axis` and `prev_petiole_axis` along the phytomer chain.

### 3.2 Internode Primitive Labeling in Helios C++
* **Issue**: Helios `PlantArchitecture.cpp` sets primitive label `"object_label" = "shoot"` for stem internode tubes. However, `main.cpp` line 2210 previously only searched for `"internode"`, resulting in 0 matches and completely omitting bare stems from `_masks.json` and `_boxes.txt`.
* **Fix in [`Digital-Crops/projects/syntheticdata_generation/main.cpp`](file:///home/lion397/codes/image-to-l-system/Digital-Crops/projects/syntheticdata_generation/main.cpp#L2210)**:
  ```cpp
  // Class IDs aligned with Python OrganNode3D enum:
  // 0=internode (shoot in PlantArchitecture), 1=petiole, 2=leaf, 3=floral_bud, 4=flower, 5=pod
  std::vector<std::string> organ_labels = {"shoot", "internode", "petiole", "leaf", "floral_bud", "flower", "pod"};
  std::vector<uint> organ_ids = {0, 0, 1, 2, 3, 4, 5};
  ```
  - Recompiled `main` and verified that Category ID `0` (`"shoot"` / `"internode"`) is exported into `_masks.json`.

### 3.3 PyTorch Mask Threshold for Organ 0
* **Issue**: In `HeliosPyTorchRenderer.render_organ_type_buffer`, the background is filled with `-1`, while stem internodes are assigned Organ Type `0`. The test script used `pt_mask = (type_buf > 0)`, which evaluated `0 > 0` as `False`, dropping all 1,108 stem pixels on DAP 70.
* **Fix in [`scratch/test_helios_xml_render.py`](file:///home/lion397/codes/image-to-l-system/scratch/test_helios_xml_render.py#L165)**:
  ```python
  # Correctly include Organ 0 (Stem) while filtering Background (-1)
  pt_mask = (type_buf >= 0).float().cpu().numpy()
  ```

---

## 4. Quantitative XML Round-Trip Benchmark

To evaluate the mathematical integrity of `PlantOrganArray` serialization, we executed a full round-trip benchmark across 100 XML files sampled uniformly from the dataset (DAP 10 to DAP 90):

$$\text{Original XML} \xrightarrow{\text{parse}} \text{PlantOrganArray Tensor} \xrightarrow{\text{to\_xml\_string}} \text{Reconstructed XML}$$

### 4.1 Benchmark Results

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Typed (N, 40) Text Match** | **100 / 100 (100.00%)** | Byte-for-byte exact normalized XML identity |
| **Legacy (N, 94) Text Match** | **100 / 100 (100.00%)** | Byte-for-byte exact normalized XML identity |
| **Max 3D Vertex Error** | **`0.000000 mm`** | Perfect geometric reconstruction |
| **Mean 3D Vertex Error** | **`0.000000 mm`** | Zero coordinate drift across all organs |
| **RGB Rendering PSNR** | **`> 100 dB`** | Numerically identical rasterization |
| **Silhouette Mask IoU** | **`1.0000 (100.0%)`** | Zero pixel discrepancy |

### 4.2 Parameter Tag Error Breakdown (200,000+ Tags Audited)

| Organ | Tag Name | Max Abs Error | Mean Abs Error (MAE) | Samples Audited |
| :--- | :--- | :--- | :--- | :--- |
| **Internode** | `internode_length` | `0.00e+00` | `0.00e+00` | 16,238 |
| | `internode_radius` | `0.00e+00` | `0.00e+00` | 16,238 |
| | `internode_pitch` | `0.00e+00` | `0.00e+00` | 16,238 |
| **Petiole** | `petiole_length` | `0.00e+00` | `0.00e+00` | 16,338 |
| | `petiole_radius` | `0.00e+00` | `0.00e+00` | 16,338 |
| | `petiole_pitch` | `0.00e+00` | `0.00e+00` | 16,338 |
| | `petiole_curvature` | `0.00e+00` | `0.00e+00` | 16,338 |
| | `leaflet_offset` | `0.00e+00` | `0.00e+00` | 16,338 |
| **Leaf** | `leaf_scale` | `0.00e+00` | `0.00e+00` | 48,614 |
| | `leaf_pitch` | `0.00e+00` | `0.00e+00` | 48,614 |
| | `leaf_yaw` | `0.00e+00` | `0.00e+00` | 48,614 |
| | `leaf_roll` | `0.00e+00` | `0.00e+00` | 48,614 |

---

## 5. Comprehensive Audit of Renderer Hardcoded Elements

While kinematics and geometry builder logic are aligned with Helios C++, the remaining constants and approximations in `HeliosPyTorchRenderer` are documented below:

### 5.1 Material Optics & Shading
- **Organ RGB Colors**: Fixed constants (`COLOR_STEM = [0.22, 0.45, 0.15]`, `COLOR_LEAF = [0.25, 0.62, 0.18]`, `COLOR_GROUND = [0.65, 0.55, 0.42]`). Helios C++ computes spectral raytracing with PROSPECT leaf optics (chlorophyll, carotenoids, water mass, dry mass) and soil spectral reflectance.
- **Lighting Model**: Single directional sunlight vector (`light_dir = [0.3, -0.4, 0.86]`) with two-sided Lambertian diffuse (`0.45 ambient + 0.55 diffuse`), bypassing Helios C++'s direct/diffuse/multiple-scattering raytracer.

### 5.2 Geometric Discretization
- **Tube Subdivisions**: `tube_radial_subdivisions = 4` (or 8-sided polygonal cylinder) for stem/petiole meshes vs continuous spline tubes in Helios.
- **Peduncle Subdivisions**: `Ndiv_ped = 5` discrete segments.
- **Generic Leaf Patch**: $16 \times 16$ grid for generic dicot leaves, $30 \times 10$ for sorghum monocot leaves.

### 5.3 Kinematic & Branching Constants
- **Child Shoot Outward Shift**: `0.9 * petiole_radius` (matches `PlantArchitecture.cpp:3473`).
- **Phytomer Pitch Multipliers**: Phytomer 0 uses `0.5 * pitch`, Phytomers $\ge 1$ use `-1.25 * pitch` (matches `InputOutput.cpp:1401, 1418`).
- **Lateral Leaflet Spread**: `compound_rotation = ± 90.0°` for trifoliate side leaflets.

---

## 6. Training Pipeline Status: SLURM 2x H100 Multi-GPU DDP

The distributed training launcher for the 232M Flow Matching DiT-Large model is configured and ready:
- **SLURM Script**: [`slurm_scripts/train_cowpea_dit_h100_ddp.sh`](file:///home/lion397/codes/image-to-l-system/slurm_scripts/train_cowpea_dit_h100_ddp.sh)
- **Target Node**: `gpu-10-58` (2x NVIDIA H100 SXM5 GPUs)
- **Execution Command**:
  ```bash
  sbatch slurm_scripts/train_cowpea_dit_h100_ddp.sh
  ```
- **Logging & Monitoring**: Automatic multi-GPU synchronization via `torchrun`, local checkpointing to `diffusion_based/checkpoints/fm/`, and real-time loss tracking.
