# Plant Organ Array & PyTorch Differentiable Renderer Handoff Document (2026-08-12)

This document provides a comprehensive technical overview, mathematical formulation, directory structure, verification commands, and operational guide for the **Plant Organ Array Differentiable Renderer** for the next agent or developer.

---

## 1. Executive Summary & Core Architectural Overview

We constructed an end-to-end PyTorch differentiable rendering pipeline for plant architectures that bridges **Helios C++ L-System XMLs**, **Plant Organ Array Tensors $(N, 93)$**, **3D Procedural Mesh Generation**, and **Image Loss Backpropagation**.

```mermaid
flowchart LR
    A[Helios XML Plant Architecture] <--> B[PlantOrganArray Tensor N x 93]
    B --> C[HeliosPlantGeometryBuilder]
    C --> D[HeliosPyTorchRenderer]
    D --> E[Rendered RGB Image / Mask]
    E --> F[Image Loss MSE / IoU]
    F -- Autograd Backprop --v B
```

### Key Technical Achievements
1. **XML $\leftrightarrow$ PlantOrganArray Roundtrip**: 100% text/attribute equivalence between Helios XML and Organ Array Tensor $(N, 93)$.
2. **Forward Kinematics & Branch Connection**: Parent shoot ID, parent node index, parent petiole attachment kinematics in 3D.
3. **Helios C++ Camera Alignment**: Spherical azimuth/elevation camera positioning, top-view UP vector alignment, `--focus-plant` HFOV auto-fitting math, and **vertical row-order flip** to match Helios exported images.
4. **Quad Triangulation & Sub-Pixel Precision**: Quad fan triangulation for OBJ assets and sub-pixel edge inclusion ($\epsilon \ge -1e-4$) eliminating all rasterization holes/gaps.
5. **Correct Leaf OBJ Orientation**: Removed erroneous Y/Z axis swap in the OBJ loader so that CowpeaLeaf assets keep their intended Helios Z-up blade plane (+x midrib, +y width, +z curvature).
6. **Autograd Backpropagation**: End-to-end gradient flow from rendered image loss back to `PlantOrganArray.tensor` (leaf 3D scale, petiole pitch, leaflet orientation angles).
7. **Rendering Time Benchmark**: 5-seed DAP 30 panel now reports per-seed Helios C++ vs PyTorch timings and speedups.

---

## 2. Codebase Directory Structure & Key Files

```text
diffusion_based/
├── models/
│   ├── plant_organ.py                   # Data class definitions for PlantOrgan (Phytomer, Internode, Petiole, Leaf)
│   ├── plant_organ_array.py             # PyTorch Tensor (N, 93) representation & XML parsing/serialization
│   ├── helios_xml_parser.py             # Fast XML ElementTree parser & generator
│   ├── helios_pytorch_geometry.py       # Forward kinematics & 3D mesh builder (internodes, petioles, compound OBJ leaves)
│   ├── helios_pytorch_renderer.py       # Differentiable PyTorch/CUDA rasterizer & camera model (Helios row-order aligned)
│   ├── test_organ_array_xml_roundtrip.py# Unit test: XML -> OrganArray -> XML 100% roundtrip text equality
│   └── test_differentiable_backprop.py   # Unit test: Autograd gradient flow from Image Loss to Organ Array Tensor
├── eval/
│   ├── test_helios_coco_mask_comparison.py # Ground truth leaf mask comparison against Helios C++ Radiation output
│   ├── test_helios_pytorch_render_quality.py # Visual quality benchmark vs Helios C++ JPEG
│   ├── dap30_multi_seed_panel.py        # 5-seed DAP 30 panel with timing analysis figure
│   ├── debug_leaf_by_leaf.py            # Step-by-step 12-frame storyboard sketchbook grid debugger (DAP 10)
│   └── output/                          # Output figures (helios_coco_mask_comparison.png, dap30_seed_panel.png, dap30_timing_analysis.png)
└── docs/
    └── 20260812_plant_organ_differentiable_renderer_handover.md # THIS HANDOVER DOCUMENT
```

---

## 3. Mathematical Principles & Implementation Details

### 3.1 PlantOrganArray Tensor Schema $(N, 93)$
Each row in `PlantOrganArray.tensor` represents a single phytomer node with 93 numeric parameters:
- `[0..4]`: Plant ID, Plant Age, Shoot ID, Shoot Type, Parent Shoot ID
- `[5..9]`: Parent Node Index, Parent Petiole Index, Shoot Pitch, Shoot Yaw, Shoot Roll
- `[10..15]`: Phytomer Index, Internode Length, Radius, Pitch, Phyllo Angle, Max Length
- `[16..20]`: Length Segments, Curvature Perturbations $(0, 1)$, Yaw Perturbations $(0, 1)$
- `[21..30]`: Petiole Length, Radius, Pitch, Curvature, Leaf Scale, Taper, Segments, Radial Subdivisions, Leaflet Scale, Offset
- `[31..42]`: Compound Leaf 0, 1, 2 (Scale, Pitch, Yaw, Roll for each leaflet)
- `[43..55]`: Bud, Peduncle, and Flower Parameters

### 3.2 3D Forward Kinematics & Outward Petiole Bending
For each shoot, the base position $\mathbf{P}_{\text{base}}$ is attached to the parent petiole tip:
$$\mathbf{P}_{\text{shoot\_base}} = \mathbf{P}_{\text{parent\_tip}} + 0.9 \cdot r_{\text{parent}} \cdot \mathbf{v}_{\text{petiole\_axis}}$$
Petiole bending vector $\mathbf{v}_{\text{petiole\_axis}}$ is rotated outward from stem internode axis $\mathbf{v}_{\text{stem}}$:
$$\mathbf{v}_{\text{petiole}} = \text{Rodrigues}\Big(\mathbf{v}_{\text{stem}}, \, \mathbf{a}_{\text{rot}}, \, \theta_{\text{pet\_pitch}}\Big)$$

### 3.3 Quad Triangulation for OBJ Assets
OBJ leaf meshes (`CowpeaLeaf_tip_highres.obj`) contain 4-vertex quad polygons. To prevent blank holes in rendered leaves, polygons are fan-triangulated:
$$\text{Quad}(v_0, v_1, v_2, v_3) \longrightarrow \text{Triangle}(v_0, v_1, v_2) + \text{Triangle}(v_0, v_2, v_3)$$

### 3.4 Camera Projection & `--focus-plant` HFOV Auto-fitting
Ported directly from Helios C++ `syntheticdata_generation/main.cpp`:
$$X_{\text{center}} = \frac{X_{\min} + X_{\max}}{2}, \quad Y_{\text{center}} = \frac{Y_{\min} + Y_{\max}}{2}$$
$$\text{max\_span} = 1.05 \cdot \max(X_{\max} - X_{\min}, Y_{\max} - Y_{\min})$$
$$\text{HFOV} = 2.0 \cdot \arctan\left(\frac{0.5 \cdot \text{max\_span}}{\text{camera\_height}}\right)$$

**Vertical Row-Order Alignment.** Helios radiation renders and COCO masks store row-0 at the bottom of the image, while the initial PyTorch screen mapper produced row-0 at the top. The renderer now applies a final vertical flip (`flip(0)` for masks, `flip(1)` for RGB) so that PyTorch outputs align with Helios GT without any post-processing.

### 3.5 Leaf OBJ Orientation
The CowpeaLeaf OBJ assets (`CowpeaLeaf_tip_highres.obj`, `CowpeaLeaf_left_highres.obj`, `CowpeaLeaf_right_highres.obj`, `CowpeaLeaf_unifoliate.obj`) are already authored in Helios Z-up convention:
- `+x` = leaf midrib (length)
- `+y` = blade width
- `+z` = curvature / normal

An earlier loader incorrectly swapped raw Y and Z axes, which made every leaf render edge-on (vertical blade). The loader now returns vertices verbatim, and the existing `roll -> pitch -> yaw -> azimuth` rotation chain places leaves in the same plane as Helios C++.

## 4. Verification Commands & How to Run

### Python Environment
Ensure you use the project virtual environment with `PYTHONPATH=.`:
```bash
export PYTHONPATH=.
PYTHON=/home/lion397/.conda/envs/digital-crops/bin/python
```

### 1. Run XML $\leftrightarrow$ PlantOrganArray Roundtrip Equivalence Test
```bash
$PYTHON diffusion_based/models/test_organ_array_xml_roundtrip.py
```
*Expected Output*: `100% Roundtrip Attribute Equivalence Passed! (0 mismatched fields)`

### 2. Run Differentiable Autograd Backprop Gradient Test
```bash
$PYTHON diffusion_based/models/test_differentiable_backprop.py
```
*Expected Output*: `Organ Array Tensor Grad Norm: 6.127346, 106 / 465 gradient channels non-zero`

### 3. Run Helios C++ Radiation Ground-Truth Leaf Mask Comparison
```bash
$PYTHON diffusion_based/eval/test_helios_coco_mask_comparison.py \
    --xml Digital-Crops/projects/syntheticdata_generation/build/output_rad_test/plot_0000_plant_0000.xml \
    --json Digital-Crops/projects/syntheticdata_generation/build/output_rad_test/plot_0000_masks.json \
    --rad-img Digital-Crops/projects/syntheticdata_generation/build/output_rad_test/plot_0000_rad.jpeg \
    --output-dir diffusion_based/eval/output
```
*Expected Output*: `Helios C++ GT Leaf Pixel Count: 135437, PyTorch Leaf Pixel Count: 98784, IoU: 0.3902, Dice: 0.5614`.
Generates figure: `diffusion_based/eval/output/helios_coco_mask_comparison.png`.

### 4. Run 5-Seed DAP 30 Panel with Timing Analysis
```bash
$PYTHON diffusion_based/eval/dap30_multi_seed_panel.py \
    --base-dir Digital-Crops/projects/syntheticdata_generation/build/output_rad_dap30 \
    --seeds 0 1 2 3 4 \
    --output-dir diffusion_based/eval/output
```
*Expected Output*:
```text
Seed 0: IoU=0.5946, Dice=0.7457
Seed 1: IoU=0.6735, Dice=0.8049
Seed 2: IoU=0.5464, Dice=0.7066
Seed 3: IoU=0.7279, Dice=0.8425
Seed 4: IoU=0.5581, Dice=0.7164
Mean IoU: 0.6201, Mean Dice: 0.7632
Mean Helios C++: 9.83 s, Mean PyTorch Total: 0.58 s, Speedup: 19.10x
```
Generates figures: `dap30_seed_panel.png` and `dap30_timing_analysis.png`.

### 5. Run DAP 10 Incremental Leaf-by-Leaf Storyboard Debug Grid
```bash
$PYTHON diffusion_based/eval/debug_leaf_by_leaf.py \
    --xml Digital-Crops/projects/syntheticdata_generation/build/output/dap10_gt_0000_plant_0000.xml \
    --output-dir diffusion_based/eval/output
```
*Expected Output*: Saves 12-frame sketchbook grid to `diffusion_based/eval/output/dap10_leaf_by_leaf_debug.png`.

---

## 5. Recent Fixes (2026-08-12)

1. **Vertical image row-order alignment** — `helios_pytorch_renderer.py` now flips RGB and organ-type buffers so PyTorch outputs match Helios exported images/masks directly. Mean DAP 30 IoU improved from ~0.47 to ~0.62 after the accompanying OBJ fix.
2. **Leaf OBJ Y/Z axis swap removed** — `helios_pytorch_geometry.py::load_obj_file()` no longer swaps Y/Z for CowpeaLeaf assets. Leaves now render with the correct broad blade plane instead of standing edge-on.
3. **DAP 30 timing benchmark** — `dap30_multi_seed_panel.py` reports a per-seed timing table and saves `dap30_timing_analysis.png`. PyTorch is ~19x faster than Helios C++ radiation rendering on average.

---

## 6. Next Steps for Next Agent / Developer

1. **Inverse Rendering Optimization Loop (Image-to-L-System)**:
   - Implement an optimization loop (`Adam` optimizer) that updates `PlantOrganArray.tensor` `(N, 93)` directly from target RGB images.
   - Loss function:
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}}(\mathbf{I}_{\text{render}}, \mathbf{I}_{\text{target}}) + \lambda_{\text{mask}} (1 - \text{IoU}) + \lambda_{\text{reg}} \mathcal{L}_{\text{smoothness}}$$
2. **Diffusion Prior Integration**:
   - Train a Score-based Diffusion Model or Denoising Diffusion Probabilistic Model (DDPM) directly on `PlantOrganArray` tensors $(N, 93)$ to act as a structural 3D plant architecture prior during image inverse optimization.
