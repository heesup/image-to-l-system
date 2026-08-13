# Hand-off Document: PyTorch Differentiable Renderer vs Ground Truth C++ Helios Pixel-to-Pixel Matching

## 1. Executive Summary & Goal
The objective of this task is to align the **PyTorch Differentiable Renderer** (`HeliosPlantGeometryTorch` & `HeliosGeometryRasterizer` in `diffusion_based/models/`) with **Ground Truth C++ Helios** outputs across growth benchmarks (DAP 10, DAP 50, DAP 90) so that organ masks (Leaves, Stems, Flowers, Pods) achieve **exact pixel-to-pixel matching**.

---

## 2. Completed Analysis & Implemented Fixes

### 2.1 Leaf Scale Factor (`current_leaf_scale_factor`)
- **Root Cause**: Young plant leaves (DAP 10) in C++ Helios use `<current_leaf_scale_factor>` (e.g. `0.65`) stored under `<petiole>` XML tags to scale leaves down during early phenological stages. `helios_geometry.py` was ignoring this factor, causing PyTorch leaves to render **1.54x oversized**.
- **Fix Applied** (`diffusion_based/models/helios_geometry.py`):
  ```python
  leaf_scale = leaf.get("scale", 0.0) * leaf.get("scale_factor", 1.0)
  ```

### 2.2 Ghost Petiole Cumulative Phyllotactic Angle (`PlantArchitecture.cpp` L1085-L1110)
- **Root Cause**: When a phytomer has no explicit petioles, C++ Helios creates a "ghost petiole" reference vector and applies a cumulative phyllotactic rotation:
  $$\text{cumulative\_rotation} = \text{parent\_node\_index} \times \text{phyllotactic\_angle}$$
- **Fix Applied** (`diffusion_based/models/helios_geometry.py`):
  Added cumulative Rodrigues rotation for ghost petiole axes on child shoots and internodes:
  ```python
  ghost = np.cross(parent_internode_axis, np.array([0.0, 0.0, 1.0]))
  if np.linalg.norm(ghost) < 0.01:
      ghost = np.array([0.0, 1.0, 0.0])
  ghost = _np_normalize(ghost)
  cum_rot = float(phyt_idx - 1) * math.radians(prev.internode_phyllotactic_angle)
  parent_petiole_axis = _np_rodrigues(ghost, parent_internode_axis, cum_rot)
  ```

### 2.3 Camera & Canopy Center Alignment
- **Root Cause**: Ground Truth C++ Helios (`main.cpp` L247) sets `canopy_center = (min_xyz + max_xyz) * 0.5f` as the camera lookat point when `--focus-plant` is enabled. If PyTorch looks at `(0, 0, 0)` ground level, a 2D translation offset occurs.
- **Centroid Realignment Benchmark** (`scratch/test_dap10_centroid_alignment.py`):
  - **Raw Unaligned DAP 10 IoU**: `0.2958` (Dice: `0.4565`)
  - **Centroid Shift**: $dy = -131$ px, $dx = 93$ px
  - **Centroid Aligned IoU**: `0.4549` (Dice: `0.6254`)
  - **Optimal Translation Search IoU**: `0.4704` (Dice: `0.6398`)

---

## 3. Key Files & Working Test Scripts

### Core Implementation Files
1. `diffusion_based/models/helios_geometry.py`: 3D Forward Kinematics (FK) and XML parsing.
2. `diffusion_based/models/helios_rasterizer_3d.py`: PyTorch differentiable rasterizer.

### Benchmark & Diagnostic Scripts (in Artifacts Scratch Directory)
- `scratch/compare_pixel_match.py`: Quantitative multi-organ IoU/Dice evaluator.
- `scratch/test_dap10_leaf_match.py`: DAP 10 leaf mask extraction and comparison.
- `scratch/test_dap10_exact_match.py`: DAP 10 full plant mask evaluator.
- `scratch/test_dap10_centroid_alignment.py`: Centroid (Center of Mass/Weight) realignment evaluator.
- `scratch/test_dap10_optimal_alignment.py`: Optimal 2D translation search evaluator.

---

## 4. Immediate Next Action Items for Continuing Agent

1. **Petiole Curvature Sign & Base Pitch/Yaw Rotation Alignment**:
   - Verify petiole curvature direction (`-deg2rad(petiole_curvatures[p] * dr_petiole)`) in `InputOutput.cpp` L1745 against `helios_geometry.py` line 591 (`_np_rodrigues(pet_axis, pet_rot_axis, -ang)`).
   - Ensure `base_rotation` pitch & yaw signs in `_reconstruct_shoot_geometry_exact` match C++ `InputOutput.cpp` L1575-L1600.

2. **3D Leaf Surface Mesh vs 2D Flat Prototype**:
   - C++ Helios uses a curved 3D leaf mesh (`6x6` subdivisions with longitudinal & lateral curvatures). Replace flat rectangle quad rendering in PyTorch with the 3D curved leaf prototype grid.

3. **Flower & Pod (Fruit) FK Attachment**:
   - Ensure peduncle node positioning and floral bud / pod rasterization match GT C++ Helios annotations in DAP 50 and DAP 90.
