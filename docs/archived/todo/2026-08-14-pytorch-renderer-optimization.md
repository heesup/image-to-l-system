# PyTorch Renderer Optimization Plan (2026-08-14)

## 📌 Problem & Profiling Summary

Late-stage plant samples (e.g. DAP 90+) render slowly in the PyTorch pipeline (`helios_pytorch_geometry.py` & `helios_pytorch_renderer.py`). Empirically, rendering a DAP 94 plant sample showed:

1. **Geometry Building Bottleneck (`build_mesh_from_organ_array`)**:
   - Takes **950 ms to 1,314 ms per plant sample** (>90% of total pipeline time).
   - Generates **245,000+ Python function calls per plant** due to sequential per-organ loops:
     - `process_petiole`: 102 calls (672 ms)
     - `generate_cone_tube_mesh_torch`: 232 calls (361 ms)
     - `rotate_vector_about_axis`: 1,334 calls (195 ms)
     - 14,700+ individual tensor allocations per plant.

2. **High Polygon Count (OBJ Leaves)**:
   - High-res OBJ leaf meshes contain ~1,400–1,900 faces per leaf.
   - A mature DAP 90 plant generates **660,884 faces**, taking **23.4 ms** on GPU.
   - Parametric generic/prototype leaves (8x8 grid) contain 128 faces per leaf, reducing face count to **152,144 faces** and cutting CUDA rasterization to **1.63 ms (14.4x faster)**.

3. **Fallback Rasterizer Loop**:
   - If running without `nvdiffrast` CUDA context (e.g., CPU fallback), triangle-by-triangle Python rasterization takes **~18 seconds**.

---

## 🎯 Action Items & Implementation Strategy

### Phase 1: Leaf Prototype Fix & Face Culling
- Fix indexing errors in `generate_texture_leaf_mesh_torch()` / parametric prototype leaf generation.
- Ensure texture-prototype leaves match C++ reference leaf surface area (109,093 px vs current 148,948 px).
- Replace high-res OBJ leaf loading with optimized prototype leaf meshes when `use_generic_leaves=True`.

### Phase 2: Vectorized Mesh Construction (`helios_pytorch_geometry.py`)
- **Batched Tube Construction**: Vectorize `generate_cone_tube_mesh_torch` across all internodes and petioles simultaneously using 3D tensor operations `(N_tubes, P_points, 3)` instead of per-organ Python loops.
- **Batched Leaf Transforms**: Batch leaf vertex transformations using 3D batch tensor matrix multiplication `(N_leaves, V_leaf, 3) @ (N_leaves, 3, 3).T` instead of looping per leaf to call `rotr_z`, `rotr_y`, `rotr_x`.
- **Unified Normal Calculation**: Compute per-vertex normals once on the merged plant mesh (`compute_face_normals_torch`) rather than 351 per-organ calls.
- **Pre-allocated Tensors**: Pre-allocate output vertex, face, normal, color, and organ ID tensors to eliminate list concatenation and repeated reallocation overhead.

### Phase 3: Renderer & Execution Verification
- Ensure `nvdiffrast` CUDA context initializes reliably.
- Verify timing on DAP 10, 30, 50, and 90 samples using `multi_dap_comparison_panel.py`.
