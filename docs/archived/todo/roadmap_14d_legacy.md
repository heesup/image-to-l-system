# Project TODO / Roadmap

## Active / High Priority

1. **Finish 14D precompute cache**
   - Location: `dataset/helios_data_14d_cache/`
   - Script: `diffusion_based/dataset/precompute_part_tensors.py`
   - Status: in progress; skips existing files on restart.

2. **Train 14D flow-matching model end-to-end**
   - Script: `diffusion_based/training/train_part_flow_matching.py`
   - Dataset: `PartArrayDataset` with `cache_dir` pointing to precomputed cache.
   - Target: 50 epochs, max_nodes=2048, image_size=128.

3. **Generate honest 14D report figures**
   - Script: `diffusion_based/eval/generate_14d_report.py`
   - Outputs: `docs/results/assets/fig3..fig7`
   - All numbers come from real 14D direct-optimization runs; no aspirational learned-method metrics.

## Deferred / Future

4. **Extend 14D to 16D by adding curvature dimensions**
   - Motivation: exact XML round-trip requires petiole/peduncle curvature (currently
     hardcoded to 0 by `part_assembly_to_xml.py`). Dataset analysis shows
     `curvature_perturbations` and `yaw_perturbations` are always `0;0`, but
     `petiole_curvature` and `peduncle_curvature` vary across plants.
   - Proposed layout:
     - 14D pose: `[type(1), base(3), rot6d(6), scale(3), exist(1)]`
     - +2 explicit dims: `petiole_curvature(1)`, `peduncle_curvature(1)`
     - Total: 16D, masked by organ type during loss/normalization.
   - Files to touch:
     - `diffusion_based/models/plant_organ_array.py` (column constants)
     - `diffusion_based/models/helios_pytorch_geometry.py` (extract curvature during mesh build)
     - `diffusion_based/models/helios_pytorch_renderer.py` (curved tube rendering)
     - `diffusion_based/models/part_assembly_to_xml.py` (write real curvature values)
     - `diffusion_based/models/part_flow_matching.py` (input/output dim 14 -> 16)
     - `diffusion_based/dataset/part_array_dataset.py` (normalize curvature dims)
     - `diffusion_based/dataset/precompute_part_tensors.py` (regenerate cache)
   - Decision: keep 14D for now; revisit after baseline 14D model is trained and evaluated.

## Done

- Migrated 40D model/training/dataset code to `diffusion_based/*/legacy/`.
- Removed active 40D imports from the main benchmark/report pipeline.
- Added 14D-only report script (`generate_14d_report.py`).
- Fixed dataset to compute 14D tensor once and render from it (no double full-mesh build).
- Added raw-depth output and affine-invariant depth loss.
- Added optional Chamfer leaf-base loss in world space.
