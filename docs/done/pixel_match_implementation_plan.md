# Implementation Plan: PyTorch Differentiable Renderer → C++ Helios Pixel-to-Pixel Matching

## Goal
Achieve per-organ IoU > 0.90 between the PyTorch differentiable renderer output and C++ Helios ground-truth masks for DAP 10, 50, 90 (cowpea), across the organs:
- `0`: Internode (main stem)
- `1`: Petiole
- `2`: Leaf
- `3`: Floral bud
- `4`: Flower
- `5`: Pod / fruit

The organ class IDs are intentionally aligned with the existing Python `OrganNode3D` enum so no fragile ID remapping is required in the benchmark harness.

---

## 1. C++ Ground Truth mask export

### Current state
`Digital-Crops/projects/syntheticdata_generation/main.cpp` exports COCO masks for only:
- `0`: plant
- `1`: flower
- `2`: pod

### Target
Replace the class list with the organ-level list aligned to the Python enum.

| C++ mask ID | Organ | Primitive labels to collect |
|------------|-------|-----------------------------|
| `0` | Internode | `"shoot"` |
| `1` | Petiole | `"petiole"`, `"peduncle"` (structural stalk) |
| `2` | Leaf | `"leaf"` |
| `3` | Floral bud | `"floral_bud"` |
| `4` | Flower | `"flower"` |
| `5` | Pod / fruit | `"fruit"`, `"pod"` |

### Implementation
1. After plant aging and primitive relabeling, collect primitive UUIDs per category by reading each UUID's `object_label` primitive data.
2. Replace the existing `writeImageSegmentationMasks` call with one that uses the six categories above.
3. Rebuild the C++ binary (`cd build && make -j`).

### Validation
- Regenerate DAP 10/50/90 samples.
- Confirm `*_masks.json` contains annotations with `category_id` 0–5.
- DAP 10 will have only classes 0–2 populated; DAP 50/90 will add 3–5.

---

## 2. Leaf scale factor (`current_leaf_scale_factor`)

### Missing fix
The hand-off document claims `helios_geometry.py` multiplies leaf scale by `current_leaf_scale_factor`. That multiplication is not currently present in any file.

### Implementation
- In `diffusion_based/models/helios_geometry.py` `_reconstruct_shoot_geometry_exact`:
  ```python
  leaf_scale = leaf.get("scale", 0.0) * leaf.get("scale_factor", 1.0)
  ```
- In `diffusion_based/models/helios_xml_parser.py` `Phytomer3D.get_organ_nodes`:
  ```python
  lnode.length = leaf.get('scale', 0.0) * leaf.get('scale_factor', 1.0)
  ```
- In `diffusion_based/models/helios_geometry.py` `_leaflet_from_node`: apply the node's `scale_factor` if present.

---

## 3. Ghost petiole cumulative phyllotactic angle

### Missing fix
When a phytomer has no explicit petiole geometry, C++ creates a ghost petiole perpendicular to the internode and rotates it by `cumulative_rotation = parent_node_index * internode_phyllotactic_angle` about the internode axis. This cumulative rotation is not currently applied in Python.

### Implementation
Add the cumulative ghost-petiole rotation in both reconstruction paths:
- `diffusion_based/models/helios_geometry.py` `_reconstruct_shoot_geometry_exact`
- `diffusion_based/models/helios_xml_parser.py` `_compute_internode_orientation`

```python
if no explicit petiole axis available:
    parent_petiole_axis = normalize(cross(parent_internode_axis, [0, 0, 1]))
    if norm < 0.01:
        parent_petiole_axis = [0, 1, 0]
    if child_shoot:
        cum_rot = parent_node_index * radians(parent_phyt.internode_phyllotactic_angle)
    else:
        cum_rot = (phyt_idx - 1) * radians(prev.internode_phyllotactic_angle)
    parent_petiole_axis = rodrigues(parent_petiole_axis, parent_internode_axis, cum_rot)
```

---

## 4. Procedural 3D curved leaf mesh

### Current state
Python uses a simple flat 8×6 grid with a tiny sinusoidal arch. C++ `GenericLeafPrototype` (Assets.cpp:21-178) produces a much richer surface.

### Target
Implement a `GenericLeafPrototype`-style procedural mesh builder.

### Parameters (cowpea trifoliate defaults)
- `subdivisions = 6` (longitudinal `Nx`)
- `aspect_ratio = 0.7` → `Ny = 6` (forced even)
- `longitudinal_curvature`
- `lateral_curvature`
- `midrib_fold_fraction`
- `petiole_roll`
- `buckle_angle`, `buckle_length`
- `wave_amplitude`, `wave_period`

### Vertex construction
For grid indices `i ∈ [0, Nx]`, `j ∈ [0, Ny]`:
1. `x = i / Nx` (length, 0 = base)
2. `y = (j / Ny - 0.5) * aspect_ratio` (width, centered)
3. Longitudinal curvature: `z += longitudinal_curvature * x^4`
4. Lateral curvature: `z += lateral_curvature * (y / aspect_ratio)^4`
5. Midrib fold: `y_fold = cos(0.5 * fold * pi) * y`, `z_fold = sin(0.5 * fold * pi) * |y|`
6. Petiole roll: small drop near base
7. Longitudinal incremental rotation: `dtheta = -atan(4 * longitudinal_curvature * x^3 * dx)` applied per row about local y.
8. Buckle: rotate distal portion about `(buckle_length, 0, 0)` parallel to y by `buckle_angle`.
9. Wave: displace along rotated normal using `wave_amplitude` and `wave_period`.
10. Faces: `(v0, v1, v2)` and `(v0, v2, v3)` per quad.

### World transform (unchanged order)
1. Uniform scale by `leaf_scale * current_leaf_scale_factor`.
2. Compound rotation about z if trifoliate.
3. Roll about local x.
4. Pitch about local y (negative angle).
5. Yaw about local z (lateral leaflets only).
6. Azimuth + compound about world z.
7. Blade-up correction (single leaves) about petiole tip axis.
8. Translate to leaf base.

### Files to update
- `diffusion_based/models/helios_geometry.py`: replace `_leaflet_local_mesh`.
- `diffusion_based/models/helios_rasterizer_3d.py`: update hard-coded `leaf_faces`.
- `diffusion_based/models/helios_xml_parser.py`: parse leaf prototype parameters if present in XML.

---

## 5. Flower / pod forward kinematics

### Current state
Python uses a simplified peduncle: start from internode axis, apply `peduncle_pitch`, and move straight to `head_pos`.

### Target
Port `recomputePeduncleOrientationVectors` from `InputOutput.cpp` L1632-1692.

### Steps
1. `parent_internode_axis` = internode axis at tip.
2. `current_petiole_axis` = petiole base axis (internode axis if no petiole).
3. `inflorescence_bending_axis = normalize(cross(parent_internode_axis, current_petiole_axis))`.
4. Apply `peduncle_pitch + base_rotation.pitch` about the bending axis.
5. Azimuthal alignment to parent petiole:
   - `parent_petiole_azimuth = -atan2(petiole_axis.y, petiole_axis.x)`
   - `current_peduncle_azimuth = -atan2(peduncle_axis.y, peduncle_axis.x)`
   - `azimuthal_rotation = current_peduncle_azimuth - parent_petiole_azimuth`
   - Rotate peduncle and bending axis about `internode_axis`.
6. Build peduncle vertices with curvature:
   - `dr = peduncle_length / n_seg`
   - Rotate axis about horizontal bend axis by `radians(peduncle_curvature * dr)` per segment toward vertical.
7. `head_pos = base + sum(dr * axis)`.
8. Use `head_pos` for flower/pod ellipsoid center.
9. Render peduncle as thin tube if length > 0.003 m.

---

## 6. Pixel-match benchmark harness

### New file
`scratch/compare_pixel_match.py`

### Inputs
- C++ GT: `*_vis.jpeg`, `*_masks.json`
- XML: `*_plant_0000.xml`
- Camera params: `*_params.json`

### Outputs
- Per-organ IoU / Dice for classes 0–5.
- Side-by-side RGB + mask overlay PNGs.
- CSV log.

### Algorithm
1. Render the XML with `HeliosGeometryRasterizer` using the same camera parameters.
2. Extract per-organ masks from PyTorch render by organ ID.
3. Load C++ masks from `*_masks.json` and rasterize polygons to binary masks.
4. Compute IoU and Dice per class.
5. Report per DAP and per class.

### Empty-class handling
If both GT and prediction are empty for a class, report `IoU = 1.0` / `Dice = 1.0`. If only one is empty, report `IoU = 0.0` / `Dice = 0.0`.

---

## 7. Execution order

1. Extend C++ mask export with organ classes 0–5.
2. Rebuild C++ binary.
3. Regenerate DAP 10/50/90 GT masks.
4. Implement `current_leaf_scale_factor` multiplication.
5. Implement ghost petiole cumulative rotation.
6. Implement procedural curved leaf mesh.
7. Implement improved flower/pod FK.
8. Create benchmark harness.
9. Run benchmark, iterate until all non-empty classes reach IoU > 0.90.

---

*Approved class order: 0=Internode, 1=Petiole, 2=Leaf, 3=Floral bud, 4=Flower, 5=Pod.*
