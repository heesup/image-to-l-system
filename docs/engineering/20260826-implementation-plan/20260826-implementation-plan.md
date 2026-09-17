---
title: "Canonical Pipeline Refactor: XML → 40D → 16D → Part Render"
date: 2026-08-26
tags: [engineering, implementation]
status: done
---

# Canonical Pipeline Refactor: XML → 40D → 16D → Part Render

## Goals

Unify all rendering pipelines into a single canonical path:
```
XML → PlantOrganArray (40D) → extract_part_tensor() → 16D → build_mesh_from_part_tensor()
```

Completely remove 94D (`NUM_FEATURES_LEGACY = 94`) from the rendering pipeline.

---

## Scope of Changes

### Component 1: `helios_pytorch_geometry.py`

#### [MODIFY] `extract_part_tensor()` — Full rewrite with native 40D Forward Kinematics (FK)

Current: `organ_array.to_legacy_tensor_diff()` → 94D → FK via COL_* constants  
Target: Read 40D typed tensor directly using T_COL_* constants to perform FK

**40D FK Algorithm:**
1. Group organ rows by `(shoot_id, phytomer_idx)`
2. Read shoot rotation (pitch/yaw/roll) and parent_shoot_id from `ORGAN_SHOOT_META` rows
3. Sort shoots in topological order (based on parent_shoot_id)
4. For each phytomer:
   - `ORGAN_INTERNODE` → `T_COL_LENGTH`, `T_COL_RADIUS`, `T_COL_PITCH`, `T_COL_PHYLLOTACTIC_ANGLE`, curv/yaw pert
   - `ORGAN_PETIOLE` (T_COL_PARENT_PETIOLE_IDX = 0 or 1) → petiole FK
   - `ORGAN_LEAF` (T_COL_CHILD_INDEX = 0,1,2) → complete R_leaf calculation
   - `ORGAN_BUD` → bud_state, is_terminal, fruit_scale
   - `ORGAN_PEDUNCLE` → peduncle FK
   - `ORGAN_FLOWER`/`ORGAN_FRUIT` (distinguished by T_COL_CHILD_INDEX) → flower pose

**Deletion:**
```python
# Completely remove this block
if organ_array.is_typed:
    legacy_tensor = organ_array.to_legacy_tensor_diff()
    organ_array = PlantOrganArray(legacy_tensor, raw_metadata=[])
```

#### [MODIFY] `build_mesh_from_organ_array()` — Convert to deprecated wrapper

Current: Generate mesh directly via 94D FK  
Target: Wrapper that delegates to `extract_part_tensor()` → `build_mesh_from_part_tensor()`

```python
def build_mesh_from_organ_array(self, organ_array, device, species=None, leaf_mode=None, ...):
    """Deprecated: Use build_mesh_from_part_tensor(organ_array.to_part_tensor()) instead."""
    import warnings
    warnings.warn("build_mesh_from_organ_array is deprecated. Use build_mesh_from_part_tensor.", DeprecationWarning)
    pt = self.extract_part_tensor(organ_array, device=device)
    return self.build_mesh_from_part_tensor(pt, device=device, leaf_mode=leaf_mode)
```

Delete existing 94D FK code (~600 lines).

---

### Component 2: Update Active Code Callers

Switch from `build_mesh_from_organ_array` → `build_mesh_from_part_tensor(arr.to_part_tensor())`:

#### [MODIFY] `plant_recon/training/train_cowpea_dit_100k_ddp.py` (L137)
#### [MODIFY] `plant_recon/training/train_cowpea_dit_100k.py` (L114)  
#### [MODIFY] `plant_recon/training/train_cowpea_vlm_scaffold_dit_ddp.py` (L172)
#### [MODIFY] `plant_recon/dataset/generate_tensor_shards.py` (L177)
#### [MODIFY] `plant_recon/models/helios_pytorch_renderer.py` (L742, L752)
#### [MODIFY] `scripts/run_cowpea_dap10_direct_opt_full.py` (L134, L272)
#### [MODIFY] `scripts/verify_40d_helios_render_comparison.py` (L107)
#### [MODIFY] `scripts/debug_larger_plant.py` (L46)
#### [MODIFY] `scripts/minimal_direct_opt_depth_chamfer_demo.py` (L137, L153, L196)
#### [MODIFY] `scripts/debug_side_view_render.py` (L47)
#### [MODIFY] `plant_recon/eval/` — Active evaluation scripts

Do not touch files in `archive/`.

---

### Component 3: `plant_organ_array.py`

#### [DEPRECATE] `to_legacy_tensor()`, `to_legacy_tensor_diff()`
- Add DeprecationWarning, keep code (may have other dependencies)
- No longer called in rendering pipeline

---

## Open Questions

> [!IMPORTANT]
> **Handling the `species` parameter**: `build_mesh_from_organ_array` takes `species="cowpea"` etc. to select leaf OBJ types. `build_mesh_from_part_tensor` currently only supports cowpea. We should add a `species` parameter to `extract_part_tensor` or configure it in the `HeliosPlantGeometryBuilder` constructor. **Can we proceed assuming cowpea-only for now?**

> [!IMPORTANT]  
> **Delete `build_mesh_from_organ_array` vs keep as deprecated wrapper**: Deleting it immediately causes import errors in archive scripts. Keeping it as a DeprecationWarning wrapper is recommended.

---

## Verification Plan

### Automated
```bash
conda run -n digital-crops python plant_recon/eval/generate_multimodal_outputs.py
```
— Confirm render outputs are identical to (or better than) current outputs

### Manual
- Verify `fig8_multimodal_depth_mask.png` generation
- Confirm `render_organ_array()` path in `helios_pytorch_renderer.py` functions normally
