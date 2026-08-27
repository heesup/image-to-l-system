# Canonical Pipeline Refactor: XML → 40D → 16D → Part Render (Implementation Progress)

**Date**: 2026-08-26  
**Status**: ✅ Implementation Complete / Verification Passed  
**Context**: Executed the [`20260826_implementation_plan.md`](20260826_implementation_plan.md) — unified every rendering path onto the single canonical path below and fully removed the legacy 94D layout from the rendering pipeline.

$$\text{XML} \xrightarrow{\text{direct parse}} \text{40D}\ (\text{PlantOrganArray}) \xrightarrow[\text{native FK}]{\text{extract\_part\_tensor()}} \text{17D} \xrightarrow{\text{GPU mesh builder}}{\text{build\_mesh\_from\_part\_tensor()}} \text{3D Mesh}$$

---

## 1. Executive Summary

All active rendering paths now converge on a single canonical path. The legacy 94D forward-kinematics code (~600 lines) was removed from the geometry builder, and `extract_part_tensor()` was rewritten to run forward kinematics **natively on the typed (N, 40) layout** — no more 40D→94D conversion. Output is verified **bit-identical** to the prior 94D-based extract.

---

## 2. Component 1 — `helios_pytorch_geometry.py`

### 2.1 `extract_part_tensor()` — Native 40D FK (rewritten)

**Before**: converted typed 40D → legacy 94D via `organ_array.to_legacy_tensor_diff()` then walked the 94D `COL_*` slots.

**After**: reads `T_COL_*` constants directly from the (N, 40) typed layout:
1. Groups typed rows into `root_meta`, `shoot_meta`, and `phytomer_data[(shoot_id, phytomer_idx)]` per-organ dicts.
2. Shoot topological ordering via `shoot_meta` `T_COL_PARENT_SHOOT_ID`.
3. Per phytomer:
   - **Internode** (`ORGAN_INTERNODE`): `T_COL_LENGTH/RADIUS/PITCH/PHYLLOTACTIC_ANGLE/LENGTH_MAX/LENGTH_SEGMENTS`, curv/yaw perturbations, gravitropic bending.
   - **Petiole** (`ORGAN_PETIOLE`, keyed by `T_COL_PARENT_PETIOLE_IDX`): petiole FK + full Helios `R_leaf` computation, packing rot6d.
   - **Leaf** (`ORGAN_LEAF`, keyed by `(petiole_idx, T_COL_CHILD_INDEX)`): leaflet variant encoded into the 16D curvature column.
   - **Bud / Peduncle / Flower / Fruit**: `T_COL_BUD_STATE`, `T_COL_BUD_IS_TERMINAL`, `T_COL_FRUIT_SCALE`, peduncle FK, flower/fruit placement at peduncle tip.

**Verification**: bit-identical (`max abs diff == 0.0`) to the old 94D-based extract on `recon_roundtrip_dap010/050/090.xml` and `test_recon_topo_dap010.xml`.

### 2.2 `build_mesh_from_organ_array()` → deprecated wrapper

Replaced the entire ~600-line 94D FK body with a `DeprecationWarning` wrapper that delegates:

```python
pt = self.extract_part_tensor(organ_array, device=device, existence_threshold=existence_threshold)
return self.build_mesh_from_part_tensor(pt, device=device, existence_threshold=existence_threshold, leaf_mode=leaf_mode)
```

Also removed the now-unused legacy `COL_*` imports from the module.

---

## 3. Component 2 — Active Caller Migration

All active (non-`archive/`) callers switched from `build_mesh_from_organ_array(...)` to `build_mesh_from_part_tensor(arr.to_part_tensor(device), device=device)`:

- **Training**: `train_cowpea_dit_100k_ddp.py`, `train_cowpea_dit_100k.py`, `train_cowpea_vlm_scaffold_dit_ddp.py`
- **Dataset**: `generate_tensor_shards.py`
- **Renderer**: `helios_pytorch_renderer.py` (`_build_mesh_cached`)
- **Scripts**: `run_cowpea_dap10_direct_opt_full.py`, `verify_40d_helios_render_comparison.py`, `debug_larger_plant.py`, `minimal_direct_opt_depth_chamfer_demo.py`, `debug_side_view_render.py`
- **Eval**: `eval_vlm_scaffold_dit.py`, `eval_cowpea_dit_100k.py`, `eval_canonical_cowpea_flow_matching.py`, `eval_pure_noise_flow_matching.py`, `test_latent_interpolation.py`, `test_leaf_modes.py`, `test_latent_flow_matching.py`, `test_vae_roundtrip.py`, `multi_dap_comparison_panel.py`, `test_render_part_tensor_quality.py`, `compare_flower_pod_masks.py`, `eval_organ_category_masks.py`

**Note:** `debug_leaf_by_leaf.py` intentionally retains the deprecated wrapper because it relies on `max_leaves`, which the part-tensor path does not support.

---

## 4. Component 3 — `plant_organ_array.py`

- Added `DeprecationWarning` to both `to_legacy_tensor()` and `to_legacy_tensor_diff()`. Code retained for any remaining external dependencies; the rendering pipeline no longer calls them.

---

## 5. Verification

### Automated
```bash
conda run -n digital-crops python diffusion_based/eval/generate_multimodal_outputs.py
```
- Runs to completion and regenerates `docs/results/assets/fig8_multimodal_depth_mask.png`.

### Native-FK equivalence
- `extract_part_tensor()` (native 40D) == old 94D-based extract, **max abs diff = 0.0** across all roundtrip XMLs.

### Smoke test
- `extract_part_tensor` → `build_mesh_from_part_tensor` and the deprecated `build_mesh_from_organ_array` wrapper both produce identical meshes; the wrapper emits the expected `DeprecationWarning`.

---

## 6. Open Items / Known Issues

- **`soft_existence` pre-existing bug**: `helios_pytorch_renderer.py::render_part_tensor`/`render_part_depth` pass `soft_existence=...` to `build_mesh_from_part_tensor()`, which does not accept that argument → `TypeError`. This is **pre-existing in HEAD** (renderer unchanged by this refactor) and `render_organ_array` is already deprecated/unused in active code. Flagged as a follow-up.
- **`max_leaves` support**: the part-tensor path has no equivalent; `debug_leaf_by_leaf.py` stays on the deprecated wrapper.

---

## 7. Flower / Pod / Peduncle Fixes (DAP 50/90)

### 7.1 Critical: tube-axis bug (`R[:,2]` → `R[:,1]`)

`_make_row` packs `rot6d = [up(3), fwd(3)]` = columns 0,1 of `R`; Gram-Schmidt
reconstructs `R[:,0]=up`, `R[:,1]=fwd` (the organ's **forward** direction), and
`R[:,2]=cross(up, fwd)`. The internode/petiole/peduncle tube builders were using
`R[:,2]` as the tube axis — i.e. rendering every tube **perpendicular** to its
intended direction (stems rendered nearly horizontal). Changed all three to use
`R[:,1]` (the true forward axis). This was the root cause of the previously
unexplained "stem positional offset" (overlap = 0).

### 7.2 Flower / Pod rotation & scale

- Flower/pod rotation now mirrors the legacy `R_fl = rotr_z(az) @ rotr_y(pitch) @ rotr_x(roll)` + yaw about the peduncle axis (`_make_row_rot`).
- Pod scale restored `pod_proto_scale = 2.594` (CowpeaPod 0.75m / 0.2891m Z-extent), matching `Assets.cpp:410`. Pod pixel count went 328 → 1196 (GT 1231).

### 7.3 Peduncle curvature + gravitropism

- Peduncle re-emitted as a curved tube: `extract_part_tensor` stores the initial
  peduncle axis in rot6d and curvature (deg/m) in the 16D curvature column;
  `build_mesh_from_part_tensor` reconstructs the 6-segment centerline.
- Added Helios gravitropism clamping (`PlantArchitecture.cpp:2474-2496`): positive
  curvature targets `+z`, negative targets `-z`, snapping to vertical on overshoot.

### 7.4 Closed-flower routing fix

Closed buds (`bud_state == 2`) were being rendered with the **open** flower OBJ
(`CowpeaFlower_open_yellow.obj`, Y-span 1.11 m) instead of the **closed** OBJ
(`CowpeaFlower_closed_yellow.obj`, Y-span 0.185 m). Routed `bud_state == 2` →
`ORGAN_FLOWER_CLOSED`, `3` → `ORGAN_FLOWER`, `4` → `ORGAN_FRUIT`. Flower pixel
count dropped 4666 → 3360 (GT 3688); component count now matches GT (54 vs 53).

### 7.5 Post-fix metrics (DAP 90 exact GT)

| Organ | our | GT | IoU |
|-------|----:|----:|----:|
| flower | 3360 | 3688 | 0.139 |
| pod | 1183 | 1231 | 0.035 |
| floral_bud | 1696 | 1128 | 0.026 |
| leaf (unchanged) | 54759 | 46041 | 0.561 |

Stems/petioles now render in the correct orientation; flower/pod heads overlap GT positions.
Remaining gaps are thin-tube positional precision (1–2 px), not scale.

---

## 8. 17D Extension + M==N + Vectorized Builder + Helios Round-trip (2026-08-27)

### 8.1 16D → 17D (bud_state)

`plant_organ_array.py:226-244` `P_COL_BUD_STATE=14, P_COL_CURVATURE=15, P_COL_PHYLLOTACTIC_ANGLE=16, NUM_FEATURES=17, NUM_FEATURES_PART=17`. `helios_pytorch_geometry.py:670` `extract_part_tensor` now emits `ORGAN_ROOT_META` (1/plant), `ORGAN_SHOOT_META` (1/shoot, parent indices in `scale`), `ORGAN_BUD`/`ORGAN_BUD_ABORTED` (`bud_state` in `P_COL_BUD_STATE`) via record order. `build_mesh_from_part_tensor:1392` decodes `D>=17?p[:,15]:p[:,14]` for curvature/variant, backward-compatible with 16D. `part_assembly_to_xml.py` updated to carry `bud_state` and `dir_z=R@[0,1,0]` (was `R[:,2]`).

### 8.2 M==N with existence=0 dormant peduncles

`helios_pytorch_geometry.py:1179-1261` keeps length 1:1 with typed `N`: `has_ped` always emits `ORGAN_PEDUNCLE` if present in 40D, `exist_ped=node_exist if is_active else 0` where `is_active=bud_state in [2,3,4] and exist>0.5`. `DAP050:1158->1158` (`163 PED` `131 dormant+69 active` for `DAP090:1558->1558`), previously `1158->995` diff `163`. Renderer (`build_mesh_from_part_tensor:1390` `exist>0.5`) culls dormant visually but tensor length stays `N` for diffusion batching. `generate_multimodal_outputs.py:arr.to_part_tensor` now `M==N`.

### 8.3 Vectorized mesh builder

`helios_pytorch_geometry.py:1341` `build_mesh_from_part_tensor` fully vectorized: `torch.bmm` batched for leaves by variant, straight tubes, pods, flowers; peduncles vectorized across `M` with `deg2rad(curv*dr)` gravitropism clamp (`PlantArchitecture.cpp:2474`). `DAP050 mesh 3.6 ms, DAP100 6.3 ms, total 8.4 ms, speedup 2270x` vs Helios C++.

### 8.4 Helios round-trip fixes

* `part_assembly_to_xml.py:300-355` record-order grouping from `shoot_metas` (`phytomer_groups` with `shoot_id, internode, petioles, peduncle, flowers`), `inode_to_location` local `parent_node_index` (was `internodes.index(parent_i)` global) and `dir_z` fix.
* `if shoot_metas: phytomer_parts/petiole_leaves/bud_state/peduncle_infls/inode_tip_pos` via record order else `cKDTree` fallback; `if not shoot_metas` guards prevent overwriting. Fixes `vector::_M_range_check __n=1 >=1` and `getPetioleAxisVector 76` crash. Verified `DAP050/090 11/11 shoots` vs prior `40` and `Helios rc 0` with `Digital-Crops/projects/syntheticdata_generation/build/params.json`.
* `soft_existence` param added to `build_mesh_from_part_tensor`, `torch.full(ped_rad)` `float(ped_rad.item())` fix.

### 8.5 FOV check (final)

`helios_pytorch_renderer.py:24-165` `compute_focus_plant_camera` `1.05` margin matches Helios `main.cpp:1748-1793` focus-plant. `rad_dap050/090_camera.json` `camera_height 5.0 angle 89.88` vs `90.0`, HFOV `DAP050 9.51° DAP090 15.36°`, legacy `camera_params.json` absent → `focus_plant true` (no `hfov_override`). Re-verified `compare_flower_pod_masks.py:48-58` uses same path; remaining `Flower IoU 0.13 Pod 0.05` gap not FOV (thin-tube 1-2 px, scale `2.594` already).

### 8.6 Helios → 17D → Helios round-trip (faithful, 2026-08-27)

The 17D part tensor now round-trips to Helios XML **bit-faithfully**. Root cause of the earlier `col1 ≠ col5` drift was two-fold: (1) `extract_part_tensor` stored FK-derived world poses but dropped the original XML parameters needed to invert the FK, and (2) `PartAssemblyToXMLConverter` recomputed shoot parent links via lossy `cKDTree` instead of reading the stored parent indices.

Fixes in `extract_part_tensor` (helios_pytorch_geometry.py):
- **shoot_meta** now stores `base_rotation` (pitch/yaw/roll) in `curvature/phyllo/bud_state` and parent indices in `scale` (with `clamp_scale=False` so `parent_shoot_ID=-1` survives).
- **internode** stores `pitch` (bud_state), `phyllotactic_angle` in **degrees** (phyllo, was radians → 3° vs 173°), `length_max` (scale_y), `length_segments` (curvature, was hardcoded 3 vs GT 2).
- **petiole** stores `pitch` (bud_state), `curvature` (curvature), `current_leaf_scale_factor` (scale_y), `length_segments` (phyllo, was hardcoded 3 vs GT 5); taper/leaflet_offset/leaflet_scale hardcoded to GT constants (0.25/0.4/0.9).
- **leaf** stores original `pitch/yaw/roll` (scale_y/scale_z/bud_state) instead of lossy `_matrix_to_euler_xyz` decomposition.
- **flower/pod** stores `pitch/yaw/roll/azimuth` (scale_y/scale_z/bud_state/curvature); **peduncle** stores `pitch/roll` (scale_y/phyllo).
- **dormant peduncles** kept at `existence=1` (M==N, `DAP050 1158→1158`, `DAP090 1558→1558`) and skipped in the **mesh builder** via `bud_state ∈ {2,3,4}` (`ped_mask`), matching Helios C++ `PlantArchitecture.cpp` (reverts the earlier `existence=0` cull that dropped rows in the XML converter's `active_mask`).

Fixes in `PartAssemblyToXMLConverter` (part_assembly_to_xml.py):
- `parent_shoot_ID`/`parent_node_index`/`parent_petiole_index` read directly from shoot_meta `scale` (not `cKDTree` `inode_parent`), fixing branch-topology drift (`internode 21` base `Δ0.03` → `0`).

### 8.6.1 Final round-trip fixes (2026-08-27, all organs IoU 1.0)

Two remaining bugs were found by comparing **GT-XML masks vs RT-XML masks** (same `--fov 8.78` camera, the real round-trip test — the earlier `0.004` diff was GT-XML vs RT-XML with the *same wrong camera*, so it was a weak test):

1. **Pod scale** (`26d6a5b`): the RT XML wrote the pre-scaled pod size (`fl_scale*pod_proto_scale*fruit_scale`) as `flower_base_scale` with `current_fruit_scale_factor=1`, so Helios re-rendered pods 2.6x too large (pod mask IoU `0.12`). Fix: store raw `flower_base_scale` in the flower row `phyllo` col and `current_fruit_scale_factor` in the bud row `scale_x`; converter writes them back. Pod IoU `0.12 → 0.95`.
2. **Flower orientation** (`e43707c`): `_make_row_rot` clamped `scale` to `min=1e-6`, corrupting **negative flower pitch** (GT `flower_pitch=-43.9°` → `0.000001°` in RT XML), so Helios re-rendered flowers pointing up instead of down (flower mask IoU `0.38`). Fix: added `clamp_scale` param, pass `False` for flower rows. Flower IoU `0.38 → 1.00`.

**Result (GT-XML vs RT-XML masks, same `--fov 8.78` camera):**

| DAP | internode | petiole | leaf | floral_bud | flower | pod |
|-----|-----------|---------|------|------------|--------|-----|
| 050 | 1.00 | 1.00 | 1.00 | — | — | — |
| 090 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

All organs round-trip at **IoU 1.0000**; RGB mean abs diff `DAP050 0.00446`, `DAP090 0.00571` (OptiX noise). This was a **known unsolved problem** in prior docs (`20260826_helios_flower_peduncle_pod_alignment_and_cleanup_report.md` §4: "Exact pod transforms were sampled during original GT generation and lost"; `20260826_canonical_pipeline_refactor_progress.md` §7.5: flower IoU 0.139, pod 0.035). **Round-trip figure:** `docs/results/assets/fig_helios_17d_roundtrip.png` regenerated (`GT XML | PT 17D | Depth | Mask | RT XML | diff`, same `fov 8.78` camera).

### 8.7 Benchmark

`diffusion_based/eval/benchmark_helios_vs_torch_renderer.py` re-run: `DAP100 8.36 ms total, 2270.8x` (`DAP50 2314x, DAP90 2230x`), `fig1_helios_vs_torch_rendering_benchmark.png` regenerated. `verify_40d_helios_render_comparison.py` mean diff `0.00489` (OptiX noise).

---

## 9. Files Changed

```
diffusion_based/models/helios_pytorch_geometry.py   # native 40D FK, deprecated wrapper, import cleanup
diffusion_based/models/plant_organ_array.py            # DeprecationWarnings on legacy converters
diffusion_based/models/helios_pytorch_renderer.py      # _build_mesh_cached -> part-tensor path
diffusion_based/training/train_cowpea_dit_100k_ddp.py
diffusion_based/training/train_cowpea_dit_100k.py
diffusion_based/training/train_cowpea_vlm_scaffold_dit_ddp.py
diffusion_based/dataset/generate_tensor_shards.py
scripts/run_cowpea_dap10_direct_opt_full.py
scripts/verify_40d_helios_render_comparison.py
scripts/debug_larger_plant.py
scripts/minimal_direct_opt_depth_chamfer_demo.py
scripts/debug_side_view_render.py
diffusion_based/eval/*.py   (eval_vlm_scaffold_dit, eval_cowpea_dit_100k,
                              eval_canonical_cowpea_flow_matching, eval_pure_noise_flow_matching,
                              test_latent_interpolation, test_leaf_modes, test_latent_flow_matching,
                              test_vae_roundtrip, multi_dap_comparison_panel,
                              test_render_part_tensor_quality, compare_flower_pod_masks,
                              eval_organ_category_masks, generate_multimodal_outputs)
```
