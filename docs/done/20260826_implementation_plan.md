# Canonical Pipeline Refactor: XML → 40D → 16D → Part Render

## 목표

모든 렌더링 경로를 단일 canonical path로 통일:
```
XML → PlantOrganArray (40D) → extract_part_tensor() → 16D → build_mesh_from_part_tensor()
```

94D (`NUM_FEATURES_LEGACY = 94`)는 렌더링 pipeline에서 완전히 제거.

---

## 변경 범위

### Component 1: `helios_pytorch_geometry.py`

#### [MODIFY] `extract_part_tensor()` — 40D 네이티브 FK로 전면 재작성

현재: `organ_array.to_legacy_tensor_diff()` → 94D → COL_* 상수로 FK  
목표: 40D typed tensor를 직접 T_COL_* 상수로 읽어 FK 수행

**40D FK 알고리즘:**
1. organ rows를 `(shoot_id, phytomer_idx)` 기준으로 그룹핑
2. `ORGAN_SHOOT_META` rows에서 shoot rotation(pitch/yaw/roll), parent_shoot_id 읽기
3. Shoot topological 순서 정렬 (parent_shoot_id 기준)
4. 각 phytomer에서:
   - `ORGAN_INTERNODE` → `T_COL_LENGTH`, `T_COL_RADIUS`, `T_COL_PITCH`, `T_COL_PHYLLOTACTIC_ANGLE`, curv/yaw pert
   - `ORGAN_PETIOLE` (T_COL_PARENT_PETIOLE_IDX = 0 or 1) → petiole FK
   - `ORGAN_LEAF` (T_COL_CHILD_INDEX = 0,1,2) → 완전한 R_leaf 계산
   - `ORGAN_BUD` → bud_state, is_terminal, fruit_scale
   - `ORGAN_PEDUNCLE` → peduncle FK
   - `ORGAN_FLOWER`/`ORGAN_FRUIT` (T_COL_CHILD_INDEX로 구분) → flower pose

**삭제:**
```python
# 이 블록 완전 제거
if organ_array.is_typed:
    legacy_tensor = organ_array.to_legacy_tensor_diff()
    organ_array = PlantOrganArray(legacy_tensor, raw_metadata=[])
```

#### [MODIFY] `build_mesh_from_organ_array()` — deprecated wrapper로 변환

현재: 94D FK로 직접 mesh 생성  
목표: `extract_part_tensor()` → `build_mesh_from_part_tensor()` 로 위임하는 wrapper

```python
def build_mesh_from_organ_array(self, organ_array, device, species=None, leaf_mode=None, ...):
    """Deprecated: Use build_mesh_from_part_tensor(organ_array.to_part_tensor()) instead."""
    import warnings
    warnings.warn("build_mesh_from_organ_array is deprecated. Use build_mesh_from_part_tensor.", DeprecationWarning)
    pt = self.extract_part_tensor(organ_array, device=device)
    return self.build_mesh_from_part_tensor(pt, device=device, leaf_mode=leaf_mode)
```

기존 94D FK 코드 (~600줄) 삭제.

---

### Component 2: 활성 코드 caller 업데이트

`build_mesh_from_organ_array` → `build_mesh_from_part_tensor(arr.to_part_tensor())` 로 전환:

#### [MODIFY] `diffusion_based/training/train_cowpea_dit_100k_ddp.py` (L137)
#### [MODIFY] `diffusion_based/training/train_cowpea_dit_100k.py` (L114)  
#### [MODIFY] `diffusion_based/training/train_cowpea_vlm_scaffold_dit_ddp.py` (L172)
#### [MODIFY] `diffusion_based/dataset/generate_tensor_shards.py` (L177)
#### [MODIFY] `diffusion_based/models/helios_pytorch_renderer.py` (L742, L752)
#### [MODIFY] `scripts/run_cowpea_dap10_direct_opt_full.py` (L134, L272)
#### [MODIFY] `scripts/verify_40d_helios_render_comparison.py` (L107)
#### [MODIFY] `scripts/debug_larger_plant.py` (L46)
#### [MODIFY] `scripts/minimal_direct_opt_depth_chamfer_demo.py` (L137, L153, L196)
#### [MODIFY] `scripts/debug_side_view_render.py` (L47)
#### [MODIFY] `diffusion_based/eval/` — 활성 eval 스크립트들

`archive/` 내 파일은 건드리지 않음.

---

### Component 3: `plant_organ_array.py`

#### [DEPRECATE] `to_legacy_tensor()`, `to_legacy_tensor_diff()`
- DeprecationWarning 추가, 코드는 유지 (다른 의존성 있을 수 있음)
- 렌더링 pipeline에서는 더 이상 호출되지 않음

---

## Open Questions

> [!IMPORTANT]
> **`species` 파라미터 처리**: `build_mesh_from_organ_array`는 `species="cowpea"` 등을 받아 잎 OBJ 종류를 선택합니다. `build_mesh_from_part_tensor`는 현재 cowpea만 지원합니다. `extract_part_tensor`에 `species` 파라미터를 추가하거나, `HeliosPlantGeometryBuilder` 생성자에서 설정해야 합니다. **현재는 cowpea만 사용한다고 가정하고 진행해도 됩니까?**

> [!IMPORTANT]  
> **`build_mesh_from_organ_array` 삭제 vs deprecated wrapper 유지**: 바로 삭제하면 archive 스크립트들이 임포트 에러를 냅니다. DeprecationWarning wrapper로 두는 것을 추천합니다.

---

## Verification Plan

### Automated
```bash
conda run -n digital-crops python diffusion_based/eval/generate_multimodal_outputs.py
```
— 렌더 결과가 현재와 동일 (또는 더 나음) 확인

### Manual
- `fig8_multimodal_depth_mask.png` 생성 확인
- `helios_pytorch_renderer.py`의 `render_organ_array()` 경로가 정상 동작 확인
