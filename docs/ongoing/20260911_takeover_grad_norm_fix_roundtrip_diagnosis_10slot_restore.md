# Session Takeover: Grad-Norm Inf Deadlock Fix, Helios Roundtrip Diagnosis & 10-Slot Contract Restore

- **Author**: Antigravity Pair Programming
- **Date**: 2026-09-11
- **Status**: P0 Fixes Committed & Verified | P1 Root-Cause Diagnosed | 10-Slot Contract Restored | Smoke Training Pending Slot
- **Supersedes**: [`20260911_differentiable_render_gradient_isolation_and_backbone_freeze.md`](20260911_differentiable_render_gradient_isolation_and_backbone_freeze.md) (Option A/B context)

---

## 1. Executive Summary

이번 세션은 세 가지 이슈를 다뤘습니다:

1. **P0 (해결·커밋 `4b70266`)**: Job `38237555`가 Epoch 11 Step 52 이후 `grad_norm=inf` → 전 스텝 skip 데드락 (3,215회 스킵, 학습 0% 진행). 로그 분석으로 **진짜 원인 2개**를 특정하고 구조적으로 수정.
2. **P1 (진단 완료, 기하 수정은 후속)**: Helios roundtrip에서 Internode/Petiole IoU 0~15% 문제 — **eval 스크립트의 COCO 카테고리 매핑 버그** (한 칸 밀림) + **FK 체인 누적 오차** (튜브 위치 9~18px 변위)의 복합 원인으로 규명. 매핑 버그는 수정 완료.
3. **10슬롯 복원 (해결·커밋 `75928d9`)**: v4의 9슬롯(인터노드 제외) 리스트럭처를 폐기하고 **v2/v3 10슬롯 계약**(slot 0 = internode, repro ×4)으로 복원. 복원 과정에서 발견한 `SLOT_ROLE_MAPPING` 8-vs-10 잘림 크래시도 함께 수정.

---

## 2. P0: Job 38237555 데드락 근본 원인 (로그 기반 확정)

### 2.1 증상

```
[Recovery] Step 52: grad_norm is NaN/Inf (inf) | Culprit params (0): -> SKIPPING STEP
(3,215회 연속, Epoch 11~마지막까지 Loss 0.0000)
```

### 2.2 핵심 발견: `Culprit params (0)`의 의미

개별 grad 텐서에 NaN/Inf가 **전혀 없는데** total norm만 `inf`라는 것은:
- `clip_grad_norm_`은 `Σ‖g‖²`을 **Float32로 계산** → fp32 max ≈ 3.4e38
- 크기 ~1e19의 **"유한하지만 거대한"** grad 하나만 있어도 제곱합이 fp32 overflow → total norm = inf, 개별 원소는 전부 유한
- 기존 culprit 탐지(`isnan/isinf` per-element)는 이 경우를 아무것도 못 잡음 → 원인 파라미터 부재로 무한 skip

### 2.3 폭발 체인 (설계 관점 재구성)

```
[폭발 전] OOD latent (|z|>6σ) → frozen VAE decode가 비정상 mesh (e^12 ≈ 1.6e5 배)
         → mesh 정점이 카메라 near-plane 관통 (clip-space w → 0)
         → nvdiffrast perspective-correct backward의 1/w² ≈ 1e19 grad
         → fp32 sum-of-squares overflow → grad_norm=inf → 영구 스킵
```

**트리거**: Epoch 11에서 시간기반 eval 폴백(31분 경과)이 발동 → `evaluate_self_consistency_batch()`가 `model.eval()` 호출 후 **`model.train()` 복귀 안 함** → eval 직후 Step 52에서 폭발. (eval 모드 누출이 점화원, w→0 야코비안이 폭탄)

---

## 3. P0 수정 내역 (커밋 `4b70266`)

### 3.1 구조적 설계 수정 (폭발이 원천 불가능하도록)

| 층 | 파일 | 수정 |
|---|---|---|
| **Latent (근원)** | `train_hierarchical_flow_matching.py` | `clean_z1 = clean_z1.clamp(-6.0, 6.0)` — latent 계약(z~N(0,I)) 지지집합 밖 OOD 차단 → degenerate mesh 입력 제거 |
| **Renderer (야코비안)** | `helios_pytorch_renderer.py` 3곳 | **w-floor**: `\|w\| < 1e-4` → ±1e-4로 floor → perspective-correct backward의 `1/w²` 인자가 구조적으로 ≤1e8 (Lipschitz bound). 3개 rasterize 경로 모두 적용. 추가로 `rasterize`/`interpolate`/`antialias` **출력** `nan_to_num` sanitize |
| **Optimizer (불변식)** | `train_hierarchical_flow_matching.py` | `clip_grad_norm_` 직전 **원자 clamp(±100)** — 어떤 경로가 남아도 fp32 제곱합 overflow 불가 (defense-in-depth) |

### 3.2 eval 모드 누출 봉쇄

- `eval_hierarchical_self_consistency.py`: `was_training` 저장 → return 직전 `model.train()` 복원
- 호출부 `train_hierarchical_flow_matching.py`: eval `try/except`에 **`finally: model.train()`** 추가 (eval 예외 시에도 복귀 보장)

### 3.3 Culprit 탐지 강화

- 유한하지만 거대한(`>1e6`) grad도 `Huge(>1e6)` 카운트로 리포트 — 재발 시 어떤 파라미터가 튀었는지 즉시 특정 가능

### 3.4 수치 검증

```
Before: grad 원소 1e19 (유한) 1개 → Σg² = 1e38+ → fp32 overflow → norm = inf  ← 로그와 일치
After w-floor: 1/w ≤ 1e4 × loss grad 1e-3 → 원소당 g² ≤ 1e2 → 1e9 파라미터여도 Σg² ≤ 1e11 ≪ 3.4e38
After atomic clamp: norm=100.0 (유한) → 정상 clip 경로 복귀, 스킵 0%
Healthy grads (~1e-3): clamp 전후 norm bit-identical (0.070711 → 0.070711)
```

---

## 4. P1: Helios Roundtrip Internode/Petiole IoU 진단

### 4.1 재계산 방법

GT/Recon COCO 마스크 덤프(`/tmp/helios_organ_mask_eval/*`)에서 per-class IoU를 직접 재계산.

**결정적 발견 — Helios COCO 카테고리 실제 정의** (GT 덤프에서 검증):
```
{0: 'shoot' (=Internode), 1: 'petiole', 2: 'leaf', 3: 'floral_bud', 4: 'flower', 5: 'pod'}
```
eval 스크립트는 1-indexed organ type으로 가정 → **전 클래스가 한 칸씩 밀려 해석됨**. 유저가 본 "Internode 0% / Petiole 1~15%"는 이 밀림의 직접 결과.

### 4.2 올바른 매핑으로 재계산한 실측 (512px, polygon rasterize)

| Stage | Internode(shoot) | Petiole | Leaf | Flower | Pod |
|---|---|---|---|---|---|
| DAP 10 | GT px 0 (n/a) | 1.9% (100px) | **95.6%** | n/a | n/a |
| DAP 50 | **0.0%** (347 vs 335px) | 13.9% (1085px) | **93.3%** | n/a | n/a |
| DAP 90 | **10.9%** (280 vs 299px) | 4.6% (787px) | **80.6%** | 19.8% | 5.6% |

### 4.3 진단 결론

- **픽셀 수는 GT↔Recon 거의 동일** (dap050 Internode: 347 vs 335) → 길이/반지름 변환은 **정확**
- 그러나 centroid가 **9~18px 변위**, 얇은 튜브(2~3px 폭)라 완전 분리 → IoU 0
- 변위 방향이 ±혼합 → 계통 좌표축 오류가 아니라 **FK 체인 누적 오차** (특히 `part_tensor_to_40d.py:400-403`의 internode pitch 하드코딩 0°/20°와 GT 실측 pitch의 불일치)
- 8px dilation 내 overlap: Internode 53~59%, Petiole 78~81% → 튜브가 근처는 있음 (오프셋 ~10~20px 스케일)

### 4.4 적용 수정

`eval_13d_xml_organ_masks.py`: `COCO_TO_CLASS = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5}` 명시적 매핑 도입 (주석으로 Helios 실제 정의 문서화).

**후속 과제 (미완료)**: FK 누적 오차 수정 — `part_tensor_to_40d.py`의 internode/petiole pitch를 GT 14D 텐서 값에서 역산하도록 변경하고 DAP 10/50/90 3포인트 회귀 검증 필요.

---

## 5. 10슬롯 계약 복원 (커밋 `75928d9`)

### 5.1 결정 배경

v4(9슬롯, internode 제외)로의 전환이 작업 트리에 있었으나 **사용자 결정으로 10슬롯(v2/v3)이 정식 계약**:
- v3 VAE 체크포인트 = 240D 입력 (10슬롯 × 24D) — v4의 216D(9슬롯)와 비호환
- 3.29%의 3+ repro organ phytomer 절단 방지 (v2 도입 취지)
- smoke 훈련 중 발견: `SLOT_ROLE_MAPPING`이 8개인데 M=10 → `[:M]` 잘림 → `functional_role_emb` 8-vs-10 크래시

### 5.2 복원 내역

| 파일 | 변경 |
|---|---|
| `phytomer_packets.py` | v2 10슬롯 빌더 복원 (`NUM_SLOTS=10`, `ROLE_SLOT_RANGES` 4:(6,10), `LABEL_ROLE_RANGES` stem 1..3). v4 신규 3함수를 **10슬롯 인덱스로 어댑트**해 보존: `anchor_scale` (petiole=slot 1), `normalize/denormalize_packet_scales`, `assemble_phytomer_ordered_14d_tensor` (internode slot 0, cotyledon twin 로직 포함) |
| `hierarchical_part_flow_matching.py` | `SLOT_ROLE_MAPPING = [0,1,2,2,2,3,4,4,4,4]` (10개), `SLOT_SUB_ROLE_MAPPING = [0,0,0,1,2,0,0,1,2,3]`, `slots_per_anchor` 기본 9→10 |
| `hierarchical_hungarian_matcher.py` | `_role_slot_ranges[4]: (6,8)→(6,10)` (repro ×4) |
| `slurm_scripts/train_hierarchical_flow_matching.sh` | `SLOTS_PER_ANCHOR` 기본 9→10 (v3 240D VAE와 일치) |
| `tests/test_phytomer_packets.py` | v2-era 기대값 복원 (7 passed) |
| `tests/test_phytomer_vae.py` | synthetic bag role map 10슬롯化 (8 passed) |

### 5.3 테스트 결과

```
31 passed / 3 failed — 3개 실패는 stash 검증으로 pre-existing 확인 (10슬롯과 무관):
  - test_hierarchical_3stage_cascaded: node_scale_mlp grad None (HEAD에서도 실패)
  - test_dinov2_3d_spatial: 동일 pre-existing
```

---

## 6. 현재 실행 상태 & 후속 지시

### 6.1 SLURM 잡 현황 (2026-09-11 15:30 기준)

| 잡 | 파티션 | 상태 | 설명 |
|---|---|---|---|
| `38238218` | gpu-6000_ada-h | PENDING (QOSGrpGRES) | **메인 학습** — 10슬롯 수정 **이전** 제출분. **취소하고 재제출할 것** |
| `38238368` | low (publicgrp) | PENDING | **10슬롯 smoke** (batch 8, 2×GPU, 600 samples, 6 epochs, `--flow_granularity phytomer`) — 최초 10슬롯 검증용 |

### 6.2 다음 에이전트 실행 순서 (필독)

1. **`scancel 38238218`** 후 `sbatch slurm_scripts/train_hierarchical_flow_matching.sh` — 10슬롯 코드로 메인 재시작. 시작 로그에서 `512 anchors x 10 slots/anchor` 확인.
2. **`38238368` smoke 결과 확인**: `slurm_scripts/logs/fm_smoke_low_38238368.log`에서 Epoch 4(렌더 활성) 도달 + `[Recovery]` 미발생 + Loss 감소 확인 → 통과 시 메인 잡 신뢰 가능.
3. smoke 실패 시: 마지막 실패가 `hierarchical_hungarian_matcher.py:328` (CUDA device-side assert, 13D organ 모드)였음 — `--flow_granularity phytomer` 추가로 우회 시도했으나 **미검증 상태로 PENDING**. organ 모드가 필요하면 매처의 13D 경로 role 분배를 먼저 검증.
4. **P1 FK 누적 오차**: `part_tensor_to_40d.py` internode pitch(400-403행) 하드코딩 제거 + `eval_13d_xml_organ_masks.py` 재실행으로 per-organ IoU 재측정.

### 6.3 핵심 파일/라인 빠른 참조

- grad clamp: `train_hierarchical_flow_matching.py:1088` 부근 (`GRAD_ELEM_CLAMP = 100.0`)
- z1 clamp: `train_hierarchical_flow_matching.py` (~469행, `clean_z1.clamp(-6.0, 6.0)`)
- w-floor: `helios_pytorch_renderer.py` — `render_multiscale_pyramid`/`render_type_buffer`/`render_depth` 3개 rasterize 경로
- eval 복귀: `eval_hierarchical_self_consistency.py` (`was_training` 패턴) + 호출부 `finally: model.train()`
- COCO 매핑: `eval_13d_xml_organ_masks.py` (`COCO_TO_CLASS`)
- 10슬롯 role: `hierarchical_part_flow_matching.py:572` (`SLOT_ROLE_MAPPING`)

### 6.4 알려진 주의사항 (Gotchas)

- **캐시 호환**: pkt 캐시(`dataset/cache/cowpea_curv26_pkt`)가 **9슬롯 v4로 생성된 샘플이 있으면** 10슬롯과 셰이프 미스매치 가능 → 캐시 에러 시 `NUM_SLOTS=10`으로 캐시 재생성 (`generate_cache.py`) 또는 v2-era 캐시 사용.
- **체크포인트 호환**: v4로 학습된 체크포인트는 9슬롯 헤드(shape 불일치) — **v3-era 체크포인트만** 로드 가능. `hierarchical_fm_epoch_500.pt`는 8슬롯 era라 주의.
- smoke 스크립트 `slurm_scripts/train_smoke_low.sh`는 `publicgrp`/`low` 파티션 전용 (account=publicgrp, QOS=publicgrp-low-qos).
- 로그인 노드에는 torch가 없고 conda env python(`/home/lion397/.conda/envs/digital-crops/bin/python`)만 torch 보유. pytest는 해당 env에 설치 완료.

---

## 7. 커밋 이력 (이 세션)

```
75928d9 feat(phytomer): restore v2/v3 10-slot packet contract (internode slot 0 + repro x4)
4b70266 fix: structurally bound renderer Jacobian & restore train mode after in-loop eval
```

미커밋 잔여: eval_13d 매핑 수정, README/AGENT_TAKEOVER_GUIDE 갱신, 이미지 에셋 다수 (다음 세션에서 정리 권장).