# Session Takeover: Grad-Norm Inf Deadlock Fix, Helios Roundtrip Diagnosis & 10-Slot Contract Restore

- **Author**: Antigravity Pair Programming
- **Date**: 2026-09-11
- **Status**: P0 Fixes Verified (local smoke, pre-§7.9 code) | P1 Root-Cause Diagnosed | 10-Slot Contract Restored | **NEW P0: PhytomerVAE rotation architecture simplified to single `head_rot` path (§7.9) — ALL existing checkpoints (v2/v3/v4/xml) now fail to load, PhytomerVAE MUST be retrained from scratch before hierarchical FM training can resume** | **NEW: shoot-topology (SHOOT_META) loss identified as the dominant cause of Helios-roundtrip collapse, unrelated to the VAE — needs its own design fix (§7.8)** | Main job 38238371 cancelled (would have crashed on VAE load)
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

## 7. NEW (같은 날, 재시작 검증 중 발견): PhytomerVAE + Helios 전체 체인 라운드트립 — 심각한 재구성 붕괴

### 7.1 배경

§5에서 10슬롯 계약을 복원한 뒤, 실제 학습 재개(§6.2) 전에 **"Helios GT → XML → 14D → 10슬롯 패킷 → PhytomerVAE encode/decode → 14D → XML → Helios raytrace"** 전체 체인을 실측하는 스크립트가 하나도 없다는 것을 확인했다. 기존 체크들은 전부 이 체인의 일부만 커버함:

| 스크립트 | 커버 범위 | VAE 포함? | 실제 Helios raytrace? |
|---|---|:---:|:---:|
| `eval_13d_xml_organ_masks.py` | XML→14D→XML→Helios | ❌ | ✅ |
| `benchmark_organ_vae_roundtrip.py` | 구 OrganLatentVAE 라운드트립 | ✅(구버전) | ❌ (PyTorch 렌더러만) |
| `tools/phytomer_vae_visualizer.py` | PhytomerVAE 라운드트립 | ✅ | ❌ (PyTorch 렌더러만) |

신규 스크립트 `diffusion_based/eval/eval_phytomer_vae_helios_roundtrip.py`를 작성해 이 공백을 메움: DAP 10/50/90 XML → `PlantOrganArray.to_part_tensor()` → `encode_fm` → `build_phytomer_packets` (XML `phytomer_ids`로 정확한 마디 그룹핑) → `PhytomerVAE.encode/decode` (v3, `mu` 결정론적) → `denormalize_packet_scales(s_a)` → `assemble_packets` (결정론적 base 재구성) → `decode_packets` → `decode_fm` → `assemble_part_tensor_to_xml` → 실제 Helios C++ raytrace. "IK-only"(VAE 없이 14D→XML만) 라운드트립과 나란히 비교해 VAE 자체의 순수 기여도를 분리.

### 7.2 결과 (fig14_phytomer_vae_helios_roundtrip.png)

| Stage | IK-only FG IoU | **VAE-RT FG IoU** | IK-only mIoU | **VAE-RT mIoU** | IK PSNR | **VAE PSNR** |
|---|---:|---:|---:|---:|---:|---:|
| DAP 10 | 95.7% | **24.0%** | 48.3% | **6.1%** | 38.90dB | **21.69dB** |
| DAP 50 | 94.1% | **7.8%** | 36.0% | **2.4%** | 26.12dB | **6.62dB** |
| DAP 90 | 87.4% | **10.5%** | 21.7% | **1.5%** | 21.43dB | **9.07dB** |

(위 표는 `use_rot_branch=False`, 즉 공유 `head_rot`로 회전을 디코딩한 결과. `use_rot_branch=True`로 돌리면 다소 덜 나쁨 — DAP10 mIoU 17.3%, DAP50 6.4%, DAP90 3.8% — 하지만 **두 설정 모두 IK-only 대비 3~8배 악화**. 처음엔 v3가 `--rot-branch` 없이 학습됐을 것이라 추정해 `use_rot_branch=False`를 기본값으로 잘못 채택했었으나, §7.4의 패킷 레벨 측정으로 이 추정이 **틀렸음이 실측으로 확인됨** — `use_rot_branch=True`가 정답이고 지금은 두 스크립트 모두 기본값을 True로 수정함.)

### 7.3 시각적 관찰 — 핵심 단서

`fig14`를 보면 GT/IK-only는 조밀한 부시(bush) 형태를 유지하지만, **VAE 라운드트립 결과는 하나의 길게 늘어진 덩굴(vine)처럼 풀어져** 원래의 조밀한 위상(topology)과 전혀 다른 모양이 된다. 잎(Leaf) 개수/타입 분류 자체는 크게 틀리지 않은 것으로 보이나(조직 수 76/82, 982/995, 1417/1427로 GT에 근접), 공간 배치가 완전히 흐트러짐.

### 7.4 근본 원인 확정 (사용자 제안으로 실측 — Helios/XML 없이 26D 패킷 레벨만 비교)

사용자 제안: "VAE가 Phytomer를 잘 복원한다면, 26D 라운드트립 오차를 먼저 재서 IK-only와 같은 입력을 만드는지 확인하자" — 정확한 지적. XML/Helios를 아예 거치지 않고 **`build_phytomer_packets`가 만든 GT 패킷(절대 아님, reference-relative 프레임)을 VAE로 encode/decode한 결과와 슬롯 단위로 직접 비교**하는 신규 스크립트 `diffusion_based/eval/eval_phytomer_vae_packet_fidelity.py`를 작성.

**결과 — `use_rot_branch=False`(공유 head, 기존 잘못된 기본값)**:

| Role | 회전 오차(°) mean/p50 | 비고 |
|---|---:|---|
| Stem | 139° / 154° | reference 슬롯이라 GT는 거의 identity인데도 이 정도 오차 → **사실상 랜덤 회전** |
| Petiole | 141° / 139° | 랜덤 수준 |
| Leaflets | 130° / 134° | 랜덤 수준 |
| Repro | 172° / 173° | 거의 180°(반대 방향) |

클래스 정확도는 100%, 스케일 오차도 2~24%로 정상 범위인데 **회전만 완전히 깨져 있음** — VAE 재구성이 전반적으로 나쁜 게 아니라 회전 헤드 하나가 사실상 작동하지 않는 것.

**같은 측정을 `use_rot_branch=True`로 재실행**:

| Role | 회전 오차(°) mean/p50 | 개선 |
|---|---:|---|
| Stem | 0.8° / 0.7° | 거의 완벽 |
| Petiole | 15~16° / 14° | 실제 학습된 수준의 오차 |
| Leaflets | 14~16° / 12~15° | 동일 |
| Repro | 1.4~13.8° (DAP90은 p50 2.2°, p90 40°— 긴 꼬리) | 대부분 양호, 일부 아웃라이어 |

**확정 결론**: v3 체크포인트는 **전용 회전 분기(`rot_branch`/`head_rot_dedicated`)로 학습되었고, 공유 헤드(`head_rot`)는 사실상 미학습 상태**다 (§8.2에서 "LayerNorm 가중치가 초기값 근처"라는 간접 증거로 반대로 추론했던 것은 **틀린 추론**이었음 — 직접 측정이 간접 추론보다 우선함을 보여주는 사례). `assemble_packets`의 `_petiole_curve_points` 커브 적분이 잎자루/화병 방향을 그대로 쓰기 때문에, 130~170°급 회전 오차가 있으면 리프/생식기관이 완전히 엉뚱한 방향으로 뻗어나가는 것은 당연한 결과(§7.3의 "덩굴" 아티팩트를 정량적으로 설명).

`use_rot_branch=True`로도 여전히 Petiole/Leaflet 13~16°, Repro 긴 꼬리(p90 40°) 정도의 **실질적인 잔여 오차**가 남아있고, 이 오차가 `assemble_packets` 이후 base 위치 오차로 나타남 (Leaflet 1.3~6cm, DAP90 Repro는 peduncle 체인을 한 번 더 거쳐 mean 25cm/p50 33cm/p90 51cm까지 증폭). 그리고 이 결과가 다시 `part_tensor_to_40d.py`의 **기존에 알려진 IK 피치 버그(§4)** 를 거쳐 XML/Helios까지 가면서 **두 개의 독립적인 오차가 곱셈적으로 겹쳐** §7.2의 심각한 최종 저하(IK-only 대비 3~8배)로 이어짐.

**다음 세션 우선순위**: (1) `--rot-branch` 없이 학습되는 공유 head를 학습 코드에서 아예 제거하거나 fallback 경로에서 빼서 향후 실수 방지, (2) Petiole/Leaflet 13~16° 잔여 오차 및 Repro 긴 꼬리의 원인 조사 (Hungarian 정렬 안 켜짐? `rot_weight` 부족? 등), (3) 이 잔여 오차만으로 IK-only 대비 얼마나 저하되는지 별도 측정 (IK 피치 버그와 분리).

### 7.5 관련 파일

- 신규: `diffusion_based/eval/eval_phytomer_vae_helios_roundtrip.py` (기본값 `use_rot_branch=True`로 수정됨)
- 신규: `diffusion_based/eval/eval_phytomer_vae_packet_fidelity.py` (26D 패킷 레벨 fidelity, Helios 불필요, 빠름)
- 결과: `docs/results/assets/fig14_phytomer_vae_helios_roundtrip.png`

### 7.7 [P0급, 실제 학습 코드에서도 동일 버그 발견 — 수정 완료]

§7.4에서 밝힌 "공유 head_rot는 사실상 미학습"이 **내 진단 스크립트만의 문제가 아니라 실제 학습/추론 코드 자체의 버그**였음을 확인:

```
grep -rn "phytomer_vae.decode(\|vae.decode(" diffusion_based/
```
결과, `use_rot_branch=True`를 빼먹은 채 `phytomer_vae.decode(...)`를 호출하는 곳이 **학습 핫패스 두 군데**에 있었음:

| 파일:라인 | 용도 | 영향 |
|---|---|---|
| `hierarchical_part_flow_matching.py:1392` | 모델 forward()가 `res["part_14d"]`(실제 지오메트리)를 만드는 곳 — 학습 중 렌더 손실과 self-consistency eval 시각화 둘 다 이 결과를 씀 | **모든 organ 회전이 깨진 공유 head로 나옴** |
| `train_hierarchical_flow_matching.py:702` | 렌더 손실용 메시를 만드는 블록 | 위와 동일 |

즉 `--flow_granularity phytomer`(기본값, 현재 학습이 쓰는 모드)에서는 **렌더 게이트가 켜진 이후(Epoch 4+) 모든 스텝에서 회전이 사실상 랜덤인 메시를 가지고 렌더 손실을 역전파하고 있었다.** 이는 §2.3에서 조사했던 "OOD latent → 비정상 mesh → 1/w² 그래디언트 폭발" 체인과도 연결 가능성이 있음(랜덤 회전 자체가 organ이 서로 뚫고 지나가거나 카메라 근접 평면을 관통하는 비정상 메시를 만들기 쉬움) — 완전히 별개의 원인이라기보다 **동일 증상의 또 다른 기여 요인일 수 있음, 다음 세션에서 재확인 권장**.

**수정**: 두 호출 모두 `use_rot_branch=True` 추가. `tests/test_hierarchical_flow_matching.py` + `tests/test_phytomer_vae.py` 14개 전부 통과 확인. 이번 세션에 로컬로 돌린 스모크 테스트(§6, `fm_smoke_local_manual.log`)는 **이 수정 전 코드로 실행됐으므로 무효** — Recovery 스킵 0건이라는 안정성 결론 자체는 유효하지만(그래디언트 클램프 검증 목적은 달성), 지오메트리 품질은 이 수정 후 다시 확인 필요. (이 수정은 §7.9에서 서술하는 아키텍처 단순화로 다시 한번 대체됨 — `use_rot_branch` 자체가 코드에서 사라짐.)

### 7.8 [더 큰 발견] "덩굴" 붕괴의 진짜 주범은 회전 오차가 아니라 SHOOT_META(가지 경계) 소실

사용자 제안으로 26D 패킷 레벨 fidelity를 측정하다가 회전 오차를 찾았지만(§7.4), 그 회전 오차만으로 §7.2의 심각한 저하(IK-only 대비 FG IoU 3~8배 악화)를 설명하기엔 부족해 보여 추가 검증을 진행:

**결정적 실험**: VAE를 아예 거치지 않고 **100% GT 14D 텐서에서 `ORGAN_SHOOT_META`/`ORGAN_ROOT_META` 메타 행만 제거**하고 그대로 `assemble_part_tensor_to_xml` → Helios 렌더:

| Stage | 원본 IK-only (메타 행 있음) | 메타 행만 제거 (그 외 100% GT) |
|---|---:|---:|
| DAP 10 | 95.7% | **53.5%** |
| DAP 50 | 94.1% | **16.5%** |

이 수치가 §7.2의 전체 VAE 라운드트립 결과(24~53% / 8~20%)와 거의 일치 — **즉 저하의 대부분은 VAE 회전 오차가 아니라 가지(shoot) 경계 정보 소실 때문**.

**원인**: 이 식물들은 DAP10에도 이미 4개, DAP50/90에는 11개의 별도 shoot(가지)를 가짐 (XML `T_COL_SHOOT_ID` 실측 확인). `PartTensorTo40DConverter.convert()`는 `ORGAN_SHOOT_META` 행을 순서대로 만나면서 "여기서 새 가지가 시작된다 + 부모 줄기의 어느 위치·각도에 붙는지"를 판단하는데, `build_phytomer_packets()`는 설계상 이 메타 행을 패킷에서 **항상 제외**한다(실제 장기만 담음). VAE 재구성 파이프라인이 packet → 조립 → flat array로 되돌아갈 때 SHOOT_META를 다시 넣어주지 않으므로, 변환기가 모든 가지를 하나로 이어붙여 **11개 가지가 한 줄로 늘어진 "덩굴"**이 됨. 가지 수가 많을수록(DAP90>DAP50>DAP10) 피해가 커지는 패턴과 정확히 일치.

**중요한 구분**: 이 문제는 미분 가능 렌더러(학습 손실)에는 영향 없음 — `HeliosPlantGeometryBuilder.build_mesh_from_part_tensor`는 각 organ의 절대 pos/rot/scale만으로 메시를 만들고 가지 위상은 필요 없음. **오직 "최종적으로 유효한 Helios XML을 만들어 물리 라이트레이싱 검증에 넣는" 단계에서만 문제가 됨.**

**더 근본적인 공백**: 현재 3-stage 생성 모델(`hierarchical_part_flow_matching.py`) 코드 전체에서 `shoot` 관련 예측이 전혀 없음 — 512개 앵커가 "하나의 shoot 축을 따라" 나열된다고만 설계되어 있어(모델 주석 원문), **애초에 여러 가지(분지)를 표현할 방법이 없음**. `phytomer_id`를 14D/26D에 추가하는 것만으로는 부족한데, `SHOOT_META` 행은 단순 ID가 아니라 "부모 줄기의 어느 지점에 어떤 각도로 붙는지"의 실제 기하 정보이기 때문. 진짜 해결은 모델이 각 앵커에 대해 (a) 새 가지 시작 여부, (b) 부모 가지/부착 지점을 예측하도록 확장하는 것 — **다음 세션 최우선 설계 논의 사항으로 남김.**

관련 스크립트: `/tmp/shoot_meta_test/test.py` (임시, 저장 안 됨 — 재현 필요시 §7.8 절차대로 재작성).

### 7.9 [사용자 결정] 회전 헤드 아키텍처 단순화 — rot_branch/head_rot_dedicated 완전 제거

§7.4/§7.7에서 `head_rot`(공유)와 `head_rot_dedicated`(전용 분기) 두 경로가 혼란과 실수(§7.2, §7.7에서 두 번이나 잘못된 쪽을 기본값으로 씀)를 유발한다는 게 드러나자, **사용자가 명시적으로 결정**: 플래그 없는 단일 경로로 정리하고, **`head_rot`(공유)만 남기고 `rot_branch`/`head_rot_dedicated`는 완전히 삭제**.

**적용 범위** (전체 `use_rot_branch`/`rot_branch`/`head_rot_dedicated` 참조 grep 후 전부 수정):
- `diffusion_based/models/phytomer_vae.py`: `rot_branch`/`head_rot_dedicated` 서브모듈 삭제, `decode()`/`forward()`에서 `use_rot_branch` 파라미터 삭제 (`head_rot` 단일 경로만 남음)
- `diffusion_based/training/train_phytomer_vae.py`: `--rot-branch` CLI 플래그 삭제
- `diffusion_based/training/train_hierarchical_flow_matching.py` (2곳), `diffusion_based/models/hierarchical_part_flow_matching.py` (1곳): `use_rot_branch=True` 인자 삭제 (더 이상 존재하지 않는 파라미터)
- `tools/phytomer_vae_visualizer.py`, `tools/precompute_phytomer_latent_pca.py`: 관련 인자/주석 제거
- `tests/test_phytomer_vae.py`: `test_rot_branch_gradients` 테스트 삭제, gradient 체크에서 rot_branch 스킵 가드 제거
- 신규 진단 스크립트 2개(`eval_phytomer_vae_helios_roundtrip.py`, `eval_phytomer_vae_packet_fidelity.py`)도 플래그 제거
- 전체 테스트 20개 통과 확인 (`test_hierarchical_flow_matching.py` + `test_phytomer_vae.py` + `test_phytomer_packets.py`)

**⚠️ 중요한 파급 효과 — 기존 체크포인트 전부 호환 불가**:
```
PhytomerVAE().load_state_dict(v3_checkpoint)
→ Unexpected key(s): "rot_branch.0.weight", ..., "head_rot_dedicated.weight", "head_rot_dedicated.bias"
```
`phytomer_vae_v2/v3/v4/xml` 체크포인트 전부 `rot_branch.*`/`head_rot_dedicated.*` 키를 가지고 있어 **strict 로드가 즉시 에러남** (의도적으로 방치 — silent partial load보다 명확한 에러가 낫다고 판단). **§7.4에서 확인했듯 v3의 실제로 학습된 회전 정보는 `head_rot_dedicated` 안에 있었고 `head_rot`(이제 유일하게 남은 경로)은 미학습 상태였으므로, 아키텍처만 단순해졌을 뿐 회전 품질은 원점(랜덤 수준)으로 돌아갔다 — PhytomerVAE는 처음부터 재학습 필요.**

**후속 조치**: 큐에 대기 중이던 메인 잡 `38238371`(4-GPU, `phytomer_vae_v3` 참조)은 시작 즉시 이 에러로 크래시할 것이 확실하므로 **취소함** (아직 시작 전이라 낭비된 컴퓨트 없음). **다음 세션 최우선 순위**: `sbatch slurm_scripts/train_phytomer_vae.sh`로 PhytomerVAE부터 처음부터 재학습 → 완료 후 `PHYTOMER_VAE_CHECKPOINT`를 새 체크포인트로 갱신하여 `train_hierarchical_flow_matching.sh` 재제출. §7.8의 shoot 위상 문제는 PhytomerVAE 재학습과 무관하게 별도로 설계 검토 필요.

---

## 8. 커밋 이력 (이 세션)

```
75928d9 feat(phytomer): restore v2/v3 10-slot packet contract (internode slot 0 + repro x4)
4b70266 fix: structurally bound renderer Jacobian & restore train mode after in-loop eval
```

미커밋 잔여: eval_13d 매핑 수정, README/AGENT_TAKEOVER_GUIDE 갱신, 이미지 에셋 다수 (다음 세션에서 정리 권장).