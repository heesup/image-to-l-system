# 2026-09-14 Stage 2 gradient burst 해결 및 dataset 식물 Helios 라운드트립 리포트

> **세션 날짜**: 2026-09-13 ~ 2026-09-14 (PDT)
> **관련 run**: 로컬 v9 run (`slurm_scripts/logs/local_v9_run2.log` → `local_v9_run2b.log`, 체크포인트 `diffusion_based/checkpoints/hierarchical_fm_v9_local2/`), 클러스터 대기 잡 `38249632` / `38250275` (`low` / `publicgrp`)
> **상세 기록**: [`docs/ongoing/20260912_stage2_stage3_boundary_and_remaining_redundancy.md`](../ongoing/20260912_stage2_stage3_boundary_and_remaining_redundancy.md) §1.9.2 (burst), §2.4-2.6 (라운드트립)
> **커밋**: `a524816` (final LayerNorm + fp32 self-attention), `ea7bb58` (ablation 문서), `0472284` (dataset 식물 라운드트립 수정), `3fc48b7`

---

## 1. 요약

두 가지가 닫혔다.

1. **Stage 2 gradient burst**는 모델 상태의 문제였고, 원인은 pre-LN coarse decoder(`nn.TransformerDecoder`, `norm_first=True`)에 **final LayerNorm이 없어서** 크기 ~1e3의 raw residual stream이 bf16 phytomer self-attention의 q/k/v로 들어간 것이다. final LayerNorm을 넣으면 frozen burst 상태에서 8/8 스텝 폭발이 0/8이 된다. 기본값으로 켜져 있고, 재시작한 v9 run은 epoch 15까지 canary 0회다.
2. **Helios 라운드트립**은 exact_gt 세 식물(fig14)에서는 해결돼 있었지만 **dataset 식물에서는 export 자체가 81.8 / 92.2 / 54.6%** (DAP 15 / 40 / 75 FG IoU)였고 VAE는 거의 손실을 더하지 않았다. exact_gt 식물은 converter가 하드코딩한 기본 각도로 생성된 식물이라 문제가 보이지 않았던 것이다. 원인 셋(처지는 shoot을 끊는 chaining, 틀린 branch point, 상수 leaf 각도)을 고쳐 **98.3 / 98.2 / 95.6%** (packet 경로), **95.1 / 96.8 / 95.6%** (VAE 경유)가 됐다. 이 중 branch point 수정은 학습 타깃에도 그대로 적용된다 (lateral의 1/3이 틀려 있었다).

---

## 2. Stage 2 gradient burst

### 2.1 재현과 해부

- v8 계열 run 두 개(`38240479` lr 1e-4, `38242849` lr 5e-5)가 같은 스텝(epoch 27-28)에서 터졌다. `DistributedSampler`가 epoch으로만 시드되므로 같은 배치를 봤다.
- 1 GPU 로컬 run에서 epoch 7 step 2062에 재현했고, 가중치를 고정한 채 어느 배치를 넣어도 터지는 것을 확인했다 → **순수한 모델 상태**의 문제. 데이터 순서, 학습률, weight decay, decoder token collapse, per-epoch eval, DDP, edge-bias mask(빼면 더 나빠짐), decoder만 fp32는 모두 원인이 아니었다.
- `FM_GRAD_PROBE=2` per-op probe: decoder 출력에서 gradient가 ~6000배, decoder layer를 지나며 ×3000 더 증폭. 증폭 지점이 phytomer self-attention 입력이었다.

### 2.2 Ablation (frozen burst 상태, 8 스텝, 두 시드)

| 설정 | burst 스텝 |
|---|---|
| baseline | 8/8, 8/8 |
| self-attention fp32만 (`FM_SELFATTN_FP32`) | 1/8, 2/8 |
| decoder final LayerNorm만 (`FM_DECODER_FINAL_NORM`) | 0/8 |
| 둘 다 (현재 기본값) | 0/8, 0/8 |

### 2.3 적용 및 결과

`diffusion_based/models/hierarchical_part_flow_matching.py`: `nn.TransformerDecoder(..., norm=LayerNorm)` (`FM_DECODER_FINAL_NORM`, 기본 1), self-attention 블록 fp32 (`FM_SELFATTN_FP32`, 기본 1). 보조 안전장치로 canary가 울린 스텝은 건너뛴다 (`d543540`).

재시작한 v9 run (v9 cache/VAE, batch 48, lr 1e-4, render loss epoch 11부터): epoch 1-15 동안 canary 0회, loss 74 → 20.0, self-consistency IoU 19.5 → 31.9%, node RMSE 1.8 cm. v8 계열이 터진 epoch 27-28을 지나면 완전히 닫힌 것으로 본다.

---

## 3. Dataset 식물 Helios 라운드트립

### 3.1 계기

`docs/results/assets/fig12_phytomer_10slot_helios_roundtrip.png` (09-11 13:01, 커밋되지 않은 스크립트로 작성)의 Helios 컬럼이 왜 나쁜지, 45° 뷰가 왜 어색한지가 질문이었다. 그 패널에는 GT의 45° 뷰가 없었고 Helios 컬럼은 shoot 분할 수정과 stem IK 이전의 export였다. 새 스크립트 `diffusion_based/eval/eval_phytomer_10slot_assembly_views.py`로 **dataset 식물**(`cowpea_dap015/040/075_seed00_..._plant_0000.xml`)에 대해 GT 메시와 10슬롯 조립을 같은 카메라(nadir, 45°, GT 메시 bounds)로 그리고, packet 경로와 VAE 경유 Helios 라운드트립을 나란히 놓았다. 옛 패널은 `_unreferenced/fig12_phytomer_10slot_helios_roundtrip_20260911.png`로 보관.

조립은 정확하다 (GT 메시 대비 RGB MAE 0.0002~0.003, 두 뷰 모두). 즉 45° 모습은 원래 식물 모양이다.

### 3.2 측정 (수정 전, FG IoU vs Helios GT)

| DAP | packet 경로 (VAE = identity) | VAE 경유 | 참고: exact_gt 식물 IK-only (fig14) |
|---|---|---|---|
| 15 | 81.8% | 82.6% | 95.7% (DAP 10) |
| 40 | 92.2% | 91.9% | 99.5% (DAP 50) |
| 75 | 54.6% | 56.1% | 96.7% (DAP 90) |

VAE 라운드트립 ≈ packet 경로. 즉 "VAE 라운드트립 = IK-only" 요건은 지켜지고 있었지만, IK-only 자체가 dataset 식물에서 GT와 멀었다. dataset 식물은 internode curvature, 180°가 아닌 phyllotaxy, yaw perturbation, 잎 각도 perturbation, 그리고 마디마다 1~3 mm씩 처지는 lateral shoot을 가진다.

### 3.3 원인과 수정

**(1) `chain_phytomers`가 부모를 자식보다 아래에만 허용했다.** cycle을 막는 장치였지만 처지는 lateral을 모두 끊었다. DAP 75: GT 12 shoot → 17 shoot, 같은 shoot 링크 99개 중 8개 손실, 한 shoot이 73 cm 이탈. 이제 각 노드는 최저 비용 후보를 가리키고, 생긴 cycle은 가장 비싼 edge에서 끊는다 (동률이면 부모가 더 낮은 쪽을 남김). 30개 dataset 식물에서 같은 shoot 링크 1988/1988. shoot 분기점에서 depth가 동률이면 방향으로 continuation을 고르고, `root_own_shoot=True`면 root 노드(떡잎 마디)를 shoot 0으로 홀로 둔다 — emitter와 converter가 shoot 0 / index 0을 특별 취급하므로 이 규칙이 없으면 DAP 15 export가 57.4%로 떨어진다.

**(2) `gt_parent_links`의 branch point 규칙("다른 shoot에서 자기보다 아래에 있는 가장 가까운 노드")이 dataset 식물 lateral의 1/3에서 틀렸다** (272개 중 66.2%, DAP 60+에서는 58%). 이 함수가 학습의 parent conditioning, depth ordinal, step loss 타깃을 만들므로 지금까지의 모든 run이 그만큼 틀린 타깃으로 학습했다. lateral의 첫 internode는 branch point에서 시작하고(부모 노드 중심에서 0.2~0.9 cm, 다른 노드까지는 1~3 cm) 캐시는 absolute packet을 저장하므로, 디코딩한 slot-0 base로 부모를 찾게 했다: `gt_parent_links(..., internode_base=)` → 97.8%. 학습 코드의 호출부도 이를 넘긴다.

**(3) converter의 leaf pitch/yaw/roll이 상수였다** (pitch 2.54, roll −15, yaw +10/0/−10). exact_gt 식물에서는 정확하지만 dataset 식물에서는 FK 기준 평균 10~18° 틀렸다. `diffusion_based/models/part_tensor_leaf_ik.py`: FK가 잎마다 frame(petiole tip 방위, petiole/internode tip 고도, roll 부호, 종류)을 돌려주고, R_leaf = Rz(azimuth)·Rz(yaw)·Ry(−pitch)·Rx(roll) 합성을 닫힌 형태로 뒤집는다. lateral leaflet은 자유도 3개라 정확히(0.00°), terminal leaflet은 pitch 하나, 단일 잎(떡잎)은 pitch+roll. `assemble_part_tensor_to_xml`이 stem IK 다음에 실행한다 (`PART_TENSOR_LEAF_IK=0`으로 끔).

부수 변경: stem IK가 petiole 축 오차도 통계에 넣고 수렴 조건에 쓴다; eval stub들은 depth ordinal + internode base를 쓰고, shoot 첫 internode는 branch 노드 중심이 아니라 자기 디코딩된 base에서 시작한다 (중심에서 그리면 모든 shoot에 0.35 cm 바닥이 남는다). 첫 마디 petiole 방위를 shoot base roll로 푸는 단계는 시도했다가 되돌렸다 — chord를 pitch/yaw 재적합보다 빨리 흔들어 DAP 15가 60%로 떨어진다.

### 3.4 결과 (수정 후)

| DAP | packet 경로 | VAE 경유 | tip 오차 | petiole 축 | leaf 회전 |
|---|---|---|---|---|---|
| 15 | **98.3%** | **95.1%** | 0.03 cm | 1.06° | 0.40° |
| 40 | **98.2%** | **96.8%** | 0.01 cm | 0.75° | 0.31° |
| 75 | **95.6%** | **95.6%** | 0.01 cm | 1.14° | 0.52° |

fig14(exact_gt)도 다시 그렸다: IK-only 95.7 / 99.5 / 96.7 → **99.9 / 99.6 / 97.7%**, VAE 92.2 / 98.4 / 95.7 → **93.7 / 98.3 / 95.8%**. 얇은 기관이 좌우하는 mean organ IoU는 IK-only 48.3 / 89.9 / 46.5 → 93.4 / 91.2 / 54.1.

남은 것: terminal leaflet(자유도 1개, 평균 1~1.6°), 첫 마디 petiole 방위(shoot base roll만이 정할 수 있음, 최대 9°), branch point 2.2%.

![fig12](assets/fig12_phytomer_10slot_helios_roundtrip.png)

![fig14](assets/fig14_phytomer_vae_helios_roundtrip.png)

---

## 4. 학습 상태와 다음 단계

- 로컬 v9 run은 09-14 09:09에 죽어 있었다(세션 중단 시점과 일치, 에러 없음). 새 코드(topology 타깃 수정 포함)로 epoch 15 체크포인트에서 `setsid nohup`으로 재시작: `local_v9_run2b.log`, 패널 `slurm_scripts/logs/run_local_20260914_095047/`.
- 클러스터: `38249632`(a100 ×4), `38250275`(4 GPU) `low`/`publicgrp` 대기, 시작 예정 9/18-19. 먼저 시작하는 쪽만 남기고 다른 하나와 로컬 run을 정리한다. `regen_*` 잡은 건드리지 않는다.
- 다음: epoch 30 통과 확인(canary, Stage 2 Scl/Ord/Ext), 렌더링 품질 추적; §2.1 남은 항목(Stage 3가 child position/roll/scale 예측, parent noise 재보정); export 잔여 오차.

## 5. 변경 파일

| 파일 | 변경 |
|---|---|
| `diffusion_based/dataset/phytomer_topology.py` | `chain_phytomers`: 최저 비용 부모 + cycle 절단, depth 동률 시 방향, `root_own_shoot`; `gt_parent_links(internode_base=)` |
| `diffusion_based/models/part_tensor_leaf_ik.py` (신규) | leaf orientation inverse |
| `diffusion_based/models/part_tensor_stem_ik.py` | petiole 축 오차 통계 + 수렴 조건 |
| `diffusion_based/models/helios_pytorch_geometry.py` | `return_node_poses`에 `leaf_frame`, `leaf_kind` |
| `diffusion_based/models/part_tensor_to_40d.py` | `assemble_part_tensor_to_xml(leaf_ik=)`, `PART_TENSOR_LEAF_IK` |
| `diffusion_based/training/train_hierarchical_flow_matching.py` | `gt_parent_links`에 디코딩한 internode base 전달 |
| `diffusion_based/eval/eval_phytomer_10slot_assembly_views.py` (신규), `eval_phytomer_vae_helios_roundtrip.py` | fig12 / fig14; depth ordinal + internode base stub |
| `tests/test_part_tensor_leaf_ik.py` (신규), `tests/test_phytomer_topology.py` | 42 tests pass across the touched suites |
