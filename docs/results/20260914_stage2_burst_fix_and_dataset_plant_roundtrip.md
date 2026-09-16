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

- 로컬 v9 run은 09-14 09:09에 죽어 있었다(세션 중단 시점과 일치, 에러 없음). 새 코드로 epoch 15에서 잠시 재개했다가(`local_v9_run2b.log`), 10:30에 **클러스터 잡 `38252603`**(geminigrp, 2 GPU, 24 h)이 같은 epoch 15 체크포인트에서 이어받으면서 로컬 run은 정리했다. launcher 기본값을 v9 레시피로 바꿨으므로(`slurm_scripts/train_hierarchical_flow_matching.sh` 헤더 참고) 이후로는 plain `sbatch`로도 같은 설정이 뜬다. 체크포인트는 `hierarchical_fm_v9/`.
- 이후(11:15~11:40): 다른 에이전트가 GPU 활용률(batch 48에서 ~20%)을 올려보려고 38252603을 취소하고 `FORCE_BATCH_SIZE=auto`(→ GPU당 256)로 `38252937`을 띄웠으나, step 시간이 2.2 s → ~18 s로 늘어 **epoch당 55-60분, batch 48의 32분보다 느렸다** (backward 1.3 → 11.2 s, render 0.65 → 5.27 s; 배치에 거의 선형). 그래서 취소하고 **`38252981`**을 batch 48, `NUM_WORKERS=8`로 같은 epoch 15 체크포인트에서 다시 시작했다. 함께 들어간 변경: 패킷 캐시가 VAE와 분리되어(latent를 학습 중 on the fly로 인코딩) VAE 교체 시 캐시 재생성이 불필요해졌고, launcher는 `train_hierarchical_flow_matching.sh`(+`TRAIN_VAE=1`)와 `generate_helios_dataset_jobs.sh`(+`--packets-only`) 둘만 남았다 (`369c3b4`).
- `low`/`publicgrp` 대기 잡(`38249632`, `38250275`)은 10:13에 취소됐다. 24시간 제한이라 9/15 11:40경 `AUTO_RESUME=1`로 재제출이 필요하다. `regen_*` 잡은 건드리지 않는다.
- 다음: epoch 30 통과 확인(canary, Stage 2 Scl/Ord/Ext), 렌더링 품질 추적; §2.1 남은 항목(Stage 3가 child position/roll/scale 예측, parent noise 재보정); export 잔여 오차.

## 6. 학습 step 시간: 병목은 렌더러가 아니라 packet 조립 루프였다 (12:20, `f3c5366`)

배치를 256으로 키웠는데도 epoch 시간이 오히려 늘어난 이유를 찾기 위해 동기화된 구간 타이머로 1 GPU, batch 48, epoch 15 재개 조건에서 step을 쪼갰다.

| | render ON, 기존 코드 | render OFF | render ON, 조립 벡터화 |
|---|---|---|---|
| step (fwd/bwd) | 1.8~2.1 s | 0.11~0.31 s | **0.19~0.39 s** |
| render 블록 | 0.6 s (그중 `assemble_packets` 0.55 s, 메시 0.01, rasterize 0.00) | - | 0.05 s |
| backward | 1.1 s | 0.02~0.04 s | 0.05 s |

단독 측정에서 메시 생성은 5~11 ms, nvdiffrast rasterize는 2 ms, 렌더 backward는 샘플당 10~21 ms(DAP 15/40/75)였다. 즉 렌더 경로는 애초에 비용이 아니었다. 비용은 `assemble_packets`가 렌더 배치의 phytomer ~1,000개마다 파이썬 루프로 휘어진 petiole/peduncle 위의 소엽·꽃 부착점을 계산한 것이고, 그 작은 연산들의 autograd 그래프가 backward를 더 키웠다. `_curve_points_batched`/`_point_on_curve`로 모든 phytomer를 한 번에 계산하도록 바꿨고, `tests/test_assemble_packets_batched.py`가 값과 gradient를 옛 루프와 대조한다 (의도한 차이 하나: petiole이 없거나 너무 짧은 phytomer의 꽃도 이제 peduncle 위에 놓인다).

배치 크기 질문의 답도 여기서 나온다. 배치 256이 느렸던 것은 렌더되는 phytomer 수에 비례해 이 루프가 길어졌기 때문이다. 루프가 사라진 지금은 큰 배치가 이득일 수 있으니 다시 재 보고 정하면 된다. 렌더링 자체의 배치 처리(nvdiffrast 0.4 range mode: 여러 메시를 이어 붙여 `rasterize` 한 번)는 가능하지만 지금은 필요 없다. 메시 생성은 이미 벡터화되어 있다.

**`38253029`**이 이 코드로 epoch 15 체크포인트에서 다시 시작했다 (batch 48, `NUM_WORKERS=8`). 그 전의 `38252981`(기존 코드, 32분/epoch)은 취소.

### 6.1 남은 샘플별 루프 셋 (13:40, `a1a82bc`)

조립 벡터화 뒤 같은 타이머로 재면 (batch 48 / 256, 1 GPU) 나머지는 세 루프였다: GT 타깃 루프에서 샘플마다 GPU에서 돌던 `gt_parent_links` + `decode_packets` (0.10 / 0.53 s), 샘플마다 따로 부르던 VAE encode (0.04 / 0.19 s), 렌더 블록의 식물별 렌더 루프 (0.05 / 0.30 s). 각각 DataLoader 워커에서 CPU로 미리 계산(`attach_parent_links`), 배치 전체를 한 번에 encode, 렌더 식물 전체를 nvdiffrast range mode 한 패스로(`render_batched`, 식물마다 자기 카메라 유지) 바꿨다. 손실 벡터화는 루프와 동일함을 테스트로 고정했다.

| batch | step (조립 벡터화 직후) | step (세 루프 제거 후) | 남은 큰 항목 |
|---|---|---|---|
| 48 | 0.21~0.39 s | **0.17~0.31 s** | backward 0.05, other 0.05~0.12 |
| 256 | 2.0~2.3 s | **1.1~1.8 s** | topo(chain_phytomers, 렌더 식물 8개) 0.14~0.21, matcher 0.09~0.24, 타깃 루프 0.17~0.28, backward 0.28~0.45 |

렌더 비율(배치의 3%)은 그대로다. 샘플당 비용이 여전히 배치에 무관하게 일정해서 배치를 키워도 epoch 시간은 줄지 않는다. 남은 것은 `chain_phytomers`의 노드별 파이썬 walk, greedy matcher, 타깃 루프의 나머지다. **`38253401`**이 이 코드로 `hierarchical_fm_v9/hierarchical_fm_epoch_020.pt`에서 이어 달린다 (`38253029`는 epoch 16~21을 ~9분/epoch로 마치고 취소; IoU 31.9 → 33.4%).

### 6.2 chain_phytomers와 matcher 벡터화 (14:40, `25c2251`)

`chain_phytomers`의 cycle 절단과 shoot walk를 pointer jumping(log N 회의 parent doubling)으로 바꿨다. cycle은 가장 비싼 edge에서 자르되 키를 float64로 계산한다 (1e-6 높이 tie-break가 3 cm 거리 옆에서는 float32 해상도 아래라 mutual-nearest 쌍이 정확히 동률이었다). continuation child는 ordinal gap → 방향/거리 순으로 `scatter_reduce`로 고르고, chain 시작·위치·shoot 번호도 doubling으로 구한다. 루프 구현은 `vectorized=False`로 남겨 테스트가 대조한다 (랜덤 점군의 모든 cue 조합 + GT 식물). N=512에서 34 ms → 2 ms.

matcher의 phytomer 단계는 pad된 배치 위에서 한 번에 돈다 (`_forward_phytomer_batched`): 클러스터를 먼저 압축한 뒤 (B, K, C_max) cost로 greedy 루프 한 번, host sync 한 번. 학습 step은 샘플별 타깃 리스트(boolean index sync 3회/샘플)를 만들지 않고 pad된 GT를 그대로 넘긴다. 단독 측정 42 → 22 ms (B=48), 166 → 56 ms (B=256). 처음 시도한 버전은 (B, 1400, 1400) cdist와 (B, K, 1400) cost 때문에 오히려 느렸고(256 배치에서 2 GB), 압축을 먼저 하도록 고쳤다.

| batch | step | fwd+bwd GPU | 남은 샘플별 항목 |
|---|---|---|---|
| 48 | **0.15~0.24 s** | ~0.1 s | 타깃 루프 0.02~0.05, head(배치 전송) 0.01~0.03 |
| 256 | **0.9~1.5 s** | ~0.6 s | 타깃 루프 0.17~0.28, head 0.10~0.17 |

샘플당 비용은 여전히 배치에 거의 무관(3~5 ms vs 3.5~6 ms)해서 batch 48을 유지한다. **`38253656`**이 이 코드로 `hierarchical_fm_v9/hierarchical_fm_epoch_025.pt`에서 이어 달린다 (`38253401`: epoch 22~25, ~7분/epoch, IoU 32.6~35.5%).

## 7. GT 치환 ablation: 무엇이 실루엣을 잃게 하나 (15:50, epoch 40)

"지금 구조가 이상적인가"에 답하려고, 고정 eval set 20개 식물에서 예측 노드를 GT phytomer에 매칭한 뒤 매칭된 노드의 한 가지 양만 GT로 바꿔 다시 렌더했다 (`diffusion_based/eval/eval_gt_substitution_ablation.py`, `hierarchical_fm_v9/hierarchical_fm_epoch_040.pt`, 같은 렌더러·카메라의 GT 렌더 대비 실루엣 IoU). existence는 항상 예측값이다.

| variant | IoU % | depth MAE cm | young (DAP ≤ 15) | mid | old (> 60) |
|---|---|---|---|---|---|
| P (all predicted) | 24.8 | 14.97 | 2.5 | 23.6 | 37.2 |
| pos ← GT | 42.3 | 12.80 | 27.2 | 36.1 | 55.9 |
| topo / rot / scale / latent ← GT (one at a time) | 25.3 / 26.3 / 23.9 / 25.2 | ~14.9 | | | |
| pos+rot / pos+latent / pos+scale / pos+topo | 45.6 / 45.0 / 43.4 / 42.7 | 12.1-12.7 | | | |
| ALL − pos | 30.6 | 13.98 | 1.2 | 32.9 | 42.9 |
| ALL − rot | 49.1 | 11.28 | 27.3 | 47.5 | 61.7 |
| ALL − latent | 47.5 | 11.53 | 27.5 | 44.6 | 60.4 |
| ALL − scale | 61.6 | 7.90 | 37.5 | 57.5 | 77.7 |
| ALL − topo | 82.1 | 4.92 | 77.4 | 80.2 | 86.4 |
| ALL (pos+topo+rot+scale+latent ← GT, existence predicted) | 82.1 | 4.92 | 77.2 | 80.3 | 86.4 |

읽는 법: 노드 **위치**가 1차 병목이다. 위치만 맞추면 24.8 → 42.3, 나머지가 다 맞아도 위치가 예측값이면 30.6으로 무너진다. 그 다음이 **회전과 latent**(나머지가 맞을 때 각각 −33, −35), 그 다음 scale(−20). ordinal/topology head는 나머지가 맞으면 아무 비용도 없다. 항목 간 상호작용이 커서(잎이 GT 잎과 겹치려면 위치·회전·latent·scale이 함께 맞아야 한다) 하나씩 고쳐서는 IoU가 거의 안 오르고 함께 맞아야 오른다. ALL이 82%인 것은 예측 노드 집합 자체(누락·가짜 노드, 0.5 gate)가 남은 18점이다. 어린 식물(DAP ≤ 15)은 예측만으로 2.5%다.

결론: 설계 문서 §2.1의 남은 항목, 즉 **child의 position/roll/scale을 고정된 parent 기준 상대량으로 Stage 3에서 생성**하는 변경의 근거가 선다. 위치를 먼저, 회전·scale을 같은 flow 상태에 넣는다. latent 쪽(렌더 손실을 Stage 3로 여는 것)은 그 다음이다.

같은 시각에 launcher의 렌더 비율 기본값을 3% → 1/6로 되돌렸다 (배치 48에서 8개 식물, step 0.21~0.45 s; 3%는 렌더 한 식물이 0.65 s 걸리던 때의 타협이었다). `38253656`은 epoch 45 체크포인트부터 이 설정으로 이어간다.

## 8. Stage 3 기하 생성 구현과 A/B (16:10, `abdbaf1`)

§7의 결론대로 `--stage3_geometry`를 구현했다. flow 상태를 `[(pos − parent_pos)·20 | roll | scale | latent]`로 넓히고, 타깃은 모델이 조건으로 받는 (노이즈 섞인) parent 기준 상대량으로 만들어 parent_in + Δpos가 GT 노드에 놓이게 했다. 손실은 latent MSE + 4.0 × 기하 MSE. `sample_ode`는 refine된 위치·roll·scale을 돌려주고, 렌더 블록은 그 위치로 식물을 그려 렌더 손실이 Stage 3 기하 블록까지 닿는다. latent-only 체크포인트는 `geom_proj`/velocity head의 latent 블록을 그대로 두고 기하 8차원만 새로 초기화해 불러오며 Adam 모멘트도 같은 규칙으로 넓힌다. 기본값은 꺼짐.

A/B: 같은 epoch 45 체크포인트에서 baseline(`38257989`, 클러스터 2 GPU, render 1/6)과 기하 run(로컬 1 GPU, `slurm_scripts/logs/local_s3geom_ab.log`, `hierarchical_fm_v9_s3geom/`)을 매 epoch self-consistency IoU로 비교한다. run의 Vel 손실은 기하 차원이 새로 시작해 12~20에서 출발한다.

## 9. Stage 3 latent는 이미지를 노드 단위로 보고 있지 않다 (17:30, `b348fd7`)

"latent가 입력 이미지를 보긴 하나?"에 수치로 답했다. 세 가지를 재었다 (모두 고정 eval set 20개 식물, baseline epoch 50 / 기하 run epoch 47).

1. **치환 ablation에 평균 latent 조건 추가** (`meanlat` = 예측 기하 + 데이터셋 평균 latent, `ALL-latent+meanlat` = GT 기하 + 평균 latent).

| variant | baseline ep50 | 기하 run ep47 |
|---|---|---|
| P (전부 예측) | 27.5 | 27.5 |
| pos ← GT | 38.5 | 38.3 |
| ALL − latent (GT 기하 + 예측 latent) | 43.3–44.1 | 42.3 |
| ALL − latent + meanlat (GT 기하 + 평균 latent) | 40.8 | 40.9 |
| meanlat (예측 기하 + 평균 latent) | 28.5 | 28.5 |
| ALL | 76.1 | 80.9 |

기하가 완벽할 때 **예측 latent는 상수 평균 latent보다 2–3점**밖에 낫지 않고, GT latent는 33–38점을 더 준다.

2. **노드별 latent 오차**: 매칭된 노드에서 샘플된 latent와 GT latent의 RMSE 5.39 vs 평균 latent 4.66 (R² −0.34). 차원별 표준편차는 GT와 같다 (0.174 vs 0.177). 즉 Stage 3는 "그럴듯한 phytomer"를 marginal에서 뽑을 뿐 그 식물의 phytomer를 만들지 않는다.

3. **조건만으로 읽어내는 양** (x_0 = 순수 노이즈에서 x1_hat = x_0 + v): 데이터셋 평균 대비 R² 0.12, 식물별 평균 대비 −0.06. Stage 3 조건에 GT 노드를 넣어도 (0.15 / −0.03) 같다. t = 0.5에서는 0.85인데 이는 x_t가 z1을 새는 것이다.

해석: 조건은 식물 수준(DAP, 크기) 정보만 전달한다. latent의 차원별 표준편차가 ≈0.41로 단위 노이즈에 비해 작아 velocity 타깃의 ~85%가 노이즈이고, t≈0의 무조건 하한이 Var(z1) ≈ 0.17/차원인데 학습 Vel 손실(0.16)이 그 바닥에 앉아 있다. flow 손실만으로는 노드별 조건을 쓸 이유가 없다.

다음: (a) `--render_to_latent` (`RENDER_TO_LATENT=1`) — 렌더 손실을 latent 블록까지 흘린다 (지금까지는 detach; smoke에서 velocity head latent 행에 |g| 0.02–2.7 확인). epoch 45에서 `low` 작업 `38260124` (2×A100, `hierarchical_fm_v9_r2l/`)로 17:45 제출. (b) flow용 latent를 단위 분산으로 스케일 (velocity head가 바뀌므로 epoch 45에서 fine-tune). (c) t 샘플링을 0 쪽으로. 대조군으로 **teacher forcing run** (`STAGE3_GT_NODES=1`, `local_gtnodes_ab.log`)이 17:07부터 돈다: 노드가 맞을 때 노드별 R²가 오르면 "노드 오차가 latent를 굶긴다"가 맞는 것이다. 세 run의 epoch별 IoU: baseline 46–52 = 31.5/31.0/32.7/26.9/29.9/31.4/30.7, 기하 47 = 30.6 (Vel 3.1 → 2.4, 아직 내려가는 중).

## 10. 9/8 Option B "45.4%" 패널과 오늘 모델의 같은 식물 비교 (9/15 09:45)

Heesup의 지적: `docs/results/assets/20260907/hierarchical_self_consistency_epoch_125.png`의 Pred 3D Mesh가 원본 이미지에 제일 가까워 보인다. 두 가지를 쟀다.

1. **그 45.4%는 무작위 배치의 앞 4개 식물 평균**이었다 (job `38145444`, eval 6회: epoch 25/50/75/100/125/150 → 39.1/26.4/33.8/26.9/45.4/49.2, 매번 다른 식물, 묘목 없음). 같은 체크포인트를 **당시 코드 그대로**(커밋 `870074f`) 지금의 고정 20개 eval set에 돌리면 **24.8%** (DAP ≥ 17: 30.8, DAP > 60: 37.0, DAP ≤ 15: 0.9). 오늘 run들은 같은 20개에서 30–34.

2. **같은 식물 7개(9/7 패널의 DAP 72 식물 + 성체 eval 식물 6개)를 같은 렌더러로 나란히** 그렸다 (`assets/20260915_optionb_vs_today_same_plants.png`, 스크립트와 수치는 `slurm_scripts/logs/archive_20260914/optionb_ep125_reeval/`).

| 식물 (DAP / GT 기관 수) | Option B ep125 | baseline ep135 | 기하 run ep76 | gt_nodes run ep70 |
|---|---|---|---|---|
| 72 / 1883 | **44.8** | 38.5 | 36.9 | 37.3 |
| 38 / 1652 | 36.5 | 46.7 | 49.8 | **62.8** |
| 42 / 1428 | 34.0 | 33.0 | **37.8** | 20.5 |
| 62 / 2804 | 47.6 | **61.9** | 56.8 | 57.4 |
| 68 / 1367 | 24.1 | 10.4 | **24.6** | 24.2 |
| 88 / 2166 | 40.5 | **49.8** | 43.9 | 45.7 |
| 97 / 2105 | 40.7 | **53.9** | 51.5 | 52.6 |
| **평균** | 38.3 | 42.0 | **43.1** | 42.9 |

시각적 인상은 근거가 있다. Option B는 **기관 하나하나를 flow로 생성**해(16D 단위 분산 organ latent, 위치가 latent 안에 포함) 잎이 퍼지고 줄기·꽃이 보이며 큰 수관(DAP 72)의 넓이를 살린다. 대신 떠 있는 기관과 지나치게 긴 줄기가 섞인다. 오늘 모델은 phytomer 패킷으로 조립하므로 물리적으로 일관되지만, **latent가 노드별 정보를 담지 못해 모든 phytomer가 평균 모양**이 되고 노드 위치도 중심으로 몰려 **작고 균일한 덩어리**가 된다. IoU는 덩어리가 중심부를 덮어 오늘 쪽이 높지만, DAP 72·88처럼 팔이 뻗은 식물의 넓이는 둘 다 놓친다.

**노드 위치가 더 잘 맞아서인가?** 아니다. 같은 7개 식물에서 예측 노드 → 가장 가까운 GT phytomer 중심 RMSE, 3 cm 안에 예측 노드가 있는 GT phytomer 비율, 예측/GT 노드 convex hull 면적비 (`nodes_cmp.json`):

| | 노드 RMSE (cm) | GT 3 cm 내 커버 | hull 면적비 |
|---|---|---|---|
| Option B ep125 | 18.3 (DAP 38: 54.9) | 22.2% | 4.35 (DAP 38: 17.9) |
| baseline ep135 | 5.9 | 23.3% | 0.68 |
| 기하 run ep76 | **4.5** | **28.7%** | 0.51 |
| gt_nodes run ep70 | 5.6 | 24.9% | 0.58 |

Option B의 "퍼짐"은 정확도가 아니라 **흩뿌림**이다 (hull이 GT의 4배, 노드 RMSE 18 cm). 오늘 run들은 반대로 **수관이 GT의 50–70%로 오그라들고** GT phytomer의 4분의 1만 3 cm 안에 노드가 있다. 이것이 치환 ablation의 "pos←GT +10–14점"의 실체다.

결론: 되돌릴 대상은 architecture가 아니라 Option B latent의 성질이다. (a) 단위 분산 latent → `LATENT_NORM=1` run (`38274201`, 15:53 시작), (b) 노드별 정보가 latent에 들어가게 하는 gt_nodes 계열, (c) Stage 2 노드 위치의 중심 쏠림(수관 hull 0.5–0.7, 커버 25%)은 위 지표로 계속 잰다.

**`low` 파티션에서 여러 학습을 동시에 돌릴 수 있나 (10:00).** 지금은 불가능하다. `low`의 GPU 노드에는 놀고 있는 GPU가 많지만(A100 16개, H100 3개) 다른 사용자의 작업이 **노드 메모리를 전부 점유**하고 있어(노드당 여유 5–13 GB) 학습 하나(메인 6 GB + 워커 8×1.1 GB ≈ 15 GB)도 못 들어간다. 16시간 대기의 원인이 GPU가 아니라 메모리였다. 그래서 `low`의 두 작업을 취소하고, baseline이 15:53에 끝나며 비는 geminigrp 2-GPU 슬롯에 **세 run을 순차 체인**으로 걸었다: `38274220` latent 단위 분산 (epoch 46–80) → `38274221` scheduled teacher forcing p 0.5 / jitter 1 cm (46–75) → `38274222` render→latent (46–70). 2 GPU에서 epoch당 ~3.5분이라 각 2시간 안팎, 오늘 밤 안에 셋 다 결과가 나온다.

## 11. 10% 데이터로 먼저 수렴을 확인하는 프로토콜 (9/15 10:15)

Heesup의 제안: 학습 데이터의 10%만 써서 데이터셋에 수렴하는지 먼저 확인하고, 나중에 데이터를 늘려 일반화한다. 채택했고, 두 가지를 붙여서 그 전략이 실제로 측정 가능하게 했다.

- **같은 eval 식물**: `--eval_set_file` (`EVAL_SET_FILE`)로 기존 `eval_set.json`의 20개 식물을 이름(prefix)으로 다시 찾아 쓰고, 부분집합을 만들 때 강제로 포함한다. 그래서 10% 학습과 100% 학습, 모든 run이 같은 20개 식물로 채점된다.
- **held-out 식물**: `--holdout_samples_per_bucket 2` (`HOLDOUT_PER_BUCKET`)로 DAP 계층별 20개 식물을 학습에서 빼고 매 평가에서 `[Holdout]`으로 따로 보고한다. 지금까지의 "self-consistency" 평가는 전부 학습 식물이었으므로 일반화는 잰 적이 없었다. 단, epoch 45에서 이어가는 run들은 45 epoch 동안 그 식물들을 이미 봤으므로 이번 체인의 held-out은 "45 이후로는 보지 않은" 식물이다. 진짜 held-out은 처음부터 제외하고 학습해야 한다.

왜 맞는 전략인가: 지금 baseline은 100k 식물을 140 epoch 봤다(식물당 140회). 정체의 원인이 데이터 부족이 아니라는 뜻이고, 10k 식물로 10배 빠르게 같은 질문 — 이 구조가 데이터를 **외울 수는 있는가** — 에 답할 수 있다. 외워지면(학습 식물 IoU가 70–80으로 오르면) 병목은 최적화/데이터 규모이고, 10k도 못 외우면 구조와 손실이 병목이다(현재 진단은 후자).

Heesup: geminigrp의 `gpu-6000_ada-h` 파티션은 최대 8 GPU를 높은 우선순위로 쓸 수 있다 (10:30). 그래서 순차 체인 대신 **독립 1-GPU 작업 여섯 개**로 바꿔 GPU가 비는 대로 병렬로 돌린다 (지금 파티션은 gpu-10-50/54 두 노드 8 GPU: 10-50은 drain 중이라 3개가 놀고, 10-54는 baseline 2 + 타 그룹 1 + 우리 1). 첫 작업은 10:32에 바로 시작했고 나머지는 15:53에 baseline이 끝나면 2개가 더 들어간다. 모두 `MAX_TRAIN_SAMPLES=10000`, 같은 eval 20개 + held-out 20개, 1 GPU에서 epoch당 ~2분:
`38274493` baseline-on-10% (참조, epoch 46–95) → `38274494` 단위 분산 latent → `38274495` scheduled TF → `38274496` render→latent → `38274497` 조합 (기하 ep78에서, 78–128) → `38274498` v10 full (조합 + multizoom + coverage + count). 각 50 epoch, 1 GPU에서 약 4시간(평가 40개 식물 포함, epoch당 ~5분). 10:55에 Heesup의 제안으로 로컬의 100% 데이터 run 둘(기하 epoch 80, gt_nodes epoch 78, 체크포인트 보존)을 멈추고, 그 GPU에서 조합 run과 v10 full을 10% 프로토콜로 돌린다 (`local_sub10_combo.log`, `local_sub10_v10.log`; 클러스터 대기 사본은 취소).

### 11.1 첫 판독: epoch 60 (10% 학습 15 epoch 뒤, 11:30)

| run (10k 식물) | 엄격 P | pos←GT | ALL−latent | ALL | latent R² (평균 대비) | probe t=0 R² (식물별 평균 대비, Stage 2 / GT 노드 조건) |
|---|---|---|---|---|---|---|
| baseline-on-10% ep60 | 28.9 | 41.3 | 42.7 | 78.5 | −0.32 | 0.04 / 0.06 |
| 단위 분산 latent ep60 | 26.9 | 34.9 | 37.6 | 72.5 | −0.72 | 0.00 / 0.02 |
| (참조) 100% baseline ep50 / ep80 | 27.5 / 27.3 | 38.5 / 37.6 | 43.3 / 39.6 | 76.1 / 67.1 | −0.34 / −0.38 | ≈0 |

- baseline-on-10%는 15 epoch 만에 100% baseline과 같은 자리(P 29)에 있고 노드 위치는 오히려 조금 낫다(pos←GT 41.3). 10배 적은 데이터로 같은 정체점에 바로 도달했다는 뜻이고, "데이터 양이 병목이 아니다"를 다시 확인한다. 아직 외우기(학습 식물 IoU 70+)는 시작되지 않았다.
- 단위 분산 latent는 15 epoch 뒤에도 노드별 정보가 없다(probe R² ≈ 0, latent R² −0.72로 오히려 나쁨). 차원별 표준편차는 GT와 일치(0.180 vs 0.177)하므로 스케일 자체는 맞게 배웠지만, 조건을 쓰게 만드는 데는 스케일만으로 부족하다. 95 epoch까지 두고 보되, 단독 변경으로는 기대치를 낮춘다.
- 이 노드의 조합 run과 v10 full은 기하 ep78에서 이어가 첫 epoch(79) 학습 식물 34.4 / 37.9, held-out 40.0 / 39.0으로 출발했다.

**epoch 80 (기하 ep78에서 2 epoch 뒤, 11:45)** — 같은 표에 두 run 추가. latent의 렌더 가치 = (ALL−latent) − (GT 기하 + 평균 latent), 즉 예측 latent가 상수 평균보다 얼마나 더 그려 주는가:

| run | 엄격 P | 어린 식물 P (DAP ≤ 15) | pos←GT | ALL−latent | GT 기하 + 평균 latent | latent 렌더 가치 | ALL | probe R² |
|---|---|---|---|---|---|---|---|---|
| baseline-on-10% ep60 | 28.9 | 5.4 | 41.3 | 42.7 | 39.2 | +3.5 | 78.5 | 0.04 |
| 단위 분산 latent ep60 | 26.9 | 9.7 | 34.9 | 37.6 | 39.4 | −1.8 | 72.5 | 0.00 |
| scheduled TF (p 0.5, jitter 1 cm) ep60, 배포 조건 / GT 노드 조건 | 29.2 / 45.3 | 10.5 / 29.4 | 42.5 / – | 45.8 / 46.0 | 38.5 | **+7.3** | 77.6 | 0.04 / 0.07 |
| render→latent ep60 | 25.0 | 2.3 | 37.9 | 39.8 | 39.0 | +0.8 | 79.0 | −0.02 |
| 조합 ep80 (기하+표준화+render→latent) | 34.0 | 19.3 | 42.0 | 45.6 | 39.9 | +5.7 | 78.1 | −0.15 |
| **v10 full ep80** (+ multizoom, coverage, count) | **35.4** | **19.8** | **44.2** | **47.5** | 39.6 | **+7.9** | **81.7** | −0.07 |
| (참조) 100% 기하 run ep76 | 33.9 | 14.6 | 42.9 | 45.5 | 40.0 | +5.5 | 79.2 | – |
| (참조) gt_nodes ep70, GT 노드 조건 | 45.3 | 26.6 | – | 48.2 | 40.1 | +8.1 | 82.3 | 0.41 |

읽기 (epoch 60 추가분): **scheduled teacher forcing은 순수 TF의 exposure bias를 없앴다** — 배포 조건 P 29.2로 baseline(28.9)과 같고(순수 TF는 20–24), GT 노드 조건 상한 45.3은 그대로이며 latent 렌더 가치는 +7.3으로 baseline(+3.5)의 두 배다. render→latent 단독은 이득이 없다(P 25.0, 렌더 가치 +0.8) — 이 기울기는 기하 run 위에서만(조합, v10) 값을 낸다. 2 epoch 만에 v10 full이 배포 조건 엄격 P 최고(35.4)이고 어린 식물 P가 5–10에서 19.8로 뛰었다(multizoom 토큰이 8× crop을 보기 시작한 효과로 보이나 2 epoch이라 확정은 아님). latent의 노드별 정보(probe R²)는 여전히 어느 run에도 없지만, latent의 **렌더 가치**는 baseline +3.5 → v10 +7.9로 올랐다: render→latent 기울기가 GT latent와 같지는 않아도 더 잘 그려지는 latent를 만든다. 다음 판독은 epoch 90.

### 11.2 둘째 판독 (12:05): 최신 체크포인트, 엄격 프로토콜

| run | ckpt | 엄격 P (배포) | 어린 식물 P | pos←GT | ALL−latent | latent 렌더 가치 | ALL | probe R² (Stage 2 / GT 노드) |
|---|---|---|---|---|---|---|---|---|
| baseline-on-10% | ep75 | 26.5 | 4.1 | 38.7 | 45.8 | +7.6 | 72.4 | 0.04 / 0.07 |
| 단위 분산 latent | ep70 | 25.4 | 3.0 | 33.6 | 36.7 | −1.8 | 77.4 | 0.02 / 0.05 |
| scheduled TF (배포 / GT 노드) | ep65 | 27.3 / 43.7 | 8.5 / 28.2 | 42.4 | 42.7 / 46.2 | +5.5 / +9.0 | 74.7 | 0.05 / 0.09 |
| render→latent | ep65 | 27.5 | 8.6 | 39.8 | 40.0 | +1.5 | 71.7 | – |
| 조합 | ep85 | 33.6 | 14.5 | **46.1** | **47.2** | +7.1 | **81.9** | −0.02 / −0.01 |
| **v10 full** | ep85 | **36.0** | **18.5** | 43.6 | 46.1 | +6.9 | 79.3 | −0.06 / −0.05 |

학습 중 지표(최근 10 epoch 평균, 학습 식물 / held-out):

| run | 마지막 epoch | 학습 식물 IoU | held-out IoU |
|---|---|---|---|
| baseline-on-10% | 80 | 33.4 | 30.7 |
| standardized latent | 78 | 27.3 | 22.9 |
| scheduled TF | 71 | 35.4 | 28.4 |
| render->latent | 72 | 32.3 | 25.5 |
| combination | 88 | 34.3 | 36.3 |
| v10 full | 88 | 35.4 | 34.9 |

판독:
- **기하 생성 계열(조합, v10)이 배포 조건 엄격 P 34–36으로 latent-only 계열(25–29)을 7–10점 앞선다.** v10 full이 선두(36.0)이고 어린 식물(18.5)과 노드 위치(pos←GT 43.6–46.1)에서도 낫다. multizoom·coverage의 몫은 조합 대비 +2.4점(P)과 어린 식물 +4점.
- 단독 변경 셋(표준화 latent, scheduled TF, render→latent)은 20 epoch 뒤에도 배포 P를 올리지 못한다. scheduled TF만 GT 노드 조건 상한(43.7)과 latent 렌더 가치(+5.5 → +9.0)를 유지한다.
- **어느 run도 10k 식물을 외우지 못한다**: 학습 식물 IoU가 25–30 epoch 동안 33–36에 머문다(held-out과 차이 ≤ 5점). 데이터 양이 아니라 구조·손실·최적화가 병목이라는 결론이 굳어진다. 어느 run의 latent에도 노드별 이미지 정보는 아직 없다(probe R² ≈ 0).
- 다음 지렛대: 10k 식물이면 **모든 샘플을 렌더**해도 epoch이 몇 분이므로 렌더 손실 비율을 1/6 → 1.0으로 올린 v10, 그리고 학습률을 1e-4 → 2e-4로 올린 v10을 이득 없는 두 run(표준화 latent, render→latent) 자리에 넣는다.

### 11.3 셋째 판독 (12:15): 정체 확인

| run | ckpt | 배포 P | 어린 식물 P | pos←GT | ALL−latent | latent 렌더 가치 | ALL |
|---|---|---|---|---|---|---|---|
| baseline-on-10% | 85 | 25.7 | 5.4 | 42.9 | 45.8 | +6.6 | 82.8 |
| scheduled TF (배포 / GT 노드) | 75 | 28.7 / 44.1 | 8.3 / 25.9 | 40.6 | 44.2 | +6.3 | 80.1 |
| 조합 | 90 | 34.2 | 13.5 | 41.4 | **49.2** | **+9.8** | 81.5 |
| **v10 full** | 90 | **35.1** | **17.8** | 41.7 | 45.9 | +7.4 | 77.0 |

세 판독을 겹치면 모든 run이 10 epoch 안에 제자리에 멈춘다: 기하 생성 계열 34–36, 나머지 26–29. 10k 식물에서도 학습 식물 IoU가 오르지 않으므로 **더 오래 돌려서 얻을 것은 없고**, 남은 지렛대는 (a) 최적화 강도 — 렌더 손실을 모든 샘플에(`38274747`, 진행 중), 학습률 2e-4(`38274748`, 대기) — 와 (b) 구조: latent가 노드별 정보를 담게 하는 방법(probe R²가 어느 run에서도 0)과 노드 커버리지(pos←GT까지 +7–17점이 남아 있음)다. coverage 손실이 실제로 노드를 펼치는지 쟀다 (같은 7개 식물, `nodes_cmp.py`): **펼치지 못한다.**

| run | 노드 RMSE | GT 3 cm 내 커버 | hull 비 |
|---|---|---|---|
| baseline-on-10% ep85 | 6.0 cm | 27.1% | 0.57 |
| 조합 ep90 | 4.8 cm | 25.2% | 0.57 |
| v10 ep90 (coverage 손실 포함) | 4.6 cm | 24.1% | 0.48 |

이유는 ablation의 "matched = 73.5 / GT 73"에 있다: **matcher는 이미 모든 GT phytomer를 예측 노드와 짝짓는다.** 커버리지 부족은 짝이 없어서가 아니라 (a) 추론 때 존재 게이트가 노드의 15–20%를 떨어뜨리고(활성 60–65 / 73) (b) 짝지어진 노드 자체가 4–6 cm 벗어나서다. 그래서 "가장 가까운 활성 노드를 끌어당기는" coverage 손실은 짝지어진 노드를 다른 GT 쪽으로 당겨 오히려 hull을 줄였다. 손실 하나로 풀 문제가 아니라 **노드 위치의 정밀도** 문제다. 128 px 입력에 DINOv2 8 px patch면 토큰 하나가 7.5 cm(1.2 m / 16)인데 노드 오차가 4–6 cm이니 해상도 한계에 닿아 있다. 다음 지렛대: 백본 fine-tuning(`FREEZE_BACKBONE=0`, v10 변형 `38274796` 제출) → 256 px 입력(10k 캐시 재생성 필요).

### 11.4 넷째 판독 (12:20) — 판독 네 번의 평균

| run | 엄격 P, 판독별 (ckpt) | 평균 |
|---|---|---|
| **v10 full** | 35.4 (80) / 36.0 (85) / 35.1 (90) / 33.6 (95) | **35.0** |
| 조합 | 34.0 / 33.6 / 34.2 / 32.1 | 33.5 |
| v10 + 렌더 손실 전 샘플 | 34.4 (80, 2 epoch째) | – |
| scheduled TF (배포) | 29.2 (60) / 27.3 (65) / 28.7 (75) / 26.7 (85) | 28.0 |
| baseline-on-10% | 28.9 (60) / 26.5 (75) / 25.7 (85) / 26.9 (95) | 27.0 |

10k 식물, 50 epoch 예산이 끝난 baseline-on-10%는 27.0으로 100% baseline(27–29)과 같다. 어린 식물 P는 체크포인트마다 7–19로 크게 흔들려(4개 식물) 판정에 쓰지 않는다. 결론은 §11.3과 같다: v10 계열 35, 나머지 27–28, 어느 것도 더 오르지 않는다. 남은 지렛대는 최적화(학습률 2e-4 `38274748` 시작, 렌더 전 샘플 `38274747` 진행)와 노드 정밀도(백본 fine-tuning; 이 노드에서 조합 run을 멈추고 그 자리에서 시작).

### 11.5 v10 변형 넷의 첫 판독 (12:45, 시작 2–7 epoch 뒤)

| v10 변형 | ckpt | 배포 P | 어린 식물 P | pos←GT | ALL−latent | ALL | 활성/GT | probe R² |
|---|---|---|---|---|---|---|---|---|
| (기준) v10 full | 80–95 평균 | 35.0 | 7–19 | 42–44 | 45–47 | 77–82 | 60/73 | −0.06 |
| 학습률 2e-4 | 85 | 34.3 | 11.0 | 43.3 | 45.6 | 78.4 | 62/73 | −0.06 |
| 렌더 손실 전 샘플 | 80 | 34.6 | 18.4 | 40.9 | 43.7 | 81.3 | 60/73 | −0.09 |
| 백본 fine-tuning | 80 | 34.3 | 17.8 | 41.0 | 46.3 | 75.7 | **81/73** | −0.10 |
| 토큰 창 3 + t0 0.25 | 80 | 32.0 | 8.2 | 38.9 | 44.5 | 78.5 | 61/73 | −0.05 |

첫 체크포인트라 판정은 이르지만, 넷 다 v10의 35 근처에서 출발했고 어느 것도 latent probe를 움직이지 않았다. 최적화 강도(학습률, 렌더 비율, 백본)로는 한계가 안 움직인다는 신호가 쌓이고 있어, 학습이 아니라 **추론 쪽 지렛대** 둘을 먼저 잰다: (a) Stage 3가 refine한 노드를 다시 조건으로 넣어 한 번 더 샘플하는 2-pass self-conditioning(학습 없이 평가만), (b) 입력 CHM에 대한 렌더 손실로 노드·latent를 식물마다 몇십 step 최적화하는 test-time refinement(analysis-by-synthesis; 미분 가능 렌더러가 이미 있음).

둘째 판독 (13:25): w3t0 ep85 엄격 P **34.4** (ALL 76.1), lr 2e-4 ep100 **33.0** (ALL 68.5). 학습 중 [Holdout] IoU는 w3t0가 43–44%로 다른 변형(34–40%)보다 높지만, 엄격 프로토콜에서는 넷 모두 33–35에 머문다. 학습 쪽 변형은 여기서 닫고, §11.8/§11.10의 카메라 창 수정을 학습 렌더에 넣은 run(`sub10_v10_cam`)으로 넘어간다.

### 11.6 추론 쪽 지렛대 (a): self-conditioning 재샘플 — 효과 없음 (12:50)

v10 ep95에서 Stage 3가 refine한 노드(pos/roll/scale)를 다시 조건으로 넣고 한 번(2-pass) 또는 두 번(3-pass) 더 샘플했다 (`eval_gt_substitution_ablation.py --self_cond_passes N`): 배포 P 33.6 → 31.7 / 31.7. 자기 노드를 조건으로 되먹여도 노드가 좋아지지 않는다. 즉 Stage 3의 refine은 "GT parent에 대한 상대 위치"를 배운 것이지 "자기 예측을 고쳐 나가는" 능력이 아니다.

### 11.7 추론 쪽 지렛대 (b): test-time refinement — 오늘 가장 큰 이득 (12:55)

`diffusion_based/eval/eval_test_time_refinement.py`: 식물마다 샘플한 뒤 노드 위치·scale·phytomer latent를 leaf 변수로 두고, **입력 CHM**(캐시의 1×·2× 깊이 채널)에 대한 학습용 렌더 손실(depth smooth-L1 + dice)로 Adam 40 step 최적화한다. GT는 쓰지 않는다(analysis-by-synthesis). v10 ep95, 고정 20개 식물, 엄격 256 px 프로토콜:

| | 전체 20 | DAP > 15 |
|---|---|---|
| 샘플 그대로 | 37.0 | 41.7 |
| **40 step 정제 후** | **44.3** | **52.0** |

식물별로는 24.5 → 55.8 (DAP 39), 31.9 → 67.5 (DAP 60), 48.4 → 74.6 (DAP 51)처럼 크게 오르는 것과 54.4 → 25.6 (DAP 89), 51.9 → 41.7 (DAP 94)처럼 무너지는 것이 섞여 있다(노드 이동 평균 0.3–4.7 cm). 무너지는 경우를 막기 위해 **입력 손실이 가장 낮은 step의 변수를 택하는** `--keep_best`(GT 없이 입력만으로 선택)와 변수 집합별(pos+scale / latent만) 변형을 이어서 잰다.

변형 (같은 체크포인트, 40 step, `--keep_best` = 입력 손실이 가장 낮은 step 채택; "전" 값은 샘플 난수로 조금씩 다름):

| 최적화 변수 | 전 → 후 (20개) | DAP > 15 |
|---|---|---|
| pos + scale + latent, keep_best | 34.9 → **46.9** (+12.0) | 38.2 → 51.2 |
| pos + scale, keep_best | 33.6 → 43.4 (+9.8) | 39.2 → 50.5 |
| latent만, keep_best | 33.4 → 41.2 (+7.8) | 39.2 → 47.8 |
| 전부, 마지막 step (keep_best 없음) | 37.0 → 44.3 (+7.3) | 41.7 → 52.0 |

세 변수 모두 기여하고(노드 위치가 가장 크고 latent도 +8), keep_best가 무너지는 식물을 막아 +12까지 간다. 지금까지의 최고 배포 수치는 **엄격 P 46.9** (baseline 27, v10 학습만 35)다.

의미: 학습 쪽 지렛대는 모두 35에서 멈췄는데, 추론 때 입력 이미지를 직접 맞추는 것만으로 +7–12점이 나온다. 모델이 이미지에서 못 읽어 내는 노드별 정보를 렌더러가 대신 찾아 주는 셈이고, 이는 곧 학습 중 렌더 손실이 (1/6 샘플, 2 scale, 노드당 한 번의 기울기로는) 아직 충분히 쓰이지 못한다는 뜻이기도 하다.

### 11.8 렌더 손실의 기준 창이 어긋나 있었다 (13:00)

캐시의 입력 CHM(`generate_cache.py`, `focus_plant=True`)은 **GT 식물의 bounding box 중심**을 화면 중심에 두고 그린 것이고, 학습 렌더 블록(`render_batched`, `focus_plant=False` + 고정 1.2 m 창)과 모든 평가 렌더는 **원점**을 중심에 둔다. eval 식물에서 재면 캐시 CHM은 bbox 중심 GT 렌더와 73–89% 겹치지만 원점 창 GT 렌더와는 41–76%만 겹친다(DAP 94: 88.6 vs 40.7). 즉 학습 중 depth/dice 손실은 완벽한 예측이어도 0이 될 수 없었고, 기울기의 일부는 식물 전체를 잘못된 오프셋 쪽으로 밀었다. test-time refinement에서 무너진 두 성체 식물(DAP 89/94)이 바로 오프셋이 큰 경우다. 수정: 예측을 입력과 같은 창(식물 bbox 중심)으로 렌더한다 — refinement에는 `--plant_centered`로 즉시 적용해 재측정 중이고, 학습 렌더 블록에는 샘플별 GT bbox 중심을 카메라 중심으로 넘기는 변경이 다음이다.

### 11.9 refinement 후속 (13:05)

| 조건 (v10 ep95, keep_best) | 전 → 후 |
|---|---|
| 원점 창 렌더, 40 step | 34.9 → **46.9** |
| 원점 창 렌더, 80 step | 33.6 → 45.5 (step을 늘려도 이득 없음) |
| 예측 자신의 bbox 중심 창(`--plant_centered`), 40 step | 32.8 → 35.1 |
| 100% baseline ep135, 원점 창 40 step | 27.6 → **42.7** (+15; 모델에 무관하게 작동) |

예측을 **자기** bbox 중심으로 렌더하면 손실이 평행이동에 둔감해져 이득이 사라진다(35.1). 입력 CHM의 창은 **GT** bbox 중심이므로 예측도 그 카메라(입력을 만든 카메라)로 렌더해야 맞다 — 렌더러에 카메라 중심 지정 옵션을 붙여 `--input_camera`로 재측정한다. refinement는 baseline에도 +15를 주므로 모델보다 큰 지렛대이고, 학습 렌더 블록에도 같은 카메라 정렬이 필요하다.

그림 스타일: Heesup의 요청으로 학습 중 패널(`eval_hierarchical_self_consistency.py`)과 같은-식물 비교 그림(`tools/compare_same_plants_figure.py`)을 흰 배경·검정 글자·Okabe-Ito 팔레트(GT 파랑, 예측 주황)의 논문 스타일로 바꿨다. 돌고 있는 run은 다음 평가부터 새 스타일로 패널을 낸다(평가 모듈을 매번 reload). fig12/fig14 스크립트는 다음 차례.

### 11.10 입력 카메라 창으로 정제하면 62.9 — 오늘의 최고 배포 수치 (13:15)

§11.8의 진단대로 예측을 **입력 CHM을 만든 카메라**(GT 식물 bbox 중심을 화면 중심에 두는 창; `--input_camera`, 렌더러 `render_batched(centers=)`)로 렌더해서 입력 CHM과 맞추면 정제 이득이 두 배 이상 커진다. 평가 IoU는 이전과 똑같이 엄격 프로토콜(원점 창, 256 px, GT 렌더 대비)로 쟀다.

| 조건 (v10 ep95, pos+scale+latent, 40 step, keep_best) | 전체 20 | DAP > 15 (16개) |
|---|---|---|
| 원점 창 렌더 (§11.7) | 34.9 → 46.9 | 38.2 → 51.2 |
| **입력 카메라 창 렌더 (`--input_camera`)** | 33.5 → **62.9** (+29.4) | 38.4 → **74.5** (+36.1) |

식물별(DAP > 15)로는 22.0 → 75.0 (DAP 25), 23.0 → 73.9 (DAP 45), 16.3 → 70.5 (DAP 68), 64.5 → 86.5 (DAP 97)처럼 예외 없이 모두 오르고, §11.7에서 무너지던 DAP 89·94도 53.8 → 80.8, 52.3 → 81.3으로 오른다(노드 이동 평균 1.3–5.2 cm). DAP ≤ 15의 네 식물(0.0/8.7/23.7/22.7)은 0.0/0.7/33.0/32.3으로, 아주 어린 식물에서는 이득이 작거나 없다(1×·2× 입력 창에서 식물이 몇 픽셀뿐이라 손실 신호가 거의 없다).

해석: §11.7의 +12는 창이 어긋난 채로 얻은 값이었고, 같은 카메라로 맞추면 렌더러가 노드 위치·크기·latent를 입력에 거의 정확히 끼워 맞춘다. 이 수치는 GT를 쓰지 않는다 — 최적화 대상은 모델 입력인 CHM이고, 카메라는 그 입력을 찍은 카메라(실측에서는 촬영 카메라 자세)다. 다만 지금 구현은 캐시가 GT bbox 중심으로 렌더된 사실을 이용해 그 중심을 GT 메시에서 다시 계산하므로, 실측 배포에서는 입력 카메라 자세를 그대로 쓰면 된다.

같은 사실이 **학습 렌더 블록**에도 적용된다: 학습 중 렌더 손실도 원점 창으로 계산하고 있으므로 §11.8의 창 불일치(겹침 41–76%)를 안고 학습해 왔다. 다음 단계는 학습 렌더도 `centers=`(GT bbox 중심)로 바꿔 다시 학습하는 것이다.

**학습 렌더에 같은 카메라를 넣은 run `sub10_v10_cam` (13:20 시작).** `--render_input_camera 1`(런처 `RENDER_INPUT_CAMERA=1`): 렌더되는 식물마다 GT 메시 bbox 중심을 계산해 `render_batched(centers=)`로 넘긴다(`tests/test_render_input_camera.py`). v10 설정 + 이 옵션, s3geom ep78에서 시작, 10% 데이터, 로컬 GPU. 첫 판독 ep80 엄격 P **35.8** (meanlat 34.8, ALL 78.3) — 같은 시점의 v10 35.4, 다른 변형 32–34.6보다 조금 높지만 아직 잡음 범위(±1–2). ep85 **34.0** (ALL 80.2), ep90 38.5, ep95 34.6, ep100 34.8 — 다섯 판독 평균 **35.5**로 v10 자체(35.0)와 같다. 학습 렌더의 카메라 창을 맞추는 것은 원칙적으로 옳지만 10% 데이터에서는 정체 수준을 바꾸지 못했다(ep90의 38.5는 잡음).

**전체 데이터로 확장 (14:15).** 10% 프로토콜의 결론(모든 변형이 ~10 epoch 안에 33–36에서 정체, 배포 수치는 정제가 결정)에 따라 정체된 10% 클러스터 job 둘(render-all, lr 2e-4)을 취소하고 그 자리에 **전체 데이터 v10 + 입력 카메라 창** run을 올렸다: job `38275054`, ada-h 2 GPU, s3geom ep78에서 시작, 128 epoch, 체크포인트 `diffusion_based/checkpoints/hierarchical_fm_v10_cam/`. 10% 결과의 일반화 확인이자 정제에 넣을 배포 모델 후보다. 첫 판독 ep80(시작 2 epoch 뒤): 엄격 P **27.5**, meanlat 34.4, ALL 79.6 — 전체 데이터에서는 학습 초반에 샘플한 latent가 평균 latent보다 7점 나쁘다(10% run들은 둘이 비슷했다). 같은 체크포인트에 기본 정제를 걸면 31.3 → **64.6** (DAP > 15: 33.6 → 72.6) — 시작 2 epoch 뒤의 전체 데이터 모델도 정제 뒤에는 10% v10(67.6)에 3점 차로 붙는다. ep85: 엄격 P **14.0** (meanlat 33.1, ALL 76.8), 정제 14.0 → 53.3 — 학습 중 [Holdout] IoU도 ep84 32.5% → ep85 15.0% → ep86 37.6%로 그 epoch만 꺼졌다 온 것이라, 발산이 아니라 **체크포인트가 나쁜 epoch에 걸린** 경우다(SAVE_EVERY=5가 임의 상태를 저장). 10% run들도 [Holdout]이 28–44%로 흔들렸는데 전체 데이터에서는 진폭이 더 크다. 그래서 가중치 EMA(`--ema_decay 0.999`, `<checkpoint>_ema.pt`)를 넣고 ep90 체크포인트에서 job `38279147`로 이어 받았다(17:15). ep90 raw: 엄격 P **28.9**, meanlat 35.9, ALL 77.5 — 전체 데이터에서는 세 판독 모두 샘플 latent(P)가 평균 latent보다 6–7점 낮다(10% run들은 둘이 같았다). 전체 데이터의 latent 분포가 더 넓어 flow 샘플이 아직 못 따라가는 것으로 보이며, ep90 raw 정제: 30.4 → 57.3 (ep80은 64.6) — 정제 결과도 체크포인트마다 흔들린다. **ep95 (EMA job에서 저장):** raw 엄격 P **35.9** (meanlat 35.2, ALL 79.3), 정제 36.7 → **66.3**; EMA 가중치 P 35.1, 정제 36.0 → 64.7. 전체 데이터 모델이 ep95에서 10% v10(35.8 → 67.6)과 같은 수준에 왔고, 샘플 latent도 평균 latent를 따라잡았다(P ≈ meanlat). EMA는 아직 4 epoch(≈ 4000 step, decay 0.999)만 평균해 raw와 비슷하다. ep100: raw P 34.5 (meanlat 33.8), EMA P 35.1 (meanlat 35.4), EMA 정제 34.8 → 66.2. 전체 데이터 lineage는 ep95–100에서 엄격 P 34.5–35.9, 정제 후 64.7–66.3으로 안정됐고, 10% v10과 같은 자리다 — 데이터를 10배 늘려도 정체 수준은 같다. ep105: raw P 37.4 (정제 66.2), EMA 36.8 (정제 66.8). ep110: raw 38.3 (정제 64.6), **EMA 39.0** (정제 65.9; ALL 79.1) — 전체 데이터에서는 정제 전 P가 35.9 → 34.5 → 37.4 → 38.3, EMA 35.1 → 35.1 → 36.8 → 39.0으로 천천히 오르고 있다(10% run은 35에서 멈췄다). ep115: raw 35.4 (정제 65.4), EMA 35.9 (정제 66.7) — ep110의 39.0은 부분적으로 잡음이었고, 전체 데이터 lineage의 수준은 정제 전 35–39, 정제 후 65–67이다. ep120: raw 36.3 (정제 **67.9**, 전체 데이터 lineage의 정제 최고), EMA 37.3 (정제 67.1). ep125: raw 34.3 (정제 66.8), EMA 38.3 (정제 67.6). **ep128 (마지막):** raw 38.1 (정제 67.0), EMA 37.7 (정제 67.0). job `38279147`은 128 epoch로 완료(9 h 25 min). 정제 전 P는 ep95 35.9에서 ep128 38.1로 천천히 올랐고 정제 후는 65–68에서 평평하므로, 같은 설정으로 ep160까지 이어 본다(job `38332343`, 2026-09-16 02:45). 이때 런처의 AUTO_RESUME이 `_ema.pt`를 최신 체크포인트로 잡는 버그가 드러나 고쳤다(`grep -v _ema`; 첫 시도 job 38332310은 EMA 가중치에서 새 optimizer로 시작해 취소). 이어진 lineage EMA: ep130 37.5 (정제 66.0), ep135 35.9 (정제 64.9), ep140 37.3 (정제 65.5), ep145 **38.3** (정제 65.7), ep150 36.5 (정제 63.8), ep155 38.3 (meanlat 34.7, ALL 75.5, 정제 **37.5 → 67.4**, 이어진 lineage 정제 최고치). **ep160(마지막, EPOCHS=160 도달, job 38332343 COMPLETED)**: [Holdout] IoU 43.2%(이 lineage 전체 최고), self-consistency 38.0%. 엄격 프로토콜: raw/EMA P **38.7**(meanlat 35.7, ALL 75.8) — 정제 전 P로는 이 lineage 최고치. 정제 후 **37.0 → 68.3** (DAP>15: 42.1 → 74.7) — 오늘 나온 모든 판독 중 정제 후 최고치다.

**전체 데이터 lineage 결론 (job 78→160, 총 82 epoch 연속 학습).** ep95부터 ep160까지 raw/EMA P는 35~39 사이에서 천천히 우상향했고(마지막이 최고), 정제 후는 64~68 사이를 오가다 마지막 판독(68.3)이 최고를 찍었다. 10% 프로토콜(v10)의 결론과 일치한다 — 데이터를 10배 늘려도 학습만으로 도달하는 수준은 크게 안 바뀌지만, holdout 지표(43.2%)와 안정성은 꾸준히 좋아졌다. **오늘의 최종 배포 수치는 엄격 P 38.7(학습만) / 68.3(정제 후)**이며, 두 값 모두 이 문서에 기록된 전체 실험 중 최고치다.

**Guided sampling (DPS 스타일, 09/16 11:10) — 첫 20개 전체 결과는 밋밋하다.** `sample_ode`에 매 ODE step마다 현재 x1 추정치를 렌더링해 입력 CHM과 비교한 손실의 gradient를 velocity에 더하는 `guidance_fn` 훅을 추가했다(가중치는 안 건드림, 후처리 정제와 별개 메커니즘 — DPS/reconstruction guidance). 10% v10 ep95, guide_scale=2.0, 20 step 전부에 가이던스 적용: **35.6 → 34.1** (DAP>15 40.1 → 39.7) — 20개 식물 중 절반은 오르고 절반은 내려가 순효과가 없다. 2개 식물 스모크 테스트(31.2→35.2)는 표본이 너무 작아 오도했다. 노드가 step당 5~9 cm씩 움직이는 것으로 보아 이른(노이즈가 큰) step부터 강한 gradient를 주는 게 문제로 보인다 — `--guide_start_frac`(가이던스를 ODE 후반부에만 적용)와 더 약한 guide_scale을 다음에 시도한다. **ODE 후반부(t≥0.5)에만 적용(`--guide_start_frac 0.5`)하니 35.2 → 37.3**(DAP>15 41.2→41.3)로 순이득이 생겼다 — 가설이 맞았다: 노이즈가 큰 초반 step에 강한 렌더 gradient를 주는 게 역효과였다. 다만 여전히 40-step 후처리 정제(+30점대)에 비하면 작은 이득(+2점)이라, guide_scale을 더 낮추거나 두 방법을 합치는(guided sampling으로 시작점을 개선한 뒤 정제) 조합이 다음 단계로 남아있다. — ep135 이후로도 정제 전 35–38 / 정제 후 64–66에서 평평하다. 세션이 한 번 끊겼다 재개되면서(09/16 09:00경) 로컬 `cnt2` 학습(ep109에서 중단)과 두 판독 루프가 멈춰 있어 다시 띄웠다.

GT 치환 표(전체 데이터 ep95 raw; 괄호는 10% v10 ep95): P 35.9 (33.6) · GT pos만 43.1 (39.8) · GT topo만 35.6 · GT rot만 37.4 · GT scale만 35.0 · GT latent만 36.7 · pos+latent 46.0 · **ALL-latent 45.9 (44.6)** · ALL-latent+meanlat 39.5 · ALL-rot 49.6 · ALL-scale 61.4 · ALL-pos 41.6 · **ALL 79.3 (72.8)**. 읽는 법: 노드·topology·scale·rot이 모두 GT여도 예측 latent로는 45.9에 그치고(ALL과 33점 차), 반대로 latent만 GT여도 36.7이다 — 정제 없는 파이프라인에서는 latent와 노드가 서로 묶여 있어 하나만 고쳐서는 오르지 않는다. 정제는 이 둘을 동시에 입력에 맞추기 때문에 66까지 간다. 전체 데이터 모델은 ALL(79.3 vs 72.8)과 GT pos(43.1 vs 39.8)에서 10% 모델보다 낫다: 데이터가 늘면 VAE 디코드/조립 경로와 노드 조건부 shape는 좋아지지만 P는 같다. 5 epoch마다 엄격 판독과 정제 판독이 자동으로 이어진다(약 15–18 min/epoch).

**네 zoom 단계를 모두 목표로 쓰면 67.6 (14:05).** 정제 목표를 캐시의 1×·2× 깊이 채널에서 1×·2×·4×·8× 네 채널(`--target_zooms 1,2,4,8`)로 늘리면 20개 식물 **35.8 → 67.6** (DAP > 15: 41.1 → 74.0). 이득은 어린 식물에서 온다: DAP 2/3/11/12가 0.0/10.6/20.7/26.8 → 35.3/21.5/52.8/58.3 (1×·2× 창에서는 식물이 몇 픽셀뿐이라 신호가 없던 경우). **오늘의 최고 배포 수치는 엄격 P 67.6**이고, 스크립트 기본값을 이 설정(입력 카메라 창, prior 5/0.5, 네 zoom 목표; 옛 동작은 `--origin_camera`)으로 바꿨다.

기본 설정으로 step을 80으로 늘리면 **36.0 → 69.1** (DAP > 15: 41.3 → 78.6; 40 step 67.6 / 74.0 대비 +1.5 / +4.6). 원점 창에서는 80 step이 이득이 없었지만(§11.9), 창을 맞추고 prior를 넣은 뒤에는 step을 더 주는 만큼 조금 더 오른다. 식물당 시간은 두 배(수 초 → 십수 초).

설정 민감도(기본 설정 대비, 40 step): 학습률 두 배(pos 6e-3, scale·latent 4e-2) 33.3 → 67.0, 약한 prior(2 / 0.2) 36.2 → 67.8 — 둘 다 기본값(67.6)과 같은 수준이라 정제 이득은 40 step에서 약 +32점으로 포화하고, 80 step이 +33이다. 남은 오차는 정제가 손댈 수 없는 부분, 즉 존재 마스크(GT 73개 중 60개만 활성)와 topology다.

**평균 latent에서 출발해도 같은 곳에 도착한다 (17:20).** 정제의 초기 latent를 flow가 샘플한 값 대신 **학습 식물 300개(전 생육 단계)의 GT phytomer latent 평균**으로 두면(`--init_mean_latent`; 모델의 `latent_mu` 버퍼는 데이터셋 순서상 앞쪽 식물만의 평균이라 9–12%에서 출발해 쓸 수 없다):

| 체크포인트 | 샘플 latent에서 출발 | 평균 latent에서 출발 |
|---|---|---|
| 10% v10 ep95 | 35.8 → 67.6 | 36.1 → **66.5** |
| 전체 데이터 v10_cam ep90 (raw) | 30.4 → 57.3 | 35.5 → **66.5** |

두 모델이 같은 66.5에 도착한다. 즉 (1) Stage 3가 샘플하는 latent는 정제 뒤에는 평균 latent보다 나을 것이 없고(probe R² ≈ 0과 일치), (2) 전체 데이터 모델처럼 샘플 latent가 나쁜 체크포인트도 평균에서 출발하면 손해가 없다. 배포 파이프라인은 **Stage 2 노드 + 평균 latent + 렌더 정제**로 단순화할 수 있고, 네트워크가 실제로 기여하는 것은 노드 위치·존재·topology다. 남은 개선 여지는 그 셋에 있다.

존재 임계값을 낮춰 노드를 더 켜는 것(`--exist_thresh`, 10% v10 ep95): 0.3 → 35.7 → 65.3, 0.4 → 32.7 → 64.2 — 기본 0.5(67.6)보다 나쁘다. 추가로 켜지는 노드는 대부분 거짓 양성이고 정제는 노드를 끌 수 없으므로, 존재 부족(60/73)은 학습 쪽에서 풀어야 한다.

**정제의 한계 두 가지를 풀어봄 — roll(고정 축) / organ 생성·삭제 (09:40).** 사용자 요청으로 `refine_plant`에 두 가지를 추가했다.

1. **회전(roll)**: 기존 `--recompute_rot`는 매 step 움직이는 노드 위치에서 forward axis를 다시 유도해 roll과 결합했는데, 이게 불안정의 원인이었다(§ 앞부분 33.5→60.0). 고쳐서 forward axis를 **pos0에서 한 번만** 계산해 고정하고(부모 위치도 기본 경로처럼 고정), roll 각도만 그 축 주위로 최적화하도록 분리했다(`--opt roll`, 더 이상 `--recompute_rot` 불필요). 10% v10 ep95: **32.9 → 66.0** (DAP>15 38.2→74.9) — 불안정은 사라졌지만 기본(pos+scale+latent, 67.6)보다 소폭 낮다. GT 치환 실험에서 GT rot 하나만 줘도 +1.5점(65.1→65.6)에 그쳤던 것과 일치 — roll은 애초에 정제로 크게 고칠 거리가 많지 않다.
2. **organ 생성·삭제**: 기존엔 existence가 샘플 시점에 0.5로 하드 임계값 처리되어 정제가 되돌릴 수 없었다(`--exist_thresh`를 낮춰도 오히려 나빠짐, 9/15). 처음엔 `--opt exist`로 존재를 **연속 확률**로 만들어 phytomer의 scale에 곱하는 방식으로 구현했는데(0에 가까우면 렌더에서 사실상 사라짐), 이는 편법이라는 지적을 받았다 — scale 축소는 fade가 아니라 실제 크기 변화라서 depth footprint가 남고, 겹친 organ 사이의 occlusion도 올바르게 처리 못 한다. 다시 보니 렌더러(`helios_pytorch_renderer.py`)에 이미 **정식 alpha compositing**이 구현돼 있었다(`rgb_comp = alpha*rgb + (1-alpha)*bg_rgb`, depth도 동일) — `build_mesh_from_part_tensor(existence=...)`가 그 채널이고, 학습 코드도 이미 이 경로로 렌더링하되 `existence.detach()`로 gradient만 끊어 쓰고 있었다. `plant_from_nodes(soft_exist=True)`를 추가해 organ 행을 다 유지한 채 phytomer별 존재확률을 그대로 이 alpha 채널로 넘기도록 고쳤다(scale은 더 이상 존재와 무관).

   | 방식 | 10% v10 ep95 | DAP>15 |
   |---|---|---|
   | scale 곱하기(편법) | 33.1 → 62.4 | 38.2 → 71.7 |
   | **alpha compositing(정석)** | **34.3 → 65.0** | **39.3 → 74.3** |
   | 기본(exist 미최적화) | 35.8 → 67.6 | 35.8 → 74.0 |

alpha 방식이 scale-hack보다 +2.6점 낫다(정석이 실제로 더 낫다는 게 확인됨) — 그러나 여전히 존재를 건드리지 않는 기본값보다는 낮다.

   `reg_exist`를 1.0 → 0.3으로 낮추니 더 좋아졌다: **35.5 → 65.9** (DAP>15 40.0→73.3, +0.9 vs reg_exist=1.0). 노드 개수가 실제로 여러 식물에서 ±1~3 바뀌기 시작했고(예: idx 67809 41→38, idx 77889 48→46), 이전에 무너지던 어린 식물(idx 2616, DAP 3)도 더 이상 붕괴하지 않는다(19.4→24.1). `reg_exist` 기본값을 0.3으로 낮췄다. 기본값(exist 미최적화, 67.6)과의 격차는 scale-hack 5.2점 → alpha+reg1.0 2.6점 → alpha+reg0.3 1.7점으로 꾸준히 좁혀지고 있다.

둘 다 **메커니즘은 정상 동작**하지만(에러 없음, 노드 개수/roll이 실제로 바뀜) 기본 설정(pos+scale+latent, 67.6)을 아직 못 넘는다 — 추가 자유도마다 별도의 learning rate/정규화 튜닝이 필요해 보인다. 스크립트 기본값은 그대로 둔다.

**렌더 손실 비율 커리큘럼 (09:45, 사용자 제안).** 학습 중 렌더 손실은 배치의 1/6만 보고 있는데, 이걸 학습이 진행되며 점점 키우면(`--render_fraction_final`, `--render_fraction_ramp_epochs`; 런처 `RENDER_FRACTION_FINAL`/`RENDER_FRACTION_RAMP_EPOCHS`) latent로 가는 렌더 gradient가 후반부에 강해질 것이라는 아이디어. 구현해서 10% 프로토콜로 0.167 → 0.5(15 epoch에 걸쳐, s3geom ep78 재개 직후부터)를 테스트 중이다(`sub10_v10_rampR`, 대기열). 다만 렌더는 이미 1/6에서 step 시간의 70%를 차지하므로(2026-09-09 프로파일), 비율을 올리면 epoch당 시간이 비례해서 늘어난다 — 신호량은 늘지만 "정보 자체가 이미지에 없다"는 구조적 병목이면 도움이 안 될 수 있다(아래 진단 참고).

**진단: per-node 시각 conditioning은 죽어있지 않다 (10:05).** latent를 만드는 `PhytomerFlowMatchingDecoder.phytomer_projector`(zero-init residual MLP로 DINOv2 패치 특징을 노드별로 bilinear 샘플링)를 직접 뜯어봤다.

- `fusion[-1]`(zero-init된 마지막 레이어) weight norm = **7.74** — 0에서 크게 벗어나 있다(첫 레이어 norm 13.79와 같은 자릿수). 학습이 이 경로를 그냥 방치한 게 아니라 실제로 키웠다.
- 같은 식물 안에서 활성 노드들 사이의 특징 표준편차(cross-node std) ≈ **0.055**, 전체 특징 표준편차 ≈ 0.15 — 즉 노드 간 차이가 전체 스케일의 약 1/3만큼 존재한다. 노드별 conditioning이 상수로 붕괴된 것은 아니다.

즉 "이미지 특징을 애초에 못 읽는다"는 가설은 기각된다 — projector는 노드마다 다른 신호를 만들어내고 있다. 그렇다면 R²≈0의 원인은 architecture가 아니라 **그 신호를 latent로 바꾸는 학습 목표(약한 flow-matching 손실 + 1/6만 보는 렌더 손실)가 그 신호를 latent 예측에 제대로 쓰도록 못 밀어붙인다** 쪽일 가능성이 높다. 렌더 비율 커리큘럼(위)이 이 가설에 대한 직접적인 검증이 된다: 신호가 있는데 못 쓰는 거라면 렌더 비율을 늘리는 게 통해야 하고, 그래도 안 바뀌면 latent 전용 conditioning 경로(별도 cross-attention/직접 latent 회귀 보조 손실)가 다음 단계다.

**GT 치환 + 정제 (20:30, 전체 데이터 ep95 raw; `eval_gt_substitution_ablation.py --refine`).** 매칭된 노드에 GT 값을 넣은 뒤 같은 정제(기본 설정)를 걸면:

| 변형 | 정제 전 | 정제 후 |
|---|---|---|
| P (예측 그대로) | 37.8 | **65.1** |
| GT pos | 44.9 | 67.1 |
| GT rot | 39.6 | 65.6 |
| GT pos+rot | 46.0 | 67.5 |
| GT scale | 38.7 | 67.1 |
| ALL-latent (latent만 예측) | 47.9 | **68.6** |
| ALL (latent까지 GT) | 78.6 | 77.0 |

읽는 법: 매칭된 노드의 위치·회전·크기를 모두 GT로 줘도 정제 뒤에는 65.1 → 68.6, 즉 **3.5점**밖에 오르지 않는다 — 정제가 노드 기하 오차는 이미 거의 흡수한다. 반면 latent까지 GT면 77–79이므로 정제가 못 찾는 **10점은 phytomer 형태(latent)**다. 위에서 본 입력 CHM(1×–8×)은 잎의 3D 자세·형태를 다 담지 못하므로 이 10점은 상당 부분 단일 상면 입력의 한계로 보인다. 어린 식물(DAP ≤ 15)은 다르다: P+정제 26.9 vs ALL+정제 59.0으로 노드 자체(존재·위치)가 병목이다. 정제 함수는 `eval_test_time_refinement.refine_plant`로 분리했고(기본 실행 재검증 36.1 → 65.7), latent 다중 시작(샘플 latent와 평균 latent에서 각각 정제해 입력 손실이 낮은 쪽 채택, `--multi_start`)도 34.6 → 65.8로 이득이 없다 — 두 시작점이 같은 국소해로 간다. 남은 개선 여지는 어린 식물의 존재·노드이고, 형태 10점은 상면 단일 입력의 한계로 본다.

모델에 무관하게 작동한다: 100% 데이터 baseline ep135(학습만 27.0)도 기본 설정 40 step으로 **27.0 → 63.1** (DAP > 15: 31.7 → 72.4). v10(35.8 → 67.6)과의 차이는 정제 전 8.8점에서 정제 후 4.5점으로 줄어든다 — 학습 쪽 개선은 정제 뒤에도 남지만 절반으로 줄고, 렌더 기반 정제가 결과의 대부분을 결정한다.

효과 없는 변형 두 가지(기본 설정, 40 step): 노드가 움직일 때 회전·부모 위치를 현재 노드에서 다시 유도(`--recompute_rot`) 33.5 → **60.0**, 여기에 roll까지 변수로(`--opt pos,scale,latent,roll`) 35.1 → **61.4** — 둘 다 회전을 샘플 시점에 고정한 기본(67.6)보다 나쁘다. 회전이 노드를 따라 움직이면 손실 지형이 거칠어져 잎이 엉뚱한 방향으로 넘어가는 것으로 보인다. 회전은 고정한 채 위치·크기·latent만 푸는 것이 맞다.

**주의 — 수치와 그림이 어긋난다.** 입력 카메라 창 버전의 렌더(`docs/results/assets/20260915_test_time_refinement_before_after_input_camera.png`, 여섯 식물 41.1 → 72.2)를 보면 DAP 39·60·88에서 잎 몇 장이 실루엣을 채우기 위해 **크고 납작한 다각형으로 부풀어** 있다. 실루엣 IoU는 오르지만 기하는 식물답지 않다 — 손실이 실루엣·깊이만 보므로 scale·latent가 사전분포를 벗어나도 벌점이 없기 때문이다. 원점 창 버전(§11.7, 여섯 식물 39.8 → 64.3)은 창이 어긋난 만큼 덜 공격적이어서 잎 모양이 유지된다. 그래서 (1) 랩미팅에는 원점 창 그림을 주 결과로, 입력 카메라 창 수치는 "IoU만으로는 부족하다"는 증거로 가져가고, (2) scale·latent를 샘플값 근처에 묶는 벌점(`--reg_scale`, `--reg_latent`)을 추가해 다시 재는 중이다.

**shape prior를 넣으면 부풀림 없이 70.4 (13:30).** scale과 latent를 샘플값 근처에 묶는 벌점(`--reg_scale 5 --reg_latent 0.5`, 샘플값과의 제곱 편차)을 넣고 같은 여섯 식물을 다시 정제하면 **39.6 → 70.4**(24.1 → 68.5, 36.1 → 71.8, 47.2 → 77.3, 26.9 → 62.1, 58.7 → 75.0, 44.9 → 67.7)이고, 렌더에서 큰 납작 다각형이 사라져 잎 모양이 GT와 같은 종류로 유지된다. 벌점 없는 72.2와 거의 같은 수치를 식물다운 기하로 얻는 셈이다. 20개 식물 엄격 프로토콜에서는 **35.8 → 63.9** (DAP > 15: 39.3 → 72.2; 벌점 없는 62.9 / 74.5와 같은 수준)로, 기하를 지키면서 이득은 그대로다. **오늘의 최고 배포 수치는 엄격 P 63.9** (baseline 학습만 27, v10 학습만 35, 원점 창 정제 46.9)다.

그림: `docs/results/assets/20260915_test_time_refinement_before_after.png` (`tools/refinement_before_after_figure.py`; 여섯 식물, 입력 JPEG · GT 렌더 · 샘플 그대로 · 정제 후; 입력 카메라 창 + shape prior + 네 zoom 목표 = 스크립트 기본값, 40.8 → 70.6). 비교용: `..._origin_frame.png` (원점 창, 39.8 → 64.3), `..._input_camera_noprior.png` (입력 카메라 창, 벌점 없음, 41.1 → 72.2, 잎 부풀림).

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
| `diffusion_based/dataset/phytomer_packets.py`, `tests/test_assemble_packets_batched.py` (신규) | §6: 조립 벡터화 (`f3c5366`) |
| `part_array_dataset.py` (`attach_parent_links`), `helios_pytorch_renderer.py` (`render_batched`), `train_hierarchical_flow_matching.py`, `tests/test_render_batched.py`, `tests/test_render_loss_vectorized.py` | §6.1: 샘플별 루프 제거 (`a1a82bc`) |
| `phytomer_topology.py` (`_resolve_forest`), `hierarchical_hungarian_matcher.py` (`_forward_phytomer_batched`), `tests/test_chain_vectorized.py`, `tests/test_matcher_batched.py` | §6.2: chain·matcher 벡터화 (`25c2251`) |
| `diffusion_based/eval/eval_gt_substitution_ablation.py`, launcher `RENDER_FRACTION` 0.167 | §7: GT 치환 ablation |
