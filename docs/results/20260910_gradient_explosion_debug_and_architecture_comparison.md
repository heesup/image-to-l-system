# 2026-09-10 그라디언트 폭발 디버깅 및 아키텍처 비교 분석

> **세션 날짜**: 2026-09-10 (PDT)  
> **관련 Job**: `38235936` (폭발 확인), `38235969` (복구 시도 중, epoch 3에서 재폭발)  
> **이전 세션 마일스톤**: [`docs/results/20260908_3d_spatial_vision_and_hierarchical_reconstruction_milestone.md`](20260908_3d_spatial_vision_and_hierarchical_reconstruction_milestone.md)

---

## 1. 오늘 확인된 버그: `.detach()` 누락으로 인한 양성 피드백 루프

### 1.1 근본 원인

`train_hierarchical_flow_matching.py`의 phytomer 브릿지 플로우 경로(`flow_granularity == "phytomer"`)에서 Stage 3의 속도 손실이 Stage 2 positional head로 역전파되는 비정상 경로가 형성되었다.

```
[fwd1 with no_grad]
    Stage 2: pos_head → pred_anchor_pos (no grad)
    Stage 3: vel_head → pred_velocity (no grad)

[z_0 재구성]
    z_0[:, :, :3] = pred_anchor_pos + 0.05 * eps   ← .detach() 되어 있음 ✓

[fwd2 with grad]
    Stage 2: pos_head → pred_anchor_pos (live grad) ← loss_anchor_pos로 수렴 ✓
    Stage 3: vel_head → pred_velocity (live grad)   ← loss_fine_vel로 수렴 ✓
```

→ fwd2에서 `pred_anchor_pos`가 live gradient를 가지므로 `loss_anchor_pos`가 올바르게 역전파됨.

→ fwd2에서 Stage 3 `fine_stage` 호출 시 `anchor_pos.detach()`로 conditioning:
  - `hierarchical_part_flow_matching.py` L1129: `anchor_pos=anchor_pos.detach()`
  - 이는 올바름 ✓

### 1.2 실제 폭발 메커니즘 (Job 38235936)

이전 버전 (`38235936` 이전)에서의 버그:

```python
# 버그: fwd2에서 pred_anchor_pos.detach() 없이 z_0 재구성
z_0[:, :, :3] = pred_anchor_pos + 0.05 * eps  # pred_anchor_pos는 live graph

# 속도 타겟 = z_1 - z_0
tgt_velocity = tgt_z1_phyto - z_0  # z_0에 pred_anchor_pos의 grad가 연결됨

# 결과: d(loss_vel)/d(pred_anchor_pos) = 2 * (v_pred - v*) * I
# pos_head가 velocity loss 최소화를 위해 좌표를 임의로 폭주시킴
```

**그라디언트 폭발 타이밍**: LR warmup(epoch 1-2)이 끝나고 본 학습률(1.0×)에 도달하는 epoch 3~4에서 피드백이 급격히 발산.

### 1.3 적용된 수정 사항 (현재 코드)

| 파일 | 수정 내용 |
| :--- | :--- |
| `train_hierarchical_flow_matching.py` L326-338 | fwd1 결과: `.detach()` 적용 후 z_0 재구성 |
| `train_hierarchical_flow_matching.py` L316-325 | fwd1 자체를 `torch.no_grad()`로 감쌈 |
| `train_hierarchical_flow_matching.py` L488-511 | fwd2 출력으로 `pred_anchor_pos/rot/logits` 갱신 (live grad) |
| `hierarchical_part_flow_matching.py` L1129 | `anchor_pos.detach()` fine_stage conditioning |

---

## 2. 9/7-9/8 아키텍처 vs. 현재 아키텍처 비교 분석

### 2.1 타임라인 정리

```
2026-09-07 22:24  커밋 ed95f45  "Option B 16D Latent Hierarchical Flow Matching"
                  → OrganLatentVAE 동결 디코더 포함, Job 38143585 시작
2026-09-08 아침   → Job 38143585의 epoch 125 결과 확인 (긍정적)

2026-09-09~10     → PhytomerVAE v3 + 76D Bridge Flow 도입
                  → 3단계 캐스케이드 아키텍처로 전환
2026-09-10        → 그라디언트 폭발 디버깅 세션 (이 문서)
```

### 2.2 아키텍처 구조 비교

| 측면 | Option B (9/7-9/8) | 현재 (76D Phytomer Bridge Flow) |
| :--- | :--- | :--- |
| **Stage 1** | CoarseSkeletalTransformer (L1 직접 지도) | MacroBiologicalHead (DAP + N_phy + Soft Margin) |
| **Stage 2** | FineBotanicalFlowMatchingDecoder | CoarseSkeletalTransformer (3D scaffold) |
| **Stage 3** | - | PhytomerFlowMatchingDecoder (76D) |
| **잠재 공간** | 16D per-organ (OrganLatentVAE, 8 슬롯/앵커) | 64D per-phytomer (PhytomerVAE v3, 10 슬롯/파이토머) |
| **z_0 초기화** | $\mathcal{N}(0, I_{16})$ 순수 가우시안 | Stage 2 scaffold pose + $\mathcal{N}(0, I_{64})$ 혼합 |
| **장기 결합** | 독립 (파이토머 내부 정합 없음) | 파이토머 단위 공동 모델링 |
| **손실 개수** | ~5개 | ~10개 |

### 2.3 9/8 epoch_125 결과가 좋았던 이유

**핵심**: Option B에서는 Stage 2 fine decoder가 `anchor_pos`와 **완전히 독립적인 순수 가우시안** $z_0$에서 출발했다.

```python
# Option B: z_0는 anchor_pos와 무관
z_0 = torch.randn(B, K, 16, device=device)   # 완전 독립
# → Stage 2 vel loss가 pos_head로 역류할 수 있는 경로가 물리적으로 없었음
# → loss_anchor_pos가 유일한 신호, 항상 100% 살아있었음
```

→ 150 에폭 수렴 + 단순한 손실 경관 + 그라디언트 안정성의 조합.

### 2.4 현재 아키텍처가 이론적으로 우월한 이유

1. **파이토머 내부 정합성**: 64D 잠재가 엽병-소엽-화경을 묶어 모델링 (r=0.783 엽병-소엽 스케일 공분산)
2. **Bridge Flow 효율**: $z_0$가 GT 위치 근방에서 시작 → ODE가 잔차만 수정
3. **명시적 스케일 학습**: `anchor_scale` 헤드가 절대 스케일 학습 (하드 클램핑 불필요)
4. **국소 시각 투영**: `AnchorVisualProjector`가 마디 위치별 국소 패치 특징 주입
5. **생물학적 사전**: Stage 1 MacroBiologicalHead의 Soft Margin이 파이토머 개수 예측

### 2.5 현재 아키텍처가 초기 수렴이 더 느린 이유

1. **복잡한 손실 경관**: 10개 손실의 균형 → 더 많은 에폭 필요
2. **Bridge coupling**: Stage 2가 부정확한 초기 에폭에서 Stage 3이 나쁜 출발점에서 시작
3. **PhyLoss 초기값**: 에폭 1: Pred 12.4 / GT 54.9 (4배 이상 차이)

**예상 수렴 마일스톤**:
- Epoch 30: `AncPos` 0.01~0.03m 안착
- Epoch 50: `PhyLoss` GT의 10% 이내로 수렴
- Epoch 100+: 장기 정합성에서 Option B를 명확히 앞섬

---

## 3. 현재 학습 상태 (Job 38235969)

에폭 3에서 재폭발 발생 중:

```
Epoch 003 | VelLoss: 1050.5 | AncPosLoss: 292.9 | Recovery: step 51~116
```

### 3.1 에폭 3 폭발 원인 분석

`fwd2`의 `pred_anchor_pos`가 live gradient를 가지므로 `loss_anchor_pos`(292m)는 올바르게 역전파되고 있다. 그런데 `VelLoss: 1050`이 가중치 2.0으로 2100이 되어 `AncPos(292 × 2 = 584)`를 3.6배 압도하는 상황이다.

에폭 1-2는 warmup이라 LR이 낮아 폭발하지 않았고, 에폭 3에서 본 LR에 도달하자마자 velocity loss 그라디언트 norm이 폭발하는 패턴이다.

**추가 조사가 필요한 지점**:
- `loss_fine_vel` 의 `tgt_velocity = tgt_z1_phyto - z_0` 에서 `tgt_z1_phyto`의 스케일
- VAE 잠재벡터 $z_{\text{phy}} \in \mathbb{R}^{64}$가 $\mathcal{N}(0, I)$ 정규화를 통과했는지 확인 필요
- velocity loss 가중치 2.0 → 0.5로 낮추거나 velocity loss에 별도 클램핑 추가 검토

---

## 4. 수정된 주요 파일 목록

| 파일 | 변경 내용 | 커밋 |
| :--- | :--- | :--- |
| `diffusion_based/training/train_hierarchical_flow_matching.py` | fwd1 no_grad, fwd2 live grad, detach z_0, nan_to_num recovery | 이 세션 |
| `diffusion_based/models/hierarchical_part_flow_matching.py` | fine_stage anchor_pos.detach() conditioning | 이 세션 |
| `diffusion_based/dataset/part_array_dataset.py` | phytomer_ids 캐시 포함 | 이전 세션 |
| `diffusion_based/dataset/phytomer_packets.py` | v3 패킷 빌더 (10-slot, scale-normalized) | 이전 세션 |
| `diffusion_based/training/hierarchical_hungarian_matcher.py` | 계층적 헝가리안 매처 | 이전 세션 |
| `slurm_scripts/train_hierarchical_flow_matching.sh` | 4xGPU 배치, epoch-gated render | 이전 세션 |

---

## 5. 참고 이미지

- [`assets/hierarchical_self_consistency_epoch_008.png`](assets/hierarchical_self_consistency_epoch_008.png) — Job 38235969, epoch 8 결과 (pending)
- [`assets/hierarchical_self_consistency_epoch_125_20260908.png`](assets/hierarchical_self_consistency_epoch_125_20260908.png) — Option B epoch 125 (9/8 아침 확인한 결과)
