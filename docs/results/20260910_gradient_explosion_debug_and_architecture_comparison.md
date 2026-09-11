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

## 6. 하이브리드 디커플링 (Hybrid Decoupled Architecture) 및 Loss 정예화 결정

### 6.1 손실 함수 진단 및 8대 핵심 손실 체계 확정

기존 10개 손실 함수에 대한 정밀 분석 결과, 노이즈를 유발하거나 중복되는 손실을 정리하고 누락된 회전 손실을 추가하여 **8대 정예 손실 체계**로 재편성:

```python
loss = (
    # Stage 1: Macro Prior
    phy_count_weight * loss_phy_count          # 1.0 : 전체 파이토머 수량 예측 (Soft Margin 마스킹)
    # Stage 2: 3D Node Scaffold (Macro Topology)
    + 2.0 * loss_anchor_pos                    # 2.0 : 마디 3D 위치 (Smooth L1)
    + 1.0 * loss_anchor_rot                    # 1.0 : [신규] 마디 6D 회전 기준계 (Smooth L1)
    + 1.0 * loss_anchor_scale                  # 1.0 : 마디 3D 스케일 (Smooth L1)
    + 1.0 * loss_anchor_exist                  # 1.0 : 마디 존재 확률 (BCE with logits)
    # Stage 3: Canonical Phytomer Latent Flow Matching (Micro Geometry)
    + 1.0 * loss_fine_vel                      # 1.0 : 64D VAE 잠재공간 속도장 MSE (z0 ~ N(0, I))
    + 1.0 * loss_fine_exist                    # 1.0 : 파이토머 내 10개 장기 슬롯 존재 확률 (BCE)
    # Stage 4: Differentiable Optical Grounding
    + depth_loss_weight * loss_depth           # 0.5 : CHM 수관 깊이 매칭 (Smooth L1)
    + silhouette_loss_weight * loss_dice       # 1.0 : 탑뷰 투영 실루엣 매칭 (Soft Dice)
)
```

**제거/비활성화 대상:**
1. `loss_cos` (0.0): 간이 CAD 메쉬와 실사 드론 RGB 간의 색상 불일치로 인한 고주파 노이즈 제거 및 렌더링 역전파 연산 가속.
2. `loss_dap` (0.0): `loss_phy_count`와 생물학적으로 90% 이상 강한 상관관계를 가져 중복되며, 기하 형성에 직접 기여하지 않음.

---

### 6.2 3대 핵심 아키텍처 질의 및 분석 결론

#### Q1. `anchor_features`와 `pos`, `rot`, `scale`, `exist`의 관계
- **`anchor_features` (384D)**: Transformer Decoder와 Self-Attention을 거친 **원천 의미론적 잠재 벡터 (Semantic Latent / Stem Cell)**. 주경/곁가지 여부, 마디 간 기하학적 연쇄 관계, 부착될 잎의 수량 등 전신 문맥을 압축.
- **`pos`, `rot`, `scale`, `exist` (13D)**: `anchor_features`로부터 각각의 전용 선형/MLP 헤드가 디코딩해 낸 **해석된 물리적 3D 뼈대(Physical Coordinates)**.
  - $\text{pos} = \text{ref\_points} + \text{pos\_head}(\text{anchor\_features})$
  - $\text{rot} = \text{rot\_head}(\text{anchor\_features})$
  - $\text{scale} = \text{softplus}(\text{scale\_head}(\text{anchor\_features}))$
  - $\text{exist} = \sigma(\text{exist\_head}(\text{anchor\_features}) + \text{macro\_prior})$

#### Q2. `scale`과 `exist`는 Stage 3에 전달 안 받는가?
- **`scale`**: 기존 76D Bridge Flow에서는 scale을 $z_t$ 플로우 벡터에 억지로 집어넣었음. 하이브리드 디커플링에서는 Stage 2가 예측한 `anchor_scale.detach()`를:
  1) Stage 3 쿼리 조건(`self.node_scale_mlp(anchor_scale.detach())`)으로 주입하여 크기에 따른 미세 형태 조절 보조,
  2) 최종 메쉬 조립 단계(`denormalize_packet_scales`)에 전달하여 절대 크기 복원.
- **`exist`**: 이미 수학적 조건부 확률 게이팅으로 결합:
  $$P(\text{organ}) = P(\text{anchor}) \times P(\text{slot} \mid \text{anchor})$$
  Soft Margin 마스킹을 통해 비활성 앵커의 불필요한 연산 및 노이즈 차단.

#### Q3. 9월 7일처럼 Pos만 하는 게 좋은가, 아니면 rot과 scale도 같이 해야 하는가?
- **9월 7일 Option B (Organ-level)**: 각 장기의 16D 잠재벡터가 **자신의 3D 회전과 크기를 내장**하여 글로벌 좌표계에서 작동했으므로 Pos만으로 충분했음.
- **현재 구조 (Phytomer-level, PhytomerVAE v3)**: 10개 장기 간 공분산($r=0.783$)을 묶기 위해 **"원점(0,0,0)에 놓인 표준 크기(1.0)의 정규화된 국소 좌표계"**로 모델링됨. 따라서 이 국소 파이토머를 3D 공간에 배치하려면:
  1) 어디에 달리는가? $\to$ **`pos`** (3D 위치)
  2) 어느 방향으로 뻗는가? $\to$ **`rot`** (6D 회전 기준계)
  3) 얼마나 큰가? $\to$ **`scale`** (3D 크기)
  이 3가지 변환 정보가 필수적임.
- **결론**: Rot과 Scale을 예측하는 것은 필수적이며, 이를 Stage 2의 단순 피드포워드 헤드로 깔끔하게 학습시키고(`loss_anchor_rot`, `loss_anchor_scale`), Stage 3은 순수 64D VAE 잠재공간에서 잎 형상 생성에만 전념하도록 분리하는 것이 최적.

---

### 6.3 하이브리드 디커플링 아키텍처 다이어그램

```
[Visual Image Tokens] (DINOv2 + PETR 3D Ray PE)
         │
         ▼
[Stage 1: MacroBiologicalHead] ──→ loss_phy_count (Soft Margin Masking)
         │
         ▼
[Stage 2: CoarseSkeletalTransformer] (Deterministic Set Transformer)
         ├─→ pos_head   ──→ anchor_pos   ──→ loss_anchor_pos
         ├─→ rot_head   ──→ anchor_rot   ──→ loss_anchor_rot (신규 추가!)
         ├─→ scale_head ──→ anchor_scale ──→ loss_anchor_scale
         ├─→ exist_head ──→ anchor_exist ──→ loss_anchor_exist
         └─→ anchor_features (384D Context)
                   │
    ┌──────────────┴────────────────────────────────────┐
    │  Conditioning: pos.detach(), rot.detach(),         │
    │                scale.detach(), anchor_features    │
    ▼                                                   │
[Stage 3: PhytomerFlowMatchingDecoder]                  │
    │  Flow Vector: 64D VAE Latent ONLY                 │
    │  Prior: z_0 ~ N(0, I_64) (표준 정규분포!)          │
    │  Velocity Target: v* = z_1 - z_0                  │
    ├─→ velocity_head ──→ pred_v ──→ loss_fine_vel       │
    └─→ exist_head    ──→ pred_slot_exist ──→ loss_fine_exist
         │
         ▼ (ODE Sampling: z_1)
[PhytomerVAE Decoder (Frozen)]
         │
         ▼ (Canonical local 10-organ packets)
[Stage 4: Assembler & Differentiable Renderer] ←────────┘
    - denormalize_packet_scales(packet, anchor_scale)
    - apply_ref_for_flow(packet, anchor_pos, anchor_rot)
    - build_mesh_from_part_tensor
    - render_multiscale_pyramid
         ├─→ loss_depth (CHM)
         └─→ loss_dice (Top-view Silhouette)
```

**수렴 안정성 보장 메커니즘:**
1. $z_0 \sim \mathcal{N}(0, I_{64})$이므로 $\|v^*\|$가 항상 $\sqrt{2 \times 64} \approx 11.3$ 바운드 내에 머물며, 차원 정규화 후 `VelLoss`는 항상 $\sim 1.0$ 수준으로 통제됨.
2. Stage 2 초반의 앵커 위치 오차가 Stage 3의 Flow Matching 속도장으로 전이(Coupling)되지 않아 그라디언트 폭발 원천 차단.

---

## 7. 참고 이미지

- [`assets/hierarchical_self_consistency_epoch_008.png`](assets/hierarchical_self_consistency_epoch_008.png) — Job 38235969, epoch 8 결과
- [`assets/hierarchical_self_consistency_epoch_125_20260908.png`](assets/hierarchical_self_consistency_epoch_125_20260908.png) — Option B epoch 125 (9/8 아침 확인한 최고 결과)
