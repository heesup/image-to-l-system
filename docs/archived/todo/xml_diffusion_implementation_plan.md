# 3D Plant Organ Part Diffusion Model 업데이트 계획

## 배경 및 목표

현재 `PlantGraphDiffuser3D`는 **완전히 합성(procedural) 데이터**(`Plant3DDataset`)로 학습됩니다.
하지만 `cowpea_0000_plant_0000.xml` (Helios XML)는 실제 L-system 시뮬레이터가 생성한 **진짜 식물 구조**를 담고 있습니다:

| 관측 | 현재 | 목표 |
|---|---|---|
| 데이터 소스 | 수작업 절차적 생성 (Plant3DDataset) | Helios XML → 실제 식물 구조 |
| Organ 표현 | 7D 단일 원시체 (x,y,z,θ,φ,len,is_leaf) | Phytomer 계층 기반 다중 organ 타입 |
| 토폴로지 | 간단한 3레벨 나무 | Shoot-Phytomer-Internode-Petiole-Leaf 계층 |
| 학습 데이터 | 4개 샘플 고정 | XML 배치 로드, DataLoader 배치 학습 |
| 이미지 조건 | 합성 렌더링 | Helios 렌더링 JPEG + XML 쌍 |

---

## Open Questions

> [!IMPORTANT]
> **Q1: Organ 표현 세분화 수준**
> XML 구조를 보면 `internode`, `petiole`, `leaf`, `floral_bud` 등 여러 organ type이 있습니다.
> 현재 7D 벡터를 어느 수준으로 확장할까요?
> - (A) 기존 7D 유지 + organ_type 원-핫 (4개 타입: internode/petiole/leaf/floral_bud) → 11D
> - (B) 각 organ에 specific한 속성 분리 (internode: length+radius+pitch; leaf: scale+pitch+yaw+roll)
> - (C) shoot-level 구조를 graph 계층으로 명시적 모델링

> [!IMPORTANT]
> **Q2: 실제 이미지-XML 쌍 데이터 규모**
> 현재 `cowpea_0000_vis.jpeg` + `cowpea_0000_plant_0000.xml` 1쌍이 있습니다.
> 학습에 사용할 DAP 범위와 샘플 수는 얼마나 생성할 계획인가요?
> - 예: DAP 5~60일, 각 DAP에 N개 샘플 (다른 seed/파라미터)
> - 현재 합성 100개 샘플로 계속 보완 학습할지, XML 전용으로 교체할지?

> [!NOTE]
> **Q3: Floral Bud 처리**
> XML에 `floral_bud` (peduncle + inflorescence)가 있습니다. DAP 10일 기준이라 fruit 없음.
> Floral bud를 별도 organ type으로 모델링할지, 무시할지 결정 필요.

---

## 제안 변경사항

### Component 1: XML Parser & 실제 데이터셋 파이프라인

#### [NEW] `dataset/helios_xml_parser.py`
Helios XML을 파싱해서 organ-level graph 표현으로 변환:
- `parse_helios_xml(xml_path)` → shoots, phytomers, nodes 계층 파싱
- **Organ 타입 매핑**: `internode(0)`, `petiole(1)`, `leaf(2)`, `floral_bud(3)`
- **노드 특성 벡터 (11D)**:
  - `x, y, z` (3D 세계 좌표, 전방 kinematics로 계산)
  - `length` (internode_length / petiole_length / leaf_scale / peduncle_length)
  - `radius` (internode_radius / petiole_radius)
  - `pitch, yaw, roll` (organ 방향)
  - `organ_type` (4D one-hot or 4D softmax)
- **Topology**: parent shoot ID → parent node 매핑으로 tree adj matrix 구성

```
Helios XML 계층:
plant_instance
  └─ shoot[0] (unifoliate)
       └─ phytomer → internode → petiole(s) → leaf(s)
  └─ shoot[1..3] (trifoliate)
       └─ phytomer → internode → petiole → leaf × 3 + floral_bud
```

#### [NEW] `dataset/helios_dataset.py`
`HeliosPlantDataset(Dataset)`:
- `data_root`: `build/output/` 폴더에서 `*_vis.jpeg` + `*_plant_*.xml` 쌍 자동 탐색
- 이미지: 기존 `transforms` 파이프라인 재사용 (256×256 normalize)
- **Camera pose**: XML이 아닌 `params.json`의 `camera_positioning` 에서 azimuth, elevation 읽기
- **Per-DAP 분리**: DAP를 조건 신호로 추가하거나 메타데이터로 기록
- `__getitem__` → `{image, raw_image, nodes (N,11), adj_matrix, parent_indices, existence_mask, camera_pose, dap, organ_types}`

---

### Component 2: 모델 아키텍처 업데이트

#### [MODIFY] `diffusion_based/models/graph_diffuser_3d.py`

**핵심 변경**: 7D → 11D node feature + organ type conditioning

```python
# 변경 전
class PlantGraphDiffuser3D:
    node_dim = 7  # (x, y, z, theta, phi, length, is_leaf)

# 변경 후  
class PlantGraphDiffuser3D:
    node_dim = 11  # (x, y, z, length, radius, pitch, yaw, roll, organ_type × 4 one-hot → 압축 4D)
```

1. **DAP Conditioning** 추가:
   ```python
   self.dap_encoder = nn.Sequential(
       nn.Linear(1, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
   )
   ```
   - `h_nodes = h_nodes + dap_emb` 로 시간 조건 주입

2. **Organ Type 예측 헤드** 분리:
   ```python
   self.organ_type_head = nn.Linear(embed_dim, 4)  # 4-class CE loss
   ```

3. **Shoot-level Self-Attention** 추가 (선택적):
   - Shoot ID 기반 grouped attention mask로 같은 shoot의 nodes끼리 더 강하게 attend

4. **kNN pruning k 조정**: 8 → 16 (더 복잡한 구조 대응)

---

### Component 3: 학습 스크립트 개선

#### [MODIFY] `diffusion_based/training/train_diffusion_3d.py`

**핵심 문제 수정**: 현재 코드는 **4개 고정 샘플만** 사용, DataLoader 없음

```python
# 변경 전 (문제)
four_samples = [dataset[i] for i in range(min(4, len(dataset)))]
# ... 매 epoch 같은 4개 샘플만 학습

# 변경 후 (DataLoader 배치 학습)
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
for epoch in range(epochs):
    for batch in loader:
        # 정상적인 배치 학습
```

**손실 함수 업데이트**:
- 기존: `loss_coord3d` (MSE on xyz) + `loss_is_leaf` + `loss_scale` + `loss_snap3d`
- 추가:
  - `loss_organ_type = F.cross_entropy(pred_organ_logits, gt_organ_types)` (4-class)
  - `loss_radius = F.mse_loss(pred_x0[:,:,4], gt_nodes[:,:,4])` (organ radius)
  - `loss_orientation = F.mse_loss(pred_x0[:,:,5:8], gt_nodes[:,:,5:8])` (pitch, yaw, roll)
  - **Snap loss 업데이트**: 11D 좌표로 kinematics 재계산

**학습 커리큘럼**:
```
Phase 1 (epoch 1-200):   합성 데이터(Plant3DDataset) + HeliosDataset 혼합 (3:1)
Phase 2 (epoch 201-500): HeliosDataset 위주 (1:3 비율 역전)
Phase 3 (epoch 501+):    HeliosDataset 전용 + augmentation
```

**체크포인트 저장**:
```python
# 기존: 마지막 epoch만 저장
# 변경: Best validation loss 기반 저장 + epoch별 저장
if val_loss < best_val_loss:
    torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
               "diffusion_based/checkpoints/best_3d_model.pt")
```

---

### Component 4: 평가 및 시각화 개선

#### [MODIFY] `diffusion_based/eval/visualize_diffusion_3d.py`

1. **실제 이미지 입력 지원**:
   ```python
   def run_inference_on_real_image(jpeg_path: str, xml_path: str = None):
       """실제 Helios JPEG + (선택적) XML GT로 추론 및 시각화"""
   ```

2. **Organ 타입별 시각화** 색상 코딩:
   - `internode`: 갈색/회색
   - `petiole`: 연두색
   - `leaf`: 초록색 (heart shape polygon 유지)
   - `floral_bud`: 노랑/보라

3. **Quantitative 평가 메트릭**:
   ```python
   def evaluate_reconstruction(pred_nodes, gt_nodes, gt_parents):
       # Chamfer Distance (3D node positions)
       # Tree Edit Distance (topology)
       # Organ Type Accuracy (F1 per class)
       # Snap Loss (joint connectivity)
   ```

4. **DAP별 재건 품질 플롯**: DAP를 x축으로 Chamfer Distance 변화 추적

---

### Component 5: 데이터 생성 자동화

#### [NEW] `Digital-Crops/projects/syntheticdata_generation/scripts/generate_dataset.sh`

```bash
#!/bin/bash
# DAP 5~60일, 각 10개 seed → 총 550개 쌍 자동 생성
for dap in 5 10 15 20 25 30 35 40 45 50 55 60; do
    for seed in $(seq 0 9); do
        ./main --radiation false --vis -n "cowpea" --dap $dap --seed $seed
    done
done
```

---

## 파일 변경 요약

```
dataset/
  [NEW] helios_xml_parser.py        # Helios XML → organ graph 파싱
  [NEW] helios_dataset.py           # HeliosPlantDataset (image+xml 쌍)
  [MODIFY] plant3d_dataset.py       # 중복 _generate_synthetic_3d_plants 버그 수정

diffusion_based/
  models/
    [MODIFY] graph_diffuser_3d.py   # 7D→11D, DAP cond, organ_type head
  training/
    [MODIFY] train_diffusion_3d.py  # DataLoader, 혼합 학습, 손실 업데이트
  eval/
    [MODIFY] visualize_diffusion_3d.py  # 실제 이미지 추론, 메트릭 추가

Digital-Crops/projects/syntheticdata_generation/scripts/
  [NEW] generate_dataset.sh         # 배치 데이터 생성 스크립트
```

---

## 검증 계획

### Automated Tests
```bash
# 1. XML 파서 유닛 테스트
python -c "from dataset.helios_xml_parser import parse_helios_xml; \
           result = parse_helios_xml('Digital-Crops/.../cowpea_0000_plant_0000.xml'); \
           print(f'Parsed {result[\"num_nodes\"]} nodes')"

# 2. 데이터셋 로드 테스트
python -c "from dataset.helios_dataset import HeliosPlantDataset; \
           ds = HeliosPlantDataset('Digital-Crops/.../build/output'); \
           print(ds[0]['nodes'].shape)"

# 3. 모델 forward pass (11D)
python -c "from diffusion_based.models.graph_diffuser_3d import PlantGraphDiffuser3D; \
           import torch; m = PlantGraphDiffuser3D(node_dim=11); \
           out = m(torch.randn(2,64,11), torch.ones(2,64,1), torch.tensor([500,500]), \
                   torch.randn(2,3,256,256)); print(out['pred_x0'].shape)"

# 4. 학습 스크립트 실행 (10 epoch 빠른 테스트)
python -m diffusion_based.training.train_diffusion_3d --epochs 10 --batch_size 4
```

### Manual Verification
- `cowpea_0000_vis.jpeg` 입력으로 추론 후 재건된 3D plant graph가 XML 구조와 유사한지 육안 확인
- DAP 10일 식물이 작은 초기 단계 구조 (2개 shoot, 각 2 phytomer)를 올바르게 재건하는지 확인
- Organ type 분류 정확도 ≥ 80% (internode/petiole/leaf 3-class)
