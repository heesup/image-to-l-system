# 3D Plant Geometry + Render Pipeline 진행 상황

> 마지막 작업 날짜: 2026-08-04 (macOS)
> 커밋: `b9fd2cd` — single Helios-style geometry+render pipeline with focus-plant HFOV and shading

## 1. 완료된 작업

### 1.1 단일 geometry pipeline 확립
- `diffusion_based/models/helios_geometry.py` 추가
  - Helios XML → 3D tube/leaflet/ellipsoid geometry 재구성
  - internode/petiole tube가 잘못 분류되던 버그 수정
  - leaf mesh 샘플링 추가 (삼각형 중심/중점) → point cloud에 leaf organ 포함
  - differentiable point-cloud sampler: 15D 노드 → Chamfer용 point cloud
  - `nodes_to_geometry()`: 15D 노드 → `HeliosTube`/`HeliosLeaflet`/`HeliosEllipsoid` (render 용)
- 기존 `plant_geometry_3d.py`, `differentiable_renderer_3d.py` 삭제

### 1.2 단일 2D rasterizer 확립
- `diffusion_based/models/helios_rasterizer_3d.py` 추가
  - Helios `Context` 카메라 모델과 동일한 투영
  - `--focus-plant` 대응: `recompute_focus_plant_hfov()` — XY bbox + 5% margin, `2*atan(span/(2*h))`
  - soft triangle rasterization에 **area normalization** 추가 → 부분 픽셀 삼각형도 예쁘게, 전체 화면 칠해지지 않음
  - tube/leaf에 simple diffuse shading 추가 (양면 leaf)

### 1.3 3D Chamfer loss
- `diffusion_based/models/pointcloud_loss_3d.py` 추가
  - `PlantPointCloudChamferLoss`: 15D 노드 → point cloud → Chamfer
  - organ-aware 가중 Chamfer
  - PLY load/write/Chamfer/normalize 유틸리티 포함

### 1.4 Helios dataset 생성기 개선
- `dataset/generate_helios_dataset.py`
  - macOS에서 offscreen env override 제거 (XQuartz 의존)
  - `--export-3d ply` 지원

### 1.5 훈련 스크립트 연동
- `diffusion_based/training/train_diffusion_3d.py`
  - legacy renderer 제거 → `HeliosGeometryRasterizer` 사용
  - 15D 예측 노드를 `nodes_to_geometry()`로 변환 후 batch 렌더
  - `PlantPointCloudChamferLoss` 연동

## 2. 검증 결과

### 2.1 5개 Helios 샘플 Chamfer (DAP5, seed 0~4, single view)
| seed | Chamfer (target-normalized) |
|------|-----------------------------|
| 00   | 0.0389                      |
| 01   | 0.0439                      |
| 02   | 0.0384                      |
| 03   | 0.0399                      |
| 04   | 0.0392                      |
| 평균 | **0.0401**                  |

### 2.2 Render 비교
- `/tmp/helios_val5/cowpea_dap005_seed*_compare.png`에 좌측 Helios `_vis.jpeg`, 우측 Python rasterizer 결과 저장
- 구조적으로 잎/줄기 배열이 Helios 이미지와 일치

## 3. 다음 컴퓨터에서 이어서 할 일

### 3.1 GPU/메모리 최적화 (우선순위 높음)
- 현재 MPS(Mac)에서 `torch.cdist(pred, target)` 메모리 버퍼 18 GB 요청 → 실패
  - 원인: point cloud가 15D 노드당 124 point × 256 node ≈ 31K, batch 2면 cdist가 큰 버퍼 요구
  - 해결 방안:
    1. `pc_loss`용 sampler 해상도 낮추기 (`n_cylinder_circ=3, n_cylinder_axis=2, n_leaf_u=3, n_leaf_v=4` 등)
    2. target subsample을 더 작게 (64~128)
    3. Chamfer를 mini-batch로 나누기 (예: target chunk 단위로 `torch.cdist` 호출)
    4. `F.normalize` 후 squared distance로 메모리 절약

### 3.2 2D render loss 훈련 검증
- `--render-loss` > 0로 1 epoch 훈련 실행
- 예측된 15D 노드가 `nodes_to_geometry()`를 거쳐 Helios `_vis.jpeg`와 비슷한 렌더 생성 확인
- 현재 `nodes_to_geometry()`에서 leaf를 단순 quad로 만들어 Helios의 trifoliate 모양과 다름 → shape는 차후 개선

### 3.3 Leaf geometry 개선 (장기)
- Helios C++ `CowpeaLeafPrototype_trifoliate_OBJ`와 동일한 OBJ prototype + 변환 체인 사용
- XML에 leaf가 1개만 저장되는 문제 해결 필요 (phytomer parameters에서 `leaves_per_petiole=3` 복원)

### 3.4 Dataset 확장
- 5개 샘플은 검증용; 실제 훈련을 위해 DAP 5~30, 다양한 시야각/태양 각도로 생성
- `dataset/generate_helios_dataset.py`로 `--workers 1 --renderer vis --export-3d ply` 실행

### 3.5 학습 실험
- render loss만, point-cloud loss만, 둘 다로 실험
- 수렴/시각화 체크
- `train_diffusion_3d.py`의 `render_loss_weight`, `pc_loss_weight` 하이퍼파라미터 튜닝

## 4. 핵심 파일/스크립트

```bash
# 5개 검증 샘플 다시 만들기
source /Users/lion397/homebrew/Caskroom/miniforge/base/bin/activate l-system
python dataset/generate_helios_dataset.py \
  --dap-start 5 --dap-end 5 --dap-step 5 --seeds 5 \
  --renderer vis --workers 1 --export-3d ply \
  --output-dir /tmp/helios_val5 \
  --main-binary Digital-Crops/projects/syntheticdata_generation/build/main

# Chamfer + render 비교
python diffusion_based/eval/compare_xml_helios_3d.py \
  --helios-ply /tmp/helios_val5/..._helios.ply \
  --xml /tmp/helios_val5/..._plant_0000.xml \
  --visualize --visualize-path /tmp/compare.png

# 훈련 (render loss)
python diffusion_based/training/train_diffusion_3d.py \
  --data-dir /tmp/helios_val5 \
  --epochs 2 --batch-size 2 --render-loss 1.0

# 훈련 (point-cloud loss, 메모리 최적화 후)
python diffusion_based/training/train_diffusion_3d.py \
  --data-dir /tmp/helios_val5 \
  --epochs 2 --batch-size 2 --pc-loss 1.0 \
  --target-ply /tmp/helios_val5/cowpea_dap005_seed00_..._helios.ply \
  --pc-samples 64
```

## 5. 알려진 이슈

- torchvision image extension/libjpeg dylib 경고 (macOS): 이미지 로딩은 PIL fallback으로 동작
- MPS에서 point-cloud Chamfer 메모리 부족: CPU 또는 GPU 메모리 큰 머신에서 `pc_loss` 우선 테스트
- `nodes_to_geometry()` leaf shape가 Helios trifoliate와 다름 (정량적 영향은 작으나 시각적 차이 있음)
