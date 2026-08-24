# Image-to-L-System: Agent Handoff Document
> **작성일**: 2026-08-21  
> **작성자**: Antigravity AI (Conversation ID: `fba72a3a-85c8-41aa-97d9-cce13575a3d8`)  
> **대상**: 이 문서를 읽고 작업을 이어받는 다음 AI Agent

---

## 1. 프로젝트 개요

단일 RGB 이미지 → 3D 식물 장기(organ) 파라미터 배열 예측 Flow Matching 모델을 학습합니다.  
최종 목표는 Cowpea(동부콩) 전 생애주기(DAP 1~100)를 다루는 **229M DiT-Large** 모델을 훈련하여, 이미지 한 장으로 3D 식물 구조를 복원하는 것입니다.

**기술 스택**
- 클러스터: UC Davis Farm HPC (`farm.hpc.ucdavis.edu`)
- 현재 활성 노드: `gpu-10-54` (NVIDIA RTX 6000 Ada, 48GB VRAM, 64 CPUs)
- Python 환경: `/home/lion397/.conda/envs/digital-crops/bin/python`
- SLURM 계정: `lion397`, 그룹: `geminigrp` / `publicgrp`

---

## 2. 현재 진행 중인 SLURM 작업 (2026-08-21 19:24 기준)

### ✅ 실행 중: Cowpea 100K GPU 렌더 데이터셋 생성 (Job Array `37829199`)

```bash
# 상태 확인
squeue -u lion397

# 진행 로그 확인 (worker 0)
cat logs/slurm_gen_37829199_0.out

# 생성된 샤드 수 확인
ls dataset/cache_cowpea_100k/ | wc -l
```

| 항목 | 내용 |
|------|------|
| Job Array | `37829199` (array `0-39`) |
| 파티션 | `low` + `gres=gpu:1` |
| 노드 | `gpu-4-54`, `gpu-5-58`, `gpu-10-50`, `gpu-3-38`, `gpu-5-46`, `gpu-12-92` 등 GPU 노드 |
| 목표 | 100,000개 샘플 → `dataset/cache_cowpea_100k/*.pt` |
| 현재 | 40개 워커 중 25개 실행 중, 14개 샤드 생성됨 (각 100개 샘플) |
| 예상 완료 | ~1시간 이내 |

> ⚠️ 작업 완료 여부를 먼저 확인하고 다음 단계 진행할 것.

---

## 3. 핵심 파일 맵

### 3.1 모델 아키텍처

| 파일 | 역할 | 파라미터 |
|------|------|---------|
| [`diffusion_based/models/canonical_cowpea_dit.py`](../diffusion_based/models/canonical_cowpea_dit.py) | 기존 73M DiT (60 에폭 학습 완료) | 73M |
| [`diffusion_based/models/canonical_cowpea_dit_large.py`](../diffusion_based/models/canonical_cowpea_dit_large.py) | **신규 229M DiT-Large** (ViT-16L + Decoder-12L, embed=768) | 229.68M |

### 3.2 데이터셋

| 파일/경로 | 역할 | 상태 |
|-----------|------|------|
| [`diffusion_based/dataset/canonical_cowpea_dataset.py`](../diffusion_based/dataset/canonical_cowpea_dataset.py) | 기존 15K 개별 `.pt` 캐시 데이터셋 | ✅ 사용 가능 (5,000개 로드) |
| [`diffusion_based/dataset/generate_cowpea_100k.py`](../diffusion_based/dataset/generate_cowpea_100k.py) | 100K GPU 렌더 생성기 (SLURM 워커 지원) | ✅ 실행 중 |
| [`diffusion_based/dataset/cowpea_shard_dataset.py`](../diffusion_based/dataset/cowpea_shard_dataset.py) | 100K 샤드 스트리밍 데이터셋 | ✅ 완성 |
| `dataset/cache/` | 기존 15K 개별 `.pt` 파일 (15,000개) | ✅ 사용 가능 |
| `dataset/cache_cowpea_100k/` | 100K 샤드 (생성 중) | ⏳ 진행 중 |

### 3.3 학습 스크립트

| 파일 | 역할 |
|------|------|
| [`diffusion_based/training/train_canonical_cowpea_flow_matching.py`](../diffusion_based/training/train_canonical_cowpea_flow_matching.py) | 기존 73M 모델 학습 (60 에폭 완료) |
| [`diffusion_based/training/train_cowpea_dit_100k.py`](../diffusion_based/training/train_cowpea_dit_100k.py) | **신규 229M 모델 학습** (100K 샤드 사용) |

### 3.4 평가 스크립트

| 파일 | 역할 |
|------|------|
| [`diffusion_based/eval/eval_pure_noise_flow_matching.py`](../diffusion_based/eval/eval_pure_noise_flow_matching.py) | 기존 73M 모델 순수 노이즈→생성 평가 |
| [`diffusion_based/eval/eval_canonical_cowpea_flow_matching.py`](../diffusion_based/eval/eval_canonical_cowpea_flow_matching.py) | 기존 73M 모델 DAP별 생성 품질 평가 |
| [`diffusion_based/eval/eval_cowpea_dit_100k.py`](../diffusion_based/eval/eval_cowpea_dit_100k.py) | **신규 229M 모델** 전 생애주기 6열 벤치마크 |

### 3.5 SLURM 스크립트 (`slurm_scripts/`)

| 파일 | 역할 | 사용법 |
|------|------|--------|
| [`slurm_scripts/generate_cowpea_100k_jobs.sh`](../slurm_scripts/generate_cowpea_100k_jobs.sh) | 100K GPU 데이터 생성 | `--submit` |
| [`slurm_scripts/generate_helios_dataset_jobs.sh`](../slurm_scripts/generate_helios_dataset_jobs.sh) | Helios C++ XML 원본 데이터 생성 | `--plant-types cowpea --submit` |

### 3.6 핵심 체크포인트

| 파일 | 모델 | 에폭 | Loss |
|------|------|------|------|
| `diffusion_based/checkpoints/fm/canonical_cowpea_dit_best.pt` | 73M DiT (기존) | 60 | ~1.2 (추정) |
| `diffusion_based/checkpoints/fm/test_large.pt` | **229M DiT-Large** | 1 (테스트) | 1131.9 |

> ⚠️ `test_large.pt`는 1 에폭 테스트 결과로 성능 미달. 100K 데이터셋 완성 후 60+ 에폭 재학습 필요.

---

## 4. 26D 노드 벡터 인코딩 (핵심 데이터 포맷)

각 식물 장기(organ)는 **512 슬롯 × 26차원** 텐서로 표현됩니다.

```
Dim  0..11: One-hot 장기 분류 (12 클래스)
             0=Root Meta, 1=Shoot Meta, 2=Internode, 3=Petiole,
             4=Leaf, 5=Bud, 6=Peduncle, 7=Flower, 8=Fruit/Pod,
             9=Flower Closed, 10=Bud Aborted, 11=Empty
Dim 12..14: 기저 위치 (x, y, z) / 20.0 정규화
Dim 15..20: 6D 연속 회전 행렬 (R_6D)
Dim 21..23: 스케일 (sx, sy, sz) / 50.0 정규화
Dim     24: 곡률 / 100.0 정규화
Dim     25: 엽서 각도(Phyllotactic angle) / 180.0 정규화
```

---

## 5. 현재 기존 모델(73M) 성능 지표

| DAP | Vertices (GT) | Chamfer Distance | Mask IoU |
|-----|--------------|-----------------|---------|
| 010 | 17,481 | **0.0529** | ~0.60 |
| 050 | 18,795 | **0.1902** | ~0.45 |
| 090 | 54,097 | **0.1711** | **0.4383** |

> 모델이 아직 완벽한 식물 구조를 복원하지 못하는 주된 이유:
> 1. 학습 데이터 부족 (15K → 100K로 확장 중)
> 2. 모델 용량 부족 (73M → 229M으로 확장)
> 3. 슬롯 정규 순서(Canonical Slot Ordering) 미적용

---

## 6. 다음 Agent가 즉시 해야 할 일

### Step 1: 100K 데이터셋 생성 완료 확인
```bash
cd /home/lion397/codes/image-to-l-system

# 작업 상태 확인
squeue -u lion397

# 생성된 샤드 수 (목표: 1000개 = 100,000개 샘플)
ls dataset/cache_cowpea_100k/ | wc -l

# 샘플 내용 검증
/home/lion397/.conda/envs/digital-crops/bin/python -c "
import torch, glob
files = sorted(glob.glob('dataset/cache_cowpea_100k/*.pt'))
print(f'Shards: {len(files)}, Total samples: {len(files)*100}')
if files:
    d = torch.load(files[0], weights_only=False)
    print('Keys:', d[0].keys())
    print('Image:', d[0]['image'].shape, 'Nodes:', d[0]['nodes'].shape)
"
```

### Step 2: 229M DiT-Large 본 학습 시작
```bash
# GPU 10-54 노드에서 직접 실행 (RTX 6000 Ada 48GB)
/home/lion397/.conda/envs/digital-crops/bin/python \
    diffusion_based/training/train_cowpea_dit_100k.py \
    --epochs 60 \
    --batch-size 32 \
    --lr 2e-4 \
    --warmup-epochs 3 \
    --cache-dir dataset/cache_cowpea_100k \
    --save-name cowpea_dit_large_150m.pt \
    --num-workers 8 \
    > logs/train_dit_large_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

> 또는 SLURM gpu-6000_ada-h 전용 파티션으로 sbatch 제출:
> ```bash
> # slurm_scripts/ 에 train_cowpea_dit_large_jobs.sh 작성 후 제출
> # #SBATCH --account=geminigrp
> # #SBATCH --partition=gpu-6000_ada-h
> # #SBATCH --gres=gpu:1
> ```

### Step 3: 학습 완료 후 전 생애주기 평가
```bash
/home/lion397/.conda/envs/digital-crops/bin/python \
    diffusion_based/eval/eval_cowpea_dit_100k.py
# 결과: docs/results/assets/fig_cowpea_100k_lifespan_benchmark.png
```

---

## 7. 알려진 이슈 및 주의사항

| 이슈 | 상세 | 해결책 |
|------|------|--------|
| **Helios C++ 바이너리 GPU 필수** | 원본 Helios 엔진(`Digital-Crops/.../main`)은 NVIDIA OptiX/CUDA 없이 실행 불가 | GPU 노드(`gres=gpu:1`)에서만 실행 |
| **PyTorch 렌더러 CPU 폴백** | `HeliosPyTorchRenderer`는 GPU 없으면 소프트웨어 렌더링 (매우 느림) | 반드시 `--device cuda` 지정 |
| **`canonical_cowpea_dataset.py` 슬롯 인덱싱** | 기존 데이터셋은 `max_slots=512` 기준으로 인코딩 | `max_slots=512` 유지 |
| **체크포인트 정리 필요** | `fm/part_flow_matching_epoch*.pt` (1.1GB × 60개 = 66GB) | 최신 + best만 유지하고 나머지 삭제 권장 |
| **슬롯 정규 순서(미구현)** | 슬롯을 organ_type으로 정렬하면 학습 안정성 향상 예상 | `canonical_cowpea_dataset.py`에서 Sort by organ_type 추가 |

---

## 8. 환경 정보

```bash
# Python 환경
/home/lion397/.conda/envs/digital-crops/bin/python  # PyTorch 2.9, CUDA 12.x

# SLURM 계정/파티션
sacctmgr show user lion397 withassoc  # 상세 계정 확인

# 사용 가능한 GPU 파티션
#   low           - publicgrp, gres=gpu:1 사용 가능
#   gpu-6000_ada-h - geminigrp, RTX 6000 Ada 전용
#   gpu-a100-h    - geminigrp, A100 80GB 전용

# 현재 세션 노드
# gpu-10-54 (RTX 6000 Ada 48GB, 64 CPUs)
# Job ID: 37820407 (OnDemand Desktop)
```

---

## 9. 문서 이력

이 handoff 문서는 다음 대화 세션의 작업을 기반으로 작성되었습니다:
- Conversation ID: `fba72a3a-85c8-41aa-97d9-cce13575a3d8`
- 주요 작업: 73M DiT 평가 → 100K 데이터셋 계획 → 슬롯 버그 수정 → Leaf scale 버그 수정 → SLURM 40노드 GPU 분산 생성 → 229M DiT-Large 설계 및 테스트
