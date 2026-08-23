# Image-to-L-System: Project Documentation

**프로젝트**: 단일 RGB 이미지 → 3D 식물 장기 파라미터 배열 예측 Flow Matching 모델  
**클러스터**: UC Davis Farm HPC | **활성 노드**: `gpu-10-54` (RTX 6000 Ada)

---

## 폴더 구조

| 폴더 | 용도 |
|------|------|
| `ongoing/` | **현재 진행 중인 작업** - 다음 Agent가 반드시 읽어야 할 문서 |
| `done/` | 완료된 구현/설계/핸드오프 문서 |
| `todo/` | 아직 시작 안 된 계획/아이디어 |
| `archived/` | 구버전 또는 폐기된 문서 |
| `results/` | 학습 결과 벤치마크 보고서 및 figure 이미지 |
| `assets/` | 문서에서 참조하는 이미지 파일 |

---

## 🔴 Ongoing (Active & Handover)

### → [`ongoing/20260823_differentiable_renderer_helios_alignment_and_roundtrip_report.md`](ongoing/20260823_differentiable_renderer_helios_alignment_and_roundtrip_report.md)
**Differentiable Renderer Alignment, COCO Internode Masking & Lossless XML Round-Trip Report (2026-08-23)**  
Latest active document. Details the dynamic gravitropic curvature alignment, child shoot reference frame fixing, COCO internode `"shoot"` mask export in C++ `main.cpp`, PyTorch mask threshold fix (`type_buf >= 0`), 100% lossless XML round-trip benchmark ($0.00\text{ mm}$ vertex error), and full renderer hardcoding audit.

### → [`ongoing/20260822_dataset_pipeline_multigpu_sharding_update.md`](ongoing/20260822_dataset_pipeline_multigpu_sharding_update.md)
**Helios Dataset Pipeline Consolidation, Multi-Arch CUDA 209 Fix & Self-Healing SLURM Sharding (2026-08-22)**  
Details the CUDA 209 resolution, 100-seed Helios C++ XML generation, SLURM faulty GPU detection & tarpit locking, decoupled 100K GPU tensor sharding engine, and next execution commands.

---

## ✅ Done (완료된 작업)

| 파일 | 내용 | 완료일 |
|------|------|--------|
| `done/20260811_helios_renderer_handover.md` | Helios PyTorch 차분 렌더러 핸드오버 (Track A/B) | 2026-08-11 |
| `done/20260812_plant_organ_differentiable_renderer_handover.md` | PlantOrganArray (N, 93) 렌더러 핸드오버 | 2026-08-12 |
| `done/20260813_refactor.md` | 전체 파이프라인 리팩터링 기록 | 2026-08-13 |
| `done/20260814_progress.md` | 3D 파이프라인 진행 기록 | 2026-08-14 |
| `done/20260814_progress_3d_pipeline.md` | 3D Chamfer + Render loss 파이프라인 완성 | 2026-08-14 |
| `done/20260814_plan.md` | 프로젝트 초기 계획 | 2026-08-14 |
| `done/20260815_40d_plant_array_representation.md` | 40D 식물 파라미터 벡터 설계 문서 | 2026-08-15 |
| `done/20260815_render_alignment_debug.md` | 렌더 정렬 디버그 기록 | 2026-08-15 |
| `done/20260818_flow_matching_scaffold_prior_analysis.md` | Flow Matching scaffold prior 분석 | 2026-08-18 |
| `done/20260819_lab_meeting_report_backprop_vs_diffusion.md` | Lab Meeting: Backprop vs Diffusion 리포트 | 2026-08-19 |
| `done/diff_renderer_pixel_match_handoff.md` | 픽셀 매칭 핸드오프 | 2026-08-15 |
| `done/lab_meeting_report.md` | Lab meeting 리포트 (직접 역전파 vs 확산 모델) | 2026-08-19 |
| `done/pixel_match_implementation_plan.md` | 픽셀 매칭 구현 계획 | 2026-08-15 |
| `done/projection_angle_explanation.md` | 카메라 투영 각도 설명 | 2026-08-13 |

---

## 📋 Todo (미착수)

| 파일 | 내용 |
|------|------|
| `todo/roadmap.md` | 장기 로드맵 |
| `todo/15_loss_reduction_strategies.md` | 15가지 손실 감소 전략 목록 |
| `todo/2026-08-14-pytorch-renderer-optimization.md` | PyTorch 렌더러 최적화 계획 |

---

## 📝 Misc (Notes & Analysis)

| File | Content |
|------|---------|
| [`misc/20260822_cluster_power_cost_estimate.md`](misc/20260822_cluster_power_cost_estimate.md) | Cluster power draw (~8.1 kW) and cloud equivalent cost (~$41.87/hr) during 40-job cowpea sharding run |

---

## 📊 Results (학습 결과)

| 파일 | 내용 |
|------|------|
| `results/15_strategies_benchmark_report.md` | 15가지 전략 벤치마크 결과 |
| `results/assets/fig_pure_noise_flow_matching.png` | 73M DiT: 순수 노이즈→생성 평가 (DAP 10/50/90) |
| `results/assets/fig_canonical_cowpea_dap10_30.png` | 73M DiT: DAP 10-30 생성 결과 |
| `results/assets/fig_*` | 각종 실험 결과 figure |

---

## 핵심 모델 체크포인트

```
diffusion_based/checkpoints/
├── fm/
│   ├── canonical_cowpea_dit_best.pt     # 73M DiT, 60 에폭 (321MB)
│   ├── part_flow_matching.pt             # 73M DiT full (1.1GB)
│   ├── test_large.pt                     # 229M DiT-Large, 1에폭 테스트 (2.5GB)
│   └── plant_organ_40d_flow_matching_best.pt  # 40D VAE 기반 (128MB)
└── canonical_cowpea_dit_best.pt          # 별도 저장된 73M best
```
