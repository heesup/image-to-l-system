# Image-to-L-System: Project Documentation

**프로젝트**: 단일 RGB 이미지 → 3D 식물 장기 파라미터 배열 예측 Flow Matching 모델
**활성 표현**: 17D Part Tensor (GPU-native, 캐노니컬) + 27D FM 노드 벡터 (one-hot + BudState; DiT 학습)
**활성 모델**: 232M DiT-Large Flow Matching (4×H100 NVL DDP + SLURM `low` partition)
**클러스터**: UC Davis Farm HPC | **학습 노드**: `gpu-10-58` (4× H100 NVL, DDP)
**2026-08-31 상태**: 17D→XML→Helios 라운드트립 완전 해결 (organ IoU 1.0) | FM **27D** 전환 + 1000 샤드 재생성·검증 완료
**학습**: `vlm_mmdit_ddp` (job **38048780**) 4×H100, global batch 128, 27D fresh shards로 **LIVE** (v-loss 13.3→6.2 하강 중)

---

## Repository & Docs Structure

### 📂 Repository Root Layout

| 디렉토리 | 설명 | 상태 |
|----------|------|------|
| `diffusion_based/` | **핵심 활성 파이프라인**: 40D XML 브릿지, 17D Part Tensor GPU 렌더러, 27D DiT 학습/평가 | ✅ Active |
| `archive/` | **통합 아카이브**: 과거 Track-A(15D), 14D, 40D VAE, 94D 및 디버그 스크립트 모음 ([`archive/README.md`](../archive/README.md)) | 📦 Archived |
| `dataset/helios_data/` | 10K XML 원본 및 1000×27D GPU 샤드 (`cowpea_shard/`; 구 26D stale 세트는 `cowpea_shard_stale_26d_20260824/` 보관) | ✅ Active |
| `slurm_scripts/` | 2×H100 DDP 분산 학습 및 데이터 생성 SLURM 배치 스크립트 | ✅ Active |
| `Digital-Crops/` | Helios C++ 시뮬레이션 엔진 서브모듈 | ✅ Active |
| `docs/` | 프로젝트 전반 설계 및 진행 문서 | 📄 Documentation |

### 📚 Docs 폴더 구조

| 폴더 | 용도 |
|------|------|
| `ongoing/` | **현재 진행 중인 작업** — 다음 Agent가 반드시 읽어야 할 문서 |
| `done/` | 완료된 구현/설계/핸드오프 문서 |
| `todo/` | 아직 시작 안 된 계획/아이디어 |
| `archived/` | 구버전 또는 폐기된 문서 |
| `results/` | 학습 결과 벤치마크 보고서 및 figure 이미지 |
| `misc/` | 기타 참고 자료 (비용 분석 등) |
| `assets/` | 문서에서 참조하는 이미지 파일 |

---

### → [`ongoing/20260831_27d_fm_layout_and_shard_regeneration.md`](ongoing/20260831_27d_fm_layout_and_shard_regeneration.md)
**27D FM Layout, Shard Regeneration & 4×H100 Training Launch (2026-08-31) — READ THIS FIRST**
27D Flow-Matching node layout (BudState channel added to the prior 26D), softplus→linear scale fix, organ-aware normalization + angle wrapping (corpus flower yaw 810° → FM explosion fix), single-source-of-truth `encode_fm`/`decode_fm`, 27D FM mesh decode (meshes bit-identical to 17D GT), 1000-shard regeneration via SLURM pipeline, launcher de-hardcoding (N-rank from SLURM allocation), and the live training run. **Next agent: read this first.**
**Live run**: job `38048780` — 4×H100 NVL, batch 32/GPU (72.8/95.8 GB), global 128, log `slurm_scripts/logs/train_vlm_scaffold_38048780.log` (`srun --overlap --jobid=38048780 nvidia-smi` for live GPU state).

### → [`ongoing/20260826_canonical_pipeline_refactor_progress.md`](ongoing/20260826_canonical_pipeline_refactor_progress.md)
**Canonical Pipeline Refactor Progress Report (2026-08-26)**
Unified all rendering paths onto the single canonical path `XML→40D→16D→Part Render`. Rewrote `extract_part_tensor()` to run native 40D forward kinematics (bit-identical to prior 94D path), converted `build_mesh_from_organ_array()` to a deprecated wrapper (removing ~600 lines of 94D FK), migrated all active training/eval/dataset/script callers to `build_mesh_from_part_tensor(arr.to_part_tensor())`, and added `DeprecationWarning` to the legacy converters.

### → [`ongoing/20260826_helios_flower_peduncle_pod_alignment_and_cleanup_report.md`](ongoing/20260826_helios_flower_peduncle_pod_alignment_and_cleanup_report.md)
**16D/26D Differentiable Renderer Alignment, Flower/Peduncle/Pod Integration & Handover Report (2026-08-26)**
Comprehensive handover on repository reorganization, removal of 94D intermediate layout, 0-indexed parent node bug fix ($0.06\ \mu\text{m}$ leaflet error), flower scaling & mesh prototype calibration, peduncle dormant bud filtering, and pod (fruit type 8) batch mesh integration guide.

### → [`ongoing/20260825_16d_part_assembly_renderer_overhaul_report.md`](ongoing/20260825_16d_part_assembly_renderer_overhaul_report.md)
**16D Part Assembly GPU Renderer Overhaul Report (2026-08-25)**
Canonical pipeline architecture (XML→40D→16D→Mesh→Render), direct 40D `extract_part_tensor()` forward-kinematic bridge (no 94D intermediary), fully vectorized `torch.bmm` GPU mesh builder (1,163x vs Helios C++), removal of legacy 94D parser/writer, root-cause analysis of broken renders, Helios C++ submodule XML reload fixes.

### → [`results/20260825_direct_optimization_cowpea_dap10_report.md`](results/20260825_direct_optimization_cowpea_dap10_report.md)
**16D Direct Optimization: Cowpea DAP 10 Benchmark (2026-08-25)**
Empirical validation of continuous soft-existence direct optimization with multi-modal supervision (RGB, Depth, Mask, Semantic Segmentation) across 15 strategies.

### → [`ongoing/20260824_helios_xml_roundtrip_fix_and_ground_clipping_report.md`](ongoing/20260824_helios_xml_roundtrip_fix_and_ground_clipping_report.md)
**Helios XML Round-Trip Invariance, Subsurface Geometry & Ground Clipping Fix (2026-08-24)**
Technical report on resolving ground collision pruning discrepancy, inflorescence/peduncle XML serialization idempotency, and `--no-ground` / `--ground-clipping` CLI flags for 100% exact botanical reconstruction.

### → [`ongoing/20260823_vlm_scaffold_dit_architecture_and_training_report.md`](ongoing/20260823_vlm_scaffold_dit_architecture_and_training_report.md)
**VLM-Scaffold-DiT: 2-Stage Neural Coarse-to-Fine Vision Diffusion Report (2026-08-23)**
Comprehensive technical report on the 2-Stage Neural Coarse-to-Fine framework: Stage 1 DETR-style Set Predictor with learnable slot queries, Stage 2 Bridge Flow Matching DiT, variable-slot batch processing via key-padding masks, hierarchical differentiable rendering losses, and active 2× H100 DDP training execution (`Job ID: 37866768`).

### → [`ongoing/20260823_vlm_scaffold_dit_architecture_design.md`](ongoing/20260823_vlm_scaffold_dit_architecture_design.md)
**VLM-Scaffold-DiT: Unified Vision-Language Scaffold Flow Matching Architecture (2026-08-23)**
Complete architecture design integrating Option 1 (Parametric Botanical Scaffold Prior) and Option 3 (Shared Pretrained DINOv3 via HuggingFace `facebook/dinov3` / SigLIP Vision Tower) for single forward-pass multi-scale 3D plant reconstruction.

### → [`ongoing/20260823_canonical_sorting_and_xml_reassembly_architectural_report.md`](ongoing/20260823_canonical_sorting_and_xml_reassembly_architectural_report.md)
**Canonical Botanical Sorting, Procedural XML Reassembly & Training Normalization Report (2026-08-23)**
Details why flat type sorting fails (-93% vertices) vs Phytomer-Preserving Tree-DFS sorting (0.000000 mm vertex error), mathematical justification for Procedural Reassembly over Deep Learning GNN topology, Loss scale normalization, and active 2× H100 DDP training execution.

### → [`ongoing/20260823_project_status_26d_dit_training.md`](ongoing/20260823_project_status_26d_dit_training.md)
**26D DiT-Large Project Status Report (2026-08-23)**
Comprehensive project status: architecture evolution timeline, current 232M DiT-Large training status, 26D encoding spec, complete active file map, checkpoints, baseline metrics, docs organization, recommended next actions.

### → [`ongoing/20260823_differentiable_renderer_helios_alignment_and_roundtrip_report.md`](ongoing/20260823_differentiable_renderer_helios_alignment_and_roundtrip_report.md)
**Differentiable Renderer Alignment, COCO Internode Masking & Lossless XML Round-Trip Report (2026-08-23)**
Gravitropic curvature alignment, child shoot reference frame fixing, COCO internode `"shoot"` mask export, PyTorch mask threshold fix, 100% lossless XML round-trip (0.000000 mm vertex error), renderer hardcoding audit.

---

## ✅ Done (완료된 작업)

| 파일 | 내용 | 완료일 |
|------|------|--------|
| `done/20260830_pr_xml_roundtrip_fix.md` | Helios C++ 무손실 XML 라운드트립 PR (peduncle/inflorescence/ground-collision pruning; docs/pr_xml_roundtrip_fix.md에서 이동) | 2026-08-30 |
| `done/20260822_dataset_pipeline_multigpu_sharding_update.md` | CUDA 209 해결, 100-seed 확장, Self-Healing SLURM, 100K GPU 샤딩 | 2026-08-22 |
| `done/20260821_cowpea_100k_dit_large_handoff.md` | 73M→229M DiT-Large 설계, 26D 인코딩, 100K 데이터셋 파이프라인 | 2026-08-21 |
| `done/20260819_lab_meeting_report_backprop_vs_diffusion.md` | Lab Meeting: Backprop vs Diffusion 리포트 | 2026-08-19 |
| `done/20260818_flow_matching_scaffold_prior_analysis.md` | Flow Matching scaffold prior 분석 | 2026-08-18 |
| `done/20260815_40d_plant_array_representation.md` | 40D 식물 파라미터 벡터 설계 문서 | 2026-08-15 |
| `done/20260815_render_alignment_debug.md` | 렌더 정렬 디버그 기록 | 2026-08-15 |
| `done/20260814_progress_3d_pipeline.md` | 3D Chamfer + Render loss 파이프라인 완성 | 2026-08-14 |
| `done/20260814_progress.md` | 3D 파이프라인 진행 기록 | 2026-08-14 |
| `done/20260814_plan.md` | 프로젝트 초기 계획 | 2026-08-14 |
| `done/20260813_refactor.md` | 전체 파이프라인 리팩터링 기록 | 2026-08-13 |
| `done/20260812_plant_organ_differentiable_renderer_handover.md` | PlantOrganArray 렌더러 핸드오버 | 2026-08-12 |
| `done/20260811_helios_renderer_handover.md` | Helios PyTorch 차분 렌더러 핸드오버 | 2026-08-11 |
| `done/20260811_next_steps_ubuntu_gpu.md` | Ubuntu GPU 세팅 가이드 | 2026-08-11 |
| `done/diff_renderer_pixel_match_handoff.md` | 픽셀 매칭 핸드오프 | 2026-08-15 |
| `done/lab_meeting_report.md` | Lab meeting 리포트 | 2026-08-19 |
| `done/pixel_match_implementation_plan.md` | 픽셀 매칭 구현 계획 | 2026-08-15 |
| `done/projection_angle_explanation.md` | 카메라 투영 각도 설명 | 2026-08-13 |

---

## 📋 Todo (미착수)

| 파일 | 내용 |
|------|------|
| `todo/roadmap_26d.md` | 26D DiT-Large 중심 로드맵 |

---

## 📝 Misc (Notes & Analysis)

| File | Content |
|------|---------| 
| [`misc/20260822_cluster_power_cost_estimate.md`](misc/20260822_cluster_power_cost_estimate.md) | Cluster power draw (~8.1 kW) and cloud equivalent cost (~$41.87/hr) during 40-job sharding run |

---

## 📊 Results (학습 결과)

| 파일 | 내용 |
|------|------|
| [`results/20260825_direct_optimization_cowpea_dap10_report.md`](results/20260825_direct_optimization_cowpea_dap10_report.md) | **Cowpea DAP 10 Direct Optimization & Differentiable PyTorch Renderer Verification Report** (RGB+Depth multi-modal inverse optimization, DAP 1 seedling growth trajectory, random seed recovery, modality ablation, 4 publication figures) |
| `results/minimal_direct_opt_chamfer_leaf_outward.md` | **Minimal 3-Organ Direct Optimization: Chamfer Pulls Leaf Outward** — focused reproduction notes for `scripts/minimal_direct_opt_depth_chamfer_demo.py` |
| `results/15_strategies_benchmark_report.md` | 14D 기준 15가지 전략 벤치마크 결과 (Historical) |
| `results/assets/fig_dap10_direct_opt_growth_trajectory.png` | DAP 1 Seedling → DAP 10 Mature Canopy 성장 최적화 궤적 Figure |
| `results/assets/fig_dap10_direct_opt_random_seed_trajectory.png` | Random Seed / Perturbed Pose → DAP 10 Target 수렴 궤적 Figure |
| `results/assets/fig_dap10_multimodal_ablation_comparison.png` | RGB vs Depth vs Multi-Modal Supervision 비교 Figure |
| `results/assets/fig_dap10_convergence_curves.png` | Loss, mSSIM, IoU, 3D Chamfer, Depth MAE 정량 수렴 곡선 Figure |
| `results/assets/fig_helios_xml_vs_differentiable_render_alignment.png` | 5-Column Helios vs PyTorch 정렬 벤치마크 |
| `results/assets/fig_pure_noise_flow_matching.png` | 73M DiT: 순수 노이즈→생성 평가 |
| `results/assets/fig_canonical_cowpea_dap10_30.png` | 73M DiT: DAP 10-30 생성 결과 |
| `results/assets/minimal_direct_opt_depth_chamfer_demo.png` | **Minimal 3-organ direct optimization**: 3D Chamfer gradient pulls a leaf outward (petiole pitch 10° → 60°) |
| `results/assets/fig_*` | 각종 실험 결과 figure (40+ images) |

---

## 📦 Archived (폐기/구버전)

| 경로 | 내용 |
|------|------|
| `archived/todo/roadmap_14d_legacy.md` | 구 14D/16D 로드맵 (폐기) |
| `archived/todo/15_loss_reduction_strategies.md` | 15가지 손실 전략 (40D 기준, 전략 자체는 유효) |
| `archived/todo/2026-08-14-pytorch-renderer-optimization.md` | PyTorch 렌더러 최적화 계획 |
| `archived/todo/xml_diffusion_implementation_plan*.md` | XML 확산 구현 계획 (폐기) |
| `archived/results/15_strategies_benchmark_report_14d.md` | 14D 벤치마크 백업 |
| `archived/report1_backprop_vs_difffusion_legacy/` | 레거시 figure 생성 스크립트 |

---

## 핵심 모델 체크포인트

```
diffusion_based/checkpoints/fm/
├── canonical_cowpea_dit_best.pt           # 73M DiT, 60 에폭 (Baseline)
├── cowpea_dit_large_2xh100_ddp.pt         # 232M DiT-Large (학습 중)
├── test_large.pt                          # 232M DiT-Large, 1 에폭 테스트 (2.5GB)
└── plant_organ_40d_flow_matching_best.pt  # 40D VAE 기반 (Legacy)
```
