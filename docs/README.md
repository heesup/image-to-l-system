# Image-to-L-System: Project Documentation

**Project**: Single-view aerial RGB/RGB-D drone image → 3D plant organ parameter reconstruction via Hierarchical Botanical Flow Matching.  
**Active Representation**: 14D canonical part tensor (`[cls, base(3), rot(6), scale(3), curv(1)]`) grouped into 10-slot phytomer packets (`[Internode | Petiole | Leaflet×3 | Peduncle | Repro×4]`); the PhytomerVAE compresses a packet to a 128D hybrid latent (48 coarse + 10×8 per-slot residual).  
**Active Model**: Hybrid-decoupled 3-stage cascaded Flow Matching (Stage 1 macro count; Stage 2 `CoarseSkeletalTransformer`, per-node position / roll / scale / depth-ordinal; Stage 3 `PhytomerFlowMatchingDecoder`, rectified flow over the 128D packet latent conditioned on the node and its fixed parent), differentiable render loss from epoch 11, Helios XML export with stem + leaf inverse kinematics.  
**Cluster Infrastructure**: UC Davis Farm HPC | training runs on the `low` partition under `publicgrp` (A100 `gpu-3-38`/`gpu-4-56`, H100 `gpu-10-58`; `--requeue` + `AUTO_RESUME=1`); the group's `geminigrp` quota is held by Heesup's dataset-regeneration jobs.  
**Latest Milestone (2026-09-14)**: the Stage 2 gradient burst is root-caused (missing final LayerNorm on the coarse decoder) and fixed; the Helios round-trip is solved on exact_gt plants (IK-only 99.9 / 99.6 / 97.7%, VAE 93.7 / 98.3 / 95.8% FG IoU, fig14) and on dataset plants (packet path 98.3 / 98.2 / 95.6%, VAE 95.1 / 96.8 / 95.6%, fig12) after fixing shoot chaining, lateral branch points (also the training targets) and leaf orientation in the export. Same day: the packet cache was decoupled from the VAE (FM encodes Stage-3 latents on the fly; the cache is now VAE-independent), the VAE/packet launchers were folded into the two current launchers (`TRAIN_VAE=1`, `--packets-only`), and a GPU-efficiency experiment (batch auto→256) showed batch 48 remains the better wall-clock — see [`results/20260914_stage2_burst_fix_and_dataset_plant_roundtrip.md`](results/20260914_stage2_burst_fix_and_dataset_plant_roundtrip.md) and `ongoing/AGENT_TAKEOVER_GUIDE.md` §0-B.  
**Active Checkpoints**: `diffusion_based/checkpoints/phytomer_vae_v9_tl_rw4_20k/phytomer_vae_128d_best.pt` (VAE) and `diffusion_based/checkpoints/hierarchical_fm_v9/` (FM, job 38253656 resumed at epoch 26 on the loop-free training step; see `ongoing/README.md`).
**Latest state (2026-09-16)**: the synthetic track has **plateaued** — strict-protocol P stays at 33–39 across every training lever tried, while post-hoc test-time refinement lifts the same checkpoints to **67–68** and is the only large gain available without retraining. A **real-image track** now exists end to end (detect plants in a rover frame → Helios plot config → per-plant architecture → differentiable-renderer refinement → rendered plot), but **its reconstructions are not yet usable**: the network's cold start collapses to a handful of organs on real crops and refinement inflates leaves rather than fitting them. Start at [`ongoing/README.md`](ongoing/README.md) — its 2026-09-16 section is the honest status of both tracks — then read [`results/20260916_multiplant_scene_pipeline.md`](results/20260916_multiplant_scene_pipeline.md) and [`results/20260916_agml_dataset_swap_real_image_test.md`](results/20260916_agml_dataset_swap_real_image_test.md).

**Handover**: this documentation tree is the takeover surface for the **Claude Code** session — start at `ongoing/AGENT_TAKEOVER_GUIDE.md` (§0-B is the 2026-09-14 ~11:30 state), then `ongoing/README.md`.

---

## Repository & Documentation Structure

### 📂 Repository Root Layout

| Directory | Description | Status |
|---|---|---|
| `diffusion_based/` | **Core Active Pipeline**: 16D Latent Hierarchical Flow Matching, `OrganLatentVAE`, 14D Canonical Part Tensor, PyTorch differentiable renderer, Hungarian matcher, evaluation suites | ✅ Active |
| `archive/` | **Integrated Archive**: Prior Track-A (15D), 14D, 40D VAE, 94D, and historical debug scripts ([`archive/README.md`](../archive/README.md)) | 📦 Archived |
| `dataset/helios_data/` | 10K synthetic Cowpea XML simulations, multi-modal RGB-D aerial renders, and pre-extracted organ tensor shards | ✅ Active |
| `slurm_scripts/` | Automated multi-GPU DDP training, evaluation, and data generation SLURM batch scripts | ✅ Active |
| `Digital-Crops/` | Helios C++ biophysical plant simulation engine submodule | ✅ Active |
| `docs/` | Comprehensive architectural specifications, ongoing research, and empirical benchmark reports | 📄 Documentation |

### 📚 Documentation Subdirectories

| Folder | Purpose |
|---|---|
| `results/` | Verified empirical training reports, quantitative benchmark tables, and visual figures |
| `ongoing/` | **Active Research & Architectural Specifications** — Mandatory starting point for contributors |
| `done/` | Completed implementations, historical handovers, and PR documentation |
| `todo/` | Planned optimizations, future feature designs, and architectural roadmaps |
| `archived/` | Deprecated designs, superseded roadmaps, and legacy reports |
| `misc/` | Reference notes, compute profiling, and HPC power cost estimates |

---

## 🌟 Key Active & Authoritative Documents

### → [`ongoing/20260912_stage2_stage3_boundary_and_remaining_redundancy.md`](ongoing/20260912_stage2_stage3_boundary_and_remaining_redundancy.md)
**Session record 2026-09-12 → 2026-09-14 — READ THIS FIRST (its §5 gives the reading order)**  
§0 status; §1.9.2 the Stage 2 gradient burst, reproduced on one GPU, dissected per op and root-caused to the coarse decoder's missing final LayerNorm (ablation: 0/8 burst steps vs 8/8); §2.1-2.3 the Stage 2/3 boundary redesign (ordinal = depth from the root, one parent rule, Stage 3 conditioned on the fixed parent); §2.4-2.5 the Helios round-trip and the stem inverse kinematics; **§2.6 the round-trip on dataset plants** (drooping-shoot chaining, branch points from the internode base, leaf orientation inverse); §6 commit log.

### → [`results/20260914_stage2_burst_fix_and_dataset_plant_roundtrip.md`](results/20260914_stage2_burst_fix_and_dataset_plant_roundtrip.md)
**2026-09-14 report (Korean): burst fix, dataset-plant round-trip, figures 12/14, training state**  
The two outcomes of 2026-09-13/14 with their tables, the three export/topology fixes, what was tried and reverted, and where training stands.

### → [`ongoing/20260911_takeover_grad_norm_fix_roundtrip_diagnosis_10slot_restore.md`](ongoing/20260911_takeover_grad_norm_fix_roundtrip_diagnosis_10slot_restore.md)
**Session Takeover: Grad-Norm Inf Deadlock Fix, Helios Roundtrip Diagnosis & 10-Slot Contract Restore (2026-09-11) — READ THIS FIRST**  
Root-cause analysis and fixes for Job `38237555` grad_norm=inf deadlock (eval-mode leak + renderer 1/w² Jacobian overflow → w-floor Lipschitz bound, latent OOD clamp, atomic grad clamp, eval model.train() restore). Per-organ Helios roundtrip IoU diagnosis (COCO category mapping bug fixed; FK pitch accumulation identified as follow-up). **Restored the canonical v2/v3 10-slot phytomer contract** (internode slot 0 + repro ×4, matches 240D `phytomer_vae_v3` checkpoint). Lists exact SLURM resubmission sequence for the next agent.

### → [`ongoing/20260908_cascaded_architecture_design_space_analysis.md`](archived/design/20260908_cascaded_architecture_design_space_analysis.md)
**Cascaded Architecture Design Space: Deterministic vs. Diffusion Analysis (2026-09-08) — READ THIS FIRST**  
Exhaustive evaluation of the $2 \times 2 = 4$ design space (Stage 2 3D node scaffold vs Stage 3 intra-phytomer organs across Deterministic ViT vs Diffusion/Flow Matching). Mathematical and botanical rationale proving why **Combination 2 ([Deterministic Scaffold] + [16D Latent Flow Matching])** achieves the optimal trade-off between metric accuracy, organic leaf curvature, fast inference (30 ms), and DepthAnything v2 compatibility.

### → [`results/20260908_3d_spatial_vision_and_hierarchical_reconstruction_milestone.md`](results/20260908_3d_spatial_vision_and_hierarchical_reconstruction_milestone.md)
**3D Spatial Vision & Hierarchical Botanical Reconstruction Milestone Report (2026-09-08)**  
Comprehensive empirical verification of 3D spatial vision (DINOv2 + PETR 3D Ray PE + DETR3D 3D Reference Queries + Intra-Phytomer Flow Matching) evaluated at Epoch 150. Achieves **49.2% Mean Silhouette IoU** (peaking at **66.5%** on DAP 64 mature bushy canopies with 1,312 organs), **2.6 cm Node RMSE**, and **13.20 cm Depth MAE**. Documents why scaling model (DINOv2-Base) and dataset will further propel performance.

### → [`results/20260907_latent_hierarchical_flow_matching_500epoch_report.md`](results/20260907_latent_hierarchical_flow_matching_500epoch_report.md)
**Option B: 16D Latent Hierarchical Botanical Flow Matching — 500-Epoch Training & 3D Reconstruction Report (2026-09-07)**  
Complete realization and empirical verification of Option B (16D Latent Botanical Flow Matching) using a frozen `OrganLatentVAE`. 500 epochs completed on 4× RTX 6000 Ada (18h 57m, job `38143585`). Velocity loss dropped by 83% ($1.71 \to 0.29$), 3D node position RMSE collapsed from 35.9 cm to 4.24 cm, and organ classification accuracy reached 85.6%. Demonstrates **55.1% Mean Silhouette IoU** (peaking at **67.9%** on mature plants), **2.49 cm Peak Height Error**, and complete elimination of ghost organ artifacts.

### → [`ongoing/20260907_latent_hierarchical_flow_matching_specification.md`](archived/design/20260907_latent_hierarchical_flow_matching_specification.md)
**Latent Hierarchical Flow Matching (Option B) Architecture & Transition Specification (2026-09-07)**  
Detailed architectural blueprint explaining the resolution of the 150× gradient scale disparity and spiky organ collapse via the 16D normalized standard Gaussian latent manifold ($\mathbf{z} \in \mathbb{R}^{16} \sim \mathcal{N}(0, I)$), backward compatibility with 14D part tensors, and frozen VAE differentiable decoding.

### → [`ongoing/20260906_hierarchical_matryoshka_botanical_flow_matching_architecture.md`](archived/design/20260906_hierarchical_matryoshka_botanical_flow_matching_architecture.md)
**Hierarchical Matryoshka Botanical Flow Matching & In-Loop Differentiable Loss Architecture (2026-09-06)**  
Two-stage hierarchical Flow Matching architecture with DAP-conditional Matryoshka node allocation, linear scaling with plant maturity ($O(K \log K)$ bipartite matching), in-loop differentiable dense depth (2.5D ICP) loss ($\lambda=0.5$), and lighting/shadow-invariant pixel-wise cosine color loss ($\lambda=0.2$). Backpropagates full autograd gradients through nvdiffrast rasterization into 3D organ geometries directly within training.

### → [`ongoing/20260903_14d_part_tensor_to_xml_dynamic_ik_report.md`](done/20260903_14d_part_tensor_to_xml_dynamic_ik_report.md)
**14D Part Tensor to Helios XML: Analytical Inverse Kinematics & Dynamic Reproductive Reconstruction Report (2026-09-03)**  
Complete closed-form Inverse Kinematics for lateral shoot insertion ($0.0000^\circ$ error), analytical dynamic phyllotaxis ($<0.85^\circ$ error), phytomer lookahead for `bud_state`, resolution of pod double scaling, and dynamic inverse kinematics for pitch ($\text{pitch} = \text{asin}(-R_{2, 0})$) and scale from 14D Part Tensor without hardcoding. State-of-the-art Helios C++ raytraced accuracy across all growth stages: DAP 10 (87.7% IoU, 34.71 dB), DAP 50 (89.6% IoU, 20.87 dB), and DAP 90 (83.3% IoU, Pod IoU 7.0%, Flower IoU 22.7%, 18.62 dB).

### → [`ongoing/AGENT_TAKEOVER_GUIDE.md`](ongoing/AGENT_TAKEOVER_GUIDE.md)
**Master Handover & Execution Manual**  
System architecture, 14D Part Tensor contract, Helios procedural kinematics rules, benchmark results, reproduction commands, known gotchas, and roadmap.

---

## 🎯 Active Model Checkpoints

```
diffusion_based/checkpoints/
├── phytomer_vae_v9_tl_rw4_20k/
│   └── phytomer_vae_128d_best.pt          # CURRENT PhytomerVAE: 128D hybrid (48 + 10x8), terminal-last packets, 20k files
├── phytomer_vae_v8/                       # VAE of the FM runs before v9 (bottom-to-top packets, cowpea_curv26_pkt cache)
├── hierarchical_fm_v9_local2/
│   └── hierarchical_fm_epoch_0{05,10,15}.pt  # CURRENT FM lineage (final-norm decoder, fp32 self-attention), training in progress
├── hierarchical_latent_fm/                # 2026-09-07 Option B lineage (16D OrganLatentVAE), kept for history
│   ├── hierarchical_fm_epoch_500.pt       # Option B 16D Latent Hierarchical FM (500 epochs, 55.1% IoU)
│   ├── hierarchical_fm_epoch_400.pt       # Checkpoint at epoch 400
│   └── hierarchical_fm_epoch_300.pt       # Checkpoint at epoch 300
├── organ_vae/
│   └── organ_latent_vae_best.pt           # Pre-trained OrganLatentVAE (16D latent, 100% cls, 1.39cm depth MAE)
└── fm/
    ├── canonical_cowpea_dit_best.pt       # 73M DiT Baseline (60 epochs)
    └── cowpea_dit_large_2xh100_ddp.pt     # 232M DiT-Large Baseline
```

---

## 📋 Planned Optimizations & Next Milestones

| Document / Task | Description | Priority |
|---|---|:---:|
| [`todo/task_speedup_xml_loading_and_kinematics.md`](todo/task_speedup_xml_loading_and_kinematics.md) | **XML Deserialization & Forward Kinematics Acceleration**: Reduce DAP 100 E2E 3.8s bottleneck via shoot chunking and vectorized forward kinematics to <100ms. | 🔴 High |
| **Non-Linear Foveated Spatial Conditioning** | Implement continuous zoom foveation (radial Log-Polar or bilinear mesh warps) to preserve 8× seedling details while retaining global canopy context without edge cropping. | 🟡 Medium |
| **Multi-Species L-System Generalization** | Expand 16D latent flow matching from Cowpea to Soybean, Maize, and Sorghum architectures. | 🟢 Future |

---

## 📊 Empirical Results & Deliverables

| File | Description |
|---|---|
| [`results/20260914_stage2_burst_fix_and_dataset_plant_roundtrip.md`](results/20260914_stage2_burst_fix_and_dataset_plant_roundtrip.md) | **2026-09-14**: Stage 2 gradient burst root-caused and fixed (ablation table); Helios round-trip on dataset plants 81.8/92.2/54.6 → 98.3/98.2/95.6% after fixing shoot chaining, lateral branch points and leaf orientation in the export |
| [`results/assets/fig14_phytomer_vae_helios_roundtrip.png`](results/assets/fig14_phytomer_vae_helios_roundtrip.png) | Helios GT vs IK-only recon vs VAE round-trip, exact_gt DAP 10/50/90 (IK-only 99.9/99.6/97.7%, VAE 93.7/98.3/95.8% FG IoU) |
| [`results/assets/fig12_phytomer_10slot_helios_roundtrip.png`](results/assets/fig12_phytomer_10slot_helios_roundtrip.png) | Dataset plants DAP 15/40/75: GT mesh and 10-slot assembly under the same nadir and 45° cameras, then both Helios round-trips |
| [`results/20260910_gradient_explosion_debug_and_architecture_comparison.md`](results/20260910_gradient_explosion_debug_and_architecture_comparison.md) | 2026-09-10: `.detach()` positive-feedback bug, hybrid-decoupled architecture decision |
| [`results/20260907_latent_hierarchical_flow_matching_500epoch_report.md`](results/20260907_latent_hierarchical_flow_matching_500epoch_report.md) | **Option B 500-Epoch Comprehensive Milestone Report**: Convergence curves, qualitative 3D reconstructions, physical loss breakdown, and ablation analysis. |
| [`results/20260825_direct_optimization_cowpea_dap10_report.md`](results/20260825_direct_optimization_cowpea_dap10_report.md) | **Cowpea DAP 10 Direct Optimization & Differentiable PyTorch Renderer Verification Report** (RGB+Depth multi-modal inverse optimization, DAP 1 seedling growth trajectory, random seed recovery, modality ablation). |
| [`results/assets/hierarchical_self_consistency_epoch_150.png`](results/assets/hierarchical_self_consistency_epoch_150.png) | **3D Spatial Vision Epoch 150 Landmark**: Ground Truth vs Predicted 3D mesh (up to 66.5% IoU), 3D point cloud and skeleton nodes (2.6 cm RMSE), and depth error heatmaps. |
| [`results/assets/hierarchical_self_consistency_epoch_501.png`](results/assets/hierarchical_self_consistency_epoch_501.png) | **SOTA 500-Epoch Visual Benchmark**: Ground Truth vs Predicted 3D reconstruction, point cloud alignment, depth maps, and difference overlays across DAP 1 to DAP 40. |
| [`results/assets/fig_organ_vae_roundtrip_comparison.png`](results/assets/fig_organ_vae_roundtrip_comparison.png) | **OrganLatentVAE Roundtrip Reconstruction**: Exact geometric and categorical preservation across all 13 plant organ classes. |
| [`results/20260908_3stage_cascaded_flow_matching_scaling_milestone.md`](results/20260908_3stage_cascaded_flow_matching_scaling_milestone.md) | **3-Stage Cascaded Flow Matching & Multi-GPU Scaling Milestone**: Architectural decoupling (Macro prior $\to$ 3D Node Scaffold $\to$ Intra-phytomer Flow Matching), 2× vegetative IoU gain (51.2%), Batch 48 / 41GB VRAM scaling, and dormant slot damping. |
| [`results/20260908_organ_vae_sparsity_gradient_balance_findings.md`](results/20260908_organ_vae_sparsity_gradient_balance_findings.md) | **Empirical Findings: Organ Vector Sparsity & Gradient Unbalance measured via frozen OrganLatentVAE** — 71× class imbalance, 61× raw gradient spread (156× peduncle L/r) → 3.3× balanced 16D latent gradients (16/16 dims alive), measured on 184,859 organs. |
| [`results/assets/fig_gt_vs_helios_verification.png`](results/assets/fig_gt_vs_helios_verification.png) | Helios C++ raytraced vs PyTorch differentiable mesh verification. |
