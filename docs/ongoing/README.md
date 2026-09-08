# Ongoing Documentation Index

This directory tracks active research, architectural decisions, mathematical derivations, and in-progress implementation milestones for the **Image-to-L-System / 3D Inverse Plant Reconstruction** project.

**Completed documents are archived in [`docs/done/`](file:///home/lion397/codes/image-to-l-system/docs/done/).**

---

## 🚨 CURRENT SYSTEM STATE (2026-09-08 ~17:00 PDT)

| Component | Status | Details |
| :--- | :---: | :--- |
| **Main Training Job** | 🟢 RUNNING | Job `38146809`, Epoch 26/500, gpu-10-54, 4× RTX 6000 Ada, 40.5 GB VRAM |
| **Helios Dataset Synthesis** | 🟡 IN PROGRESS | 40 shards (38147132–38147171), 51,880/~120,000 XMLs done |
| **Cache Tensors Available** | 22,600 `.pt` | In `dataset/cache/cowpea_curv26/` |
| **Per-Epoch Eval Panels** | ⚠️ NOT YET | Job 38146809 started on old code. Restart with INIT_CHECKPOINT at Epoch 25 |
| **OnDemand Desktop** | 🟢 RUNNING | Job `38147219`, gpu-5-58 — **DO NOT CANCEL** |

### ⚡ Most Urgent Action
```bash
# Check if Epoch 25 checkpoint is ready
ls -lh diffusion_based/checkpoints/hierarchical_latent_fm/hierarchical_fm_epoch_025.pt

# If it exists → cancel 38146809 → re-submit with resume
scancel 38146809
INIT_CHECKPOINT=diffusion_based/checkpoints/hierarchical_latent_fm/hierarchical_fm_epoch_025.pt \
  sbatch slurm_scripts/train_hierarchical_flow_matching.sh
```

---

## Active & Authoritative Documents (Start Here)

| Document | Purpose | Key Content |
| :--- | :--- | :--- |
| **[AGENT_TAKEOVER_GUIDE.md](file:///home/lion397/codes/image-to-l-system/docs/ongoing/AGENT_TAKEOVER_GUIDE.md)** | **Master Handover & Execution Manual (UPDATED 2026-09-08)** | Full system architecture, SLURM job status, all 2026-09-08 changes (dormant slot damping, batch 192, eval_every 1, --resume, 24h Helios jobs), next steps, gotchas. |
| **[20260908_cascaded_architecture_design_space_analysis.md](file:///home/lion397/codes/image-to-l-system/docs/ongoing/20260908_cascaded_architecture_design_space_analysis.md)** | **Cascaded Architecture Design Space (ADR)** | Evaluation of 4 architectural combinations (Det vs Diff). Mathematical proof why [Det + Diff] is optimal. |
| **[20260907_latent_hierarchical_flow_matching_specification.md](file:///home/lion397/codes/image-to-l-system/docs/ongoing/20260907_latent_hierarchical_flow_matching_specification.md)** | **Option B: 16D Latent Botanical Flow Matching Spec** | Frozen `OrganLatentVAE` bridge, regularized Gaussian latent ($\mathbf{z} \in \mathbb{R}^{16} \sim \mathcal{N}(0, I)$), Hungarian matcher integration. |
| **[20260906_hierarchical_matryoshka_botanical_flow_matching_architecture.md](file:///home/lion397/codes/image-to-l-system/docs/ongoing/20260906_hierarchical_matryoshka_botanical_flow_matching_architecture.md)** | **Hierarchical Matryoshka Botanical Flow Matching Architecture** | Full two-stage hierarchical model, DAP-conditional Matryoshka anchor allocation ($O(K \log K)$ bipartite matching), in-loop 1-step differentiable dense depth loss. |

---

## Current Status Summary

| Phase / Milestone | Description | Status |
| :--- | :--- | :---: |
| **Phase 1** | Minimal 5-Organ Fixed-Topology Benchmark (ICP / Diff Render / FM) | ✅ DONE |
| **Phase 2** | Variable Organ Topology (Over-alloc pruning, spawning) | ✅ DONE |
| **Phase 3 - Curvature 26D FM** | Extended FM with curvature channel, pyramid 16-ch conditioning | ✅ DONE |
| **Fig 10** | Full lifecycle Helios C++ raytracing (DAP 10/50/90) | ✅ DONE — 95.1%/92.8%/86.5% |
| **Option B 500-Epoch** | 16D Latent Hierarchical FM (Job `38143585`, Commit `ed95f45`) | ✅ DONE — 55.1% mean IoU, 67.9% peak |
| **3-Stage Cascaded FM** | New arch: DINOv2-PETR + 3-stage coarse/fine FM (Job `38146809`) | 🔄 TRAINING Epoch 26/500 |
| **Helios Dataset Scale-Up** | 40-shard parallel synthesis to ~120k samples (24h SLURM) | 🔄 IN PROGRESS 51,880/~120k |
| **Per-Epoch Validation Panels** | `--eval_every 1` diagnostic panels every epoch | ⚠️ PENDING restart at Epoch 25 |

---

## Key Visual Deliverables (in `docs/results/assets/`)

### 2026-09-08 Session (3-Stage Cascaded, Job 38146809)
- **[hierarchical_self_consistency_epoch_025.png](file:///home/lion397/codes/image-to-l-system/docs/results/assets/hierarchical_self_consistency_epoch_025.png)**: First milestone checkpoint — 7-column panel (Helios Raytrace | RGB | GT Depth | Pred Depth+MAE | Real 3D Composite | 3D Skeleton).
- **[hierarchical_self_consistency_epoch_050.png](file:///home/lion397/codes/image-to-l-system/docs/results/assets/hierarchical_self_consistency_epoch_050.png)**: Epoch 50 panel.
- **[hierarchical_self_consistency_epoch_075.png](file:///home/lion397/codes/image-to-l-system/docs/results/assets/hierarchical_self_consistency_epoch_075.png)**: Epoch 75 panel with increased mesh contrast.
- **[hierarchical_self_consistency_epoch_100.png](file:///home/lion397/codes/image-to-l-system/docs/results/assets/hierarchical_self_consistency_epoch_100.png)**: Epoch 100 panel.

### 2026-09-07 (Option B 500-Epoch)
- **[hierarchical_self_consistency_epoch_501.png](file:///home/lion397/codes/image-to-l-system/docs/results/assets/hierarchical_self_consistency_epoch_501.png)**: SOTA 500-Epoch Option B evaluation.

### Earlier Milestones
- **[fig10_helios_per_organ_mask_comparison.png](file:///home/lion397/codes/image-to-l-system/docs/results/assets/fig10_helios_per_organ_mask_comparison.png)**: 7-column Helios raytrace comparison DAP 10/50/90.
- **[fig12_back_to_basics_benchmark_summary.png](file:///home/lion397/codes/image-to-l-system/docs/results/assets/fig12_back_to_basics_benchmark_summary.png)**: Phase 1 synthesis comparison.

---

## Completed Today (2026-09-08)

| Task | Commit | Notes |
| :--- | :--- | :--- |
| ✅ Dormant slot damping + backbone LR 6e-5 + anchor loss 4.0 | `0389ca5` | Fixes ghost organ in reproductive slots |
| ✅ Batch size 48 → global batch 192 (82.5% VRAM utilization) | `0389ca5` | Doubled training throughput |
| ✅ Cache-first XML filtering in `PartArrayDataset` | `027118d` | Prevents 3ch vs 16ch collision |
| ✅ Helios shard time limit → 24 hours | `97420ee` | Prevents DAP 73–100 timeout |
| ✅ 40 Helios shards resubmitted (38147132–38147171) | `97420ee` | Covers DAP 1–100, 1000 seeds each |
| ✅ `--eval_every 1` validation decoupled from checkpoint saves | `c587767` | Every epoch generates diagnostic panel |
| ✅ `--resume` + `INIT_CHECKPOINT` checkpoint resume | `0435325` | Full epoch + optimizer state restoration |
| ✅ Pushed all 6 commits to `origin/main` | — | Up to date |
