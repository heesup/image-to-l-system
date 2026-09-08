# Ongoing Documentation Index

This directory tracks active research, architectural decisions, mathematical derivations, and in-progress implementation milestones for the **Image-to-L-System / 3D Inverse Plant Reconstruction** project.

**Completed documents are archived in [`docs/done/`](file:///home/lion397/codes/image-to-l-system/docs/done/).**

---

## Active & Authoritative Documents (Start Here)

| Document | Purpose | Key Content |
| :--- | :--- | :--- |
| **[../results/20260908_3d_spatial_vision_and_hierarchical_reconstruction_milestone.md](file:///home/lion397/codes/image-to-l-system/docs/results/20260908_3d_spatial_vision_and_hierarchical_reconstruction_milestone.md)** | **3D Spatial Vision & Hierarchical Botanical Reconstruction Milestone (Epoch 150 Benchmark)** | Visual and quantitative breakthrough in 3D reconstruction (`hierarchical_self_consistency_epoch_150.png`): 49.2% mean silhouette IoU (peak 66.5%), 2.6 cm node RMSE, 13.20 cm depth MAE, DINOv2-PETR-DETR3D 3D scaffold, and scaling laws. |
| **[20260907_latent_hierarchical_flow_matching_specification.md](file:///home/lion397/codes/image-to-l-system/docs/ongoing/20260907_latent_hierarchical_flow_matching_specification.md)** | **Option B: 16D Latent Botanical Flow Matching Specification** | Resolution of 150× gradient scale disparity via regularized standard Gaussian latent manifold ($\mathbf{z} \in \mathbb{R}^{16} \sim \mathcal{N}(0, I)$), frozen `OrganLatentVAE` bridge, 100% categorical preservation, and Hungarian matcher integration. |
| **[../results/20260907_latent_hierarchical_flow_matching_500epoch_report.md](file:///home/lion397/codes/image-to-l-system/docs/results/20260907_latent_hierarchical_flow_matching_500epoch_report.md)** | **Option B 500-Epoch Comprehensive Milestone Report** | Empirical training convergence on 4× RTX 6000 Ada, 55.1% mean IoU (peak 67.9%), 2.49 cm peak height error, biological organ budget calibration, and elimination of ghost artifacts. |
| **[20260906_hierarchical_matryoshka_botanical_flow_matching_architecture.md](file:///home/lion397/codes/image-to-l-system/docs/ongoing/20260906_hierarchical_matryoshka_botanical_flow_matching_architecture.md)** | **Hierarchical Matryoshka Botanical Flow Matching & In-Loop Differentiable Loss** | Full two-stage hierarchical model, DAP-conditional Matryoshka anchor allocation ($O(K \log K)$ bipartite matching), in-loop 1-step differentiable dense depth (2.5D ICP) loss, and lighting/shadow-invariant pixel-wise cosine color loss. |
| **[AGENT_TAKEOVER_GUIDE.md](file:///home/lion397/codes/image-to-l-system/docs/ongoing/AGENT_TAKEOVER_GUIDE.md)** | **Master Handover & Execution Manual** | Full system architecture, 14D Part Tensor contract, Helios procedural kinematics rules, active benchmark results, reproduction commands, gotchas, and Phase 2 roadmap. |
| **[20260903-back-to-basics.md](file:///home/lion397/codes/image-to-l-system/docs/ongoing/20260903-back-to-basics.md)** | **Phase 1–3 Benchmark & Phase 2 Active Roadmap** | Comparative benchmark across 3 paradigms (ICP vs Differentiable Rendering vs Flow Matching), multi-scale pyramid verification, and variable organ topology next steps. |
| **[20260905_fm_curv26_handoff.md](file:///home/lion397/codes/image-to-l-system/docs/ongoing/20260905_fm_curv26_handoff.md)** | **FM Curvature-26D Session Handoff (2026-09-05)** | 26D FM training (curvature channel + pyramid 16-ch conditioning), trainer NaN/DDP fixes, coverage audits, verification commands, and prioritized next steps. |

---

## Current Status Summary

| Phase / Milestone | Description | Status |
| :--- | :--- | :---: |
| **Phase 1** | Minimal 5-Organ Fixed-Topology Benchmark (ICP / Diff Render / Flow Matching) | **✅ DONE** |
| **Phase 3** | Multi-Scale Concentric Zoom Pyramid ($1\times \to 4\times \to 8\times$) | **✅ DONE** |
| **Fig 10** | Full lifecycle Helios C++ raytracing verification (DAP 10/50/90) | **✅ DONE** — DAP 10: 95.1%, DAP 50: 92.8%, DAP 90: 86.5% |
| **Option B (Phase 2)** | 16D Latent Hierarchical Botanical Flow Matching with frozen `OrganLatentVAE` (500 Epochs, Job `38143585`) | **✅ DONE & VERIFIED** (55.1% mean IoU, peak 67.9%, 2.49cm height error, 0 ghost organs) |
| **Next Step** | Continuous Foveated Zoom (Log-Polar / Bilinear Mesh Warp) & Multi-Species Extension | **🔜 PLANNED** |

---

## Key Visual Deliverables (in `docs/results/assets/`)

- **[hierarchical_self_consistency_epoch_501.png](file:///home/lion397/codes/image-to-l-system/docs/results/assets/hierarchical_self_consistency_epoch_501.png)**: SOTA 500-Epoch Option B Evaluation — GT vs Predicted RGB, 3D Point Clouds, Depth Maps, and Difference Overlays across DAP 1 to DAP 40.
- **[fig_organ_vae_roundtrip_comparison.png](file:///home/lion397/codes/image-to-l-system/docs/results/assets/fig_organ_vae_roundtrip_comparison.png)**: OrganLatentVAE 16D round-trip fidelity across 13 organ classes.
- **[fig10_helios_per_organ_mask_comparison.png](file:///home/lion397/codes/image-to-l-system/docs/results/assets/fig10_helios_per_organ_mask_comparison.png)**: 7-column multi-modal comparison across DAP 10, 50, 90 (GT RGB, COCO masks, Depth vs Reconstructed XML RGB, masks, Depth vs PyTorch direct).
- **[fig12_back_to_basics_benchmark_summary.png](file:///home/lion397/codes/image-to-l-system/docs/results/assets/fig12_back_to_basics_benchmark_summary.png)**: Phase 1 synthesis comparison — Ground Truth, Method 1 (ICP), Method 2 (Diff Renderer), Method 3 (Flow Matching).
- **[fig13_progressive_multiscale_pyramid.png](file:///home/lion397/codes/image-to-l-system/docs/results/assets/fig13_progressive_multiscale_pyramid.png)**: Progressive concentric zoom pyramid ($1.0\times, 2.0\times, 4.0\times, 8.0\times$).
