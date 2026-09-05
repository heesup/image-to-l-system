# Ongoing Documentation Index

This directory tracks active research, architectural decisions, mathematical derivations, and in-progress implementation milestones for the **Image-to-L-System / 3D Inverse Plant Reconstruction** project.

**Completed documents are archived in [`docs/done/`](file:///home/lion397/codes/image-to-l-system/docs/done/).**

---

## Active & Authoritative Documents (Start Here)

| Document | Purpose | Key Content |
| :--- | :--- | :--- |
| **[AGENT_TAKEOVER_GUIDE.md](file:///home/lion397/codes/image-to-l-system/docs/ongoing/AGENT_TAKEOVER_GUIDE.md)** | **Master Handover & Execution Manual** | Full system architecture, 14D Part Tensor contract, Helios procedural kinematics rules, active benchmark results, reproduction commands, gotchas, and Phase 2 roadmap. |
| **[20260903-back-to-basics.md](file:///home/lion397/codes/image-to-l-system/docs/ongoing/20260903-back-to-basics.md)** | **Phase 1–3 Benchmark & Phase 2 Active Roadmap** | Comparative benchmark across 3 paradigms (ICP vs Differentiable Rendering vs Flow Matching), multi-scale pyramid verification, and variable organ topology next steps. |
| **[20260905_fm_curv26_handoff.md](file:///home/lion397/codes/image-to-l-system/docs/ongoing/20260905_fm_curv26_handoff.md)** | **FM Curvature-26D Session Handoff (2026-09-05)** | 26D FM training (curvature channel + pyramid 16-ch conditioning), trainer NaN/DDP fixes, coverage audits (exp6/7), verification commands, known issues, and prioritized next steps. |

---

## Current Status Summary

| Phase | Description | Status |
| :--- | :--- | :---: |
| **Phase 1** | Minimal 5-Organ Fixed-Topology Benchmark (ICP / Diff Render / Flow Matching) | **✅ DONE** |
| **Phase 3** | Multi-Scale Concentric Zoom Pyramid ($1\times \to 4\times \to 8\times$) | **✅ DONE** |
| **Fig 10** | Full lifecycle Helios C++ raytracing verification (DAP 10/50/90) | **✅ DONE** — DAP 10: 95.1%, DAP 50: 92.8%, DAP 90: 86.5% |
| **Phase 2** | Variable Organ Topology — Over-allocation with existence pruning ($N_{\max}=10$) | **🔜 NEXT** |

---

## Visual Deliverables (in `docs/results/assets/`)

- **[fig10_helios_per_organ_mask_comparison.png](file:///home/lion397/codes/image-to-l-system/docs/results/assets/fig10_helios_per_organ_mask_comparison.png)**: 7-column multi-modal comparison across DAP 10, 50, 90 (GT RGB, COCO masks, Depth vs Reconstructed XML RGB, masks, Depth vs PyTorch 14D direct).
- **[fig12_back_to_basics_benchmark_summary.png](file:///home/lion397/codes/image-to-l-system/docs/results/assets/fig12_back_to_basics_benchmark_summary.png)**: Phase 1 synthesis comparison — Ground Truth, Method 1 (ICP), Method 2 (Diff Renderer), Method 3 (Flow Matching).
- **[fig13_progressive_multiscale_pyramid.png](file:///home/lion397/codes/image-to-l-system/docs/results/assets/fig13_progressive_multiscale_pyramid.png)**: Progressive concentric zoom pyramid ($1.0\times, 2.0\times, 4.0\times, 8.0\times$).
- **[exp1_icp_alignment.png](file:///home/lion397/codes/image-to-l-system/docs/results/assets/exp1_icp_alignment.png)**: 3D point cloud segmentation and ICP alignment diagnostic.
- **[exp2_diff_render_progression.png](file:///home/lion397/codes/image-to-l-system/docs/results/assets/exp2_diff_render_progression.png)**: 75-step multi-scale pyramid convergence progression.
- **[exp3_flow_matching_trajectory.png](file:///home/lion397/codes/image-to-l-system/docs/results/assets/exp3_flow_matching_trajectory.png)**: 15-step Euler ODE reverse trajectory.
