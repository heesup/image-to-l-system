---
title: "Image-to-L-System — Documentation Map"
date: 2026-09-16
tags: [index, moc]
status: active
---

# 🗺️ Image-to-L-System — Documentation Map

Single RGB-D drone image → 3D plant organ parameter reconstruction via Hierarchical Botanical Flow Matching.

---

## 🚀 Start Here

- [Current State](current-state/current-state.md) — What is running now, what is being built, and what the evidence says
- [Agent Handover Guide](agent-handover-guide/agent-handover-guide.md) — Master manual; §0 is the one-page orientation for a new agent
- [Stage 2/3 Boundary Redesign](engineering/20260912-stage2-stage3-boundary/20260912-stage2-stage3-boundary.md) — Primary engineering record since 2026-09-12
- [Sim-to-real Assessment & Plan (2026-09-16)](experiments/20260916-sim-to-real-assessment/20260916-sim-to-real-assessment.md) — Ranked next steps for both tracks and the appearance-gap measurement

---

## 🏗️ Architecture & Reference

_System design documents, mathematical specifications, and camera geometry_

- [Cascaded Design Space Analysis (2026-09-08)](architecture/cascaded-design-space-analysis/cascaded-design-space-analysis.md)
- [Current Architecture (2026-09-10)](architecture/current-architecture/current-architecture.md)
- [Repository Structure & Code Map (2026-09-16)](architecture/code-structure/code-structure.md)
- [Hierarchical Matryoshka Flow Matching (2026-09-06)](architecture/hierarchical-matryoshka-flow-matching/hierarchical-matryoshka-flow-matching.md)
- [Latent Hierarchical Fm Spec (2026-09-07)](architecture/latent-hierarchical-fm-spec/latent-hierarchical-fm-spec.md)
- [Projection Angle & Camera Geometry Explanation](architecture/projection-angle-explanation/projection-angle-explanation.md)
- [Cluster Power & Compute Cost (2026-08-22)](architecture/cluster-power-cost.md)

---

## 🧪 Experiments & Results

_Training runs, benchmark results, and milestone reports (newest first)_

- [Stage 3 Conditioning, Settled (2026-09-18)](experiments/20260918-stage3-conditioning-settled/20260918-stage3-conditioning-settled.md)
- [20260916 Multiplant Scene (2026-09-16)](experiments/20260916-multiplant-scene/20260916-multiplant-scene.md)
- [20260916 Agml Dataset Swap (2026-09-16)](experiments/20260916-agml-dataset-swap/20260916-agml-dataset-swap.md)
- [20260915 Real Image First Test (2026-09-15)](experiments/20260915-real-image-first-test/20260915-real-image-first-test.md)
- [20260914 Stage2 Burst Fix Roundtrip (2026-09-14)](experiments/20260914-stage2-burst-fix-roundtrip/20260914-stage2-burst-fix-roundtrip.md)
- [20260910 Gradient Explosion Debug (2026-09-10)](experiments/20260910-gradient-explosion-debug/20260910-gradient-explosion-debug.md)
- [20260908 Skeleton Geometry Chamfer (2026-09-08)](experiments/20260908-skeleton-geometry-chamfer/20260908-skeleton-geometry-chamfer.md)
- [20260908 Organ Vae Sparsity (2026-09-08)](experiments/20260908-organ-vae-sparsity/20260908-organ-vae-sparsity.md)
- [20260908 3Stage Cascaded Milestone (2026-09-08)](experiments/20260908-3stage-cascaded-milestone/20260908-3stage-cascaded-milestone.md)
- [20260908 3D Spatial Vision Milestone (2026-09-08)](experiments/20260908-3d-spatial-vision-milestone/20260908-3d-spatial-vision-milestone.md)
- [20260907 Latent Fm 500Epoch (2026-09-07)](experiments/20260907-latent-fm-500epoch/20260907-latent-fm-500epoch.md)
- [20260825 Direct Opt Cowpea Dap10 (2026-08-25)](experiments/20260825-direct-opt-cowpea-dap10/20260825-direct-opt-cowpea-dap10.md)
- [Minimal Direct Opt Chamfer (2026-08-25)](experiments/minimal-direct-opt-chamfer/minimal-direct-opt-chamfer.md)
- [15 Strategies Benchmark (2026-08-25)](experiments/15-strategies-benchmark/15-strategies-benchmark.md)

---

## 🔧 Engineering Sessions

_Implementation sessions, refactors, and PR documentation_

- [20260811 Helios Renderer Handover (2026-08-11)](engineering/20260811-helios-renderer-handover/20260811-helios-renderer-handover.md)
- [20260811 Next Steps Ubuntu Gpu (2026-08-11)](engineering/20260811-next-steps-ubuntu-gpu/20260811-next-steps-ubuntu-gpu.md)
- [20260812 Plant Organ Diff Renderer (2026-08-12)](engineering/20260812-plant-organ-diff-renderer/20260812-plant-organ-diff-renderer.md)
- [20260813 Refactor (2026-08-13)](engineering/20260813-refactor/20260813-refactor.md)
- [20260814 Plan (2026-08-14)](engineering/20260814-plan/20260814-plan.md)
- [20260814 Progress 3D Pipeline (2026-08-14)](engineering/20260814-progress-3d-pipeline/20260814-progress-3d-pipeline.md)
- [20260814 Progress (2026-08-14)](engineering/20260814-progress/20260814-progress.md)
- [20260815 40D Plant Array (2026-08-15)](engineering/20260815-40d-plant-array/20260815-40d-plant-array.md)
- [20260815 Render Alignment Debug (2026-08-15)](engineering/20260815-render-alignment-debug/20260815-render-alignment-debug.md)
- [20260818 Flow Matching Scaffold (2026-08-18)](engineering/20260818-flow-matching-scaffold/20260818-flow-matching-scaffold.md)
- [20260821 Cowpea 100K Dit (2026-08-21)](engineering/20260821-cowpea-100k-dit/20260821-cowpea-100k-dit.md)
- [20260822 Dataset Multigpu Sharding (2026-08-22)](engineering/20260822-dataset-multigpu-sharding/20260822-dataset-multigpu-sharding.md)
- [20260823 Canonical Sorting Xml (2026-08-23)](engineering/20260823-canonical-sorting-xml/20260823-canonical-sorting-xml.md)
- [20260823 Diff Renderer Helios Alignment (2026-08-23)](engineering/20260823-diff-renderer-helios-alignment/20260823-diff-renderer-helios-alignment.md)
- [20260823 Project Status 26D Dit (2026-08-23)](engineering/20260823-project-status-26d-dit/20260823-project-status-26d-dit.md)
- [20260823 Vlm Scaffold Dit Design (2026-08-23)](engineering/20260823-vlm-scaffold-dit-design/20260823-vlm-scaffold-dit-design.md)
- [20260823 Vlm Scaffold Dit Training (2026-08-23)](engineering/20260823-vlm-scaffold-dit-training/20260823-vlm-scaffold-dit-training.md)
- [20260824 Botanical Mmdit Training (2026-08-24)](engineering/20260824-botanical-mmdit-training/20260824-botanical-mmdit-training.md)
- [20260824 Helios Xml Roundtrip Fix (2026-08-24)](engineering/20260824-helios-xml-roundtrip-fix/20260824-helios-xml-roundtrip-fix.md)
- [20260825 16D Part Assembly Renderer (2026-08-25)](engineering/20260825-16d-part-assembly-renderer/20260825-16d-part-assembly-renderer.md)
- [20260826 Canonical Pipeline Refactor (2026-08-26)](engineering/20260826-canonical-pipeline-refactor/20260826-canonical-pipeline-refactor.md)
- [20260826 Helios Flower Peduncle Pod (2026-08-26)](engineering/20260826-helios-flower-peduncle-pod/20260826-helios-flower-peduncle-pod.md)
- [20260826 Implementation Plan (2026-08-26)](engineering/20260826-implementation-plan/20260826-implementation-plan.md)
- [20260830 Pr Xml Roundtrip Fix (2026-08-30)](engineering/20260830-pr-xml-roundtrip-fix/20260830-pr-xml-roundtrip-fix.md)
- [20260831 27D Fm Layout Shard Regen (2026-08-31)](engineering/20260831-27d-fm-layout-shard-regen/20260831-27d-fm-layout-shard-regen.md)
- [20260831 Pr Gravitropic Curvature (2026-08-31)](engineering/20260831-pr-gravitropic-curvature/20260831-pr-gravitropic-curvature.md)
- [20260901 17D Xml Helios Roundtrip (2026-09-01)](engineering/20260901-17d-xml-helios-roundtrip/20260901-17d-xml-helios-roundtrip.md)
- [20260903 14D Part Tensor Ik (2026-09-03)](engineering/20260903-14d-part-tensor-ik/20260903-14d-part-tensor-ik.md)
- [20260903 Back To Basics (2026-09-03)](engineering/20260903-back-to-basics/20260903-back-to-basics.md)
- [20260905 Fm Curv26 Handoff (2026-09-05)](engineering/20260905-fm-curv26-handoff/20260905-fm-curv26-handoff.md)
- [20260905 Implementation Plan (2026-09-05)](engineering/20260905-implementation-plan/20260905-implementation-plan.md)
- [20260908 Phytomer Capacity Recalibration (2026-09-08)](engineering/20260908-phytomer-capacity-recalibration/20260908-phytomer-capacity-recalibration.md)
- [20260909 Phytomer Latent Matching (2026-09-09)](engineering/20260909-phytomer-latent-matching/20260909-phytomer-latent-matching.md)
- [20260911 Diff Render Gradient Isolation (2026-09-11)](engineering/20260911-diff-render-gradient-isolation/20260911-diff-render-gradient-isolation.md)
- [20260911 Hybrid Vae Rotation Topology (2026-09-11)](engineering/20260911-hybrid-vae-rotation-topology/20260911-hybrid-vae-rotation-topology.md)
- [Diff Renderer Pixel Match](engineering/diff-renderer-pixel-match/diff-renderer-pixel-match.md)
- [Pixel Match Implementation Plan](engineering/pixel-match-implementation-plan/pixel-match-implementation-plan.md)
- [Pr Helios Xml Roundtrip Fix](engineering/pr-helios-xml-roundtrip-fix/pr-helios-xml-roundtrip-fix.md)

---

## 🛠️ Tools

_Standalone tool documentation_

- [Phytomer Vae Latent Visualizer (2026-09-09)](tools/phytomer-vae-latent-visualizer/phytomer-vae-latent-visualizer.md)

---

## 📊 Lab Meetings

_Presentation reports and meeting notes_

- [20260819 Backprop Vs Diffusion (2026-08-19)](lab-meetings/20260819-backprop-vs-diffusion/20260819-backprop-vs-diffusion.md)
- [Lab Meeting Report](lab-meetings/lab-meeting-report/lab-meeting-report.md)

---

## 📋 Planning & Roadmaps

_Future plans and optimization roadmaps_

- [Phase3 Advanced Depth](planning/phase3-advanced-depth.md)
- [Roadmap 26D](planning/roadmap-26d.md)
- [Task Speedup Xml Kinematics](planning/task-speedup-xml-kinematics.md)


---

## 📦 Archive

_Superseded designs, deprecated roadmaps, legacy reports_

- [Readme](archive/design/README.md)
- [15_Strategies_Benchmark_Report_14D](archive/results/15_strategies_benchmark_report_14d.md)
- [15_Loss_Reduction_Strategies](archive/todo/15_loss_reduction_strategies.md)
- [2026 08 14 Pytorch Renderer Optimization](archive/todo/2026-08-14-pytorch-renderer-optimization.md)
- [Roadmap_14D_Legacy](archive/todo/roadmap_14d_legacy.md)
- [Xml_Diffusion_Implementation_Plan](archive/todo/xml_diffusion_implementation_plan.md)
- [Xml_Diffusion_Implementation_Plan_Updated](archive/todo/xml_diffusion_implementation_plan_updated.md)

---
