# Image-to-L-System: Project Documentation

**Project**: Single-view aerial RGB/RGB-D drone image → 3D plant organ parameter reconstruction via Hierarchical Botanical Flow Matching.  
**Active Representation**: 16D Organ Latent Space ($\mathbf{z} \in \mathbb{R}^{16} \sim \mathcal{N}(0, I)$ via pre-trained `OrganLatentVAE`) + 14D Canonical Part Tensor (`[cls, base(3), rot(6), scale(3), curv(1)]`).  
**Active Model**: Two-Stage Hierarchical Botanical Flow Matching (Stage 1: Coarse Set Transformer with Matryoshka Phytomer Anchor Queries; Stage 2: Fine Flow Matching Decoder over 16D Latents + Frozen Differentiable VAE Decoder).  
**Cluster Infrastructure**: UC Davis Farm HPC | **Active Nodes**: 4× NVIDIA RTX 6000 Ada Generation (192 GB VRAM) & 4× NVIDIA H100 NVL.  
**Latest Milestone (2026-09-07)**: Option B (16D Latent Hierarchical Flow Matching) completed 500 epochs (Job `38143585`). Velocity loss collapsed by 83% ($1.71 \to 0.29$), 3D anchor position RMSE reduced to 4.24 cm, organ classification accuracy reached 85.6%, achieving **55.1% mean silhouette IoU** (peaking at **67.9%** on mature canopies), **2.49 cm peak height error**, and complete elimination of ghost organ artifacts.  
**Active Checkpoints**: `diffusion_based/checkpoints/hierarchical_latent_fm/hierarchical_fm_epoch_500.pt` & `diffusion_based/checkpoints/organ_vae/organ_latent_vae_best.pt`.

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

### → [`results/20260908_3d_spatial_vision_and_hierarchical_reconstruction_milestone.md`](results/20260908_3d_spatial_vision_and_hierarchical_reconstruction_milestone.md)
**3D Spatial Vision & Hierarchical Botanical Reconstruction Milestone Report (2026-09-08) — READ THIS FIRST**  
Comprehensive empirical verification of 3D spatial vision (DINOv2 + PETR 3D Ray PE + DETR3D 3D Reference Queries + Intra-Phytomer Flow Matching) evaluated at Epoch 150. Achieves **49.2% Mean Silhouette IoU** (peaking at **66.5%** on DAP 64 mature bushy canopies with 1,312 organs), **2.6 cm Node RMSE**, and **13.20 cm Depth MAE**. Documents why scaling model (DINOv2-Base) and dataset will further propel performance.

### → [`results/20260907_latent_hierarchical_flow_matching_500epoch_report.md`](results/20260907_latent_hierarchical_flow_matching_500epoch_report.md)
**Option B: 16D Latent Hierarchical Botanical Flow Matching — 500-Epoch Training & 3D Reconstruction Report (2026-09-07)**  
Complete realization and empirical verification of Option B (16D Latent Botanical Flow Matching) using a frozen `OrganLatentVAE`. 500 epochs completed on 4× RTX 6000 Ada (18h 57m, job `38143585`). Velocity loss dropped by 83% ($1.71 \to 0.29$), 3D anchor position RMSE collapsed from 35.9 cm to 4.24 cm, and organ classification accuracy reached 85.6%. Demonstrates **55.1% Mean Silhouette IoU** (peaking at **67.9%** on mature plants), **2.49 cm Peak Height Error**, and complete elimination of ghost organ artifacts.

### → [`ongoing/20260907_latent_hierarchical_flow_matching_specification.md`](ongoing/20260907_latent_hierarchical_flow_matching_specification.md)
**Latent Hierarchical Flow Matching (Option B) Architecture & Transition Specification (2026-09-07)**  
Detailed architectural blueprint explaining the resolution of the 150× gradient scale disparity and spiky organ collapse via the 16D normalized standard Gaussian latent manifold ($\mathbf{z} \in \mathbb{R}^{16} \sim \mathcal{N}(0, I)$), backward compatibility with 14D part tensors, and frozen VAE differentiable decoding.

### → [`ongoing/20260906_hierarchical_matryoshka_botanical_flow_matching_architecture.md`](ongoing/20260906_hierarchical_matryoshka_botanical_flow_matching_architecture.md)
**Hierarchical Matryoshka Botanical Flow Matching & In-Loop Differentiable Loss Architecture (2026-09-06)**  
Two-stage hierarchical Flow Matching architecture with DAP-conditional Matryoshka anchor allocation, linear scaling with plant maturity ($O(K \log K)$ bipartite matching), in-loop differentiable dense depth (2.5D ICP) loss ($\lambda=0.5$), and lighting/shadow-invariant pixel-wise cosine color loss ($\lambda=0.2$). Backpropagates full autograd gradients through nvdiffrast rasterization into 3D organ geometries directly within training.

### → [`ongoing/20260903_14d_part_tensor_to_xml_dynamic_ik_report.md`](ongoing/20260903_14d_part_tensor_to_xml_dynamic_ik_report.md)
**14D Part Tensor to Helios XML: Analytical Inverse Kinematics & Dynamic Reproductive Reconstruction Report (2026-09-03)**  
Complete closed-form Inverse Kinematics for lateral shoot insertion ($0.0000^\circ$ error), analytical dynamic phyllotaxis ($<0.85^\circ$ error), phytomer lookahead for `bud_state`, resolution of pod double scaling, and dynamic inverse kinematics for pitch ($\text{pitch} = \text{asin}(-R_{2, 0})$) and scale from 14D Part Tensor without hardcoding. State-of-the-art Helios C++ raytraced accuracy across all growth stages: DAP 10 (87.7% IoU, 34.71 dB), DAP 50 (89.6% IoU, 20.87 dB), and DAP 90 (83.3% IoU, Pod IoU 7.0%, Flower IoU 22.7%, 18.62 dB).

### → [`ongoing/AGENT_TAKEOVER_GUIDE.md`](ongoing/AGENT_TAKEOVER_GUIDE.md)
**Master Handover & Execution Manual**  
System architecture, 14D Part Tensor contract, Helios procedural kinematics rules, benchmark results, reproduction commands, known gotchas, and roadmap.

---

## 🎯 Active Model Checkpoints

```
diffusion_based/checkpoints/
├── hierarchical_latent_fm/
│   ├── hierarchical_fm_epoch_500.pt       # SOTA Option B 16D Latent Hierarchical FM (500 epochs, 55.1% IoU)
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
| [`results/20260907_latent_hierarchical_flow_matching_500epoch_report.md`](results/20260907_latent_hierarchical_flow_matching_500epoch_report.md) | **Option B 500-Epoch Comprehensive Milestone Report**: Convergence curves, qualitative 3D reconstructions, physical loss breakdown, and ablation analysis. |
| [`results/20260825_direct_optimization_cowpea_dap10_report.md`](results/20260825_direct_optimization_cowpea_dap10_report.md) | **Cowpea DAP 10 Direct Optimization & Differentiable PyTorch Renderer Verification Report** (RGB+Depth multi-modal inverse optimization, DAP 1 seedling growth trajectory, random seed recovery, modality ablation). |
| [`results/assets/hierarchical_self_consistency_epoch_150.png`](results/assets/hierarchical_self_consistency_epoch_150.png) | **3D Spatial Vision Epoch 150 Landmark**: Ground Truth vs Predicted 3D mesh (up to 66.5% IoU), 3D point cloud and skeleton nodes (2.6 cm RMSE), and depth error heatmaps. |
| [`results/assets/hierarchical_self_consistency_epoch_501.png`](results/assets/hierarchical_self_consistency_epoch_501.png) | **SOTA 500-Epoch Visual Benchmark**: Ground Truth vs Predicted 3D reconstruction, point cloud alignment, depth maps, and difference overlays across DAP 1 to DAP 40. |
| [`results/assets/fig_organ_vae_roundtrip_comparison.png`](results/assets/fig_organ_vae_roundtrip_comparison.png) | **OrganLatentVAE Roundtrip Reconstruction**: Exact geometric and categorical preservation across all 13 plant organ classes. |
| [`results/assets/fig_gt_vs_helios_verification.png`](results/assets/fig_gt_vs_helios_verification.png) | Helios C++ raytraced vs PyTorch differentiable mesh verification. |
