# Project Roadmap — 26D VLM-Scaffold-DiT Flow Matching

> **Last Updated**: 2026-08-23
> **Active Model**: 176.2M VLM-Scaffold-DiT (DINOv3 ViT-B/14 Backbone + Bridge Flow Matching, 4096 Slots × 26D, 100K Drone Orthophoto Dataset)
> **Active Training**: 2× NVIDIA H100 SXM5 80GB NVLink (SLURM Job: `37866562`)

---

## Phase 1: Foundational 26D Representation & Geometry Engine (Completed)

- [x] 26D canonical organ vector encoding (4096 slots × 26D)
- [x] Helios C++ ↔ PyTorch differentiable renderer geometry alignment (0.000 mm vertex error)
- [x] 100% lossless XML round-trip verified (0.000000 mm vertex error)
- [x] Multi-arch CUDA nvdiffrast support (`sm_70` through `sm_90+PTX`)
- [x] Self-healing SLURM fault tolerance and automated job recovery
- [x] Canonical Botanical Phytomer-Preserving Tree Sorting (Shoot 0 Phytomers 0..K -> Branches sorted by vertical height $Z$)
- [x] Organ category semantic segmentation color LUT bug fix (0: Stem, 1: Petiole, 2: Leaf, 3: Peduncle, 4: Flower, 5: Pod)
- [x] True botanical vertical canopy height calculation ($Z_{\max} - Z_{\min}$) with 8cm ~ 85cm realistic field bounds

---

## Phase 2: High-Throughput Sharding & Drone Orthophoto Dataset (Completed)

- [x] **Georeferenced Drone Orthophoto Pipeline**:
  - [x] Fixed 90.0° Nadir Top-down camera elevation (`cel = 90.0°`)
  - [x] Fixed 5.0m drone flight altitude (`cam_h = 5.0m`)
  - [x] Fixed 0.0° North-aligned azimuth (`caz = 0.0°`, Orthophoto standard)
  - [x] Variable natural solar illumination angle augmentation (`saz = 0~360°`, `sel = 30~85°`)
- [x] **GPU In-Memory Mesh Caching Acceleration**:
  - [x] Pre-loaded and cached 3D plant meshes in GPU/RAM memory to eliminate disk XML I/O bottlenecks
  - [x] Achieved **295.8 samples/sec per GPU** rendering throughput
  - [x] 100,000 full dataset samples generated across 1,000 shards in **under 2 minutes** (40 parallel SLURM jobs)

---

## Phase 3: VLM-Scaffold-DiT Architecture & Training (In Progress)

- [x] **Unified Multi-Modal VLM-Scaffold-DiT Architecture**:
  - [x] Shared Pretrained Vision Backbone (DINOv3 ViT-B/14, 768-dim embeddings, `diffusion_based/models/vlm_vision_tower.py`)
  - [x] Global Token ➔ Multi-Task Macro Phenotyping Heads (`[DAP, Height, Radius, Active_Count]`)
  - [x] Dynamic Fibonacci Botanical Scaffold Generator ($x_{\text{scaffold}}$ with phyllotactic petiole/leaf anchors)
  - [x] Spatial Patch Tokens $(B, L_v, 768)$ ➔ DiT Decoder Cross-Attention for fine geometric detail reconstruction
  - [x] **Bridge Flow Matching**: Optimal transport trajectory starting from structural botanical scaffold ($x_{\text{scaffold}} \to x_{\text{target}}$)
  - [x] Unified single-pass DDP autograd forward pass avoiding dual-call gradient race conditions
- [/] **2× H100 SXM5 DDP Training (Job `37866562`, 60 Epochs)** — **Actively Running**
  - [x] W&B real-time multi-modal logging: [`vlm_scaffold_dit_2xh100_b64_0823_1731`](https://wandb.ai/lion395-university-of-california-davis/cowpea-vlm-scaffold-dit/runs/c6jxjox2)
  - [x] 6-Column Evaluation Suite: GT vs Gen (RGB, Depth with colorbar, Multi-color Organ Segmentation with semantic legend)

---

## Phase 4: Downstream Quality & Multi-Species Expansion (Upcoming)

- [ ] Tweedie DPS / Differentiable Rendering Guidance at test-time inference
- [ ] End-to-end differentiable Depth / Mask rendering loss supervision (mSSIM + FG-IoU)
- [ ] **Accelerate XML Deserialization & Forward Kinematics Pipeline** (Reduce 3.8s DAP 100 E2E bottleneck to <100ms via shoot-level chunked assembly / vectorized FK; see [`task_speedup_xml_loading_and_kinematics.md`](task_speedup_xml_loading_and_kinematics.md))
- [ ] Extend 26D pipeline to Sorghum and Common Bean
- [ ] Field drone orthomosaic test-time adaptation with real multi-spectral imagery
- [ ] Gaussian-Splatting / NeRF 3D mesh reconstruction cross-validation

