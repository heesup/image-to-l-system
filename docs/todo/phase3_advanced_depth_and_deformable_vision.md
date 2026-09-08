# Phase 3: Advanced Depth Pyramid & Deformable 3D Vision Roadmap

## 1. Overview & Dataset Cache Analysis

In Phase 2 (the current 4-in-1 architecture), the model combines:
1. **Pretrained DINOv2-S/14 Backbone** (`embed_dim=384`).
2. **3D Camera Ray Positional Embedding (PETR-style)**.
3. **3D Anchor Queries with 3D Reference Points (DETR3D-style)**.
4. **Phytomer Block Flow Matching Decoder ($M=8$ slots)**.

This roadmap documents the remaining experimental modules deferred to Phase 3 to preserve engineering simplicity, fast training throughput, and ease of ablation.

---

## 2. Dataset Cache: 16-Channel Multi-Zoom Depth Pyramid

The preprocessed dataset cache located at `dataset/cache/cowpea_curv26/*.pt` contains pre-rendered multi-zoom image tensors with shape `[16, 256, 256]` (in `float16`):

| Channels | Modality | Zoom Level | Description |
| :--- | :--- | :--- | :--- |
| **0, 1, 2** | RGB | **1x** | Full-canopy global view (normalized ImageNet stats) |
| **3** | **Depth / CHM** | **1x** | **Ground-truth Canopy Height Model / Metric Depth (0.0 to ~0.50m)** |
| **4, 5, 6** | RGB | **2x** | Centered 2x zoom on main stem and primary crown |
| **7** | **Depth / CHM** | **2x** | **Ground-truth Depth at 2x resolution** |
| **8, 9, 10** | RGB | **4x** | 4x zoom on active apical meristem & upper nodes |
| **11** | **Depth / CHM** | **4x** | **Ground-truth Depth at 4x resolution** |
| **12, 13, 14**| RGB | **8x** | 8x microscopic zoom on petiole base & flower buds |
| **15** | **Depth / CHM** | **8x** | **Ground-truth Depth at 8x resolution** |

### Key Insight: Zero-Rendering-Cost Depth Supervision
Because the ground-truth Depth/CHM maps (Channels 3, 7, 11, 15) are **already pre-rendered and cached on disk**, future auxiliary depth supervision does **not** require any real-time rendering via `HeliosPyTorchRenderer`. The depth maps can be loaded directly from memory alongside the RGB channels.

---

## 3. Phase 3 Proposed Enhancements

### A. DPT (Dense Prediction Transformer) Multi-Scale Reassembly Neck
* **Concept**: Tap 4 intermediate transformer layers of DINOv2 (`[2, 5, 8, 11]`), reassemble them using convolutional residual blocks at 4 spatial scales ($1/4, 1/8, 1/16, 1/32$), and fuse them via feature pyramid connections.
* **Goal**: Merge high-resolution local leaf boundary features with low-resolution semantic canopy features.

### B. Auxiliary Multi-Scale Depth Supervision Head
* **Concept**: Attach a lightweight $1 \times 1$ conv head onto the DPT neck to predict a multi-scale depth map $\hat{D}_{1\times}, \hat{D}_{2\times}$.
* **Loss**: Supervise against cached Channel 3 & Channel 7 using Scale-Invariant Logarithmic (SILog) loss:
  $$\mathcal{L}_{\text{depth}} = \frac{1}{N} \sum_i d_i^2 - \frac{\lambda}{N^2} \left(\sum_i d_i\right)^2, \quad d_i = \log \hat{D}_i - \log D_i^*$$
* **Goal**: Force the visual backbone to maintain strict physical depth calibration in real-world metric units.

### C. Deformable Cross-Attention for 3D Queries
* **Concept**: Given a 3D anchor reference point $\mathbf{p}_k = (x, y, z)$, project it into the 2D image plane via camera matrices:
  $$(u_k, v_k) = \mathbf{K} \cdot [\mathbf{R} | \mathbf{t}] \cdot \mathbf{p}_k$$
* Sample $P$ local offsets around $(u_k, v_k)$ across the multi-scale DPT feature maps using bilinear interpolation.
* **Goal**: Focus cross-attention exclusively on the local visual foliage surrounding each specific botanical node, minimizing background noise.
