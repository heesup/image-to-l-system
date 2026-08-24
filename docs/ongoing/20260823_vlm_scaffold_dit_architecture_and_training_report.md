# VLM-Scaffold-DiT: 2-Stage Neural Coarse-to-Fine Vision Diffusion for 3D Plant Reconstruction

> **Document Status**: Active / Ongoing  
> **Date**: August 23, 2026  
> **Target Species**: Cowpea (*Vigna unguiculata*)  
> **Current Active Training Run**: 2× NVIDIA H100 SXM5 80GB (SLURM Job: `37866768`)  
> **W&B Live Tracker**: [vlm_scaffold_dit_2xh100_b64_0823_2023](https://wandb.ai/lion395-university-of-california-davis/cowpea-vlm-scaffold-dit/runs/iw0iexb0)

---

## 1. Executive Summary & Paradigm Shift

Previous procedural approaches used synthetic mathematical rules (such as Fibonacci phyllotaxis and idealized vertical stem columns) to initialize 3D plant structures. While computationally simple, rule-based geometric priors cannot represent the asymmetrical growth, vine wandering, wind-induced curvature, and complex canopy occlusions typical of real agricultural field crops.

To resolve this fundamental limitation, we introduce **VLM-Scaffold-DiT**, an end-to-end data-driven **2-Stage Neural Coarse-to-Fine Vision Diffusion Framework**:
1. **Stage 1 (Direct Neural Coarse Set Predictor)**: A DETR-style Transformer Decoder with learnable botanical slot queries that cross-attends with DINOv3 visual patch tokens to directly predict a coarse 3D Plant Organ Array $\mathbf{x}_{\text{coarse}} \in \mathbb{R}^{B \times N \times 26}$ directly from a single 2D drone orthophoto.
2. **Stage 2 (Continuous Bridge Flow Matching DiT Refiner)**: Treats $\mathbf{x}_{\text{coarse}}$ as the neural bridge starting point ($t=0$) and learns a continuous velocity vector field $\mathbf{v}_\theta$ via a 12-layer Diffusion Transformer (DiT) with spatial cross-attention to resolve fine geometric details (leaf curvature, micro-petiole twisting, organ collision resolution) in 10–15 Euler ODE integration steps.

---

## 2. Complete End-to-End System Architecture

```
                      [ 2D Drone Orthophoto Input (3, 128, 128) ]
                     (Fixed 5.0m Nadir Top-View, Fixed 0.0° North)
                                       │
                                       ▼
                        ┌───────────────────────────┐
                        │    DINOv3 Vision Tower    │
                        │      (ViT-B/14, 768d)     │
                        └─────────────┬─────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
   [ [CLS] Global Token ]                          [ Spatial Patch Tokens ]
          (1, 768)                                     (L_v = 81, 768)
              │                                               │
              ▼                                               │
 ┌─────────────────────────┐                                  │
 │   Macro Trait Predictor │                                  │
 ├─────────────────────────┤                                  │
 │ • DAP (1~100 days)      │                                  │
 │ • Height (8~85 cm)      │                                  │
 │ • Canopy Radius (cm)    │                                  │
 │ • Active Slot Count     │                                  │
 └────────────┬────────────┘                                  │
              │                                               │
              ▼                                               │
 ┌─────────────────────────────────────────────────────────┐  │
 │     Stage 1: Neural Coarse Set Predictor (DETR-style)   │  │
 │  • Learnable Botanical Slot Queries Q ∈ R^(N×768)       │  │
 │  • Cross-Attention with DINOv3 Spatial Visual Patches   │◄─┘
 │  • Output: Data-Driven 3D Coarse Scaffold x_coarse      │
 └────────────────────────────┬────────────────────────────┘
                              │
                              ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                    Stage 2: Bridge Flow Matching DiT                   │
 │  • Interpolation: x_t = (1 - t) * x_coarse + t * x_target + σ(t) ε     │
 │  • Self-Attention across N=4096 Canonical Organ Slots                  │
 │  • Multi-Head Cross-Attention to DINOv3 Spatial Visual Patches         │
 │  • Conditioning: Time embedding + Macro Phenotype tokens               │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
                     [ 26D Vector Velocity Field v_θ ]
                                     │ (10~15 ODE Euler Steps)
                                     ▼
                     [ 3D Plant Organ Array (4096, 26) ]
                                     │
                                     ▼
                 ┌───────────────────────────────────────┐
                 │  Lossless Round-Trip XML Synthesizer  │
                 │   (0.000 mm Vertex Round-Trip Error)  │
                 └───────────────────┬───────────────────┘
                                     │
                                     ▼
                 ┌───────────────────────────────────────┐
                 │ PyTorch Differentiable 3D Mesh Engine │
                 │   (Vectorized GPU Tube & Leaf Mesh)   │
                 └───────────────────┬───────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
  [ 3D RGB Render ]          [ 3D Depth Map ]        [ 6-Class Organ Seg Mask ]
 (Soil Ground, Green)      (True Z Extent + Bar)    (Stem, Petiole, Leaf, Pod)
```

---

## 3. Mathematical Formulations & Component Details

### 3.1. Variable Slot Batching via Key-Padding Masks
Agricultural plants undergo extreme morphological expansion: an early vegetative seedling at DAP 10 contains $\approx 50$ organs, whereas a full-canopy mature plant at DAP 90 contains over $2,000$ organs. To handle variable active slot counts $N_i$ within mini-batches on GPUs:

1. **Fixed Maximum Capacity**: Tensors are allocated with capacity $N_{\text{max}} = 4096$ slots: $\mathbf{x} \in \mathbb{R}^{B \times N_{\text{max}} \times 26}$.
2. **Boolean Key-Padding Mask**:
   $$\text{mask}_{b, j} = \begin{cases} \text{False} (0) & \text{if slot } j < N_b \quad (\text{active botanical organ}) \\ \text{True} (1) & \text{if slot } j \ge N_b \quad (\text{padded empty slot}) \end{cases}$$
3. **Attention & Loss Gating**:
   * Transformer attention ignores padded tokens via `tgt_key_padding_mask=mask`.
   * Losses are normalized strictly over active organs:
     $$\mathcal{L} = \frac{\sum_{b, j} \mathcal{L}_{b, j} \cdot (1 - \text{mask}_{b, j})}{\sum_{b, j} (1 - \text{mask}_{b, j})}$$
4. **Soft Existence Recovery**: For any slot, organ existence probability is parameterized by the empty-class logit: $p(\text{exist}) = 1.0 - \text{Softmax}(\mathbf{x}_{:, :12})[\text{EMPTY\_IDX}]$.

---

### 3.2. Stage 1: Direct Neural Coarse Set Predictor
* **Input**: Spatial visual tokens $\mathbf{H}_{\text{patches}} \in \mathbb{R}^{B \times 81 \times 768}$ and global token $\mathbf{h}_{\text{global}} \in \mathbb{R}^{B \times 768}$ from DINOv3.
* **Learnable Botanical Queries**: $\mathbf{Q} \in \mathbb{R}^{N_{\text{max}} \times 768}$ encoding canonical phytomer structural hierarchy.
* **Architecture**: 4-layer Transformer Decoder with Pre-LayerNorm and GELU activations.
* **Outputs**:
  * $\mathbf{x}_{\text{coarse}} \in \mathbb{R}^{B \times N_{\text{max}} \times 26}$: Coarse organ position, orientation, scale, and type.
  * Macro Phenotypic Predictions: $\widehat{\text{DAP}}$, $\widehat{\text{Height}}$, $\widehat{\text{Radius}}$, $\widehat{N}_{\text{active}}$.
* **Stage 1 Loss**:
  $$\mathcal{L}_{\text{Stage1}} = \mathcal{L}_{\text{coarse\_geom}} + 0.5 \cdot \mathcal{L}_{\text{macro}}$$
  $$\mathcal{L}_{\text{coarse\_geom}} = \frac{1}{\sum (1 - \text{mask})} \sum_{b, j} \mathbf{w}_{\text{active}} \odot \| \mathbf{x}_{\text{coarse}, b, j} - \mathbf{x}_{\text{target}, b, j} \|_2^2$$

---

### 3.3. Stage 2: Continuous Bridge Flow Matching DiT Refiner
Instead of diffusing from pure Gaussian white noise (which requires 50–100 diffusion steps), Stage 2 uses the Neural Coarse Scaffold as the bridge trajectory origin:

1. **Neural Bridge Interpolation**:
   For random continuous timestep $t \sim \mathcal{U}(0, 1)$ and noise $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$:
   $$\mathbf{x}_t = (1 - t) \cdot (\mathbf{x}_{\text{coarse}} + \sigma_{\min} \boldsymbol{\epsilon}) + t \cdot \mathbf{x}_{\text{target}}$$
2. **Analytical Target Velocity**:
   $$\mathbf{u}_t = \frac{d\mathbf{x}_t}{dt} = \mathbf{x}_{\text{target}} - (\mathbf{x}_{\text{coarse}} + \sigma_{\min} \boldsymbol{\epsilon})$$
3. **DiT Flow Matching Loss**:
   $$\mathcal{L}_{\text{velocity}} = \frac{1}{\sum (1 - \text{mask})} \sum_{b, j} \mathbf{w}_{\text{active}} \odot \| \mathbf{v}_\theta(\mathbf{x}_t, t, \mathbf{H}_{\text{patches}}, \mathbf{h}_{\text{macro}})_{b, j} - \mathbf{u}_{t, b, j} \|_2^2$$

---

### 3.4. Hierarchical Differentiable Rendering Losses
With vectorized GPU mesh assembly (`HeliosPyTorchGeometry`) and hardware rasterization (`nvdiffrast`), 3D meshes are rendered in **< 0.5 ms per plant on H100**, enabling real-time gradient backpropagation:

1. **Stage 1 Coarse Mask / Silhouette Loss**:
   $$\mathcal{L}_{\text{render, Stage1}} = 1 - \text{IoU}(\text{Render}_{\text{mask}}(\mathbf{x}_{\text{coarse}}), \mathbf{M}_{\text{ground\_truth}})$$
   * *Role*: Enforces macro canopy spread, vertical height boundary, and main stem placement.
2. **Stage 2 Fine Photometric RGB + SSIM Loss**:
   $$\widehat{\mathbf{x}}_1 = \mathbf{x}_t + (1 - t) \cdot \mathbf{v}_\theta \quad (\text{Instantaneous 3D Endpoint})$$
   $$\mathcal{L}_{\text{render, Stage2}} = \| \text{Render}_{\text{rgb}}(\widehat{\mathbf{x}}_1) - \mathbf{I}_{\text{drone}} \|_1 + \lambda_{\text{ssim}} (1 - \text{SSIM}) + \mathcal{L}_{\text{seg}}$$
   * *Role*: Polishes individual leaf curvatures, leaflet pitch/yaw rotations, and organ intersection avoidance at sub-millimeter precision.

---

## 4. Georeferenced Drone Orthophoto Dataset

To mirror standard UAV agricultural surveys, the synthetic dataset was generated with exact orthophoto specifications:

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Camera Elevation (`cel`)** | `90.0°` | Fixed Nadir Top-Down View |
| **Drone Flight Altitude (`cam_h`)** | `5.0m` | Standard agricultural UAV survey altitude |
| **Camera Azimuth (`caz`)** | `0.0°` | Fixed True-North aligned (Orthophoto standard) |
| **Solar Illumination (`sel`, `saz`)** | `30.0° ~ 85.0°` | Diverse natural sun angles and diurnal lighting |
| **Organ Slot Capacity** | `4,096` | Accommodates full mature lifespan (DAP 1 ~ 100) |
| **Dataset Size** | `100,000` samples | 1,000 shards × 100 samples |
| **Rendering Acceleration** | `295.8 samples/s/GPU` | In-memory 3D template caching |

---

## 5. Standardized 6-Column Evaluation Benchmark

Every epoch, the evaluation pipeline generates and logs a full 6-column benchmark figure across benchmark stages (DAP 10, 30, 50, 70, 90):

| Column | Title | Visual Representation |
| :---: | :--- | :--- |
| **Col 1** | **GT RGB** | Differentiable PyTorch RGB rendering of ground-truth XML plant on soil ground. |
| **Col 2** | **GT Depth** | Physical metric depth map with continuous colorbar and true height $H_{\text{cm}} = Z_{\max} - Z_{\min}$. |
| **Col 3** | **GT Organ Seg** | Multi-color semantic organ segmentation map with discrete legend (Stem, Petiole, Leaf, Pod). |
| **Col 4** | **Gen RGB** | Differentiable PyTorch RGB rendering of model-generated 3D plant. |
| **Col 5** | **Gen Depth** | Model-generated depth map with metric colorbar and predicted height $\widehat{H}_{\text{cm}}$. |
| **Col 6** | **Gen Organ Seg** | Model-generated semantic organ segmentation map with discrete category legend. |

* **Local Evaluation Figure Path**: [`docs/results/assets/fig_cowpea_100k_lifespan_benchmark.png`](file:///home/lion397/codes/image-to-l-system/docs/results/assets/fig_cowpea_100k_lifespan_benchmark.png)
* **Latest Training Epoch Visuals**: [`docs/results/assets/fig_vlm_scaffold_latest_eval.png`](file:///home/lion397/codes/image-to-l-system/docs/results/assets/fig_vlm_scaffold_latest_eval.png)

---

## 6. Active Distributed Training Specifications

* **Model**: `VLMScaffoldDiTModel` (2-Stage Neural Coarse-to-Fine)
* **Total Parameters**: **218.8M Parameters** (100% trainable end-to-end)
* **Hardware**: 2× NVIDIA H100 SXM5 80GB (Node: `gpu-10-58`, NVLink interconnect)
* **SLURM Job ID**: `37866768`
* **Global Batch Size**: 64 (Micro-batch 16 × 2 GPUs × 2 Gradient Accumulation Steps)
* **Optimizer**: AdamW (Learning Rate: `2.00e-04`, Cosine Annealing, Weight Decay: `1e-4`)
* **Epoch Length**: 3,125 batches/epoch per GPU (100,000 samples)
* **W&B Live Monitor**: [https://wandb.ai/lion395-university-of-california-davis/cowpea-vlm-scaffold-dit/runs/iw0iexb0](https://wandb.ai/lion395-university-of-california-davis/cowpea-vlm-scaffold-dit/runs/iw0iexb0)
