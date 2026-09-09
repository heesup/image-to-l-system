# Latent Hierarchical Flow Matching (Option B) Architecture & Transition Specification

**Author**: Google DeepMind Antigravity Pair Programmer  
**Date**: September 7, 2026  
**Status**: Approved for Implementation  
**Target Repository**: `image-to-l-system`

---

## 1. Executive Summary & Motivation

In previous iterations, the Botanical Flow Matching pipeline operated directly on raw continuous geometry parameters (13D: 3D base coordinates, 6D continuous rotation, 3D scale dimensions, 1D gravitropic curvature). While conceptually straightforward, empirical training revealed severe failure modes:

1. **The 150x Gradient Scale Disparity**:
   - Real-world plant organ dimensions span multiple orders of magnitude. A flower peduncle or young petiole has a radius of $r \approx 0.00225\text{ m}$ ($2.25\text{ mm}$), while main stem internodes and petioles reach lengths of $l \approx 0.35\text{ m}$ ($35\text{ cm}$).
   - Direct regression losses placed $17.5$ on length and only $0.1125$ on radius.
   - Consequently, optimization collapsed fine stalks to zero thickness ($2\text{ }\mu\text{m}$, vanishing on camera) and caused leaflet surfaces to degenerate into spiky, needle-like "porcupine" artifacts.
2. **Dimension Heterogeneity & Boundary Violations**:
   - Raw geometric parameters mix unconstrained coordinates ($x, y, z \in \mathbb{R}$), $SO(3)$ rotation manifolds ($r \in \mathbb{R}^6$), strictly positive scales ($s_x, s_y, s_z > 0$), and bounded curvature ($\kappa \in [-60, 60]^\circ/\text{m}$).
   - Direct flow ODE integration frequently wandered into non-physical parameter regimes (e.g., negative leaf scales or self-intersecting rotation representations).

To permanently resolve these issues, we transition to **Option B: Latent Hierarchical Flow Matching**. The flow matching velocity field is learned over a regularized 16-dimensional spherical standard Gaussian latent manifold ($z \in \mathbb{R}^{16} \sim \mathcal{N}(0, I)$) pre-trained and verified via `OrganLatentVAE`.

---

## 2. Addressing the Organ One-Hot Vector & Compatibility

> **User Question**:
> *"After converting to the latent space, there is no explicit plant organ one-hot vector. Can the existing downstream algorithms still be used identically?"*

### **Answer: Yes, 100% Compatible and Mathematically Unified.**

Here is the exact operational mechanism:

```
[GT Organ in Dataset]
  (26D: One-hot Organ Type 13D + Continuous Geometry 13D)
         │
         ▼
[Pretrained OrganLatentVAE Encoder]
         │
         ▼
[Unified 16D Latent Space z ~ N(0, I)]
  - Seamlessly encodes BOTH semantic organ category and physical shape.
  - Latent coordinates naturally form separable botanical clusters.
         │
         ▼ (Stage 2 Flow Matching ODE: Prior z_0 -> Predicted z_1)
[Pretrained OrganLatentVAE Decoder (Frozen, Differentiable)]
         │
         ├────────────────────────────────────────┬────────────────────────────────────────┐
         ▼                                        ▼                                        ▼
  `pred_cls_logits`                         `recon_26d`                           `part_14d` (Canonical)
  (13 classes, 100.0% accuracy)            (Full 26D vector)                 [cls, base, rot, scale, curv]
         │                                        │                                        │
         ▼                                        ▼                                        ▼
  [Hungarian Matcher &                     [PartArrayDataset &                 [HeliosPyTorchRenderer &
   Empty Slot Pruning]                      Evaluation Metrics]                 3D Differentiable Mesh]
```

### Key Clarifications:
1. **The VAE Encoder Consumes the One-Hot Vector**:
   - The input to `OrganLatentVAE.encode()` is the full 26-dimensional node layout: `[one-hot organ type (13), base*20 (3), rot6d (6), scale*50 (3), curv*60 (1)]`.
   - The encoder maps this heterogeneous 26D representation into $z \in \mathbb{R}^{16}$.
2. **The VAE Decoder Restores the One-Hot / Class Logits with 100% Accuracy**:
   - `OrganLatentVAE.decode(z)` has dedicated linear prediction heads:
     - `head_cls`: produces 13-class logits with **100.0% empirical accuracy** on test organs.
     - `head_base`: 3D coordinates.
     - `head_rot`: $SO(3)$ continuous representation.
     - `head_scale`: guaranteed positive via $\text{Softplus}(s) + 10^{-4}$.
     - `head_curv`: 1D gravitropic curvature.
3. **Downstream Pipeline Invariance**:
   - **Stage 1 (Coarse Skeletal Transformer)**: Unchanged. Operates on 3D anchor positions ($x, y, z$) and phytomer cluster existence.
   - **Stage 2 (Fine Decoder)**: Predicts velocity $v_\theta \in \mathbb{R}^{16}$. Clean 1-step prediction $\hat{z}_1 = z_t + (1-t)v_\theta$ is passed to `vae.decode()`.
   - **Hungarian Matcher**: Stage 1 matches coarse anchors by 3D position distance; Stage 2 matches fine intra-cluster slots using latent distance $\|\hat{z}_1 - z_{\text{tgt}}\|_1$ and class log-likelihood.
   - **Helios Differentiable Renderer**: Takes the exact 14D part tensor and soft class probabilities produced by `vae.decode()`. Autograd gradients flow backward smoothly through the frozen VAE decoder weights into $v_\theta$.
   - **L-System / XML Reassembly**: Consumes the canonical 14D/26D reconstructed representation identically to previous checkpoints.

---

## 3. Empirical Verification of Pretrained OrganLatentVAE

Before proceeding with pipeline integration, `OrganLatentVAE` was trained and rigorously benchmarked on 210,452 physical plant organs from the cowpea dataset using [`diffusion_based/eval/benchmark_organ_vae_roundtrip.py`](file:///home/lion397/codes/image-to-l-system/diffusion_based/eval/benchmark_organ_vae_roundtrip.py):

| Metric | Raw Geometry (Previous) | OrganLatentVAE (16D Latent) |
| :--- | :---: | :---: |
| **Top-View Silhouette IoU** | 74.2% (spiky) | **91.94%** (DAP 51: **97.8%**, DAP 31: **95.1%**) |
| **3D Depth MAE** | 20.4 cm | **1.39 cm** |
| **Organ Classification Accuracy** | 89.1% | **100.0%** |
| **Scale MAE** | 1.82 cm | **0.24 cm** (Radius MAE: **0.04 cm**) |
| **Flower Peduncle Rendering** | Disappeared ($r \to 2\,\mu\text{m}$) | **Fully Restored** ($r = 2.25\text{ mm}$ stalks visible) |
| **Latent Distribution** | N/A (Arbitrary scale) | $\mu \in [-0.08, +0.06], \sigma \in [0.89, 1.08]$ (Standard Gaussian) |

---

## 4. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Vision Perception
        Img["Drone Imagery (RGB + CHM Depth, 4ch)"] --> ViT["ViT Image Encoder (384D tokens)"]
    end

    subgraph Stage 1: Coarse Phytomer Skeleton
        ViT --> S1["Coarse Skeletal Transformer (Matryoshka Queries K <= 512)"]
        S1 --> AncPos["Predicted Anchor 3D Pos (B, K, 3)"]
        S1 --> AncFeat["Anchor Latent Features (B, K, 384)"]
        S1 --> AncExist["Anchor Existence Logits (B, K, 1)"]
        S1 --> DAP["Auxiliary DAP Regressor (B, 1)"]
    end

    subgraph Stage 2: Fine Latent Botanical Flow
        AncFeat --> S2["Fine Botanical Flow Matching Decoder"]
        ViT --> S2
        Prior["Prior Noise z_0 ~ N(0, I) in R^16"] -->|Interpolate z_t| S2
        S2 -->|Predicts Velocity v_theta| Vel["Velocity Field v_theta in R^16"]
        Vel --> CleanZ["Analytical Clean Prediction z_hat_1 = z_t + (1-t)v_theta"]
    end

    subgraph Differentiable Optical Grounding
        CleanZ --> VAE["Frozen OrganLatentVAE Decoder"]
        VAE --> Geom["Reconstructed 14D Part Tensor (Base, Rot, Scale, Curv)"]
        VAE --> Probs["Organ Class Probabilities (13 classes)"]
        Geom --> Mesh["Helios Plant Geometry Builder (Differentiable Mesh)"]
        Probs --> Mesh
        Mesh --> Rend["Helios 4-Scale Multiscale Pyramid Renderer (1x, 2x, 4x, 8x)"]
        Rend --> LossRend["Multiscale Depth SmoothL1 + Silhouette Dice + Cosine Color Loss"]
    end

    subgraph Backpropagation
        LossFM["Flow Matching Velocity MSE Loss ||v_theta - (z_1 - z_0)||^2"] --> Optim["AdamW Optimizer"]
        LossRend -.->|Autograd via Frozen VAE Decoder| Vel
        LossRend --> Optim
    end
```

---

## 5. Detailed Component Changes

### 5.1. `diffusion_based/models/organ_latent_vae.py`
Add unified batch conversion utilities:
- `decode_to_part_tensor(z: torch.Tensor, existence: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]`:
  - Decodes $(B, N, 16)$ latent vectors.
  - Unpacks base position (scaled by $1/20$), rotation (6D), scale (scaled by $1/50$), and curvature (scaled by $60$).
  - Returns `(part_14d, organ_probs)` ready for instant ingestion by `HeliosPlantGeometryBuilder`.

### 5.2. `diffusion_based/models/hierarchical_part_flow_matching.py`
- Modify `node_dim` parameter:
  - Default `node_dim: int = 16` (formerly `GEOM_NODE_DIM = 13`).
- `FineBotanicalFlowMatchingDecoder`:
  - `geom_proj`: `nn.Linear(16, embed_dim)`.
  - `velocity_head`: `nn.Linear(embed_dim, 16)`.
  - `geom_context_proj`: `nn.Linear(16, geom_feat_dim)`.
- `sample_ode`:
  - Initialize prior: $x_0 \sim \mathcal{N}(0, I)^{B \times N_{\text{fine}} \times 16}$.
  - Integrate 2nd-Order Heun Predictor-Corrector ODE across $T=20$ steps:
    $$\hat{z}_{k+1} = z_k + \frac{\Delta t}{2} \left( v_\theta(z_k, t_k) + v_\theta(z_k + \Delta t \, v_\theta(z_k, t_k), t_{k+1}) \right)$$
  - Pass final $\hat{z}_1$ to `vae.decode()` to obtain clean 3D physical geometries without needle artifacts.

### 5.3. `diffusion_based/training/hierarchical_hungarian_matcher.py`
- Adapt Stage 2 fine intra-cluster matching cost:
  $$C_{ij} = \lambda_{\text{cls}} \left( -\log P_i(c_j) \right) + \lambda_{\text{latent}} \|\hat{z}_{1, i} - z_{\text{tgt}, j}\|_1$$
  where $z_{\text{tgt}, j} = \text{vae.encode}(x_j)[0] \in \mathbb{R}^{16}$.
- Provides sub-millisecond assignment that matches organs based on both geometric topology and latent botanical identity.

### 5.4. `diffusion_based/training/train_hierarchical_flow_matching.py`
- Add `--organ_vae_checkpoint` CLI argument (defaults to `diffusion_based/checkpoints/organ_vae/organ_latent_vae_best.pt`).
- Initialize and freeze `vae = OrganLatentVAE(latent_dim=16, hidden_dim=256).to(device)`. Set `vae.eval()` and `requires_grad=False`.
- For each batch:
  1. Encode active physical organs (classes $\ge 3$) to $z_1 = \text{vae.encode}(nodes)[0]$.
  2. Sample prior noise $z_0 \sim \mathcal{N}(0, I)$.
  3. Interpolate $z_t = (1 - t) z_0 + t z_1$, target velocity $v^* = z_1 - z_0$.
  4. Predict velocity $v_\theta = \text{model}(z_t, t, \dots)$.
  5. Compute 1-Step analytical clean latent: $\hat{z}_1 = z_t + (1 - t) v_\theta$.
  6. Decode $\hat{z}_1$ via frozen VAE, assemble mesh, and render 4-scale pyramid.
  7. Compute multi-scale Depth, Soft Dice, and Cosine losses, backpropagating through frozen VAE weights directly into $v_\theta$.

### 5.5. `slurm_scripts/train_hierarchical_flow_matching.sh`
- Add `--organ_vae_checkpoint diffusion_based/checkpoints/organ_vae/organ_latent_vae_best.pt`.
- Add `--node_dim 16`.
- Direct checkpoint destination to `diffusion_based/checkpoints/hierarchical_latent_fm/`.

---

## 6. Execution & Verification Steps

1. **Step 1: Code Implementation**:
   - Update `organ_latent_vae.py` with batch decoding helpers.
   - Update `hierarchical_part_flow_matching.py` for 16D latents and sample ODE decoding.
   - Update `hierarchical_hungarian_matcher.py` for 16D latent distance.
   - Update `train_hierarchical_flow_matching.py` with frozen VAE grounding.
   - Update SLURM submission script.
2. **Step 2: Local Verification (`scratch/test_latent_flow_step.py`)**:
   - Run 1-batch end-to-end forward/backward test on local TITAN RTX GPU.
   - Confirm finite gradients on all Stage 1 and Stage 2 parameters.
   - Confirm `sample_ode` generates valid, non-spiky organ parameters.
3. **Step 3: SLURM Launch**:
   - Cancel old 13D training job (`scancel 38143159`).
   - Submit new 16D Latent Flow Matching training job via `sbatch slurm_scripts/train_hierarchical_flow_matching.sh`.
   - Verify logs for loss progression (`VelLoss`, `AncPosLoss`, `DepthLoss`, `DiceLoss`, `CosLoss`).
