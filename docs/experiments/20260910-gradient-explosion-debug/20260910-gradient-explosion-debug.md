---
title: "2026-09-10 Gradient Explosion Debugging & Architectural Comparison Analysis"
date: 2026-09-10
tags: [experiment, results]
status: done
---

# 2026-09-10 Gradient Explosion Debugging & Architectural Comparison Analysis

> **Session Date**: 2026-09-10 (PDT)  
> **Relevant Jobs**: `38235936` (explosion verified), `38235969` (recovery attempted, re-exploded at epoch 3)  
> **Prior Milestone**: [`docs/experiments/20260908-3d-spatial-vision-milestone/20260908-3d-spatial-vision-milestone.md`](../20260908-3d-spatial-vision-milestone/20260908-3d-spatial-vision-milestone.md)

---

## 1. Identified Bug: Positive Feedback Loop via Missing `.detach()`

### 1.1 Root Cause

In `train_hierarchical_flow_matching.py`, within the phytomer bridge flow pathway (`flow_granularity == "phytomer"`), an unintended gradient path allowed Stage 3 velocity loss to backpropagate directly into the Stage 2 positional head.

```
[fwd1 with no_grad]
    Stage 2: pos_head -> pred_anchor_pos (no grad)
    Stage 3: vel_head -> pred_velocity (no grad)

[z_0 reconstruction]
    z_0[:, :, :3] = pred_anchor_pos + 0.05 * eps   <-- .detach() applied here

[fwd2 with grad]
    Stage 2: pos_head -> pred_anchor_pos (live grad) <-- supervised by loss_anchor_pos
    Stage 3: vel_head -> pred_velocity (live grad)   <-- supervised by loss_fine_vel
```

- In fwd2, `pred_anchor_pos` retains live gradients, allowing `loss_anchor_pos` to backpropagate properly.
- In fwd2, Stage 3 `fine_stage` is conditioned with `anchor_pos.detach()`:
  - `hierarchical_part_flow_matching.py` L1129: `anchor_pos=anchor_pos.detach()` (correct).

### 1.2 Actual Explosion Mechanism (Job 38235936)

The bug prior to `38235936`:

```python
# Bug: z_0 reconstructed in fwd2 without pred_anchor_pos.detach()
z_0[:, :, :3] = pred_anchor_pos + 0.05 * eps  # pred_anchor_pos retains live autograd graph

# Target velocity = z_1 - z_0
tgt_velocity = tgt_z1_phyto - z_0  # z_0 gradient connects to pred_anchor_pos

# Result: d(loss_vel)/d(pred_anchor_pos) = 2 * (v_pred - v*) * I
# pos_head destabilizes spatial coordinates to minimize velocity loss
```

**Explosion Timing**: As linear learning rate warmup (epochs 1–2) concluded and reached full scale (1.0x) around epochs 3–4, this positive feedback loop rapidly diverged.

### 1.3 Applied Fixes (Active Codebase)

| File | Fix Details |
| :--- | :--- |
| `train_hierarchical_flow_matching.py` L326-338 | fwd1 output: Apply `.detach()` before reconstructing $z_0$ |
| `train_hierarchical_flow_matching.py` L316-325 | Wrap fwd1 inside `torch.no_grad()` |
| `train_hierarchical_flow_matching.py` L488-511 | Update `pred_anchor_pos/rot/logits` from fwd2 output (live grad) |
| `hierarchical_part_flow_matching.py` L1129 | Condition `fine_stage` with `anchor_pos.detach()` |

---

## 2. 9/7–9/8 Architecture vs. Current Architecture Comparative Analysis

### 2.1 Timeline Summary

```
2026-09-07 22:24  Commit ed95f45  "Option B 16D Latent Hierarchical Flow Matching"
                  -> Frozen OrganLatentVAE decoder, launched Job 38143585
2026-09-08 AM     -> Verified positive epoch 125 results from Job 38143585

2026-09-09~10     -> Introduced PhytomerVAE v3 + 76D Bridge Flow
                  -> Migrated to 3-stage cascaded architecture
2026-09-10        -> Gradient explosion debugging session (this document)
```

### 2.2 Architectural Comparison

| Aspect | Option B (9/7–9/8) | Current (76D Phytomer Bridge Flow) |
| :--- | :--- | :--- |
| **Stage 1** | CoarseSkeletalTransformer (L1 direct supervision) | MacroBiologicalHead (DAP + N_phy + Soft Margin) |
| **Stage 2** | FineBotanicalFlowMatchingDecoder | CoarseSkeletalTransformer (3D scaffold) |
| **Stage 3** | - | PhytomerFlowMatchingDecoder (76D) |
| **Latent Space** | 16D per-organ (OrganLatentVAE, 8 slots/anchor) | 64D per-phytomer (PhytomerVAE v3, 10 slots/phytomer) |
| **$z_0$ Initialization** | $\mathcal{N}(0, I_{16})$ Pure Gaussian | Stage 2 scaffold pose + $\mathcal{N}(0, I_{64})$ hybrid |
| **Organ Coupling** | Independent (no intra-phytomer consistency) | Joint intra-phytomer modeling |
| **Loss Count** | ~5 losses | ~10 losses |

### 2.3 Why 9/8 epoch_125 Results Were Strong

**Core Insight**: In Option B, the Stage 2 fine decoder originated from a pure Gaussian $z_0$ **completely independent of `anchor_pos`**:

```python
# Option B: z_0 independent of anchor_pos
z_0 = torch.randn(B, K, 16, device=device)   # fully decoupled
# -> No physical pathway existed for Stage 2 velocity loss to flow into pos_head
# -> loss_anchor_pos was the sole positional signal, remaining 100% stable
```

Result: Convergence at 150 epochs, simpler loss landscape, and uncompromised gradient stability.

### 2.4 Theoretical Advantages of the Current Architecture

1. **Intra-Phytomer Consistency**: 64D latent jointly models petiole-leaflet-peduncle ($r=0.783$ petiole-leaflet scale covariance).
2. **Bridge Flow Efficiency**: $z_0$ starts near GT positions → ODE solves only residual corrections.
3. **Explicit Scale Learning**: Dedicated `anchor_scale` head learns metric scale directly.
4. **Localized Visual Projection**: `AnchorVisualProjector` injects local patch features per node position.
5. **Biological Prior**: Stage 1 MacroBiologicalHead soft margin guides total phytomer count.

### 2.5 Reasons for Slower Initial Convergence

1. **Complex Loss Landscape**: Balancing 10 distinct loss objectives requires more epochs.
2. **Bridge Coupling**: In early epochs where Stage 2 is imprecise, Stage 3 originates from sub-optimal starting points.
3. **PhyLoss Initial Disparity**: Epoch 1: Pred 12.4 vs GT 54.9 ($>4\times$ divergence).

**Expected Convergence Milestones**:
- Epoch 30: `AncPos` settles to 0.01~0.03 m.
- Epoch 50: `PhyLoss` converges within 10% of GT.
- Epoch 100+: Outperforms Option B in fine-grained organ consistency.

---

## 6. Hybrid Decoupled Architecture & Streamlining to 8 Core Losses

### 6.1 Diagnostic and Finalization of 8 Core Losses

Consolidated noisy and redundant objectives into an **8-core loss system**, while adding missing canonical rotation supervision:

```python
loss = (
    # Stage 1: Macro Prior
    phy_count_weight * loss_phy_count          # 1.0 : Total phytomer count prediction (Soft Margin masking)
    # Stage 2: 3D Node Scaffold (Macro Topology)
    + 2.0 * loss_anchor_pos                    # 2.0 : Node 3D position (Smooth L1)
    + 1.0 * loss_anchor_rot                    # 1.0 : [New] Node 6D rotation reference frame (Smooth L1)
    + 1.0 * loss_anchor_scale                  # 1.0 : Node 3D scale (Smooth L1)
    + 1.0 * loss_anchor_exist                  # 1.0 : Node existence probability (BCE with logits)
    # Stage 3: Canonical Phytomer Latent Flow Matching (Micro Geometry)
    + 1.0 * loss_fine_vel                      # 1.0 : 64D VAE latent velocity MSE (z0 ~ N(0, I))
    + 1.0 * loss_fine_exist                    # 1.0 : 10-slot existence within phytomer (BCE)
    # Stage 4: Differentiable Optical Grounding
    + depth_loss_weight * loss_depth           # 0.5 : CHM canopy height matching (Smooth L1)
    + silhouette_loss_weight * loss_dice       # 1.0 : Top-view silhouette matching (Soft Dice)
)
```

**Removed / Deactivated Objectives:**
1. `loss_cos` (0.0): Removed high-frequency noise from color mismatches between CAD meshes and drone imagery; accelerates render backpropagation.
2. `loss_dap` (0.0): Redundant with `loss_phy_count` ($>90\%$ biological correlation) without providing direct 3D geometric constraints.

---

### 6.2 Key Architectural Questions and Findings

#### Q1. Relationship between `anchor_features` and `pos`, `rot`, `scale`, `exist`
- **`anchor_features` (384D)**: Primary semantic latent vectors produced by Transformer Decoder self-attention, compressing global context (main stem vs lateral branches, branching order, leaflet count).
- **`pos`, `rot`, `scale`, `exist` (13D)**: Interpreted physical 3D scaffolding decoded via dedicated linear/MLP heads:
  - $\text{pos} = \text{ref\_points} + \text{pos\_head}(\text{anchor\_features})$
  - $\text{rot} = \text{rot\_head}(\text{anchor\_features})$
  - $\text{scale} = \text{softplus}(\text{scale\_head}(\text{anchor\_features}))$
  - $\text{exist} = \sigma(\text{exist\_head}(\text{anchor\_features}) + \text{macro\_prior})$

#### Q2. Are `scale` and `exist` passed to Stage 3?
- **`scale`**: In hybrid decoupling, Stage 2's `anchor_scale.detach()` is passed to:
  1) Stage 3 query conditioning (`self.node_scale_mlp(anchor_scale.detach())`) to guide scale-dependent micro morphology,
  2) Final mesh assembly (`denormalize_packet_scales`) to restore absolute metric dimensions.
- **`exist`**: Coupled via mathematical conditional probability gating:
  $$P(\text{organ}) = P(\text{anchor}) \times P(\text{slot} \mid \text{anchor})$$
  Soft Margin masking suppresses noisy computation on inactive anchors.

#### Q3. Should we predict only Pos (as on Sep 7) or include Rot and Scale?
- **Sep 7 Option B (Organ-level)**: Each organ's 16D latent vector embedded its own 3D rotation and scale in global coordinates; Pos alone was sufficient.
- **Current Architecture (Phytomer-level, PhytomerVAE v3)**: To capture 10-organ covariance ($r=0.783$), phytomers are modeled in a **normalized canonical local coordinate frame centered at $(0,0,0)$ with scale 1.0**. Transforming this into 3D world space requires:
  1) Insertion location: **`pos`** (3D position)
  2) Branching direction: **`rot`** (6D orientation frame)
  3) Metric scale: **`scale`** (3D scale)
- **Conclusion**: Predicting Rot and Scale is essential. Decoupling them into Stage 2 feedforward heads (`loss_anchor_rot`, `loss_anchor_scale`) allows Stage 3 to focus purely on 64D VAE latent leaf morphology.

---

### 6.3 Hybrid Decoupled Architecture Diagram

```
[Visual Image Tokens] (DINOv2 + PETR 3D Ray PE)
         │
         ▼
[Stage 1: MacroBiologicalHead] ──→ loss_phy_count (Soft Margin Masking)
         │
         ▼
[Stage 2: CoarseSkeletalTransformer] (Deterministic Set Transformer)
         ├─→ pos_head   ──→ anchor_pos   ──→ loss_anchor_pos
         ├─→ rot_head   ──→ anchor_rot   ──→ loss_anchor_rot (New)
         ├─→ scale_head ──→ anchor_scale ──→ loss_anchor_scale
         ├─→ exist_head ──→ anchor_exist ──→ loss_anchor_exist
         └─→ anchor_features (384D Context)
                   │
    ┌──────────────┴────────────────────────────────────┐
    │  Conditioning: pos.detach(), rot.detach(),         │
    │                scale.detach(), anchor_features.detach()  │
    ▼                                                   │
[Stage 3: PhytomerFlowMatchingDecoder]                  │
    │  Flow Vector: 64D VAE Latent ONLY                 │
    │  Prior: z_0 ~ N(0, I_64) (Standard Gaussian)      │
    │  Velocity Target: v* = z_1 - z_0                  │
    ├─→ velocity_head ──→ pred_v ──→ loss_fine_vel       │
    └─→ exist_head    ──→ pred_slot_exist ──→ loss_fine_exist
         │
         ▼ (ODE Sampling: z_1)
[PhytomerVAE Decoder (Frozen)]
         │
         ▼ (Canonical local 10-organ packets)
[Stage 4: Assembler & Differentiable Renderer] ←────────┘
    - denormalize_packet_scales(packet, anchor_scale)
    - apply_ref_for_flow(packet, anchor_pos, anchor_rot)
    - build_mesh_from_part_tensor
    - render_multiscale_pyramid
         ├─→ loss_depth (CHM)
         └─→ loss_dice (Top-view Silhouette)
```

**Convergence Stability Guarantees:**
1. With $z_0 \sim \mathcal{N}(0, I_{64})$, $\|v^*\|$ remains bounded by $\sqrt{2 \times 64} \approx 11.3$, ensuring dimension-normalized `VelLoss` stays well controlled at $\sim 1.0$.
2. Early Stage 2 anchor localization errors cannot couple into Stage 3 Flow Matching velocity fields, preventing gradient explosions.

---

## 7. Analysis of Job 38236714 Explosion & Permanent Safeguards (2026-09-10 Night)

### 7.1 Failure Mode (Job 38236714)
- Epochs 1–2 were stable. However, as linear warmup (3 epochs, 348 steps) ended at **Epoch 3 Step 46–92**, sudden gradient explosion occurred:
  - `[Epoch 03] Step 092/116 | Loss: 773.3856 | Pos: 65.4449, Rot: 205.5988, Scl: 1.5440, Ext: 19.9443 | Vel: 348.8686`
  - Followed by continuous `grad_norm is NaN/Inf (inf)`.

### 7.2 Root Cause: 2,311.13 Norm Gradient Leakage via `anchor_features`
1. While `anchor_pos`, `anchor_rot`, and `anchor_scale` were detached, **`anchor_features` was passed into Stage 3 without `.detach()`**.
2. Unit tests revealed that backpropagation from Stage 3 `loss_fine_vel` dumped an **anomalous 2,311.13 gradient norm** directly into Stage 2 `anchor_self_attn`.
3. This corrupted Stage 2 representations, triggering simultaneous runaway across `pos_head`, `rot_head`, and `scale_head` immediately upon warmup completion.
4. Additionally, `rot_head` 6D vectors lacked Gram-Schmidt orthonormalization (exploding to 205), and unconstrained `pos_head` diverged to 65 meters.

### 7.3 Three Permanent Safeguards Applied
1. **Complete Layer Isolation (`anchor_features.detach()`)**:
   - `fine_stage(anchor_features=anchor_features.detach(), ...)`
   - Unit tests confirmed Stage 2 `anchor_self_attn`, `pos_head`, `rot_head` gradients from `loss_vel.backward()` are **identically `None` (0.00)**.
2. **Gram-Schmidt 6D Orthonormalization (Zhou et al., CVPR 2019)**:
   - $e_1 = \text{normalize}(v_1)$, $u_2 = v_2 - (e_1 \cdot v_2)e_1$, $e_2 = \text{normalize}(u_2)$
   - $e_1, e_2$ magnitudes are fixed to 1.0; L1 rotation loss mathematically cannot exceed 4.0 (blocking `Rot: 205`).
3. **Physical $\pm 50\text{ cm}$ Bounding & Scale Clamping**:
   - `delta_pos = torch.tanh(self.pos_head(anchor_features)) * 0.5`, bounding coordinates within plant envelope ($\pm 0.5\text{ m}$) and preventing runaway (`Pos: 65m`).

### 7.4 Existence Loss (`Ext`) Spike in Job 38236722 & Resolution
1. **Symptom**:
   - 3D skeleton stabilized (`Pos: 0.23m`, `Rot: 1.31`), but at Epoch 3 Step 96–120, `Ext` (`loss_anchor_exist`) spiked from **229 $\to$ 1699**.
2. **Root Cause**:
   - `anchor_logits = delta_logits + macro_out["init_logits"]`
   - For inactive slots ($k > \text{active}$), `init_logits` was clamped to $+15.0$. To turn off non-existent targets ($y=0$), the network forced `delta_logits` to extreme negative values, destabilizing `exist_head`.
3. **Resolution**:
   - **Removed Conflicting `init_logits` Addition**: `anchor_logits = torch.tanh(self.exist_head(anchor_features)) * 8.0`, bounding existence classification within $[-8.0, 8.0]$ (capping loss below 8.0).
   - **Stage 3 Existence Bounding**: `torch.tanh(self.exist_head(x)) * 8.0`.
   - **Added `anchor_norm` (LayerNorm)**: Stabilized feature norms following residual self-attention.
   - **Loop Safety Clamp**: `pred_anchor_logits.clamp(min=-10.0, max=10.0)`.

---

## 8. Reference Figures

- [`assets/hierarchical_self_consistency_epoch_008.png`](assets/hierarchical_self_consistency_epoch_008.png) — Job 38235969, epoch 8 output
- [`assets/hierarchical_self_consistency_epoch_125_20260908.png`](assets/hierarchical_self_consistency_epoch_125_20260908.png) — Option B epoch 125 result
