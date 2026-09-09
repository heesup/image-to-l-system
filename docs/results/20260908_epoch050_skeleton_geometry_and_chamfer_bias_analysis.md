# Epoch 050 Botanical Skeleton Geometry & Metric Analysis

**Date:** 2026-09-08 (~18:40 PDT)  
**Evaluated Artifact:** `docs/results/assets/hierarchical_self_consistency_epoch_050.png`  
**Checkpoint:** `diffusion_based/checkpoints/hierarchical_latent_fm/hierarchical_fm_epoch_050.pt`  
**Author:** Antigravity Autonomous Agent (Pair programming with Heesup Yun)  

---

## 1. Executive Summary

At Epoch 50 of the 3-Stage Cascaded Hierarchical Flow Matching model (`hierarchical_fm` Job `38146809`), training logs showed seemingly exceptional geometric metrics:
- **`AncPosLoss`**: `0.0089` (~0.009, Smooth L1 in normalized coordinate space)
- **`Node RMSE`**: `1.7 cm` (One-way Chamfer nearest-neighbor distance)
- **`Depth MAE`**: `8.25 cm` (Down from 15.73 cm at Epoch 25)
- **`Silhouette IoU`**: `9.7%` (Sampled on challenging young/intermediate canopy)

However, visual inspection of Column 6 (**"3D Botanical Skeleton: Stem Trees & Nodes"**) in the diagnostic panel showed that the reconstructed plant skeleton appeared distorted, clustered at the center, or had stem vectors extending into unnatural orientations.

This technical report documents the mathematical and architectural reasons for this apparent discrepancy, distinguishing between **metric artifacts**, **loss objectives**, and **botanical structural convergence dynamics**.

---

## 2. Root Cause Analysis: Metric vs. Visual Discrepancy

### 2.1 Anchor Position Loss vs. Stem Vector Geometry

- **What `AncPosLoss` Measures:**  
  `AncPosLoss` strictly evaluates the 3D translation offset of coarse anchor query points:
  $$\mathcal{L}_{\text{anchor\_pos}} = \frac{1}{|\mathcal{M}_{\text{anc}}|} \sum_{(i, j) \in \mathcal{M}_{\text{anc}}} \text{SmoothL1}(\mathbf{p}_{\text{anc}}^{(i)}, \mathbf{t}_{\text{anc}}^{(j)})$$
  Because positions are normalized by `BASE_SCALE = 20.0` (or metric units), a loss of `0.0089` indicates that the coarse insertion points (nodes) are positioned within ~1–2 cm of ground truth cluster centroids.

- **What Column 6 Plots:**  
  Column 6 does **not** simply connect anchor points. It plots the physical internode segments decoded from Stage 3 Fine Flow Matching (`Type 3: Internode`):
  $$\mathbf{x}_{\text{tip}} = \mathbf{x}_{\text{base}} + \mathbf{R}_{6\text{D}}[:, 1] \times L_{\text{internode}}$$
  where:
  - $\mathbf{x}_{\text{base}}$: 3D base coordinate of the internode.
  - $\mathbf{R}_{6\text{D}}[:, 1]$: Forward axis of the 6D continuous rotation matrix.
  - $L_{\text{internode}}$: Stem physical length in centimeters.

**Key Finding:** Even if node center coordinates ($\mathbf{x}_{\text{base}}$) are accurate within 1.7 cm, a 20°–30° error in the 6D orientation matrix $\mathbf{R}_{6\text{D}}$ causes stem segments to project into the ground or sideways into empty air. (Observed in DAP 17, Row 1).

---

### 2.2 Metric Bias of One-Way Chamfer Distance (`Node RMSE`)

The evaluation script computes `Node RMSE` via nearest-neighbor distance:
```python
dists = np.linalg.norm(pred_nodes[:, None, :] - gt_nodes[None, :, :], axis=-1).min(axis=1)
node_rmse_cm = float(np.sqrt(np.mean(dists ** 2)) * 100.0)
```
- This metric is **pred $\to$ GT one-way**: for every predicted node, it finds the distance to the closest GT node.
- **Mode Collapse / Central Clustering Bias:**  
  In mature plants (e.g., DAP 55, Row 3), GT nodes radiate outward across a 25 cm radius. At Epoch 50, predicted nodes are still concentrated within a 5–10 cm central cluster.
  Since every central predicted node is indeed close to at least one central GT node, the average distance remains tiny (**2.3 cm for DAP 55, 1.7 cm mean overall**).
  However, the outer branches are completely unpopulated (false negatives), resulting in an incomplete tree skeleton that looks collapsed visually.

---

### 2.3 Organ Classification Accuracy & Missing Kinematic Chain

1. **Classification Accuracy (`ClsAcc: ~66%`):**  
   At Epoch 50, organ classification accuracy across active slots is approximately 65–66%. When a petiole or leaflet slot is misclassified as an internode (Type 3), spurious stem segments appear in the canopy. Conversely, when true internodes are misclassified, stem continuity is severed.
2. **Independent Slot Flow Matching vs. Explicit Tree Constraints:**  
   Unlike procedural L-systems that enforce topological parent-child connectivity ($\text{Base}_{k+1} \equiv \text{Tip}_k$), the current Flow Matching operates over slot tokens. Exact end-to-end alignment requires further training so that intra-phytomer self-attention learns to snap internode endpoints together.

---

## 3. Training Convergence Trajectory & Next Expectations

In hierarchical botanical generation, convergence proceeds in distinct temporal phases:
1. **Phase 1: Macro Anchor Positioning (Epochs 1–40) [COMPLETE]**  
   Coarse plant size (DAP), organ budget, and node cluster centroids settle to <2 cm RMSE.
2. **Phase 2: Fine Flow Orientation & Class Separation (Epochs 40–150) [CURRENT STAGE]**  
   `ClsAcc` rises from 66% $\to$ 85%+, 6D continuous rotation matrices align with plant gravitropic/phototropic directions, and outward branch expansion begins.
3. **Phase 3: Differentiable Photometric Refinement (Epochs 150–500) [UPCOMING]**  
   Silhouette Dice and CHM Depth loss tighten canopy margins, resolving leaf pitch, stem curvature, and eliminating gap errors.

---

## 4. Takeaway for Agent Takeover

- Do **not** alter the anchor loss weights; `AncPosLoss: 0.009` is performing as designed.
- Do **not** interpret the low Node RMSE as full geometric reconstruction; always cross-reference Column 5 (Oblique 3D Mesh Composite) and Column 6 (3D Botanical Skeleton).
- If evaluating mature plants, consider reporting bidirectional Chamfer distance ($\text{Pred} \leftrightarrow \text{GT}$) to penalize missing outer branches.
