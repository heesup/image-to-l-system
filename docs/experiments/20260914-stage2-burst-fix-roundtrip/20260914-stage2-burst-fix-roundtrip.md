---
title: "2026-09-14 Stage 2 Gradient Burst Resolution & Dataset Plant Helios Roundtrip Report"
date: 2026-09-14
tags: [experiment, results]
status: done
---

# 2026-09-14 Stage 2 Gradient Burst Resolution & Dataset Plant Helios Roundtrip Report

> **Session Date**: 2026-09-13 ~ 2026-09-14 (PDT)  
> **Relevant Runs**: Local v9 run (`outputs/logs/local_v9_run2.log` → `local_v9_run2b.log`, checkpoint `outputs/checkpoints/hierarchical_fm_v9_local2/`), cluster queue jobs `38249632` / `38250275` (`low` / `publicgrp`)  
> **Detailed Record**: [`docs/../../engineering/20260912-stage2-stage3-boundary/20260912-stage2-stage3-boundary.md`](../../engineering/20260912-stage2-stage3-boundary/20260912-stage2-stage3-boundary.md) §1.9.2 (burst), §2.4-2.6 (roundtrip)  
> **Commits**: `a524816` (final LayerNorm + fp32 self-attention), `ea7bb58` (ablation documentation), `0472284` (dataset plant roundtrip fix), `3fc48b7`

---

## 1. Summary

Two major issues have been successfully closed:

1. **Stage 2 Gradient Burst**: Root-caused to model architecture rather than training data. In the pre-LN coarse decoder (`nn.TransformerDecoder`, `norm_first=True`), the **absence of a final LayerNorm** caused raw residual stream values ($\sim 10^3$) to feed directly into the bf16 phytomer self-attention $Q/K/V$ projections. Adding a final LayerNorm reduced step explosions from 8/8 to 0/8 in frozen burst states. Enabled by default; the restarted v9 run experienced 0 canary triggers through Epoch 15.
2. **Helios Roundtrip**: While resolved for three exact_gt prototype plants (fig14), **dataset plant export originally achieved only 81.8 / 92.2 / 54.6%** (DAP 15 / 40 / 75 FG IoU), with the VAE adding almost zero degradation. Prototype plants did not expose the issue because they used hardcoded default angles. Remediating three flaws (chaining breaking drooping shoots, incorrect branch points, and constant leaf orientations) raised performance to **98.3 / 98.2 / 95.6%** (packet path) and **95.1 / 96.8 / 95.6%** (via VAE). The branch point fix directly applies to training targets (where 1/3 of laterals were previously corrupted).

---

## 2. Stage 2 Gradient Burst

### 2.1 Reproduction & Dissection

- Two v8 lineage runs (`38240479` lr 1e-4, `38242849` lr 5e-5) exploded at the exact same step (Epoch 27–28). Because `DistributedSampler` was seeded only by epoch, both saw identical batches.
- Reproduced on 1 GPU locally at Epoch 7 Step 2062; verified that holding weights frozen triggered explosions on any arbitrary batch $\implies$ **pure model state problem**. Ruled out data ordering, learning rate, weight decay, decoder token collapse, per-epoch eval, DDP, edge-bias masking, and decoder-only fp32.
- `FM_GRAD_PROBE=2` per-op telemetry: Gradients amplified $\sim 6000\times$ at decoder output, and another $3000\times$ traversing decoder layers. The amplification point was the phytomer self-attention input.

### 2.2 Ablation (Frozen Burst State, 8 Steps, Two Random Seeds)

| Configuration | Burst Steps |
|---|---|
| Baseline | 8/8, 8/8 |
| Self-attention fp32 only (`FM_SELFATTN_FP32`) | 1/8, 2/8 |
| Decoder final LayerNorm only (`FM_DECODER_FINAL_NORM`) | 0/8, 0/8 |
| Both (current default) | 0/8, 0/8 |

### 2.3 Implementation & Results

`plant_recon/models/hierarchical_part_flow_matching.py`: `nn.TransformerDecoder(..., norm=LayerNorm)` (`FM_DECODER_FINAL_NORM`, default 1), fp32 self-attention blocks (`FM_SELFATTN_FP32`, default 1). As a secondary guard, canary triggers skip problematic steps (`d543540`).

Restarted v9 run (v9 cache/VAE, batch 48, lr 1e-4, render loss from Epoch 11): 0 canaries over Epochs 1–15, loss descended 74 $\to$ 20.0, self-consistency IoU improved 19.5 $\to$ 31.9%, node RMSE 1.8 cm.

---

## 3. Dataset Plant Helios Roundtrip

### 3.1 Background

Investigated why the Helios column in `docs/experiments/20260914-stage2-burst-fix-roundtrip/assets/fig12_phytomer_10slot_helios_roundtrip.png` performed poorly and why 45° views looked distorted. The panel lacked GT 45° reference views, and the Helios column preceded shoot splitting fixes and stem IK. Using `eval_phytomer_10slot_assembly_views.py`, evaluated **dataset plants** (`cowpea_dap015/040/075_seed00_..._plant_0000.xml`) by plotting GT meshes alongside 10-slot assembly under identical camera perspectives (nadir, 45°, GT mesh bounds).

Assembly was confirmed accurate (RGB MAE 0.0002~0.003 vs GT mesh across both views). The 45° appearance accurately represents true plant morphology.

### 3.2 Measurements (Pre-Fix, FG IoU vs. Helios GT)

| DAP | Packet Path (VAE = identity) | Via VAE | Reference: exact_gt plant IK-only (fig14) |
|---|---|---|---|
| 15 | 81.8% | 82.6% | 95.7% (DAP 10) |
| 40 | 92.2% | 91.9% | 99.5% (DAP 50) |
| 75 | 54.6% | 56.1% | 96.7% (DAP 90) |

VAE roundtrip closely matched the packet path. The requirement "VAE roundtrip $\approx$ IK-only" held, but IK-only itself diverged from ground truth on dataset plants due to internode curvature, non-180° phyllotaxis, yaw perturbations, leaf angle jitter, and drooping lateral shoots (1~3 mm drop per node).

### 3.3 Root Causes & Remediation

**(1) `chain_phytomers` restricted parents to positions strictly below children.** Designed to prevent cycles, this rule severed drooping laterals. At DAP 75: GT 12 shoots $\to$ 17 shoots, 8 of 99 intra-shoot links broken, with one shoot displaced by 73 cm. Now, each node points to the lowest-cost candidate; cycles are broken at the highest-cost edge (breaking ties by keeping lower parent). Across 30 dataset plants, intra-shoot links recovered to 1988/1988. At shoot junctions where depth ties, continuity is chosen by orientation. When `root_own_shoot=True`, the root node (cotyledon) remains isolated as shoot 0 — required because emitters and converters treat shoot 0 specially (without this, DAP 15 export dropped to 57.4%).

**(2) `gt_parent_links` branch point heuristic ("closest node in another shoot below current node") was wrong for 1/3 of lateral shoots in dataset plants** (66.2% of 272, 58% for DAP 60+). Because this function generates training targets for parent conditioning, depth ordinals, and step losses, all previous runs trained on corrupted targets. Because lateral internodes originate at branch points (0.2~0.9 cm from parent node center, vs 1~3 cm to others), parents are now resolved via decoded slot-0 bases: `gt_parent_links(..., internode_base=)` $\implies$ 97.8% accuracy.

**(3) Converter leaf pitch/yaw/roll angles were hardcoded constants** (pitch 2.54, roll −15, yaw +10/0/−10). Accurate for synthetic prototypes, but deviated by 10~18° mean on dataset plants. In `plant_recon/models/part_tensor_leaf_ik.py`: forward kinematics returns per-leaf local frames (petiole tip azimuth, petiole/internode tip elevation, roll sign, organ type) and inverts $R_{\text{leaf}} = R_z(\text{azimuth}) R_z(\text{yaw}) R_y(-\text{pitch}) R_x(\text{roll})$ in closed form. Lateral leaflets (3 DOF) solve exactly (0.00° error); terminal leaflets (1 DOF) solve pitch; cotyledons solve pitch+roll. Runs after stem IK in `assemble_part_tensor_to_xml`.

### 3.4 Post-Remediation Results

| DAP | Packet Path | Via VAE | Tip Error | Petiole Axis | Leaf Rotation |
|---|---|---|---|---|---|
| 15 | **98.3%** | **95.1%** | 0.03 cm | 1.06° | 0.40° |
| 40 | **98.2%** | **96.8%** | 0.01 cm | 0.75° | 0.31° |
| 75 | **95.6%** | **95.6%** | 0.01 cm | 1.14° | 0.52° |

Re-evaluated fig14 (exact_gt): IK-only 95.7 / 99.5 / 96.7 $\to$ **99.9 / 99.6 / 97.7%**, VAE 92.2 / 98.4 / 95.7 $\to$ **93.7 / 98.3 / 95.8%**. Mean organ IoU improved from 48.3 / 89.9 / 46.5 $\to$ 93.4 / 91.2 / 54.1.

Remaining residual: Terminal leaflets (1 DOF, 1~1.6° mean error), first-node petiole azimuth ($\le 9^\circ$), and 2.2% branch point ambiguity.

![fig12](assets/fig12_phytomer_10slot_helios_roundtrip.png)

![fig14](assets/fig14_phytomer_vae_helios_roundtrip.png)

---

## 4. Training Execution & Evolution to v10

- Migrated cluster launcher to canonical v9/v10 configuration.
- Decoupled packet caches from VAE latents (encoding latents on-the-fly during training), eliminating cache invalidation when swapping VAE weights.

---

## 5. Performance Profiling & Vectorization Breakthroughs

### 5.1 Bottleneck Isolation: Packet Assembly Loop (`f3c5366`)

Profiling with synchronized GPU timers under 1 GPU, batch 48, Epoch 15:

| Metric | Render ON (Legacy Loop) | Render OFF | Render ON (Vectorized Assembly) |
|---|---|---|---|
| Step (fwd/bwd) | 1.8~2.1 s | 0.11~0.31 s | **0.19~0.39 s** |
| Render Block | 0.6 s (`assemble_packets` 0.55 s, mesh 0.01 s, rasterize 2 ms) | - | 0.05 s |
| Backward | 1.1 s | 0.02~0.04 s | 0.05 s |

Mesh synthesis required 5~11 ms, `nvdiffrast` rasterization 2 ms, and render backward 10~21 ms/sample. The primary overhead stemmed from sequential Python loops in `assemble_packets` computing curved petiole/peduncle attachment points across $\sim 1,000$ phytomers per batch. Vectorized via `_curve_points_batched`/`_point_on_curve`.

### 5.2 Elimination of Remaining Sample Loops (`a1a82bc`, `25c2251`)

- Precomputed `attach_parent_links` on CPU within DataLoader workers.
- Encoded entire batches through VAE concurrently.
- Replaced sequential plant rendering with `nvdiffrast` range-mode batched rasterization (`render_batched`).
- Vectorized `chain_phytomers` via pointer jumping (log $N$ parent doubling), reducing execution time from 34 ms $\to$ 2 ms for $N=512$.
- Batchified Hungarian matching (`_forward_phytomer_batched`).

---

## 6. Ground Truth Substitution Ablation Analysis (Epoch 40)

To determine structural bottlenecks, evaluated predictions on 20 fixed eval plants, matching predicted nodes against GT phytomers and substituting individual predicted variables with GT:

| Variant | IoU % | Depth MAE (cm) | Young (DAP ≤ 15) | Mid | Old (> 60) |
|---|---|---|---|---|---|
| P (All Predicted) | 24.8 | 14.97 | 2.5 | 23.6 | 37.2 |
| Pos $\leftarrow$ GT | 42.3 | 12.80 | 27.2 | 36.1 | 55.9 |
| Topo / Rot / Scale / Latent $\leftarrow$ GT | 25.3 / 26.3 / 23.9 / 25.2 | ~14.9 | - | - | - |
| Pos + Rot / Pos + Latent / Pos + Scale | 45.6 / 45.0 / 43.4 | 12.1~12.7 | - | - | - |
| ALL − Pos | 30.6 | 13.98 | 1.2 | 32.9 | 42.9 |
| ALL − Rot | 49.1 | 11.28 | 27.3 | 47.5 | 61.7 |
| ALL − Latent | 47.5 | 11.53 | 27.5 | 44.6 | 60.4 |
| ALL − Scale | 61.6 | 7.90 | 37.5 | 57.5 | 77.7 |
| ALL − Topo | 82.1 | 4.92 | 77.4 | 80.2 | 86.4 |
| ALL (GT geometry + latent, predicted existence) | 82.1 | 4.92 | 77.2 | 80.3 | 86.4 |

**Interpretation**: **Node position is the primary bottleneck**. Providing GT positions alone raises IoU from 24.8 $\to$ 42.3%. When all variables except position are GT, performance collapses to 30.6%. Secondary bottlenecks are **orientation and latent features** ($-33$ and $-35$ when withheld).

---

## 7. Stage 3 Geometry Flow Matching & Latent Limitations

### 7.1 Stage 3 Geometry Modeling (`--stage3_geometry`)
Extended flow state to `[(pos - parent_pos) * 20 | roll | scale | latent]`, generating child relative pose and scale in Stage 3.

### 7.2 Latent Information Bottleneck
Evaluation revealed Stage 3 latents did not condition strongly on input images:
- Predictions achieved $R^2 \approx 0.12$ relative to dataset means, but $R^2 \le 0$ relative to plant-specific means.
- Substituting dataset mean latents (`meanlat`) yielded 28.5% IoU vs 27.5% for predicted latents; GT latents provided $+33\sim 38\%$.
- Model latents captured broad plant scale and DAP, but lacked fine-grained node-specific differentiation.

---

## 8. Test-Time Refinement: Analysis-by-Synthesis Breakthrough

### 8.1 Concept and Validation
Implemented `plant_recon/eval/eval_test_time_refinement.py`: Optimizes node positions, scales, and phytomer latents for 40 AdamW steps directly against the input Canopy Height Model (CHM) using render losses (no GT labels required at inference).

### 8.2 Input Camera Alignment (`--input_camera`)
Identified coordinate window mismatch: cache CHMs were centered on GT bounding boxes (`focus_plant=True`), whereas rendering evaluated from the origin. Aligning the virtual camera to match input acquisition bounds doubled optimization efficiency.

### 8.3 Benchmark Results

| Protocol (20 Eval Plants, 256px Strict Evaluation) | Full 20 Plants | Mature (DAP > 15) |
|---|---|---|
| Raw Model Prediction (v10 ep95) | 35.8% | 41.1% |
| Origin-Centered Refinement (40 steps) | 46.9% | 51.2% |
| **Input Camera Refinement (`--input_camera`, 40 steps)** | **63.9%** | **72.2%** |
| **Input Camera + Multi-Zoom Targets (1x, 2x, 4x, 8x)** | **67.6%** | **74.0%** |
| **80-Step Multi-Zoom Refinement** | **69.1%** | **78.6%** |
| **160-Epoch Lineage + Refinement (Job 38332343)** | **68.3%** | **74.7%** |

### 8.4 Shape Regularization (`--reg_scale 5 --reg_latent 0.5`)
Without regularization, unconstrained optimization inflated leaves into planar polygons to maximize silhouette overlap. Quadratic penalties against predicted priors retained realistic botanical geometry while achieving **67.6% FG IoU**.

---

## 9. Comprehensive Conclusions

1. **Training Plateau (P ~ 35–39)**: Pure model predictions consistently plateau around 35–39 across all model and data scales (10% vs 100% data, unfreezing backbones, multiscale tokens).
2. **Analysis-by-Synthesis Resolution (P ~ 68)**: Test-time photometric refinement provides the critical performance breakthrough (+30~35%), adjusting metric positions and latent morphology to fit observed CHM depth in real time.
3. **Helios Physical Roundtrip**: Reaches **95.6%~98.3%** fidelity across the complete physical raytracing pipeline once topological branching metadata is preserved.
