---
title: "Session Takeover: Grad-Norm Inf Deadlock Fix, Helios Roundtrip Diagnosis & 10-Slot Contract Restore"
date: 2026-09-11
tags: [handover, session]
status: active
---

# Session Takeover: Grad-Norm Inf Deadlock Fix, Helios Roundtrip Diagnosis & 10-Slot Contract Restore

- **Author**: Antigravity Pair Programming
- **Date**: 2026-09-11
- **Status**: P0 Fixes Verified (local smoke, pre-§7.9 code) | P1 Root-Cause Diagnosed | 10-Slot Contract Restored | **NEW P0: PhytomerVAE rotation architecture simplified to single `head_rot` path (§7.9) — ALL existing checkpoints (v2/v3/v4/xml) now fail to load, PhytomerVAE MUST be retrained from scratch before hierarchical FM training can resume** | **NEW: shoot-topology (SHOOT_META) loss identified as the dominant cause of Helios-roundtrip collapse, unrelated to the VAE — needs its own design fix (§7.8)** | Main job 38238371 cancelled (would have crashed on VAE load)
- **Supersedes**: [`20260911_differentiable_render_gradient_isolation_and_backbone_freeze.md`](../../engineering/20260911-diff-render-gradient-isolation/20260911-diff-render-gradient-isolation.md) (Option A/B context)

---

## 1. Executive Summary

This session addressed three primary issues:

1. **P0 (Resolved · Commit `4b70266`)**: Job `38237555` entered `grad_norm=inf` deadlock starting at Epoch 11 Step 52, skipping every step (3,215 consecutive skips, 0% training progress). Log forensics pinpointed **two distinct root causes**, which were structurally remediated.
2. **P1 (Diagnosed, Geometry Fix Ongoing)**: Internode/Petiole IoU collapsed to 0~15% in Helios roundtrips — determined to be a compound effect of an **eval script COCO category off-by-one mapping bug** plus **FK kinematic chain drift** (tube centroid offset by 9~18 px). The mapping bug was resolved.
3. **10-Slot Contract Restoration (Resolved · Commit `75928d9`)**: Discarded the 9-slot (excluding internodes) restructure attempted in v4 and restored the canonical **v2/v3 10-slot contract** (slot 0 = internode, repro ×4). Fixed a related `SLOT_ROLE_MAPPING` 8-vs-10 truncation crash uncovered during smoke testing.

---

## 2. P0: Job 38237555 Deadlock Root Cause (Log Verification)

### 2.1 Symptoms

```
[Recovery] Step 52: grad_norm is NaN/Inf (inf) | Culprit params (0): -> SKIPPING STEP
(3,215 consecutive skips, Loss frozen at 0.0000 from Epoch 11 onward)
```

### 2.2 Key Finding: Meaning of `Culprit params (0)`

When individual gradient tensors contain **zero NaNs or Infs**, but total norm evaluates to `inf`:
- `clip_grad_norm_` computes $\sum \|g\|^2$ in **Float32** (fp32 max $\approx 3.4 \times 10^{38}$).
- A single finite but massive gradient element ($\sim 10^{19}$) causes the sum of squares to overflow fp32 $\implies$ total norm = `inf`, despite every parameter tensor being strictly finite.
- Prior per-element `isnan/isinf` culprit checks caught nothing, repeatedly skipping without identifying culprit parameters.

### 2.3 Failure Chain (System Dynamics)

```
[Prior to Crash] OOD latent (|z| > 6σ) -> frozen VAE decodes degenerate mesh (e^12 ≈ 1.6e5 scale)
                 -> Mesh vertices pierce camera near-plane (clip-space w -> 0)
                 -> nvdiffrast perspective-correct backward yields 1/w² ≈ 1e19 gradients
                 -> fp32 sum-of-squares overflow -> grad_norm=inf -> permanent skip loop
```

**Trigger**: At Epoch 11, time-based eval fallback (31 min elapsed) triggered `evaluate_self_consistency_batch()`, which switched to `model.eval()` and **failed to restore `model.train()`** $\implies$ explosion occurred immediately on Step 52 post-eval. (Eval mode leak acted as ignition; $w \to 0$ Jacobian acted as explosive).

---

## 3. P0 Remediation (Commit `4b70266`)

### 3.1 Structural Design Fixes

| Layer | File | Modification |
|---|---|---|
| **Latent (Source)** | `train_hierarchical_flow_matching.py` | `clean_z1 = clean_z1.clamp(-6.0, 6.0)` — enforces latent support set $z \sim \mathcal{N}(0, \mathbf{I})$, eliminating degenerate mesh inputs |
| **Renderer (Jacobian)** | `helios_pytorch_renderer.py` (3 sites) | **w-floor**: bounds $\|w\| < 10^{-4} \implies \pm 10^{-4}$, bounding perspective-correct backward $1/w^2 \le 10^8$ (Lipschitz bound). Applied to all 3 rasterize paths; outputs sanitized with `nan_to_num` |
| **Optimizer (Invariant)** | `train_hierarchical_flow_matching.py` | Atomic clamp ($\pm 100$) before `clip_grad_norm_` — prevents fp32 sum-of-squares overflow even if extreme gradients emerge (defense-in-depth) |

### 3.2 Sealing Eval Mode Leaks

- `eval_hierarchical_self_consistency.py`: Stored `was_training` state $\implies$ restored `model.train()` prior to return.
- Caller `train_hierarchical_flow_matching.py`: Wrapped eval in `try/finally: model.train()` to guarantee restoration under all exit conditions.

### 3.3 Enhanced Culprit Telemetry

- Reports finite but large ($>10^6$) gradients via `Huge(>1e6)` counter, pinpointing unstable parameters immediately upon emergence.

### 3.4 Numerical Verification

```
Before: Single finite grad element 1e19 -> sum(g²) >= 1e38 -> fp32 overflow -> norm = inf (matches logs)
After w-floor: 1/w <= 1e4 x loss grad 1e-3 -> element g² <= 1e2 -> sum(g²) <= 1e11 << 3.4e38 for 1e9 params
After atomic clamp: norm = 100.0 (finite) -> restores normal clip path, 0% skips
Healthy grads (~1e-3): Norms pre/post clamp are bit-identical (0.070711 -> 0.070711)
```

---

## 4. P1: Helios Roundtrip Internode/Petiole IoU Diagnostics

### 4.1 Methodology

Recomputed per-class IoU directly from GT/Reconstruction COCO mask dumps (`/tmp/helios_organ_mask_eval/*`).

**Critical Discovery — True Helios COCO Category Schema** (Verified on GT dumps):
```
{0: 'shoot' (=Internode), 1: 'petiole', 2: 'leaf', 3: 'floral_bud', 4: 'flower', 5: 'pod'}
```
The evaluation script assumed 1-indexed organ types, shifting **all classes by one position**. The reported "Internode 0% / Petiole 1~15%" was directly caused by this index mismatch.

### 4.2 Recalculated Metrics (512px Polygon Rasterization)

| Stage | Internode(shoot) | Petiole | Leaf | Flower | Pod |
|---|---|---|---|---|---|
| DAP 10 | GT px 0 (n/a) | 1.9% (100px) | **95.6%** | n/a | n/a |
| DAP 50 | **0.0%** (347 vs 335px) | 13.9% (1085px) | **93.3%** | n/a | n/a |
| DAP 90 | **10.9%** (280 vs 299px) | 4.6% (787px) | **80.6%** | 19.8% | 5.6% |

### 4.3 Diagnostic Conclusions

- **Pixel areas match closely between GT and Reconstruction** (dap050 Internode: 347 vs 335 px) $\implies$ length/radius conversions are **accurate**.
- Centroids exhibit a **9~18 px spatial displacement**; thin tubular structures (2~3 px wide) become disjoint under this offset $\implies$ IoU drops to 0.
- Offset directions vary with mixed $\pm$ signs $\implies$ reflects **FK kinematic chain drift** (notably hardcoded 0°/20° internode pitch in `part_tensor_to_40d.py:400-403` vs ground truth pitch).
- Within an 8 px dilation window, overlap rises to 53~59% (Internode) and 78~81% (Petiole), confirming tubes exist in the immediate vicinity.

### 4.4 Implemented Fixes

`eval_13d_xml_organ_masks.py`: Introduced explicit `COCO_TO_CLASS = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5}` mapping with documentation.

**Follow-up Requirement**: Eliminate hardcoded internode pitch in `part_tensor_to_40d.py`, inverting pitch directly from 14D tensors and validating across DAP 10/50/90.

---

## 5. Restoration of 10-Slot Contract (Commit `75928d9`)

### 5.1 Rationale

While a 9-slot draft (omitting internodes) existed in the working tree, the **10-slot contract (v2/v3) was confirmed as canonical**:
- v3 VAE checkpoints require 240D input (10 slots × 24D), incompatible with 216D (9 slots).
- Prevents truncation of the 3.29% of phytomers bearing $\ge 3$ reproductive organs.
- Fixed `SLOT_ROLE_MAPPING` truncation bug where length 8 caused crashes against $M=10$.

### 5.2 Summary of Changes

| File | Changes |
|---|---|
| `phytomer_packets.py` | Restored 10-slot builder (`NUM_SLOTS=10`, `ROLE_SLOT_RANGES` 4:(6,10), `LABEL_ROLE_RANGES` stem 1..3). Adapted helper functions: `phytomer_scale`, `normalize/denormalize_packet_scales`, `assemble_phytomer_ordered_14d_tensor` |
| `hierarchical_part_flow_matching.py` | Set `SLOT_ROLE_MAPPING = [0,1,2,2,2,3,4,4,4,4]` (10 elements), `SLOT_SUB_ROLE_MAPPING = [0,0,0,1,2,0,0,1,2,3]`, default `slots_per_phytomer` 9 $\to$ 10 |
| `hierarchical_hungarian_matcher.py` | Set `_role_slot_ranges[4]: (6,8) -> (6,10)` (repro x4) |
| `scripts/train_hierarchical_flow_matching.sh` | Set `SLOTS_PER_PHYTOMER` default 9 $\to$ 10 |
| `tests/test_phytomer_packets.py` | Restored v2 expected values (7 passed) |
| `tests/test_phytomer_vae.py` | Updated synthetic bag role map to 10 slots (8 passed) |

---

## 6. Execution Status & Next Steps

### 6.1 SLURM Job Status (as of 2026-09-11 15:30)

| Job | Partition | Status | Notes |
|---|---|---|---|
| `38238218` | gpu-6000_ada-h | PENDING | Submitted before 10-slot fix $\implies$ cancel and resubmit |
| `38238368` | low (publicgrp) | PENDING | 10-slot smoke validation run |

### 6.2 Action Order for Next Agent

1. Resubmit main training with 10-slot configuration: verify `512 nodes x 10 slots/node` in startup logs.
2. Confirm smoke job `38238368`: check Epoch 4 render activation, absence of `[Recovery]` skips, and monotonic loss descent.
3. Address FK kinematic chain drift in `part_tensor_to_40d.py` (lines 400–403).

---

## 7. PhytomerVAE + Helios End-to-End Roundtrip Analysis

### 7.1 Motivation

Prior to restarting main runs, end-to-end fidelity across the entire chain was evaluated:
```
Helios GT -> XML -> 14D -> 10-slot packet -> PhytomerVAE encode/decode -> 14D -> XML -> Helios raytrace
```
Implemented `plant_recon/eval/eval_phytomer_vae_helios_roundtrip.py` to compare against IK-only baselines and isolate VAE contributions.

### 7.2 Results (`fig14_phytomer_vae_helios_roundtrip.png`)

| Stage | IK-only FG IoU | **VAE-RT FG IoU** | IK-only mIoU | **VAE-RT mIoU** | IK PSNR | **VAE PSNR** |
|---|---:|---:|---:|---:|---:|---:|
| DAP 10 | 95.7% | **24.0%** | 48.3% | **6.1%** | 38.90 dB | **21.69 dB** |
| DAP 50 | 94.1% | **7.8%** | 36.0% | **2.4%** | 26.12 dB | **6.62 dB** |
| DAP 90 | 87.4% | **10.5%** | 21.7% | **1.5%** | 21.43 dB | **9.07 dB** |

### 7.3 Visual Diagnosis

While GT/IK-only plants maintain dense canopy morphology, VAE roundtrip reconstructions unravel into elongated vines. Organ counts closely match ground truth (76/82, 982/995, 1417/1427), indicating structural layout and spatial orientation are the primary failure points.

### 7.4 Root Cause Analysis: Packet-Level Fidelity

Comparing raw 26D packets without XML/Helios conversion via `eval_phytomer_vae_packet_fidelity.py`:

**Under `use_rot_branch=False` (Shared Head):**
- Mean rotation error: Stem 139°, Petiole 141°, Leaflets 130°, Repro 172° $\implies$ **essentially random rotations**.
- Confirmed that v3 checkpoints were trained using `rot_branch`/`head_rot_dedicated`, leaving the shared `head_rot` untrained.

**Under `use_rot_branch=True` (Dedicated Head):**
- Mean rotation error: Stem 0.8°, Petiole 15~16°, Leaflets 14~16°, Repro 1.4~13.8° $\implies$ significant recovery.
- Residual angular errors (13~16°) still compound across kinematic integration in `assemble_packets`, amplifying base offsets (Leaflet 1.3~6 cm, DAP 90 Repro up to 25~51 cm).

### 7.7 Training Path Bug Remediation

Identified two sites where `phytomer_vae.decode()` was called without `use_rot_branch=True`:
1. `hierarchical_part_flow_matching.py:1392`
2. `train_hierarchical_flow_matching.py:702`

In both cases, render losses had been backpropagating through random rotations.

### 7.8 Dominant Factor: SHOOT_META (Branch Topology) Elimination

Evaluating a 100% GT 14D tensor with only `ORGAN_SHOOT_META`/`ORGAN_ROOT_META` rows removed:

| Stage | Full IK-only (with meta rows) | Meta Rows Removed (100% GT geometry) |
|---|---:|---:|
| DAP 10 | 95.7% | **53.5%** |
| DAP 50 | 94.1% | **16.5%** |

Matches the magnitude of degradation in VAE roundtrips: **the primary driver of the "vine" artifact is the omission of branch boundary metadata**, causing 11 lateral shoots to concatenate into a single linear axis.

### 7.9 Architectural Simplification: Unifying Rotation Heads

To eliminate dual-path ambiguity, **the rotation architecture was simplified to a single `head_rot` path, completely excising `rot_branch` and `head_rot_dedicated`**:
- Removed all dual-branch flags across `phytomer_vae.py`, `train_phytomer_vae.py`, and evaluation scripts.
- **Impact**: All legacy checkpoints (`v2/v3/v4/xml`) contain obsolete keys and cannot be loaded. **PhytomerVAE must be retrained from scratch** before resuming hierarchical FM training.

---

## 8. Commit Log (This Session)

```
75928d9 feat(phytomer): restore v2/v3 10-slot packet contract (internode slot 0 + repro x4)
4b70266 fix: structurally bound renderer Jacobian & restore train mode after in-loop eval
```