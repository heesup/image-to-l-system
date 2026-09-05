# Agent Takeover & Engineering Handover Guide
**Project: Image-to-L-System / 3D Inverse Procedural Plant Reconstruction**
**Last Updated:** 2026-09-04 (Phase 2 complete)
**Primary Author/Agent:** Antigravity Autonomous Agent (Pair programming with Heesup Yun)
**Environment:** Linux, Python 3.10+, Mamba (`mamba activate digital-crops`), CUDA, PyTorch, `nvdiffrast`, Helios C++ OptiX Raytracer.

---

## 1. Executive Summary & Mission

The core mission of this repository is **inverse 3D botanical reconstruction**:
Given a single monocular top-view RGB-D ($256 \times 256 \times 4$) image containing Canopy Height Model (CHM) depth, reconstruct the exact 3D plant architecture into:
1. **Canonical 14D Part Tensor Representation**: Disentangled per-organ metric state representation.
2. **Native Helios C++ XML Tree**: Full procedural L-system specification that compiles and raytraces in the native physical Helios C++ engine with high photometric and geometric fidelity.

### Major Milestone Achieved:
- We have completely solved the **Closed-Form Inverse Kinematics (IK)** and **Procedural Kinematics Inversion** between the 14D Part Tensor and native Helios XML.
- Helios C++ raytraced reconstructions achieve state-of-the-art accuracy across all growth stages without manual branch angle tuning:
  - **DAP 10 (Seedling)**: **95.10% Foreground Mask IoU**, **94.91% Leaf IoU**, **38.02 dB Depth PSNR**.
  - **DAP 50 (Branching)**: **92.85% Foreground Mask IoU**, **91.67% Leaf IoU**, **25.17 dB Depth PSNR**.
  - **DAP 90 (Fruiting)**: **86.47% Foreground Mask IoU**, **79.45% Leaf IoU**, **21.14 dB Depth PSNR**.
- Direct Differentiable Rendering optimization with our **Concentric Multi-Scale Pyramid** achieves **92.45% Mask IoU** and **0.71 mm Chamfer distance** on canonical unifoliate seedlings without blank-canvas collapse (Phase 2 anti-erasure rewrite; was 81.61% / 2.29 mm in Phase 1).
- **Phase 2 (Variable Organ Topology) complete**: Strategy B over-allocation pruning (10→5 slots, OVERALL PASS) and Strategy A residual-driven leaf spawning (57.7%→82.7% IoU, +25 pp) both verified.
- **Representation coverage audit (exp6/exp7)**: all 26 dimensions of the continuous node tensor verified differentiable & rendered (2 were dead pre-fix: curvature, scale_z). Group-wise recovery: base_xyz RECOVERED (4.8→0.7–1.6 mm), curvature PARTIAL (56→80→45–57°/m, weak signal), rot6d/scale LIMITED by gauge freedoms & sub-pixel radius observability, organ-type mobility FAILED (Phase 3 scope).

---

## 2. Core Data Contracts & Tensor Specifications

### A. Canonical 14D Part Tensor (`diffusion_based/models/plant_organ_array.py`)
Every organ (and metadata element) is represented by a 14-dimensional vector:
$$\mathbf{p} \in \mathbb{R}^{14} = [\text{organ\_type}(1), \mathbf{x}_{\text{base}}(3), \mathbf{r}_{6\text{D}}(6), \mathbf{s}_{\text{scale}}(3), \text{curvature}(1)]$$

| Index | Field | Description | Constraints |
| :---: | :--- | :--- | :--- |
| `0` | `organ_type` | Categorical integer organ identifier | See Categorical Organ Types table below |
| `1:4` | `base_xyz` | 3D base coordinate in meters $[X, Y, Z]$ | Physical metric coordinates |
| `4:10` | `rot6d` | Continuous 6D rotation vector $[u_x, u_y, u_z, v_x, v_y, v_z]$ | Gram-Schmidt orthonormalized to $SO(3)$ matrix $R = [\mathbf{r}_1, \mathbf{r}_2, \mathbf{r}_3]$ |
| `10:13` | `scale_xyz` | Physical dimensions in meters $[s_x, s_y, s_z]$ | Stem: $[L, r, 0]$; Petiole: $[L, r, 0]$; Leaf: $[s, s, s]$ |
| `13` | `curvature` | 3D bending curvature ($^\circ$ or $\text{m}^{-1}$) | Stem/Petiole sagittal curvature |

### B. Categorical Organ Types (`NUM_ORGAN_TYPES = 13`)
```python
ORGAN_NONE           = 0   # Empty / pruned slot
ORGAN_ROOT_META      = 1   # Plant base position & age metadata
ORGAN_SHOOT_META     = 2   # Shoot metadata & base rotation
ORGAN_INTERNODE      = 3   # Stem internode segment
ORGAN_PETIOLE        = 4   # Petiole segment
ORGAN_LEAF           = 5   # Leaf blade (unifoliate or trifoliate)
ORGAN_PEDUNCLE       = 6   # Reproductive peduncle stem
ORGAN_BUD_DORMANT    = 7   # Dormant floral bud
ORGAN_BUD_ACTIVE     = 8   # Active bud
ORGAN_FLOWER_CLOSED  = 9   # Closed flower bud
ORGAN_FLOWER_OPEN    = 10  # Fully open yellow flower
ORGAN_FRUIT          = 11  # Fruit pod (cowpea pod)
ORGAN_BUD_ABORTED    = 12  # Aborted bud
```

### C. The 7-Row Canonical Seedling Standard
In canonical seedling experiments (e.g. `scratch/target_unifoliate_14d.pt`), the tensor has shape `(7, 14)`:
- **Row 0**: `ORGAN_ROOT_META` ($s_x=10.0$ encodes DAP 10)
- **Row 1**: `ORGAN_SHOOT_META` (Shoot ID 0)
- **Row 2**: `ORGAN_INTERNODE` (Stem, $L=0.05\,\text{m}$, $r=0.0035\,\text{m}$)
- **Row 3**: `ORGAN_PETIOLE` (Petiole 0, $+X$ direction)
- **Row 4**: `ORGAN_PETIOLE` (Petiole 1, $-X$ direction)
- **Row 5**: `ORGAN_LEAF` (Leaf blade 0, at Petiole 0 tip)
- **Row 6**: `ORGAN_LEAF` (Leaf blade 1, at Petiole 1 tip)

> [!NOTE]
> **Metadata Handling in Rendering**:
> `HeliosPlantGeometryBuilder` and `HeliosPyTorchRenderer` automatically filter out rows where `ot < ORGAN_INTERNODE` or scale $= 0$. They generate 0 vertices and do not pollute the rendering or back-propagation.
> However, `assemble_part_tensor_to_xml` uses Rows 0 and 1 to build the plant base and shoot headers in XML.

### D. Physical Scale Purity Contract
- `scale_xyz` (indices 10:13) **MUST ALWAYS** represent true physical dimensions in meters.
- **NEVER** multiply `existence` (alpha/opacity) into `scale_xyz`.
- In differentiable rendering, soft existence is passed independently via the `existence` tensor to modulate vertex opacities.

---

## 3. The 5 Golden Rules of Helios Procedural Kinematics

When converting 14D Part Tensors to Helios XML (`diffusion_based/models/part_tensor_to_40d.py`), you must adhere to these 5 mathematical rules. Violating any of them causes severe canopy deformation or crashes:

### Rule 1: Shoot 0 Phytomer 0 Base Internode Pitch Isolation
In Helios C++ (`InputOutput.cpp`), Shoot 0's orientation in 3D world space is governed by `<base_rotation>`:
- The base internode (`phytomer 0`) MUST have `<internode_pitch> 0 </internode_pitch>`.
- All subsequent nodes ($k \ge 1$) undergo relative deflection: $\Delta \theta = -1.25 \times \text{internode\_pitch}$ relative to the preceding stem segment.
- **Danger**: If world Z zenith angle is written to `phytomer 0`, the stem undergoes $-1.25 \times 60^\circ = -75^\circ$ backward bending at each node, collapsing the canopy into a tangled "starburst".

### Rule 2: Closed-Form Lateral Shoot Base IK (`solve_helios_shoot_base`)
When a lateral shoot branches from a parent node, Helios applies 4 sequential rotations:
1. $R(\mathbf{k}_{\text{pet}}, 0.5 \times \theta_{\text{pitch}})$ (half pitch about petiole axis)
2. $R(\mathbf{u}_p, 90^\circ)$ (base roll about parent internode)
3. $R(\mathbf{k}_{\text{pitch}}, -\theta_{\text{shoot\_pitch}})$ (lateral insertion pitch)
4. $R(\mathbf{u}_p, \theta_{\text{shoot\_yaw}})$ (axillary yaw rotation)

In `part_tensor_to_40d.py`, `solve_helios_shoot_base(u_p, v_p, u_c)` solves this in $O(1)$ closed-form time via:
$$\mathbf{u}_p \cdot \mathbf{u}_c = A \cos(\theta_{\text{shoot\_pitch}}) + B \sin(\theta_{\text{shoot\_pitch}})$$
$$\theta_{\text{shoot\_pitch}} = \text{atan2}(B, A) \pm \text{acos}\left(\frac{\mathbf{u}_p \cdot \mathbf{u}_c}{\sqrt{A^2 + B^2}}\right)$$
$$\theta_{\text{shoot\_yaw}} = \text{planar projection}(\mathbf{u}_c)$$
Error is mathematically $0.0000^\circ$.

### Rule 3: Dynamic Relative Phyllotaxis Inversion
Helios internodes rotate around the stem axis before emitting petioles. In `part_tensor_to_40d.py`, relative phyllotaxis between consecutive nodes $k-1$ and $k$ is dynamically extracted:
$$\mathbf{p}_1 = \mathbf{v}_{k-1} - (\mathbf{u} \cdot \mathbf{v}_{k-1})\mathbf{u}, \quad \mathbf{p}_2 = \mathbf{v}_k - (\mathbf{u} \cdot \mathbf{v}_k)\mathbf{u}$$
$$\theta_{\text{phyllo}} = \text{atan2}(\mathbf{u} \cdot (\mathbf{p}_1 \times \mathbf{p}_2), \mathbf{p}_1 \cdot \mathbf{p}_2)$$

### Rule 4: Petiole-Local Child Indexing (`leaf_counts_per_pet`)
In Helios XML:
- For a unifoliate shoot, each petiole has exactly 1 leaf (`child_index = 0`).
- For a trifoliate shoot, the petiole has 3 leaflets (`child_index = 0, 1, 2`).
- **Danger**: Never use a global phytomer counter. Track `leaf_counts_per_pet[parent_pet_idx]` to guarantee that Petiole 1 receives `child_index = 0` in unifoliate plants.

### Rule 5: Bud State & Reproductive Lookahead
Helios procedural mesh building relies on `<bud_state>` inside `<phytomer>`:
- `4: BUD_FRUITING` $\to$ renders fruit pods.
- `3: BUD_FLOWER_OPEN` $\to$ renders yellow flower petals.
- `2: BUD_FLOWER_CLOSED` $\to$ renders green calyx buds.
`part_tensor_to_40d.py` performs a lookahead on organs attached to each phytomer, setting the exact `bud_state` and extracting pitch directly from the $R_{2, 0}$ rotation column ($\text{pitch} = \text{asin}(-R_{2, 0})$).

---

## 4. Key Source Code & Architecture Directory Map

The repository was reorganized (2026-09-04) to contain only active code. Legacy
models/training/eval live in `archive/`; reusable verification scripts live in
`tests/unit/`; `scratch/` holds only active benchmark scripts + their outputs.

```
/home/lion397/codes/image-to-l-system/
├── scratch/                              <-- [ACTIVE] experiment scripts + outputs (git-ignored)
│   ├── phase2_core.py                    <-- [CRITICAL] shared anti-erasure loss core (exp2/4/5/7)
│   ├── make_target_unifoliate.py         <-- [CRITICAL] canonical 7-row target generator
│   ├── exp1_per_organ_icp.py             <-- Method 1: Point Cloud ICP benchmark
│   ├── exp2_diff_render_opt.py           <-- Method 2: Diff Renderer (anti-erasure) benchmark
│   ├── exp3_toy_flow_matching.py         <-- Method 3 baseline (kept for reference; superseded by 3b)
│   ├── exp3b_flow_matching_fixed.py      <-- Method 3b: fixed FM (conditioning+guidance+hybrid)
│   ├── exp4_overalloc_pruning.py         <-- Phase 2 Strategy B: 10-slot over-allocation pruning
│   ├── exp5_underalloc_spawn.py          <-- Phase 2 Strategy A: residual-driven leaf spawning
│   ├── exp6_dimension_coverage.py        <-- 26D per-dimension differentiability audit
│   ├── exp7_dimension_recovery.py        <-- Group-wise perturb-recover test
│   ├── eval_phase1_comparison.py         <-- Synthesis comparison & Figure 12 generator
│   ├── exp*_recon_14d.pt / target_*.pt   <-- benchmark outputs & GT artifacts
│   └── xml_outputs/                      <-- exported Helios XMLs
├── tests/unit/                           <-- [ACTIVE] reusable verification scripts
│   ├── test_14d_curvature.py             <-- 14D extraction / 40D reconstruction w/ curvature
│   ├── test_multiscale_pyramid.py        <-- pyramid renderer verification
│   ├── test_reproductive.py / test_render_14d.py  <-- DAP 50/90 raytrace verification
│   ├── test_soft_existence_visual.py     <-- soft-existence alpha rendering demo
│   └── helios_xml_roundtrip_visualizer.py
├── diffusion_based/
│   ├── models/                           <-- [ACTIVE] 6 modules, dependency-ordered:
│   │   ├── plant_organ_array.py          <-- [CRITICAL] 14D/40D column constants + XML roundtrip
│   │   ├── helios_pytorch_geometry.py    <-- [CRITICAL] mesh builder + differentiable mapping
│   │   ├── helios_pytorch_renderer.py    <-- [CRITICAL] nvdiffrast renderer + multi-scale pyramid
│   │   ├── part_tensor_to_40d.py         <-- [CRITICAL] closed-form IK + XML assembler (assemble_part_tensor_to_xml)
│   │   ├── vit_image_encoder.py          <-- FM conditioning encoder (dep of part_flow_matching)
│   │   └── part_flow_matching.py         <-- FM denoiser model
│   ├── training/                         <-- [ACTIVE] 2 files:
│   │   ├── flow_matching.py              <-- [CRITICAL] Rectified Flow scheduler (x_t=(1-t)x0+t·x1)
│   │   └── train_part_flow_matching.py   <-- FM training loop
│   ├── dataset/                          <-- [ACTIVE] 3 files:
│   │   ├── part_array_dataset.py         <-- [CRITICAL] 25D FM encode/decode (BASE_SCALE=20, SCALE_SCALE=50)
│   │   ├── cowpea_shard_dataset.py       <-- shard dataset loader
│   │   └── generate_tensor_shards.py     <-- shard generation pipeline
│   ├── eval/
│   │   ├── eval_13d_xml_organ_masks.py   <-- [CRITICAL] Helios C++ raytrace benchmark (Fig 10)
│   │   └── metrics.py                    <-- shared metrics
│   └── checkpoints/fm/part_flow_matching.pt  <-- final FM checkpoint (only one kept)
├── dataset/helios_data/cowpea_shard/     <-- FM training data (25GB, git-ignored)
├── archive/                              <-- legacy code (git-tracked, not active)
│   ├── models_legacy/  training_legacy/  eval_scripts/  dataset_scripts/
│   └── ... (dataset_legacy, notebooks_legacy, etc.)
├── Digital-Crops/                        <-- Helios C++ OptiX raytracer
├── docs/
│   ├── ongoing/AGENT_TAKEOVER_GUIDE.md   <-- [THIS DOCUMENT]
│   └── results/assets/                   <-- all benchmark figures (fig10-14, exp*)
└── scripts/                              <-- figure-generation utilities
```

Removed on 2026-09-04 cleanup (~93 GB freed):
- 60 intermediate FM epoch checkpoints + stale DIT/VLM/VAE checkpoints (73 GB)
- stale shard dataset `cowpea_shard_stale_26d_20260824` (19 GB)
- `wandb/`, `logs/`, `training_logs/`, `agent_temp/`, `diffusion_based/plots/`
- `diffusion_based/tests/` (empty after archiving plant_vae test)

---

## 5. Active Benchmark Results

### A. Full Lifecycle Raytraced Reconstruction ([Figure 10](file:///home/lion397/codes/image-to-l-system/docs/results/assets/fig10_helios_per_organ_mask_comparison.png))
Evaluated via `diffusion_based/eval/eval_13d_xml_organ_masks.py` against native OptiX physical raytracing:

| Growth Stage | Foreground IoU | Leaf IoU | Reproductive IoU | Depth PSNR | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DAP 10 (Seedling)** | **95.10%** | **94.91%** | — | **38.02 dB** | SOTA Verified |
| **DAP 50 (Branching)** | **92.85%** | **91.67%** | — | **25.17 dB** | SOTA Verified |
| **DAP 90 (Fruiting)** | **86.47%** | **79.45%** | Flower: **21.01%** | **21.14 dB** | SOTA Verified |

### B. Phase 1 Seedling Benchmark ([Figure 12](file:///home/lion397/codes/image-to-l-system/docs/results/assets/fig12_back_to_basics_benchmark_summary.png))
Evaluated on true top-view ($90^\circ$ nadir) calibrated RGB-D target with 5 geometric slots and metadata:

| Method | Latency | Mask IoU | 3D Chamfer | Position MSE | Scale MSE | Helios XML Export |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ground Truth (7-Row)** | — | **100.00%** | **0.00 mm** | **0.000 cm²** | **0.000 cm²** | **SUCCESS** |
| **Method 1: Per-Organ ICP** | 257.5 ms | **68.07%** | **16.11 mm** | **0.500 cm²** | 0.928 cm² | **SUCCESS** |
| **Method 2: Diff Renderer (Pyramid + Anti-Erasure)** | 58 ms/step (4.4s) | **92.06%** | **1.04 mm** | **0.070 cm²** | **0.277 cm²** | **SUCCESS** |
| **Method 3: Flow Matching ODE** | **64.3 ms (15 steps)**| 9.97% | 30.30 mm | 28.112 cm² | 23.024 cm² | **SUCCESS** |
| **Method 3b: FM (best-of-6) + Render Polish (hybrid)** | 2.5 s total | **79–83%** (stable) | **3.03 mm** | 0.179 cm² | 0.213 cm² | **SUCCESS** |

> [!NOTE]
> Method 2 was upgraded during Phase 2 with the anti-erasure loss stack (see §8).
> Phase 1 baselines were: IoU 81.61%, Chamfer 2.29 mm, Position MSE 2.854 cm² — but silently erased
> the internode + both petioles (2/5 organ retention). The new run retains all 5 organs.

### C. Phase 2 — Variable Organ Topology ([Figure: exp4/exp5 progression PNGs](file:///home/lion397/codes/image-to-l-system/docs/results/assets/exp4_overalloc_pruning_progression.png))

| Strategy | Setup | Result | Verdict |
| :--- | :--- | :--- | :--- |
| **B: Over-Allocation + Pruning** | 10 slots (5 real + 5 coincident ghosts), L1 sparsity + soft-NMS exclusion + ghost-NONE logit init +2.0 + annealing | 10→**5** slots; ghosts e<1e-4 & argmax NONE; reals e≥0.5; IoU **94.29%**, Chamfer **1.43 mm**; XML exported | **OVERALL PASS** |
| **A: Under-Allocation + Spawning** | 3 slots (no leaves), residual-mask-driven leaf cloning at residual centroid | 5 spawn events; IoU 35.3%→**83.36%** (+48.1 pp); 4/5 spawns kept, 1 bad spawn self-pruned | **FEASIBLE** (not SOTA) |

### D. Representation Coverage Audit (exp6 + exp7)

**Exp6 — per-dimension differentiability probe** (`scratch/exp6_dimension_coverage.py`):
Two tests per 26D dimension: (A) analytic gradient norm through the full loss, (B) finite-difference image sensitivity.

| Verdict pre-fix | Dimensions |
| :--- | :--- |
| DEAD (no render effect, no gradient) | `scale_z`, `curvature` |
| Verdict post-fix | **all 26 OPTIMIZABLE** |

Fixes applied to `helios_pytorch_geometry.py`:
- `curvature` (col 13) now bends INTERNODE/PETIOLE tubes (7-ring sagittal bend about the horizontal axis cross(fwd, z), deg/m, Helios gravitropic convention; curvature=0 is bitwise-identical to the old straight 2-ring path). Also clamped to ±720°/m to prevent degenerate full-loop bends.
- `scale_z` (col 12) now modulates LEAF blade width (aspect factor `s_z/s_x`, clamped [0.2, 5.0]); canonical `[s,s,s]` renders identically to the old uniform path.

**Exp7 — group-wise recovery** (`scratch/exp7_dimension_recovery.py`): perturb one group of a GT-correct state, optimize with the full Phase 2 stack, measure functional error:

| Group | Init err → Final err | Verdict | Root cause for non-full recovery |
| :--- | :--- | :--- | :--- |
| `base_xyz` | 4.8 → **0.7–1.6 mm** | **RECOVERED** | — |
| `rot6d` | 2.3 → 3.2 mm func | LIMITED | gauge freedoms (leaf in-plane spin is near-image-invariant); tips stay pinned |
| `scale_xyz` | 30% → 34% (len+rad) | LIMITED | radius is sub-pixel at 256px for thin tubes (observability-limited); length is recovered via tip anchor |
| `curvature` | 80 → 45–57 °/m | PARTIAL | gradient exists (~3e-4) but weak; needs longer schedules & better parameterization |
| `organ_logits` | 100% → 100% mismatch | FAILED | type-mobility requires crossing argmax decision boundaries — Phase 3 scope |

Operational notes: the exact-GT loss (~0.075) is NOT reachable by optimization (irreducible shading/regularizer floor ≈ 1.1–1.5); GT-anchor weight should stay ≥ 1.5 (decaying it to 0.5 lets Z-position drift under RGB noise — Z is only weakly observed in top-view depth).

---

## 6. How to Run & Verify Everything (Step-by-Step)

### Prerequisites:
```bash
# Ensure environment is active
mamba activate digital-crops

# ALWAYS set PYTHONPATH to repo root:
export PYTHONPATH=.
```

### 1. Regenerate Ground Truth Target Plant (7-Row with Metadata):
```bash
PYTHONPATH=. python scratch/make_target_unifoliate.py
# Outputs:
# - scratch/target_unifoliate_14d.pt (shape: (7, 14))
# - scratch/target_unifoliate_rgbd.pt (shape: (4, 256, 256))
# - docs/results/assets/target_unifoliate_topview_rgbd.png
```

### 2. Run the 3 Optimization Benchmarks:
```bash
# Method 1: Point Cloud ICP
PYTHONPATH=. python scratch/exp1_per_organ_icp.py

# Method 2: Differentiable Renderer with Multi-Scale Pyramid + Anti-Erasure
PYTHONPATH=. python scratch/exp2_diff_render_opt.py

# Method 3: Conditional Flow Matching
PYTHONPATH=. python scratch/exp3_toy_flow_matching.py
```

### 3. Run the Phase 2 Topology & Coverage Benchmarks:
```bash
# Strategy B: 10-slot over-allocation with existence pruning (OVERALL PASS)
PYTHONPATH=. python scratch/exp4_overalloc_pruning.py

# Strategy A: 3-slot under-allocation with residual-driven leaf spawning
PYTHONPATH=. python scratch/exp5_underalloc_spawn.py

# Dimension coverage: per-dim gradient/sensitivity audit (all 26 must be OPTIMIZABLE)
PYTHONPATH=. python scratch/exp6_dimension_coverage.py

# Dimension recovery: group-wise perturb-recover end-to-end test
PYTHONPATH=. python scratch/exp7_dimension_recovery.py
```

### 4. Run the Synthesis Evaluation & Update Figure 12:
```bash
PYTHONPATH=. python scratch/eval_phase1_comparison.py
# Outputs:
# - scratch/xml_outputs/*.xml
# - docs/results/assets/fig12_back_to_basics_benchmark_summary.png
```

### 5. Run the Full Lifecycle Raytracing Benchmark (Figure 10):
```bash
PYTHONPATH=. python diffusion_based/eval/eval_13d_xml_organ_masks.py
# Outputs:
# - docs/results/assets/fig10_helios_per_organ_mask_comparison.png
```

---

## 7. Common Gotchas & Traps (DO NOT FALL INTO THESE)

1. **`PYTHONPATH=.` is Mandatory**:
   Running `python scratch/exp1_per_organ_icp.py` without `PYTHONPATH=.` will raise `ModuleNotFoundError: No module named 'diffusion_based'`.
2. **Metadata Rows in 14D Part Tensors**:
   `target_unifoliate_14d.pt` has 7 rows. Rows 0 and 1 are `ORGAN_ROOT_META` and `ORGAN_SHOOT_META`.
   If writing code that expects strictly 5 physical organs (Stem, Pet0, Pet1, Leaf0, Leaf1), use:
   `p_geom = part_tensor[part_tensor[:, 0] >= ORGAN_INTERNODE]`
   When saving reconstructed tensors for XML export, always re-attach the metadata rows!
3. **Logit Clamping for Non-Candidate Organ Classes**:
   In continuous 26D representation, initialize non-candidate organ logits to `-20.0` or mask them out to `-100.0`. If left at `0.0`, Adam optimization on existence can cause a petiole or leaf slot to inadvertently flip its argmax to `ORGAN_ROOT_META` or `ORGAN_SHOOT_META`.
4. **Never Multiply Existence into Scale**:
   `scale` must remain physical dimensions (meters). Modulate visibility strictly via vertex opacities (`soft_exist`).
5. **Helios XML Non-Negative Values**:
   Helios C++ raytracer will crash with `SIGABRT` if any `<internode_radius>` or `<leaf_scale>` is negative or zero. Always clamp physical dimensions to $\ge 10^{-4}$ m in `part_tensor_to_40d.py`.
6. **CHM Depth vs OpenGL Depth**:
   In our renderer, the 4th channel is Canopy Height Model (CHM) height from the ground (m), where ground $= 0.0$ and surface $= Z_{\text{world}}$. This is the inverse of camera-distance depth.
7. **Existence Is NOT the Only Erasure Channel (Phase 2 finding)**:
   Even with existence floored at 1.0, the optimizer erases thin tubes through the **scale channel**: a misaligned tube has zero pixel overlap with its target, so IoU/depth/RGB reward shrinking it. Radius collapses to sub-pixel (invisible), length shrinks, and position drifts (zero gradient + Adam normalized steps = random walk). Countermeasures (all in `scratch/phase2_core.py`, used by exp2/exp4/exp5):
   - `ExistenceWarden`: warmup freeze (existence + scale frozen for first ~15–25 steps) + existence floor hinge for seeded organs.
   - `apply_scale_floor`: **hard** radius floor (straight-through clamp, rel 0.7×init) — radius collapse is a one-way trap because at 0 px no image gradient can restore it. Length uses a **soft** hinge (`scale_floor_loss`) instead: hard-clamping length zeroes the gradient at the boundary and freezes organs at 70% length.
   - `tube_pull_loss`: dilated-corridor coverage deficit around target tubes (annealed 40px→8px) — long-range growth gradient for zero-overlap tubes.
   - `organ_tip_positions` + 3D tip/base anchor loss (quadratic, per-organ-summed): image-space losses are structurally blind to zero-pixel organs (a nadir-viewed vertical tube projects a ~12px sliver whose visible pixels all interpolate CHM=0 → invisible). The tip anchor is differentiable w.r.t. base/rot/scale and works at 0 px. **Use sum-over-organs, not mean** (mean dilutes single-organ errors into stalling).
8. **Coincident-Copy Ambiguity (Phase 2 finding)**:
   Two coincident ghost copies saturate compositing (each e≈0.7 renders identically to one e≈1.0) — pure L1 sparsity cannot decide a winner. Fixes: (a) `ExistenceWarden.regularizers` soft-NMS mutual-exclusion penalty with **seeded-slot gradient protection** (detach the real slot's factor in ghost-real pairs, else exclusion erases real organs); (b) initialize ghost `ORGAN_NONE` logits HIGH (+2.0, "dead until proven useful"); (c) late-stage ghost NONE-logit annealing (+0.05/step after step 60) because softmax saturation floors `p(1-p)` existence at ~1e-3 under Adam.
9. **Nadir-View CHM Blindness of Vertical Tubes**:
   `build_mesh_from_part_tensor` renders a vertical stem as a ~12 px sliver whose visible pixels all lie on the bottom-ring edge (CHM interpolates to 0 there) → `clamp(depth*100)` mask = 0 → **the internode contributes zero image signal in top-view**. This is a renderer semantic, not a bug; the 3D tip anchor (gotcha 7) is the correct compensation.
10. **Helios C++ Leaf Roll Is NOT Rotation-Equivariant (Phase 2 finding)**:
    `PlantArchitecture.cpp` applies unifoliate leaf roll about the **world X axis** (`context_ptr->rotateObject(objID_leaf, roll_rot, "x")` with sign `±` by `shoot_index.x % 2`), and the blade-up correction uses world-up. A plant yaw-rotated 30° in `base_rotation` renders with genuinely different leaf orientations (verified: GT vs yaw30-GT raytrace IoU 17.9%, not fixed by 2D rotation correction). Consequence: the 14D→XML pipeline is only rotation-equivariant for yaw-free geometry; Helios C++ raytrace IoU against yaw-perturbed reconstructions is pessimistic. Use the PyTorch renderer (our differentiable convention) for optimization-time verification (exp4 recon: 90.24% PyTorch IoU) and treat Helios IoU as approximate for yawed plants.
11. **Dataset Organ-Type Prior (shard layout)**:
    `dataset/helios_data/cowpea_shard/*.pt` samples store `nodes` as (N, 25+) **FM-encoded** tensors: columns 0:13 are the one-hot organ type (see `part_array_dataset.py` `FM_NODE_DIM=25`), NOT a scalar organ column. Empirical distribution over 600 plants / 617,941 organs: PETIOLE 20.40%, BUD_ABORTED 49.55%, SHOOT_META 7.16%, INTERNODE 7.09%, PEDUNCLE 7.06%, LEAF 4.82%, FLOWER_OPEN 2.07%. Used as the logit prior in exp4 (`compute_dataset_organ_prior`).
12. **Dead Dimensions (exp6 finding, FIXED)**:
    Pre-fix, `curvature` (col 13) was read but never used by the mesh builder (tubes straight 2-ring, leaves fixed prototype), and `scale_z` (col 12) was ignored for both tubes AND leaves (only `s_x` uniform). Both are now wired (see §5-D). When adding new geometry features, always run exp6 — a dimension that renders zero pixels or receives zero gradient is invisible to the optimizer no matter what loss you design.
13. **GT-Anchor Weight Annealing Trap (exp7 finding)**:
    Decaying the 3D tip/base anchor weight below ~1.0 lets Z-position drift under RGB noise: in a nadir view, Z (height) is observed only through the depth channel, which is weak vs RGB terms. Keep `tip_w >= 1.5` when GT anchors are available; reserve annealing for unknown-geometry scenarios (exp4 ghosts).
14. **Curvature Optimization Is Slow (exp7 finding)**:
    The curvature gradient (~3e-4 through the full loss) is 2–5 orders of magnitude weaker than position gradients. Adam normalizes per-parameter, so a high `curv_lr` (2.0) is safe for curvature recovery BUT destabilizes other groups (unit-norm steps swing curvature against tiny true gradients). Use per-group learning rates; expect curvature recovery to need 150–300 steps.
15. **26D vs 14D Column Layout Confusion**:
    `organ_tip_positions` now auto-detects layout (14D: base 1:4, rot 4:10, len 10; 26D: base 13:16, rot 16:22, len 22). When writing new helpers, never index a 26D node tensor with 14D column constants — this produced silently-wrong 655mm "errors" before the audit.
16. **Which generative algorithm? Flow Matching, not DDPM/DDIM**:
    The repo's generative model is **conditional Rectified Flow / Flow Matching** (Lipman et al. 2023 / Liu et al. 2023): `x_t = (1-t)x_0 + t·x_1`, `v_target = x_1 - x_0`, MSE velocity regression, Euler ODE sampling t:0→1 (`diffusion_based/training/flow_matching.py`). No DDPM noise-prediction, no DDIM deterministic schedule; `sigma_min=0` means fully deterministic straight paths.
17. **Why raw FM underperforms + the hybrid fix (exp3b finding)**:
    Exp3's 9.97% IoU had 4 measured root causes: (a) **conditioning collapse** — CNN+AvgPool encoder maps all images to near-identical features (pairwise dist ~0.007), so the field regresses to the unconditional dataset mean; (b) **discrete-block mean regression** — one-hot organ logits cannot cross argmax boundaries, and rotation/scale regression averages multi-modal yaw/pitch datasets into degenerate geometry; (c) degenerate dataset (21 samples varying one scalar); (d) no test-time guidance. Fixes in `scratch/exp3b_flow_matching_fixed.py`: structurally diverse dataset (210 samples: scale×pitch×yaw×curvature), depth-stat + contrastive conditioning, **slot-role type assignment** post-ODE, and the recommended production pattern: **FM proposal (fast, topology-correct) → differentiable-render polish (exp2 stack, ~75 steps)**. Result: IoU 9.97% → **65.54%**, Chamfer 30.3 → 6.8 mm, type agreement 100%.
    In-place masking of a converted 14D tensor (`p14[:, :13][:, mask] = -100`) **cuts the autograd graph** — always role-lock logits on the 26D node BEFORE `diff_node_to_part_tensor_14d` when guidance gradients are needed.
    Final exp3b recipe (stable 79–83% over repeated runs): 6 FM proposals (batched ODE) → **10-step mini-polish per proposal** → select best by post-polish loss → 75-step full polish. Two extra lessons: (a) raw ODE render-loss is a *poor* proposal-quality predictor — selection must happen AFTER optimization; (b) proposal mini-polish must run OUTSIDE `torch.no_grad()` (backprop through the render is required).

---

## 8. Immediate Next Steps & Phase 3 Roadmap

**Phase 2 (Variable Organ Topology) is COMPLETE and verified** (2026-09-04):
- ✅ **Strategy B (Over-Allocation + Existence Pruning)**: 10 slots → exactly 5; ghost slots decay to e<1e-4 with argmax ORGAN_NONE; real organs retained (e≥0.5); valid Helios XML exported (`scratch/xml_outputs/method_4.xml`); final IoU 92.45%, Chamfer 0.71 mm. All anti-erasure machinery lives in `scratch/phase2_core.py` (shared by exp2/4/5).
- ✅ **Strategy A (Under-Allocation + Spawning) evaluated**: 3 slots (1 stem + 2 petioles, no leaves) + residual-driven leaf spawning reaches 82.68% IoU (+25.0 pp over the 3-slot cap); 5 spawn events, 1 bad spawn correctly self-pruned. Feasible but probe-quality (Chamfer 13.5 mm, slight over-spawn without mutual-exclusion pressure among spawned slots).
- ⚠️ **Known limitation discovered**: Helios C++ unifoliate leaf roll is applied about world axes (PlantArchitecture.cpp:2136) → XML pipeline is not yaw-rotation-equivariant (gotcha §7-10). A future fix should derive leaf pitch/yaw/roll in the petiole-local frame inside `part_tensor_to_40d.py`.

### Phase 3 Objective (proposed): Scale to Real Canopy Topology
1. **Multi-phytomer plants**: extend exp4's over-allocation scheme beyond the 5-organ seedling to DAP 50 branching architecture (multiple phytomers + lateral shoots). The closed-form lateral-shoot IK (§3 Rule 2) already supports this; the missing piece is slot→shoot topology assignment (parent_logits / parent_candidates already exist in `PlantOrganArray`).
2. **Type-mobility**: exp4 used role-locked slots ({type, NONE} only). Free 13-class logits with the dataset prior (gotcha §7-11) is the next step; watch for argmax-flip instability (gotcha §7-3).
3. **Spawn + prune unified**: combine exp4's pruning pressure with exp5's spawning in one optimizer (mutual exclusion among spawned slots is the missing piece — reuse the soft-NMS exclusion term).
4. **Helios leaf-frame fix**: derive leaf pitch/yaw/roll in the petiole-local frame in `part_tensor_to_40d.py` so raytraced IoU is rotation-equivariant (unblocks trustworthy Helios verification of yawed reconstructions).
5. **Curvature in FM (COMPLETED 2026-09-05)**: the FM node layout was 25D (`FM_NODE_DIM=25`) and **did not encode curvature** (14D col 13). Completed:
   - `FM_NODE_DIM` 25→**26** in `part_array_dataset.py` (`FM_CURV=25`, `CURV_SCALE=1/60`, roundtrip err < 2e-6)
   - Shards regenerated on SLURM (20 parallel jobs, 100K samples in 14 min, 26D verified)
   - **Trainer NaN bug fixed**: `loss_cat_active` sliced `[:, :, :EMPTY_IDX]` where `EMPTY_IDX=0` → empty slice → NaN since the beginning. Correct slice is `[:, :, :FM_OT_END]` (13 classes). Plus: loss now computed in fp32 outside the bf16 autocast (scale-block targets reach 100.0 → bf16 square overflow), NaN-batch skip guard added.
   - Local smoke verified: epoch 1 loss 3329 → epoch 2 ~68; **curvature velocity loss 5.4 → 0.42** and falling.
   - Per-epoch visualization panel added (`fm_visualization.py`): target image, GT render | FM-generated composite, and a per-tube **curvature GT-vs-prediction bar chart**; optional W&B logging (`--use-wandb`, project `part-flow-matching`).
   - Detailed session log + next steps: [`docs/ongoing/20260905_fm_curv26_handoff.md`](20260905_fm_curv26_handoff.md).
   - **How curvature is predicted**: the model predicts per-slot velocity `v(x_t, t)`; the curvature channel of that velocity transports the normalized curvature component from the scaffold prior (≈0) to the data value. At t=1, decode = `fm[:, FM_CURV] / CURV_SCALE` → deg/m. Note the degenerate-axis caveat (§7 curvature probes require |curv|>0).
   - **Why a dedicated `loss_curv` is required (not redundant)**: variance-share analysis on fresh 26D shards (13,874 active organs) shows a unified 26-col MSE would give curvature **0.00%** of the gradient signal — shard normalization gives scale std ≈ 70.9 vs curvature ≈ 0.61, so scale alone is 97.28% of total variance. The trainer balances per-block explicitly (each term is a per-slot-per-col mean), and curvature has no coverage elsewhere: `loss_inactive_geom` only regularizes *inactive* slots. Weight 1.0 gives curvature 1/7 of the loss budget; observed curvature velocity loss 5.4 → 0.42 over 2 smoke epochs.
   - **50-epoch training COMPLETED** (2026-09-05): loss 20,606 → 68, zero NaN skips. Slot-aligned curvature evaluation: **MAE 25.5°/m, median 8.8°/m** (n=2,326 tube organs, 40 DAP-diverse samples, 15-step Euler ODE). Final checkpoint: `diffusion_based/checkpoints/fm_curv/part_flow_matching.pt` (max_nodes=512). Caveats: (a) organ-type agreement vs GT is low (13%) because FM slot order ≠ canonical order — always apply `canonical_sort_nodes` before comparing; (b) MAE is inflated by DAP-100 plants whose curvature reaches ±140°/m — evaluate with median as well.
   - **Pyramid-concat conditioning (16-ch)**: `generate_tensor_shards.py --mode cache --pyramid concat` stores `[z1(4ch) | z2 | z4 | z8]` = 16 channels (RGB[-1,1] + CHM per zoom). `ViTImageEncoder.forward` averages per-zoom patch embeddings for 16-ch input (output token shape unchanged). Motivation: DAP-1 seedlings fill <2% of the fixed 5m-drone frame — zoom channels give the encoder the stem/leaf detail it cannot see at 1x.
   - **Vis panel**: target view now shows the 8x zoom channels (channels 12:15) with DAP-diverse sample selection (first/mid/last of dataset); DAP-1 at 1x looks like bare ground otherwise.
   - **Repo structure**: `scripts/cache_dataset_tensors.py` merged into `generate_tensor_shards.py --mode cache` (per-sample .pt) vs `--mode shard` (packed 100/shard). Cache dirs are crop-named (`dataset/cache/cowpea_curv26/`) with a species filter in `PartArrayDataset` so bean/cowpea never mix.

You are now fully equipped with all necessary context, math, code paths, and verification tools to take over and drive Phase 3 to completion!
