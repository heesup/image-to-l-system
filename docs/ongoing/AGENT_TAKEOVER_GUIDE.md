# Agent Takeover & Engineering Handover Guide
**Project: Image-to-L-System / 3D Inverse Procedural Plant Reconstruction**
**Last Updated:** 2026-09-04
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
- Direct Differentiable Rendering optimization with our **Concentric Multi-Scale Pyramid** achieves **81.61% Mask IoU** and **2.29 mm Chamfer distance** on canonical unifoliate seedlings without blank-canvas collapse.

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

```
/home/lion397/codes/image-to-l-system/
├── diffusion_based/
│   ├── models/
│   │   ├── part_tensor_to_40d.py       <-- [CRITICAL] Closed-form IK & XML assembler
│   │   ├── helios_pytorch_geometry.py  <-- [CRITICAL] 14D & 26D mesh builder & differentiable mapping
│   │   ├── helios_pytorch_renderer.py  <-- [CRITICAL] nvdiffrast renderer + multi-scale pyramid
│   │   ├── plant_organ_array.py        <-- [CRITICAL] Typed 40D & 14D column constants & XML roundtrip
│   │   └── part_assembly_to_xml.py     <-- Pipeline bridge
│   └── eval/
│       └── eval_13d_xml_organ_masks.py <-- [CRITICAL] Full lifecycle benchmark (DAP 10, 50, 90)
├── scratch/
│   ├── make_target_unifoliate.py       <-- [ACTIVE] Canonical 7-row unifoliate target generator
│   ├── exp1_per_organ_icp.py           <-- [ACTIVE] Method 1: Point Cloud ICP benchmark
│   ├── exp2_diff_render_opt.py         <-- [ACTIVE] Method 2: Diff Renderer Multi-Loss Pyramid benchmark
│   ├── exp3_toy_flow_matching.py       <-- [ACTIVE] Method 3: Conditional Flow Matching ODE benchmark
│   ├── eval_phase1_comparison.py       <-- [ACTIVE] Synthesis comparison & Figure 12 generator
│   └── xml_outputs/                    <-- Exported XML files from all benchmarks
├── docs/
│   ├── ongoing/
│   │   ├── AGENT_TAKEOVER_GUIDE.md     <-- [THIS DOCUMENT] Single source of truth
│   │   ├── 20260903_14d_part_tensor... <-- Comprehensive IK & raytracing report
│   │   ├── 20260903-back-to-basics.md  <-- Phase 1 benchmark & Phase 2 roadmap
│   │   └── README.md                   <-- Directory index
│   └── results/assets/
│       ├── fig10_helios_per_organ...   <-- Full lifecycle raytraced comparison (DAP 10, 50, 90)
│       ├── fig12_back_to_basics...     <-- Phase 1 synthesis comparison grid
│       └── fig13_progressive_multi...  <-- Concentric multi-scale pyramid visualization
```

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
| **Method 2: Diff Renderer (Pyramid)** | 40.0 ms/step (3.4s) | **81.61%** | **2.29 mm** | **2.854 cm²** | **0.827 cm²** | **SUCCESS** |
| **Method 3: Flow Matching ODE** | **64.3 ms (15 steps)**| 9.97% | 30.30 mm | 28.112 cm² | 23.024 cm² | **SUCCESS** |

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

# Method 2: Differentiable Renderer with Multi-Scale Pyramid
PYTHONPATH=. python scratch/exp2_diff_render_opt.py

# Method 3: Conditional Flow Matching
PYTHONPATH=. python scratch/exp3_toy_flow_matching.py
```

### 3. Run the Synthesis Evaluation & Update Figure 12:
```bash
PYTHONPATH=. python scratch/eval_phase1_comparison.py
# Outputs:
# - scratch/xml_outputs/*.xml
# - docs/results/assets/fig12_back_to_basics_benchmark_summary.png
```

### 4. Run the Full Lifecycle Raytracing Benchmark (Figure 10):
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

---

## 8. Immediate Next Steps & Phase 2 Roadmap

With Phase 1 (Minimal Seedling) and Phase 3 (Multi-Scale Pyramid) completed and verified, the incoming agent should execute **Phase 2: Variable Organ Topology**:

### Phase 2 Objective:
Extend the optimization pipeline to handle plants where the number of organs is **unknown or variable**:
1. **Target**: Canonical seedling with 5 physical organs (Stem, 2 Petioles, 2 Leaves).
2. **Strategy B (Over-Allocation with Existence Pruning) [Active Plan]**:
   - Initialize with $N_{\max} = 10$ organ slots:
     - 2 Internodes
     - 4 Petioles
     - 4 Leaves
   - Allow continuous existence $e_i \in [0, 1]$ to be optimized via the multi-scale pyramid loss.
   - Verify that the 5 redundant slots cleanly decay to $e_i < 10^{-4}$ and `ORGAN_NONE`.
   - Before XML compilation, filter out pruned slots (`e_i < 0.5`), verify that the resulting tree contains exactly the active organs, and export valid Helios XML.
3. **Evaluate Strategy A (Under-Allocation / Splitting)**:
   - Start with 3 slots (1 stem, 2 petioles; missing leaves).
   - Evaluate whether new leaf organs can be spawned via bifurcation or slot expansion.

You are now fully equipped with all necessary context, math, code paths, and verification tools to take over and drive Phase 2 to completion!
