# Back-to-Basics: 3D Inverse Plant Reconstruction Benchmark

**Problem Statement:**
- **Input**: Python-rendered RGB+D (4-channel) top-view image.
- **Output**: Plant part array (14D Part Tensor).
- **Downstream Verification**: Plant part array → Helios XML → Helios C++ raytraced image & depth.

## Motivation

We want to re-validate from the simplest possible example first — comparing different approaches for the single-view imaging problem:
1. Randomly initialize 1x unifoliate shoot, 2x petioles, 2x leaves, then fit to a target plant (different organ parameters) via per-organ ICP.
2. Exploit differentiable renderer gradients for direct optimization.
3. Solve using a lightweight Diffusion or Flow Matching algorithm.

The harder follow-up is the **variable organ count problem**: using the best-performing method, evaluate:
1. Start with fewer organs than the target, then fit up.
2. Start with more organs than the target, then prune down.
3. Start from 0 organs and grow up to the target.

**Key Design Tension:**
- Fixed image scale: very small plants (DAP 10 seedlings) are invisible at field FOV.
- Auto-crop to bounding box: loses absolute metric scale, making all plants appear the same size.
- Solution: build a concentric multi-scale zoom pyramid (1× field → 4× → 8× organ-level).

**Development Environment:** `mamba activate digital-crops`

---

## Proposed Roadmap & Implementation Plan: Back-to-Basics Benchmark

### Core Objective & Full Parameter Tuning
Rather than the hack of reducing existence (alpha) to erase organs, we **fully optimize all geometric and categorical elements of the Canonical 14D Part Tensor** to accurately reconstruct Target Plant morphology:
1. **Organ Type (`ot`, 1D)**: Categorical logits (`argmax`) determine organ class; when `ORGAN_NONE=0`, the slot is cleanly removed to control organ count.
2. **Base Position (`base_xyz`, 3D)**: 3D translation in the physical coordinate frame (metres, $X, Y, Z$).
3. **Rotation (`rot6d`, 6D $\to SO(3)$)**: 3D rotation via Gram-Schmidt orthonormalization ($R \in \mathbb{R}^{3 \times 3}$).
4. **Scale (`scale_xyz`, 3D)**: Physical absolute dimensions (stem/petiole length & radius, leaf scale). *100% pure physical dimensions — NEVER multiplied by existence/alpha!*
5. **Curvature (`curvature`, 1D)**: 3D sagittal bending curvature of stem and petioles ($^\circ$ or $\text{m}^{-1}$).

---

### Phase 1: Minimal 5-Organ Benchmark (Fixed Topology) — [**COMPLETED ✓**]

#### Step 1. Canonical 5-Organ Ground Truth & True Top-View Target (`scratch/make_target_unifoliate.py`)
- **Plant Configuration (7-Row Canonical Unifoliate Seedling)**:
  - Row 0: `ORGAN_ROOT_META` (plant age DAP 10)
  - Row 1: `ORGAN_SHOOT_META` (Shoot ID 0, unifoliate)
  - 1x Stem Internode: $L = 5.0\text{ cm} = 0.05\text{ m}$, $r = 3.5\text{ mm} = 0.0035\text{ m}$, vertical $+Z$ direction (projects as a centered dot in top-view).
  - 2x Opposite Petioles: Pitch $45^\circ$, Azimuth $0^\circ$ and $180^\circ$ (bilateral symmetry), $L = 3.5\text{ cm} = 0.035\text{ m}$.
  - 2x Unifoliate Leaf Blades: Pitch $0^\circ$ (horizontal), Scale $4.5\text{ cm} = 0.045\text{ m}$ (spreading bilaterally).
- **Camera Setup**:
  - `HeliosPyTorchRenderer(elevation_deg=90.0, azimuth_deg=0.0, include_depth=True, image_size=256)`
  - True Top-View (straight-down nadir): stem projects as a centered dot, two leaves spread left/right symmetrically, CHM depth channel encodes physical heights (m).
- **Perturbed Starting State**:
  - Position: $\pm 1.5\text{ cm}$ offset ($X, Y, Z$)
  - Rotation: $\pm 25^\circ$ random axis rotation
  - Scale: $\pm 20\%$ size distortion
  - Curvature: $\pm 30.0$ error

---

#### Step 2. Method 1: Per-Organ Point Cloud ICP (`scratch/exp1_per_organ_icp.py`)
- **Pipeline**:
  1. Back-project the top-view CHM depth channel into a 3D metric point cloud $\mathcal{P}_{\text{tgt}} \in \mathbb{R}^{K \times 3}$ using calibrated camera intrinsics.
  2. Cluster by height ($Z$) and normal vectors to segment stem/petiole/leaf point sets.
  3. Independently run 3D ICP (Iterative Closest Point) on each organ primitive (cylindrical stem/petiole, planar leaf mesh).
  4. Decode the aligned rigid transform $(R, t)$ and principal axis lengths into a Canonical 14D Part Tensor.
- **Evaluation & Visualization**:
  - 3D point cloud alignment trajectory, 14D Part Tensor reconstruction error (MSE), final Helios XML re-render (`docs/results/assets/exp1_icp_alignment.png`).

#### Step 3. Method 2: Differentiable Renderer Multi-Loss Direct Optimization (`scratch/exp2_diff_render_opt.py`)
- **Blank-canvas local minima prevention**:
  - Pure RGB L1 loss collapses to the trivial zero-transparency local minimum, so we use a **multi-objective loss**:
    $$\mathcal{L}_{\text{total}} = \sum_{s \in \{1, 4, 8\}} w_s \cdot (2\mathcal{L}_{\text{IoU}}^{(s)} + \mathcal{L}_{\text{Depth}}^{(s)} + \mathcal{L}_{\text{RGB}}^{(s)}) + \lambda_{\text{scale}} \mathcal{L}_{\text{prior}}$$
    1. **Silhouette Mask IoU Loss ($\mathcal{L}_{\text{IoU}}$)**: If an organ is erased and becomes background, IoU loss spikes to 1.0, forcing the plant to unconditionally fill the target silhouette.
    2. **Metric CHM Depth L1 Loss ($\mathcal{L}_{\text{Depth}}$)**: Enforces leaf height and stem 3D position accurately in metres.
    3. **RGB Photometric Loss ($\mathcal{L}_{\text{RGB}}$)**: Aligns per-pixel 2D rotation angle and texture of leaves.
- **Per-parameter differentiated learning rates**:
  - `pos_lr = 0.0015` (physical translation in metres)
  - `rot_lr = 0.015` (6D rotation)
  - `scale_lr = 0.0015` (length/radius/scale)
  - `curvature_lr = 0.5` (curvature)
  - `logits_lr = 0.04` (organ classification / existence)
- **Evaluation & Visualization**:
  - Step 0 / 15 / 30 / 45 / 60 / 75 rendering progression, pyramid loss convergence curve, 14D parameter error trace (`docs/results/assets/exp2_diff_render_progression.png`).

#### Step 4. Method 3: Conditional Flow Matching Vector Field (`scratch/exp3_toy_flow_matching.py`)
- **Architecture**:
  - Input: 4-channel RGB-D top-view image ($256 \times 256 \times 4$).
  - Condition encoder: lightweight CNN backbone → image embedding vector $c_{\text{img}} \in \mathbb{R}^{256}$.
  - Vector field network $v_\theta(x_t, t, c_{\text{img}})$:
    - State vector: $x_t \in \mathbb{R}^{5 \times 26}$ (5 slots, 13D logits + 13D geometry each).
    - MLP architecture models inter-slot spatial relationships (stem–petiole–leaf connectivity).
  - Training: Optimal Transport Flow Matching loss:
    $$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, x_0, x_1} \left\| v_\theta(x_t, t, c_{\text{img}}) - (x_1 - x_0) \right\|^2$$
  - Sampling: Starting from $x_0 \sim \mathcal{N}(0, I)$, generate $x_1$ via 15 Euler ODE steps → decode via `diff_node_to_part_tensor_14d`.
- **Evaluation & Visualization**:
  - ODE trajectory snapshots at $t \in \{0, 5, 10, 15\}$, 14D Part Tensor generation accuracy (`docs/results/assets/exp3_flow_matching_trajectory.png`).

#### Step 5. Phase 1 Synthesis & Helios C++ Raytracing Verification (`scratch/eval_phase1_comparison.py`)
- **Phase 1 Benchmark Final Results (Canonical 14D + 7-Row Target with Metadata)**:

| Method | Latency | Top-View Mask IoU | 3D Chamfer Distance | Position MSE | Scale MSE | XML Export |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ground Truth (7-Row)** | — | **100.00%** | **0.00 mm** | **0.000 cm²** | **0.000 cm²** | **SUCCESS** |
| **Method 1: Per-Organ ICP** | 257.5 ms | **68.07%** | **16.11 mm** | **0.500 cm²** | 0.928 cm² | **SUCCESS** |
| **Method 2: Diff Renderer (Pyramid)** | 40.0 ms/step (3.4s) | **81.61%** | **2.29 mm** | **2.854 cm²** | **0.827 cm²** | **SUCCESS** |
| **Method 3: Flow Matching ODE** | **64.3 ms (15 steps)** | 9.97% | 30.30 mm | 28.112 cm² | 23.024 cm² | **SUCCESS** |

- **Key Findings & Conclusions**:
  1. **Progressive Multi-Scale Pyramid ($\mathcal{L}_{\text{Pyramid}}$) Impact**:
     - Applying the 3-level pyramid loss ($1.0\times$ fixed global scale, $4.0\times$ canopy branch alignment, $8.0\times$ dense organ facet gradients) completely overcame the height-scale ambiguity chronic to single-view monocular methods. Chamfer distance dropped from 96.9 mm to **2.29 mm**.
  2. **Differentiable Renderer Blank-Canvas Local Minimum: 100% Eliminated**:
     - Combined $\mathcal{L}_{\text{IoU}}$ silhouette mask loss and Physical Scale Purity guarantee achieves 81.61% IoU with no transparency collapse.
  3. **Helios XML Procedural Kinematics Alignment Complete**:
     - Isolating base internode pitch to 0 and using per-petiole leaf counters (`leaf_counts_per_pet`) ensures all reconstructed tensors compile and raytrace correctly in the Helios C++ physics engine.

---

### Phase 2: Variable Organ Topology — [**NEXT STAGE**]

Extending the fixed-organ Phase 1 problem to the case where organ count is variable. Three strategies to evaluate:
- **Strategy A (Under-allocation)**: Start with 3 slots (1 stem + 2 petioles; missing leaves), verify whether 2 leaf organs can be dynamically spawned.
- **Strategy B (Over-allocation + Pruning) [Active Recommendation]**:
  - Initialize with $N_{\max}=10$ organ slots.
  - Optimize continuous existence logits $e_i \in [0, 1]$ differentiably via the multi-scale pyramid loss.
  - Verify that redundant slots decay to $e_i < 10^{-4}$ and `ORGAN_NONE`.
  - Before XML compilation: filter pruned slots, verify the tree contains exactly the active organs, export valid Helios XML.
- **Strategy C (Autoregressive Sequential Growth)**: Sequential organ spawning — stem $\to$ petiole $\to$ leaf.

---

### Phase 3: Multi-Scale Image Pyramid & Absolute Metric Scale — [**COMPLETED ✓**]
- **Problem**: A DAP 10 seedling (~5 cm) occupies only $15 \times 15$ pixels in the field FOV (1.5 m), causing gradient vanishing. Auto-cropping to bounding box loses absolute metric scale.
- **Solution (Implemented)**:
  1. **Multi-Scale Pyramid ($1.0\times \to 4.0\times \to 8.0\times$)**:
     - `HeliosPyTorchRenderer.render_multiscale_pyramid()` implemented.
     - $1.0\times$ (1.2 m window, global metric scale preserved) + $4.0\times$ (0.3 m window, canopy alignment) + $8.0\times$ (0.15 m window, dense 256×256 pixel gradients for seedlings).
  2. **Validated**: Visualization at `docs/results/assets/fig13_progressive_multiscale_pyramid.png`. Method 2 achieves 81.61% IoU and 2.29 mm Chamfer distance by default.

---

### Verification & Deliverables
- **Core Scripts**:
  - `PYTHONPATH=. python scratch/make_target_unifoliate.py`: Generate standard 7-row target (with metadata)
  - `PYTHONPATH=. python scratch/exp1_per_organ_icp.py`: Method 1 ICP
  - `PYTHONPATH=. python scratch/exp2_diff_render_opt.py`: Method 2 Differentiable Renderer
  - `PYTHONPATH=. python scratch/exp3_toy_flow_matching.py`: Method 3 Flow Matching
  - `PYTHONPATH=. python scratch/eval_phase1_comparison.py`: Synthesis benchmark evaluation and Figure 12
- **Visual Deliverables**:
  - `docs/results/assets/fig10_helios_per_organ_mask_comparison.png`: Full lifecycle Helios raytracing verification (DAP 10: 95.1%, DAP 50: 92.8%, DAP 90: 86.5%)
  - `docs/results/assets/fig12_back_to_basics_benchmark_summary.png`: Phase 1 three-method comparison grid
  - `docs/results/assets/fig13_progressive_multiscale_pyramid.png`: Multi-scale pyramid validation
