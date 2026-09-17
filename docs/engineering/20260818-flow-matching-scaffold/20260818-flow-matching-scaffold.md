---
title: "Flow Matching Prior Analysis & 3D Botanical Scaffold Design Report"
date: 2026-08-18
tags: [engineering, implementation]
status: done
---

# Flow Matching Prior Analysis & 3D Botanical Scaffold Design Report

**Date**: 2026-08-20  
**Document Path**: `docs/engineering/20260818-flow-matching-scaffold/20260818-flow-matching-scaffold.md`  
**Core Topics**: Analysis of Zero Plant Array ($x_0 = \mathbf{0}$) limitations, empirical organ ratio statistics, 2048-node scaling, and Uniform Botanical Scaffold Prior design.

---

## 1. Core Problem Diagnosis: Limitations of Zero Plant Array ($x_0 = \mathbf{0}$)

### 1.1 Phase Space Singularity and Mode Collapse
- **Singularity Point Collapse**: At $x_0 = \mathbf{0}$, all $N$ slots originate from the exact single point $(0, 0, 0)$.
- **Gradient Competition**: From the perspective of Flow Matching velocity field $v_\theta(x_t, t, I)$, all slot queries at $t=0$ share identical states. Dispersing them into distinct 3D canopy locations induces severe gradient competition within transformer self-attention.
- **Averaging Bottleneck**:
  - **DAP 10 (Seedling)**: Small juvenile leaves inflate excessively to fill the canvas.
  - **DAP 50/90 (Mature Plants)**: Slots fail to expand outward sufficiently, clumping around the central mean location.

### 1.2 Violation of Optimal Transport Flow Matching (OT-CFM) Mathematical Principles
- Optimal Transport Flow Matching (Lipman et al., 2023) learns straight-line transport paths from a **spatially distributed prior distribution $p_0(x)$** toward the **target data distribution $p_1(x)$**.
- Having initial states $x_0$ pre-dispersed throughout 3D space minimizes trajectory crossings and enables each slot to take shortest-path straight-line trajectories to its nearest organ destination.

---

## 2. Empirical Dataset Statistics

Statistical aggregation across 376 sample plants and 214,796 organ nodes sampled from ~15,000 plant XMLs in the dataset:

### 2.1 Node Count Statistics and Rationale for 2048-Node Scaling

| Statistic | Full Dataset | Bean | Cowpea | Sorghum |
|---|---|---|---|---|
| **Min (Seedling DAP 10)** | 8 | 12 | 10 | 8 |
| **Median** | 152 | 465 | 310 | 50 |
| **Mean** | 571.3 | 824.6 | 843.4 | 43.8 |
| **75th Percentile** | 606.2 | 1,120 | 980 | 50 |
| **90th Percentile** | **1,966.0** | 2,150 | 2,400 | 50 |
| **95th Percentile** | **2,633.0** | 2,626 | 3,852 | 50 |
| **Max** | **4,163** | 2,770 | 4,163 | 50 |

> **Conclusion**:
> - The previous `max_nodes = 512` was sufficient for seedlings (DAP 10) and sorghum ($\le 50$), but **caused severe canopy truncation for mature DAP 50–90 legumes (mean 800–2,000+ organs)**.
> - Scaling to **`max_nodes = 2048`** accommodates **over 90% of the dataset (the vast majority of mature plant canopies)** without truncation.

### 2.2 Empirical Organ Type Distribution

Empirical breakdown across all 214,796 organ nodes:

| Rank | Organ Type | Actual Count | Percentage (%) | 2048-Slot Budget Allocation |
|---|---|---|---|---|
| **1** | **Leaf** | 86,984 | **40.50%** | **830** |
| **2** | **Petiole** | 30,490 | **14.19%** | **290** |
| **3** | **Internode** | 30,239 | **14.08%** | **288** |
| **4** | **Peduncle** | 28,247 | **13.15%** | **270** |
| **5** | **Bud** | 20,505 | **9.55%** | **196** |
| **6** | **Bud Aborted** | 7,742 | **3.60%** | **74** |
| **7** | **Fruit** | 3,459 | **1.61%** | **33** |
| **8** | **Flower / Closed** | 4,585 | **2.13%** | **44** |
| **9** | **Shoot / Root Meta** | 2,545 | **1.19%** | **23** |
| **Total** | **All Organs** | **214,796** | **100.00%** | **2,048** |

```
┌───────────────────────────────────────────────────────────┐
│              Macro Biological Composition                 │
├──────────────────────────────────────┬────────────────────┤
│ • Assimilation Organs (Leaves):      │ 40.5% (~40%)       │
│ • Skeletal Organs (Stem/Petiole/Ped):│ 41.4% (~41%)       │
│ • Reproductive Organs (Flower/Fruit):│ 16.9% (~18%)       │
│ • Base Metadata (Shoot/Root Meta):   │  1.2% (~ 1%)       │
└──────────────────────────────────────┴────────────────────┘
```

---

## 3. Uniform Botanical Scaffold Prior ($x_0 \sim p_{\text{scaffold}}$) Architecture

Rather than zero initialization, $x_0$ is initialized as a **Uniform Botanical Scaffold reflecting empirical statistical proportions and 3D plant morphology**.

```
[Baseline: Zero Initialization]
x0 = (0, 0, 0) single point  --->  All slots start at center and radially expand (trajectory crossings & distortions)

[Proposed: Uniform Botanical Scaffold Prior]
x0 = 2,048 slots uniformly distributed in 3D cylindrical/conical canopy volume via Fibonacci phyllotaxis and empirical organ ratios
   --->  Model observes 2D image, aligns organs in occupied regions, and prunes empty regions via p(empty) -> 1.0
```

### 3.1 3D Spatial Geometric Placement Rules
1. **Stem / Internode (14%)**:
   - Arranged vertically along the central $z$-axis cylinder ($r \le 0.05\text{ m}, z \in [0.0, 0.7]\text{ m}$).
2. **Petiole / Peduncle (27%)**:
   - Branches radially outward from the central stem at $30^\circ \sim 60^\circ$ angles.
3. **Leaf (40%)**:
   - Arranged in a Fibonacci golden-angle ($\theta = 137.5^\circ$) spiral at petiole terminals and outer canopy volume ($r \in [0.08, 0.45]\text{ m}, z \in [0.05, 0.65]\text{ m}$).
4. **Bud / Flower / Fruit (18%)**:
   - Distributed uniformly across stem axils and branching junctions.

### 3.2 Expected Benefits
1. **DAP 10 Seedlings**:
   - Only central juvenile leaf slots are fine-aligned; outer surplus slots are cleanly extinguished with $p(\text{empty}) \rightarrow 1.0$, producing **sharp, compact seedling leaves**.
2. **DAP 50/90 Mature Plants**:
   - Slots are already pre-distributed throughout the 3D canopy volume; rather than forcibly pushing leaves outward from the origin, the network forms large, dense canopies in-place.
3. **Accelerated Training Convergence & Fidelity**:
   - Transport distance ($\|x_1 - x_0\|$) is substantially shortened, simplifying Flow Matching trajectories and maximizing transformer velocity prediction accuracy.

---

## 4. Next Action Items

1. **Implement `BotanicalScaffold` Module (`diffusion_based/models/botanical_scaffold.py`)**:
   - Tensor generator supporting 2048 nodes with 3D Fibonacci phyllotaxis and statistical organ proportions.
2. **Scale `PartArrayDataset` and `PartFlowMatchingModel` to 2048 Nodes**:
   - Support `max_nodes = 2048` and update Flow Matching training loop with Scaffold Prior.
3. **Retrain and Evaluate Flow Matching**:
   - Verify trajectory generation across Cowpea, Bean, and Sorghum: $t=0$ (Scaffold) $\rightarrow t=0.5$ (Pruning & Alignment) $\rightarrow t=1.0$ (3D Plant).
