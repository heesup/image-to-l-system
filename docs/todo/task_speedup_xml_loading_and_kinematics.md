# Task: Accelerate XML Deserialization & Forward Kinematics (FK) Pipeline

> **Status**: Proposed / High-Priority Optimization  
> **Target Components**: [`diffusion_based/models/helios_pytorch_geometry.py`](../../diffusion_based/models/helios_pytorch_geometry.py) (`extract_part_tensor`), [`diffusion_based/models/plant_organ_array.py`](../../diffusion_based/models/plant_organ_array.py) (`from_xml_file`)  
> **Related Figure**: [`docs/results/assets/fig1_helios_vs_torch_rendering_benchmark.png`](../results/assets/fig1_helios_vs_torch_rendering_benchmark.png)

---

## 1. Problem Statement & Empirical Bottleneck

In our empirical performance benchmarks on the NVIDIA A100 GPU (Figure 1), we observed a striking asymmetry between the **14D Direct Part Renderer** and **End-to-End (`XML → Image`) loading**:

| Pipeline Stage | Mechanism | DAP 10 Latency | DAP 100 Latency ($N=2,546$) | % of E2E Time (DAP 100) |
| :--- | :--- | :---: | :---: | :---: |
| **XML Deserialization** | `PlantOrganArray.from_xml_file()` | 1.4 ms | 30.3 ms | 0.8% |
| **Forward Kinematics (FK)** | `extract_part_tensor()` | **99.2 ms** | **3,769.1 ms** (~3.8 s) | **98.8%** |
| **Vectorized GPU Mesh Build** | `build_mesh_from_part_tensor()` | 6.3 ms | 13.7 ms | 0.36% |
| **GPU Rasterization (512×512)** | `renderer.forward()` (Nvdiffrast) | 1.8 ms | 2.7 ms | 0.07% |
| **Total 14D Direct Render** | **Mesh Build + Rasterize** | **8.16 ms** | **16.42 ms** | **0.43%** |
| **Total End-to-End** | **XML $\to$ FK $\to$ Render** | **108.7 ms** | **3,815.8 ms** (~3.8 s) | 100.0% |

### Key Insight:
- The GPU Part Renderer is **1,156x faster** than Helios C++ (16.4 ms vs 18.99 s) because it is fully vectorized with `torch.bmm`.
- However, **98.8% of total E2E latency** is spent in `extract_part_tensor()` traversing the procedural tree in Python, launching ~50,000 tiny individual PyTorch CUDA operations inside a sequential loop.

---

## 2. Root Cause Analysis

1. **Sequential Phytomer Walk in Python**:
   - Plants branch hierarchically: Shoots $\to$ Phytomers $\to$ Internodes $\to$ Petioles $\to$ Leaves / Inflorescences.
   - `extract_part_tensor` steps node-by-node down each shoot to accumulate internode pitch deflections, phyllotactic yaw rotations, and lateral branch angles.
2. **Micro-Kernel Dispatch & PyTorch Dispatcher Overhead**:
   - For every organ row ($N = 2,546$ at DAP 100), the function calls `torch.zeros(14, device=device)`, `torch.linalg.norm()`, `torch.linalg.cross()`, and small matrix multiplications.
   - Launching tens of thousands of micro-kernels from Python incurs severe CPU dispatch latency and driver synchronization stalls.

---

## 3. Proposed Architectures & Implementation Strategies

### Strategy A: Shoot-Level Independence & Chunked Assembly (Recommended)
**Core Idea**: Plants have few shoots ($\sim 10-30$) but many organs ($2,500+$). Lateral shoots are independent kinematic sub-trees attached to specific parent anchors.

1. **Phase 1: Shoot-Local Kinematics (Parallel / Vectorized)**
   - Evaluate each shoot $s$ in its own canonical local reference frame (base at $(0, 0, 0)$, initial stem pointing along $+Z$).
   - Because shoots do not depend on each other's internal geometry, local poses can be computed concurrently using Python multiprocessing / thread pools or batched NumPy arrays.
2. **Phase 2: Shoot-Base Anchor DAG (Shallow Tree)**
   - Build a lightweight Directed Acyclic Graph (DAG) connecting only shoot bases:
     $$\text{Shoot } 0 \longrightarrow \text{Shoot } 1, 2, \dots \longrightarrow \text{Sub-branches}$$
   - Tree depth is shallow ($\le 3$). Computing global $4 \times 4$ base transformation matrices $\mathbf{T}_s$ for all 10–30 shoots takes $< 1\text{ ms}$.
3. **Phase 3: Vectorized Global Pose Assembly**
   - For each shoot $s$, transform all its local organ positions and $SO(3)$ rotation frames to world coordinates in a single batched matrix multiplication:
     $$\mathbf{X}_{\text{world}}^{(s)} = \mathbf{T}_s \times \mathbf{X}_{\text{local}}^{(s)}$$
     $$\mathbf{R}_{\text{world}}^{(s)} = \mathbf{R}_s \times \mathbf{R}_{\text{local}}^{(s)}$$
   - Concatenate shoot slices into the final $(N, 14)$ part tensor.

---

### Strategy B: Vectorized NumPy / PyTorch Phytomer Cumulative Rollout
**Core Idea**: Internodes along a shoot follow a recurrence relation:
$$\mathbf{R}_{k} = \mathbf{R}_{k-1} \cdot \Delta \mathbf{R}(\theta_{\text{pitch}}, \theta_{\text{phyllo}})$$
- Instead of looping phytomer-by-phytomer, represent relative transformations as a batch of rotation matrices or unit quaternions.
- Use associative scan / cumulative matrix products (`torch.cumprod` on quaternions or batched prefix products) to compute all phytomer orientations across a shoot in parallel.

---

### Strategy C: Lightweight C++ / Cython Extractor (`helios_fk_fast`)
**Core Idea**: Move the XML parser and tree kinematics into native compiled C++:
- Native C++ (using `pugixml` or Helios's existing `InputOutput.cpp` AST) processes 2,500 nodes and matrix math in contiguous memory in $< 2\text{ ms}$.
- Expose a simple pybind11 function:
  ```python
  part_tensor = helios_fast_io.xml_to_part_tensor("plant.xml", device="cuda")
  ```
- This completely bypasses Python interpreter overhead and achieves the theoretical maximum speed.

---

### Strategy D: Automatic Binary Sidecar Caching (`.part14d.pt`)
**Core Idea**: Avoid re-parsing XML during recurring experiments:
- When `PlantOrganArray.from_xml_file(xml_path)` or `to_part_tensor()` is called, automatically write a compressed sidecar `.part14d.pt` if not already present.
- Subsequent calls load the raw tensor in $< 1\text{ ms}$, entirely eliminating runtime FK overhead during multi-epoch dataset iterations.

---

## 4. Work Breakdown & Action Items

- [ ] **Task 1: Benchmark Baseline Profiler**:
  - Profile `extract_part_tensor` using `cProfile` / `torch.profiler` to quantify exact time spent in `torch.linalg.norm`, `rotr_*`, and dictionary lookups.
- [ ] **Task 2: Implement Strategy A (Shoot-level chunking)**:
  - Add `extract_shoot_local_tensors()` and `assemble_shoot_hierarchy()`.
  - Validate numerical parity against ground-truth `extract_part_tensor` ($< 10^{-6}$ error on base positions and rot6d).
- [ ] **Task 3: Implement Strategy D (Sidecar caching)**:
  - Add `cache_part_tensor=True` flag to `PlantOrganArray.from_xml_file()`.
- [ ] **Task 4 (Stretch Goal): Fast C++ pybind11 Bridge**:
  - Create a standalone header-only XML-to-14D extractor inside `Digital-Crops/` or `diffusion_based/csrc/`.

---

## 5. Acceptance Criteria

1. **Runtime Target**:
   - DAP 100 XML $\to$ 14D Part Tensor reduced from **3,769 ms to $< 100\text{ ms}$** (pure Python/vectorized) or **$< 5\text{ ms}$** (compiled C++).
2. **Numerical Invariance**:
   - Reconstructed 14D Part Tensor must have $0.000\text{ mm}$ position drift and $< 0.01^\circ$ angular divergence compared to the existing canonical pipeline.
