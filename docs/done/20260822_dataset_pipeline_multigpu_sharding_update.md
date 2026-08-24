# Image-to-L-System: Dataset Pipeline & Multi-GPU Sharding Update
> **Date**: 2026-08-22  
> **Author**: Antigravity AI  
> **Status**: Ongoing (Unified 100-Seed Batch + 100K 26D Sharding Running on SLURM)

---

## 1. Overview & Background

During the setup of the large-scale Cowpea dataset synthesis pipeline for 100K Flow Matching (26D DiT-Large) training, critical cluster issues and codebase redundancies were identified and resolved:
1. **CUDA Error 209 (`cudaErrorNoKernelImageForDevice`)** caused by microarchitecture mismatch in `nvdiffrast`.
2. **Unresponsive GPU hardware failure** on a specific node slot (`gpu-5-58` GPU 0).
3. **Pipeline Consolidation & Cleanup**: Unified disparate C++ XML simulation and Python GPU sharding into a single master SLURM orchestrator, with root directory alignment under `dataset/helios_data/`.

---

## 2. Key Accomplishments & Technical Solutions

### 2.1 CUDA 209 (`cudaErrorNoKernelImageForDevice`) Resolution
* **Root Cause Analysis**:
  * Node `gpu-4-54` executes on an **RTX A5500 (`sm_86`)** GPU.
  * Running `cuobjdump` on `_nvdiffrast_c.so` revealed that the initial `pip install` only compiled a single architecture Fatbin (`sm_89`, corresponding to the build node `gpu-10-54`'s RTX 6000 Ada). When executed on `sm_86` nodes, the runtime failed with `Cuda error: 209 [cudaFuncGetAttributes(&attr, (void*)fineRasterKernel);]`.
* **Permanent Multi-Architecture Fatbin Build**:
  * Built and installed `nvdiffrast` directly with explicit space-separated architecture flags:
    ```bash
    cd /tmp/nvdiffrast_src
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX" python setup.py install --force
    ```
  * `cuobjdump` inspection verified that native fatbin ELF/PTX codes are now embedded for every target GPU in the cluster:
    * `sm_70` (Tesla V100)
    * `sm_75` (TITAN RTX)
    * `sm_80` (A100)
    * `sm_86` (RTX A5500)
    * `sm_89` (RTX 6000 Ada)
    * `sm_90` (H100)
    * `compute_90` (PTX forward compatibility)
* **Direct Multi-Node Verification**:
  * Verified directly on `gpu-4-54` (A5500 `sm_86`) via `srun`:
    ```
    CUDA dev: NVIDIA RTX A5500 Arch: (8, 6)
    Rasterize SUCCESS on node gpu-4-54! Output: torch.Size([1, 128, 128, 4])
    ```
  * Relaunched all 34 jobs; verified that workers across all node architectures (`gpu-4-54`, `gpu-5-46`, `gpu-3-38`, `gpu-10-58`, `gpu-10-50`) are generating `.pt` shards simultaneously with 0 errors.

### 2.2 Unification of Helios XML Generator & Expansion to 100 Seeds
* **Master Script**: Unified all generation workflows into **`slurm_scripts/generate_helios_dataset_jobs.sh`**.
* **Seed 100 Expansion**: Configured default parameters to `PLANT_TYPES="cowpea"`, `SEEDS=100` (expanded from 50), and `DAP=1..100`, producing 10,000 unique 3D plant XML structural templates.
* **Incremental Generation**: Automatically skips pre-existing samples (`seeds 0..49`) and synthesizes only new seeds (`seeds 50..99`).

### 2.3 Self-Healing SLURM Fault-Tolerance Pattern
* **Failure Analysis**: Node `gpu-5-58` has multiple GPUs. Only GPU 0 (`0000:61:00.0`) suffered an unresponsive driver state (`Failed to get device handle / Unknown Error`), while the other 3 TITAN RTX GPUs remained fully operational.
* **Fault-Tolerant Implementation**:
  * Automated health check probing `nvidia-smi` and PyTorch CUDA tensor allocation on job startup.
  * When a faulty GPU is detected:
    1. **Auto-Resubmission**: Spawns a replacement job (`sbatch "$JOB_SCRIPT"`) which SLURM routes to a healthy GPU.
    2. **Fault Slot Tarpit (Lock)**: Keeps the failed process asleep (`sleep 28800`) to hold the broken GPU slot and prevent other jobs from landing on it.
  * **Live Verification**: `helios_pipe_5_dap26-30` (`37859574`) landed on `gpu-5-58` GPU 0, immediately triggered self-healing auto-resubmit, and spawned replacement `37859589` on `gpu-3-38` (A100) while holding the broken slot.

### 2.4 Codebase Cleanup & Dataset Component Roles
The roles of all dataset-related files are clearly decoupled:
1. **`scripts/generate_helios_dataset.py`** [Phase 1 Engine]: Calls C++ Helios engine to simulate 3D plant growth and write XMLs to `dataset/helios_data/cowpea/`.
2. **`diffusion_based/dataset/generate_tensor_shards.py`** [Phase 2 Engine]: Reads XMLs, performs GPU multi-view rendering + 26D organ encoding, and writes `.pt` tensor shards to `dataset/helios_data/cowpea_shard/`.
3. **`diffusion_based/dataset/cowpea_shard_dataset.py`** [PyTorch DataLoader]: `PlantShardDataset` / `CowpeaShardDataset` streaming loader and dynamic collation for model training.
4. **Deleted Obsolete Files**:
   * Removed `diffusion_based/dataset/generate_cowpea_100k.py` (legacy stub).
   * Removed `slurm_scripts/generate_tensor_shards_jobs.sh` (merged into master pipeline).
   * Removed `slurm_scripts/generate_cowpea_dataset_jobs.sh` (merged into master pipeline).

### 2.5 Large-Scale Canopy Scaling ($N > 2048 \to 4096+$ Slots)
* Upgraded `CanonicalCowpeaDiTLargeModel` (232.43M parameters) and sharding pipeline to `max_slots = 4096`.
* Implemented dynamic sinusoidal position embedding continuation in `_get_slot_pos_embed()`, guaranteeing out-of-the-box support for any canopy size even if $N > 4096$.
* Dynamic mini-batch collation ensures memory efficiency ($N \approx 50$ for young plants does not pay $N=4096$ attention overhead).

---

## 3. Directory & Pipeline Architecture

```
image-to-l-system/
├── slurm_scripts/
│   ├── generate_helios_dataset_jobs.sh     # [Master] Unified C++ XML Synthesis + GPU 26D Sharding Pipeline
│   └── logs/                               # Cluster execution logs
├── dataset/
│   └── helios_data/
│       ├── cowpea/                         # [Raw XMLs] DAP 1~100 × 100 Seeds base 3D plant XMLs
│       └── cowpea_shard/                   # [Shards] 100K 26D Flow Matching .pt tensor shards
└── diffusion_based/
    ├── dataset/
    │   ├── generate_tensor_shards.py       # Standalone GPU rendering & tensor sharding engine
    │   ├── cowpea_shard_dataset.py         # PlantShardDataset (auto-fallback & dynamic batching)
    │   └── part_array_dataset.py           # 26D Flow Matching node layout definition
    ├── models/
    │   ├── helios_pytorch_renderer.py      # Multi-Arch CUDA nvdiffrast PyTorch renderer
    │   └── canonical_cowpea_dit_large.py   # 232.43M DiT-Large architecture (max_slots=4096)
    └── training/
        └── train_cowpea_dit_100k.py        # 100K Shard-based DiT-Large training pipeline
```

---

## 4. Current Cluster Execution Status (2026-08-22 14:08 PDT)

34 unified pipeline jobs are actively executing across all available Farm HPC GPU nodes via dynamic FIFO scheduling:

```bash
# Monitor jobs in real time:
squeue -u $USER
gpu-free
```

* **Full Cluster Utilization**:
  * **27 GPU jobs actively running simultaneously** across `gpu-4-56` (8 A100s), `gpu-5-46` (4 A100s), `gpu-3-38` (4 A100s), `gpu-10-58` (4 H100s), `gpu-10-50` (4 Ada 6000), `gpu-4-54` (A5500), `gpu-12-92` (V100).
  * 8 pending jobs queued in FIFO, automatically launching as earlier workers finish.
  * Self-healing fault tolerance verified: Job `37861674` on `gpu-5-58` auto-spawned `37861703` while holding the bad slot.
* **Pipeline Flow**:
  * Each worker generates ~3 DAPs (300 XMLs) into `dataset/helios_data/cowpea/`, then immediately renders and encodes 2,500 samples (25 shards) into `dataset/helios_data/cowpea_shard/`.
  * Pre-existing 7,810 XML templates are instantly skipped via `_complete()` checks.

---

## 5. Next Steps Workflow Guide

### Step 1: Monitor End-to-End Pipeline
```bash
# Check SLURM job status
squeue -u $USER

# Count generated XML templates (Target: 10,000 files)
ls -1 dataset/helios_data/cowpea/*_plant_*.xml | wc -l

# Count generated 26D tensor shards (Target: 1,000 files = 100K samples)
ls -1 dataset/helios_data/cowpea_shard/*.pt | wc -l
```

### Step 2: Launch 232M DiT-Large Flow Matching Model Training
Once shards are generated (or dynamically streamed):
```bash
python diffusion_based/training/train_cowpea_dit_100k.py \
    --epochs 60 \
    --batch-size 32 \
    --lr 2e-4 \
    --cache-dir dataset/helios_data/cowpea_shard
```
