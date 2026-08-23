# Cluster Power & Cost Estimate — 2026-08-22

> **Context**: During the cowpea 100k dataset generation pipeline run (40 unified SLURM jobs,
> 22 actively running at time of capture, 18 PENDING in FIFO queue).
> Captured at: `2026-08-22 14:24 PDT`

---

## 1. Active GPU Inventory (22 SLURM jobs × gres=gpu:1)

| Node | GPU Model | Architecture | GPUs Used | TDP | Estimated Load | Power Draw |
|------|-----------|-------------|-----------|-----|----------------|------------|
| `gpu-5-58` | TITAN RTX | sm_75 | 5 | 280 W | ~40% (Phase 1 CPU) | **~560 W** |
| `gpu-10-58` | H100 SXM | sm_90 | 3 | 700 W | ~60% | **~1,260 W** |
| `gpu-12-92` | Tesla V100 | sm_70 | 1 | 300 W | ~50% | **~150 W** |
| `gpu-4-56` | A100 80GB | sm_80 | 4 | 400 W | ~65% | **~1,040 W** |
| `gpu-3-38` | A100 80GB | sm_80 | 4 | 400 W | ~65% | **~1,040 W** |
| `gpu-5-46` | A100 80GB | sm_80 | 4 | 400 W | ~65% | **~1,040 W** |
| `gpu-4-54` | RTX A5500 | sm_86 | 2 | 230 W | **77 W** (observed via `nvidia-smi`) | **~154 W** |
| `gpu-10-54` | RTX 6000 Ada | sm_89 | 1 | 300 W | ~15% (desktop session) | **~45 W** |
| **Total GPU** | | | **24** | | | **~5,289 W** |

> **Note**: Phase 1 (Helios C++ XML generation) is CPU-bound — GPU utilization is
> 20–40% during this phase. Phase 2 (nvdiffrast differentiable rendering) is GPU-bound
> at 60–80% utilization. The A5500 observed draw of 77 W was captured live from
> `nvidia-smi` during Phase 2 execution.

---

## 2. Server Baseline Power (CPU + RAM + Networking + Cooling)

| Item | Estimate |
|------|----------|
| Active nodes | 8 nodes |
| Per-node baseline (Dual-Xeon + 512 GB RAM + NVLink/PCIe) | ~350 W |
| **Server subtotal** | **~2,800 W** |

---

## 3. Total Estimated Power Draw

```
GPU load:        ~5,289 W
Server baseline: ~2,800 W
─────────────────────────
Total:           ~8,089 W  ≈  8.1 kW
```

---

## 4. Cloud Market Equivalent Cost (Lambda Labs / CoreWeave rates, Aug 2026)

| GPU Model | Qty | Rate ($/hr/GPU) | Subtotal |
|-----------|-----|----------------|----------|
| H100 SXM | 3 | $3.99 | $11.97/hr |
| A100 80GB | 12 | $2.00 | $24.00/hr |
| Tesla V100 | 1 | $0.80 | $0.80/hr |
| TITAN RTX | 5 | $0.40 | $2.00/hr |
| RTX A5500 | 2 | $0.80 | $1.60/hr |
| RTX 6000 Ada | 1 | $1.50 | $1.50/hr |
| **Total** | **24** | | **$41.87/hr** |

> ~$41.87/hr ≈ **₩57,500/hr** (KRW at Aug 2026 rate ~1,374 KRW/USD)

---

## 5. Full Pipeline Cost Projection

| Scenario | Duration | Estimated Cost |
|----------|----------|----------------|
| Optimistic (all jobs fast, no faults) | ~3 hrs | ~$126 |
| Realistic (FIFO queue, Phase 2 rendering) | ~4–5 hrs | ~$167–$209 |
| Worst-case (re-queues + faults) | ~6–8 hrs | ~$251–$335 |

> ✅ **Actual cost: $0** — this compute is provided free of charge via university HPC cluster allocation.

---

## 6. Reference: `gpu-free` Snapshot

```
gpu-5-50      titan        4/ 4 free  DOWN+DRAIN+INVALID_REG
gpu-10-54     6000_ada     3/ 4 free  ALLOCATED
gpu-4-54      a5500        2/ 4 free  MIXED+PLANNED
gpu-5-58      titan        0/ 5 free  ALLOCATED
gpu-5-46      a100         0/ 4 free  MIXED+PLANNED
gpu-4-56      a100         0/ 8 free  MIXED+PLANNED
gpu-3-38      a100         0/ 4 free  MIXED+PLANNED
gpu-12-92     v100         0/ 1 free  MIXED+PLANNED
gpu-10-58     h100         0/ 4 free  MIXED+PLANNED
gpu-10-50     6000_ada     0/ 4 free  MIXED+PLANNED
```
