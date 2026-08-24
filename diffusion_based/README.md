# Diffusion-Based 3D Plant Architecture Reconstruction

**26D Organ-Vector Flow Matching** — reconstructs 3D cowpea plant architecture from a single RGB image across the entire lifecycle (DAP 1–100).

---

## Active Pipeline: 232M DiT-Large Flow Matching

| Component | File | Description |
|-----------|------|-------------|
| **Model** | `models/canonical_cowpea_dit_large.py` | 232M DiT-Large (ViT-16L encoder + Decoder-12L, embed=768, max_slots=4096) |
| **Geometry** | `models/helios_pytorch_geometry.py` | 26D organ array → 3D mesh builder (kinematics, gravitropic curvature) |
| **Renderer** | `models/helios_pytorch_renderer.py` | Multi-modal GPU rasterizer via nvdiffrast (RGB + Depth + Mask + Organ Map) |
| **Parser** | `models/helios_xml_parser.py` | Helios XML → `PlantOrganArray` tensor |
| **Core** | `models/plant_organ_array.py` | 26D organ array data structure, normalization, serialization |
| **XML Export** | `models/part_assembly_to_xml.py` | `PlantOrganArray` → Helios XML reconstruction |
| **Dataset** | `dataset/cowpea_shard_dataset.py` | Streaming `.pt` shard loader with dynamic collation |
| **Sharding** | `dataset/generate_tensor_shards.py` | XML → GPU render → 26D `.pt` tensor shards |
| **Training** | `training/train_cowpea_dit_100k_ddp.py` | 2×H100 DDP training with W&B logging |
| **Eval** | `eval/eval_cowpea_dit_100k.py` | Full lifespan 6-column benchmark |
| **Metrics** | `eval/metrics.py` | Masked SSIM, FG-IoU, Chamfer Distance |

---

## 26D Node Vector Encoding

Each plant organ is a **512 slots × 26D** tensor:

```
Dim  0–11:  One-hot organ type (12 classes)
             0=Root Meta, 1=Shoot Meta, 2=Internode, 3=Petiole,
             4=Leaf, 5=Bud, 6=Peduncle, 7=Flower, 8=Fruit/Pod,
             9=Flower Closed, 10=Bud Aborted, 11=Empty
Dim 12–14:  Base position (x, y, z) / 20.0
Dim 15–20:  6D continuous rotation (R₆D, Gram-Schmidt SO(3))
Dim 21–23:  Scale (sx, sy, sz) / 50.0
Dim     24: Curvature / 100.0
Dim     25: Phyllotactic angle / 180.0
```

---

## Quick Start

### Train (2×H100 DDP)

```bash
sbatch slurm_scripts/train_cowpea_dit_h100_ddp.sh
```

### Train (Single GPU)

```bash
python diffusion_based/training/train_cowpea_dit_100k.py \
    --epochs 60 --batch-size 32 --lr 2e-4 \
    --cache-dir dataset/helios_data/cowpea_shard
```

### Evaluate

```bash
python diffusion_based/eval/eval_cowpea_dit_100k.py
```

---

## Dataset

| Path | Content | Count |
|------|---------|-------|
| `dataset/helios_data/cowpea/` | Raw Helios C++ XML templates | 10,000 (100 seeds × 100 DAPs) |
| `dataset/helios_data/cowpea_shard/` | 26D GPU tensor shards | 100,000 samples |

---

## Legacy

Superseded code (40D, 14D, 15D, 5D representations) is preserved in `*/legacy/` subdirectories:

- `models/legacy/` — Old model architectures
- `training/legacy/` — Old training scripts
- `dataset/legacy/` — Old dataset formats
- `eval/legacy/` — Old evaluation scripts
- `legacy/README_15d_legacy.md` — Original 15D README
- `legacy/README_TECHNICAL_5d_legacy.md` — Original 5D technical design
