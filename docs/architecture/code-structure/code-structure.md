---
title: "Repository Structure & Code Map (2026-09-16)"
date: 2026-09-16
tags: [architecture, code-map, structure]
status: active
---

# Repository Structure & Code Map

Canonical layout after the 2026-09-16 restructure. Before that date the main package was
`diffusion_based/`, checkpoints lived under `diffusion_based/checkpoints/`, job logs under
`slurm_scripts/logs/`, the real-image track was `real_world/` and Digital-Crops sat at the repo
root. Everything below uses the new paths; historical docs have been bulk-updated to match.

---

## 1. Top-level layout

```
image-to-l-system/
├── plant_recon/                  # ★ the library: the whole active pipeline
│   ├── models/                   # encoders, flow-matching model, VAE, renderer, export
│   ├── training/                 # trainers, flow scheduler, Hungarian matcher
│   ├── dataset/                  # dataset/cache/packet code (NOT the data itself)
│   └── eval/                     # evaluation + figure scripts, metrics
├── use_cases/real_world/         # sim-to-real application: detectors, real-field datasets, multi-plant pipeline
├── submodules/Digital-Crops/     # git submodule: Helios C++ OptiX engine (shared with other projects)
├── dataset/                      # DATA ONLY: helios_data/, cache/ (gitignored, generated)
├── outputs/                      # ALL generated artifacts (gitignored)
│   ├── checkpoints/              #   model checkpoints (was plant_recon/checkpoints/)
│   ├── logs/                     #   slurm + local run logs and panels (was slurm_scripts/logs/)
│   ├── wandb/                    #   wandb runs (WANDB_DIR exported by the training launcher)
│   ├── weights/                  #   detector base weights (yolo11n*.pt)
│   └── eval/                     #   eval script output dirs (debug_outputs/, output/, roundtrip_outputs/)
├── scripts/                      # cluster launchers ONLY (train_hierarchical_flow_matching.sh, ...)
├── tools/                        # standalone utilities (organize_logs.py, VAE visualizer, figure scripts)
├── tests/                        # pytest suite (imports plant_recon)
├── docs/                         # this documentation tree (see docs/_index.md)
├── archive/                      # superseded code, kept for lineage (see archive/README.md)
│   ├── lm_based/                 #   archived VLM/grammar-token track (+ its L-system dataset code)
│   ├── notebooks_legacy/          #   Track-A notebooks and figures
│   ├── dataset_legacy/           #   Track-A 40D / helios dataset loaders
│   └── ...                       #   earlier generations (root_legacy, models_legacy, ...)
├── environment.yml
└── README.md
```

## 2. What lives where

| Concern | Location |
| :--- | :--- |
| Model definitions (Stages 0–4, VAE, renderer, XML export) | `plant_recon/models/` |
| Training loops & the Hungarian matcher | `plant_recon/training/` |
| Dataset, packet & cache **code** | `plant_recon/dataset/` |
| Evaluation / benchmark / figure scripts | `plant_recon/eval/` |
| Real-image (sim-to-real) application | `use_cases/real_world/` — imports `plant_recon`, never the reverse |
| Helios C++ engine | `submodules/Digital-Crops/` (git submodule, pinned commit) |
| Generated datasets & caches (data) | `dataset/helios_data/`, `dataset/cache/` — see `dataset/README.md` |
| Checkpoints, logs, wandb, weights, eval dumps | `outputs/` (everything gitignored) |
| Cluster job launchers | `scripts/*.sh` (logs are NOT here — they are in `outputs/logs/`) |
| One-off experiments / debug repros | `scratch/<YYYYMMDD_topic>/` (gitignored; see `scratch/README.md`) |
| Superseded code | `archive/` with an index in `archive/README.md` |

## 3. Import conventions

- `PYTHONPATH=.` from the repo root (as before). The library imports as
  `from plant_recon.models...`, `from plant_recon.dataset...`, `from plant_recon.eval...`.
- `use_cases/real_world/` scripts import the library and run as plain scripts
  (`python use_cases/real_world/eval/run_multiplant_scene.py`).
- Archived tracks import from their archive location, e.g.
  `from archive.lm_based.dataset import LSystem` — they remain importable for reference.

## 4. Storage conventions (`outputs/`)

- **One rule**: anything regenerable or generated goes under `outputs/`; nothing inside it is
  committed. Job logs keep the launcher's one-folder-per-start-date layout (`outputs/logs/<YYYYMMDD>/`);
  `python tools/organize_logs.py --apply` files stray logs (see `outputs/logs/README.md`).
- `scripts/train_hierarchical_flow_matching.sh` exports `WANDB_DIR=outputs/wandb` and
  writes checkpoints to `outputs/checkpoints/` (default `OUTPUT_DIR`).
- Detector weights live in `outputs/weights/` (defaults in
  `use_cases/real_world/detector/train_yolo_detector.py` and `scripts/train_real_plant_detector.sh`).

## 5. Promotion rules (from `scratch/`)

- Reusable tool → `tools/`
- Evaluation script → `plant_recon/eval/` (or `use_cases/real_world/eval/` for the real-image track)
- Report figure → `docs/<section>/<topic>/assets/` next to its report
- Everything else → `archive/scratch/<date>_<topic>/` + a row in `archive/README.md`

## 6. Digital-Crops submodule

- Pinned at a commit under `submodules/Digital-Crops`; update with
  `git submodule update --init --recursive` (`.gitmodules` points at `GEMINI-Breeding/Digital-Crops`).
- All code resolves it relative to the repo root (`REPO_ROOT / "submodules/Digital-Crops"`), never
  via a hardcoded home-directory path.
- Its generated build outputs (`projects/syntheticdata_generation/build/...`) are gitignored.
