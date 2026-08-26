# Archive Directory

This directory consolidates legacy, historical, and exploratory scripts accumulated during the evolution of the Image-to-L-System project.

> [!NOTE]
> All code in this archive was working at the time of its archival. Files are placed here because they have been superseded by the canonical **16D Part Assembly & 26D DiT-Large Flow Matching Pipeline**, not because they were broken.

---

## 🏛️ Architectural Evolution Timeline

```
[ Track-A: 15D Graph Diffuser ] (Jun - Aug 11, 2026)
           │
           ▼
[ 14D Part Direct Assembly ] (Aug 12 - 14, 2026)
           │
           ▼
[ 94D Fixed Phytomer-Slot ] (Aug 15 - 18, 2026)
           │
           ▼
[ 40D Typed Organ Array (XML Bridge) ] (Aug 19, 2026 - Present)
           │
           ├─▶ [ 16D Part Assembly (GPU Mesh / Diff. Renderer) ] (Active)
           │
           └─▶ [ 26D Canonical Node Vector (DiT-Large FM) ] (Active)
```

---

## 📁 Archive Subdirectory Index

| Directory | Original Path | Era | Description |
|-----------|---------------|-----|-------------|
| [`root_legacy/`](root_legacy/) | `legacy/` | Track-A / Aug 12 | Original differentiable renderer, 15D graph diffuser, and early verification scripts. |
| [`models_legacy/`](models_legacy/) | `diffusion_based/models/legacy/` | 40D / Track-A | Historical models: `plant_global_vae.py`, `plant_shoot_vae.py`, `plant_pure_transformer_vae.py`, `vit_grpo_policy_40d.py`, `organ_array_diffuser_40d.py`. |
| [`training_legacy/`](training_legacy/) | `diffusion_based/training/legacy/` | 40D / Track-A | Historical training scripts: `train_40d_flow_matching.py`, `train_plant_vae.py`, `train_vit_grpo_40d.py`, `train_vit_backprop_40d.py`. |
| [`eval_legacy/`](eval_legacy/) | `diffusion_based/eval/legacy/` | 40D / Track-A | Historical evaluation: 40D backprop demo, 15-strategy deep benchmark, VAE comparison scripts. |
| [`eval_scripts/`](eval_scripts/) | `diffusion_based/eval/test_comparison_*.py` | 40D / Aug 23 | Standalone renderer comparison scripts (`test_comparison_all_renderers_script.py`, `test_comparison_including_40d_script.py`). |
| [`dataset_legacy/`](dataset_legacy/) | `diffusion_based/dataset/legacy/` | 40D | `organ_array_dataset_40d.py` (40D dataset loader). |
| [`dataset_raw_legacy/`](dataset_raw_legacy/) | `dataset/legacy/` | Track-A | `generate_helios_dataset_track_a.py` and old `helios_xml_parser.py`. |
| [`notebooks_legacy/`](notebooks_legacy/) | `notebooks/legacy/` | Track-A | Timing comparisons, leaflet mesh benchmarks, and DAP stability tests. |
| [`scripts_legacy/`](scripts_legacy/) | `scripts/legacy/` | Track-A | Legacy notebook creation helpers. |
| [`tests_render_legacy/`](tests_render_legacy/) | `diffusion_based/tests/legacy/` | 40D | `verify_rendering_equivalence_40d.py`. |
| [`tests_unit_legacy/`](tests_unit_legacy/) | `tests/unit/legacy/` | 40D | 40D unit test suite (`test_part_representation_40d.py`, `test_plant_organ_array_image_backprop_40d.py`, `test_plant_organ_array_xml_roundtrip_40d.py`). |
| [`scratch/`](scratch/) | `scratch/` | Aug 23 - Aug 25 | Active working snapshots: ground clipping verification, multi-species round-trip scripts, XML PR comparison tools, and direct optimization debug scripts. |

---

## 🔄 Canonical Active Pipeline Mapping

For current production code, refer to:
- **XML Parsing & Serialization**: `diffusion_based/models/plant_organ_array.py` (40D Typed format)
- **16D Part Extraction & FK**: `diffusion_based/models/helios_pytorch_geometry.py` (`extract_part_tensor`)
- **Fast GPU Mesh Building**: `diffusion_based/models/helios_pytorch_geometry.py` (`build_mesh_from_part_tensor`)
- **Multi-Modal Differentiable Renderer**: `diffusion_based/models/helios_pytorch_renderer.py` (`render_part_tensor`, `render_multimodal`)
- **DiT-Large Flow Matching Training**: `diffusion_based/training/train_cowpea_dit_100k_ddp.py`
