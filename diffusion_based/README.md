# Diffusion-Based 3D Plant Architecture Reconstruction

This module implements a **15D organ-typed graph diffusion model** that reconstructs a 3D botanical structure from a single RGB plant image.

---

## Node Representation (15D)

| Dim | Meaning |
|-----|---------|
| 0-2 | `x, y, z` base position |
| 3   | `length` / scale |
| 4   | `radius` / thickness |
| 5-7 | `pitch, yaw, roll` (degrees) |
| 8-11| one-hot organ type: `[internode, petiole, leaf, floral_bud]` |
| 12  | `shoot_id` |
| 13  | `phytomer_idx` |
| 14  | `existence` confidence |

---

## Key Components

- **`models/graph_diffuser_3d.py`**: Vision-conditioned transformer decoder with O(N·k) k-NN self-attention, sparse parent prediction, organ-type head, and DAP-budget head.
- **`models/differentiable_renderer_3d.py`**: Differentiable 2D renderer matching Helios `PlantArchitecture` camera/projection.
- **`models/plant_geometry_3d.py`**: Explicit 3D geometry generator from 15D organ graphs / Helios XML. Outputs PLY point clouds and supports a fully-differentiable PyTorch point-cloud sampler.
- **`models/pointcloud_loss_3d.py`**: Differentiable Chamfer 3D loss and PLY loader for target point clouds.
- **`training/train_diffusion_3d.py`**: DDPM training loop with multi-objective graph losses and optional 3D point-cloud Chamfer loss against a target PLY.
- **`eval/compare_xml_helios_3d.py`**: Compare Helios-generated PLY with XML-derived point cloud using Chamfer distance.
- **`dataset/helios_dataset.py` / `helios_xml_parser.py`**: Load Helios `*_vis.jpeg` + `*_plant_*.xml` + `*_params.json` pairs and parse them into 15D tensors.

---

## Quick Start

### Train

```bash
# 3D point-cloud supervised training
python diffusion_based/training/train_diffusion_3d.py \
    --data-dir Digital-Crops/projects/syntheticdata_generation/build/output \
    --epochs 200 \
    --batch-size 2 \
    --pc-loss 1.0 \
    --target-ply data/gaussian_splat/2025-06-17-bed1tier2plant1.ply \
    --pc-samples 1024
```

### Render a parsed plant graph

See `differentiable_renderer_3d.py::DifferentiablePlantRenderer3D.forward`:

```python
renderer = DifferentiablePlantRenderer3D(image_size=720).cuda()
img = renderer(nodes_15d, parent_indices, cam_azimuth_deg=0.0,
               focus_plant=True, camera_params={'camera_height': 1.0, ...},
               background='ground')
```

---

## macOS / Local Dev Notes

- Data generation requires the `Digital-Crops` Helios build (Linux/GPU). On a Mac, copy a pre-generated `build/output/` dataset.
- The model/renderer run on CPU or MPS; use smaller `--batch-size` / `--image-size` if memory is limited.

---

## Status

Implemented and functional. The 3D point-cloud supervision path is new; validate the XML-derived geometry against Helios-generated PLYs with `eval/compare_xml_helios_3d.py` before relying on the Chamfer training loss.
