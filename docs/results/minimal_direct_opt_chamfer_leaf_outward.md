# Minimal 3-Organ Direct Optimization: Chamfer Pulls Leaf Outward

**Script**: [`scripts/minimal_direct_opt_depth_chamfer_demo.py`](../../scripts/minimal_direct_opt_depth_chamfer_demo.py)  
**Figure**: [`assets/minimal_direct_opt_depth_chamfer_demo.png`](assets/minimal_direct_opt_depth_chamfer_demo.png)  
**Metrics JSON**: [`assets/minimal_direct_opt_depth_chamfer_demo.json`](assets/minimal_direct_opt_depth_chamfer_demo.json)

---

## What it shows

A self-contained verification that the differentiable PyTorch renderer + geometry
builder can directly optimize a continuous botanical parameter using only a 3D
vertex Chamfer distance.

- **Plant template**: one internode (stem) + one petiole + one leaf.
- **Perturbation**: the initial petiole pitch is 10°, so the leaf is nearly
  upright/flat.
- **Target**: petiole pitch = 60° (leaf opened outward).
- **Optimized parameter**: only `petiole_pitch`.
- **Loss**: 3D Chamfer distance between the current and target vertex point clouds.
- **Result**: petiole pitch converges 10° → 60° and Chamfer distance drops from
  ~303 mm to ~0.07 mm in 200 Adam steps (LR = 0.5°/step).

---

## Reproduction

```bash
conda activate digital-crops
python scripts/minimal_direct_opt_depth_chamfer_demo.py
```

Output:

```text
Device: cuda
Target plant: 3 organs

Experiment B2: Chamfer pulls leaf outward
  Initial petiole pitch: 10.0°  ->  Final: 60.0°
  Initial Chamfer: 303.47 mm  ->  Final: 0.07 mm
Saved figure: docs/results/assets/minimal_direct_opt_depth_chamfer_demo.png
Saved metrics JSON: docs/results/assets/minimal_direct_opt_depth_chamfer_demo.json
Demo complete.
```

---

## Figure layout

The generated figure is a compact, publication-friendly 2×6 panel:

- **Top row**: target RGB (90° top-down), target canopy-height map, target side
  view (20° elevation), initial RGB, initial side view, 3D Chamfer distance curve.
- **Bottom row**: optimized RGB, optimized side view, initial 3D x-z point-cloud
  overlay, optimized 3D x-z overlay, zoomed leaf-tip panel, parameter trajectory.
- **Green arrows** mark the Chamfer gradient direction on the rendered side view
  and on the 3D overlays.

The right-hand metrics column shows the Chamfer distance decay and the petiole
pitch trajectory from 10° to the target 60°.

---

## Key takeaway

Even with a tiny 3-organ plant, the 3D Chamfer loss supplies a strong gradient
that rotates the leaf outward until the rendered point cloud matches the target.
This confirms that the differentiable geometry builder correctly backpropagates
through the petiole-pitch joint angle.
