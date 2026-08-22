# Differentiable Renderer vs Helios Visual Alignment Debug

> Date: 2026-08-04  
> Environment: Ubuntu + NVIDIA RTX 6000 Ada (48 GB), PyTorch 2.9.1+cu128, `DISPLAY=:1.0`  
> Commit base: `729fab8` (`b9fd2cd` single Helios-style geometry+render pipeline)

## 1. Goal

Verify that the new Python differentiable renderer (`HeliosGeometryRasterizer`) produces a plant image that is structurally and visually aligned with the Helios C++ OpenGL visualizer output, before using it as a supervision signal in `train_diffusion_3d.py`.

## 2. Method

- **Input XML**: `Digital-Crops/projects/syntheticdata_generation/build/output/plot_0000_plant_0000.xml`
- **Reference image**: `plot_0000_vis.jpeg` (Helios C++ visualizer, 1920×1080)
- **Python renderers**:
  1. Render from explicit XML-derived geometry (`build_helios_geometry_from_xml`)
  2. Render from 15D organ graph (`nodes_to_geometry` from `helios_xml_parser`)
- **Resolution**: 256×256 for fast comparison
- **Camera**: `focus_plant=True`, `camera_height=1.0`, `azimuth=0°`

Images and diff maps saved under `/tmp/helios_render_test/`.

## 3. Quantitative Results

| Metric | XML geometry render | 15D nodes render |
|--------|---------------------|------------------|
| Metric | XML geometry render | 15D nodes render | XML + ground bg |
|--------|---------------------|------------------|-----------------|
| MAE vs Helios (full image) | 0.319 | 0.295 | **0.237** |
| MSE vs Helios (full image) | 0.126 | 0.110 | — |
| SSIM | 0.286 | 0.304 | **0.439** |
| Plant mask IoU vs Helios | **0.851** | **0.851** | **0.851** |
| Mask precision / recall | 0.851 / 1.000 | 0.851 / 1.000 | — |

> Mask extraction: Helios reference plant region ≈ pixels darker than 0.65 in grayscale; Python render plant region ≈ pixels brighter than 0.08 in grayscale (black background).

## 4. Visual Results

### 4.1 Side-by-side comparison (left to right: Helios ref, XML render, 15D nodes render)

![renders](file:///tmp/helios_render_test/compare/renders.png)

### 4.2 Effect of ground background

Adding a tan/brown ground color (`background="ground"`) substantially improves full-image metrics:

![background_comparison](file:///tmp/helios_render_test/compare/background_comparison.png)

### 4.3 Per-pixel absolute difference maps

![diff_maps](file:///tmp/helios_render_test/compare/diff_maps.png)

### 4.4 Plant masks and IoU

![mask_metrics](file:///tmp/helios_render_test/compare/mask_metrics.png)

### 4.5 Multi-view stability (azimuth 0°, 45°, 90°, 135°)

![azimuth_grid](file:///tmp/helios_render_test/azimuth_grid.png)

The renderer is view-consistent; the same structural biases appear at all azimuths.

## 5. Key Observations

### 5.1 Plant silhouette is well aligned (IoU ≈ 0.85)

The overall bounding box, branching structure, and leaf placement match Helios reasonably well. This means the **camera model, focus-plant HFOV recomputation, and 3D geometry reconstruction are basically correct**.

### 5.2 Major visual gaps

| # | Issue | Helios reference | Python renderer | Impact on render loss |
|---|-------|------------------|-----------------|-----------------------|
| 1 | **Background** | Textured soil/ground with pot edges | Solid dark gray/black | Large MAE contributor; easy to fix with a ground texture or ignore via mask |
| 2 | **Leaf shape** | Trifoliate compound leaves (3 leaflets, serrated/oval) | Single elliptical/quad leaf per node | Medium; leaf silhouette differs, affects edge gradients |
| 3 | **Leaf density / occlusion** | Many small leaflets create fine texture | Fewer, larger leaf quads create a “leafy blob” look | Medium; makes gradients coarser |
| 4 | **Shading & shadows** | Ground shadows, ambient occlusion, darker leaf undersides | Simple diffuse term only, no shadows | Medium; removes important depth cues |
| 5 | **Color palette** | Brown stems + yellow-green leaves + soil | Darker green leaves + slightly brown stems | Small; mostly correctable with color tuning |

### 5.3 15D nodes render is slightly closer than XML geometry render

Surprisingly, the 15D-nodes render has lower MAE/MSE and higher SSIM than the explicit-XML-geometry render. This is likely because:

- `nodes_to_geometry()` uses a **single quad leaf** with less overdraw/occlusion, producing a cleaner, slightly less “noisy” silhouette.
- The explicit XML geometry carries all Helios leaflets and triangle faces, which exaggerates the shape mismatch when rendered with the simple Python rasterizer.

This suggests that improving the **leaf geometry prototype** in `nodes_to_geometry()` is more important for render-loss quality than matching every Helios mesh triangle.

## 6. Bugs / Code Fixes Applied During Debug

1. `helios_rasterizer_3d.py` leaf triangle rendering OOM at 512×512 → chunked leaf triangle renderer added.
2. Bud rendering `RuntimeError: The size of tensor a (N) must match the size of tensor b (H)` → broadcasted `r_norm` explicitly.
3. `IndexError: list index out of range` in `_composite_by_depth` for buds → removed custom `organ_colors` so default 4-class list is used.
4. Return type of `render_numpy_geometry` was already `np.ndarray`; downstream code attempted `.permute()` on it → comparison script adapted.

## 7. Recommendations

### Short term (before training with render loss)

1. **Add a ground background** matching Helios soil color / simple texture. This alone will drop MAE significantly. ✅ Implemented: `background="ground"` reduces MAE from 0.319 → 0.237 and SSIM from 0.286 → 0.439.
2. **Mask the loss** to plant-only pixels so background mismatch does not dominate gradients.
3. **Use `image_size=256` or smaller** for render-in-the-loop to stay within GPU memory.

### Medium term (improve visual fidelity)

4. **Implement trifoliate leaf geometry** in `nodes_to_geometry()` using the Helios `CowpeaLeafPrototype_trifoliate_OBJ` or a parametric 3-leaflet model.
5. **Add soft ground shadow** by projecting plant mask onto a ground plane and darkening.
6. **Tune stem/leaf colors** to match Helios average RGB after background removal.

### Long term

7. **Connect Helios OBJ leaf prototypes** directly so predicted leaves share the exact mesh shape.
8. **Investigate why `leaves_per_petiole=3` is collapsed to one leaf in the XML** and restore the correct phytomer parameters if needed.

## 8. Conclusion

The differentiable renderer is **structurally aligned** with Helios (plant mask IoU ≈ 0.85), which is sufficient to provide a meaningful render supervision signal. A simple tan/brown ground background reduces the full-image MAE from 0.319 to **0.237** and raises SSIM from 0.286 to **0.439**. The remaining gap is mainly from simplified leaf shapes and missing shadows, not camera geometry.

## 9. Artifact Paths

```text
/tmp/helios_render_test/xml_render.png
/tmp/helios_render_test/nodes_render.png
/tmp/helios_render_test/render_bg_None.png
/tmp/helios_render_test/render_bg_ground.png
/tmp/helios_render_test/compare/helios_ref.png
/tmp/helios_render_test/compare/side_by_side.png
/tmp/helios_render_test/compare/renders.png
/tmp/helios_render_test/compare/background_comparison.png
/tmp/helios_render_test/compare/diff_maps.png
/tmp/helios_render_test/compare/mask_metrics.png
/tmp/helios_render_test/azimuth_grid.png
```

## 10. Next Steps

- [x] Renderer visual alignment debugged
- [x] Add ground background to renderer
- [ ] Wire `background="ground"` and plant-masked render loss into `train_diffusion_3d.py`
- [ ] Run 1–2 epochs with `--render-loss 1.0`
- [ ] Generate multi-DAP dataset (DAP 5–30)
- [ ] Improve trifoliate leaf shape for higher-fidelity render loss
