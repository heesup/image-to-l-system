# PhytomerVAE-64 Latent Visualizer GUI (2026-09-09)

**Status**: DONE — Gradio web app at `tools/phytomer_vae_visualizer.py`.
Implements the §4.2 handoff spec (PCA latent cloud + 64D sliders → render),
extended with organ-combo coloring, leaf-quality toggle, and Web 3D view.

---

## Features

| Feature | Detail |
| :--- | :--- |
| **PCA latent cloud (2D)** | Clickable matplotlib scatter (215,509 packet latents) — click → nearest packet's z loads into sliders |
| **PCA latent cloud (3D)** | Rotatable plotly Scatter3d, hover shows packet idx |
| **Color by** | `slots` (presence count) / `dap` (1–100) / `combo` (organ-type combination, 24 unique) / `none` |
| **64D sliders** | Data-range bounds (min/max + 10% pad), Random z ~ N(0,I), Reset |
| **Reference frame** | `identity` (true 6D identity `[1,0,0,0,1,0]`) or `real` (a packet's own ref + center) |
| **Render** | `HeliosPyTorchRenderer` RGB + depth (bbox auto-focus, `zoom_factor=1.0`) |
| **Leaf quality** | `high` = highres cowpea OBJ leaves (~1.4k verts) / `low` = lightweight alpha-cutout (~80 verts) |
| **Web 3D** | `gr.Model3D` tab — mesh exported to GLB (vertex colors, trimesh) and served through the gradio queue |
| **Packet info** | Per-slot organ type / base / scale / curvature text panel |

## Files

| File | Purpose |
| :--- | :--- |
| `tools/precompute_phytomer_latent_pca.py` | One-time cache build: encode packet cache → z (mu), PCA(64→3), optional DAP labels → `/tmp/opencode/phytomer_gui_cache/` |
| `tools/phytomer_vae_visualizer.py` | Gradio app (loads cache + frozen `phytomer_vae_structural` ckpt) |
| `diffusion_based/models/helios_pytorch_geometry.py` | + `"lowpoly"` leaf mode (~80-vert alpha-cutout leaf, `get_generic_leaf_mesh(Nx=8,Ny=8)`) |

## Run

```bash
# 1. Precompute latent cloud + PCA (one-time, ~1-2 min encode + ~7 min DAP labels)
.../bin/python tools/precompute_phytomer_latent_pca.py --with-dap

# 2. Launch GUI
.../bin/python tools/phytomer_vae_visualizer.py \
    --server-name 0.0.0.0 --server-port 7860
```

Access: VNC browser `localhost:7860` or SSH tunnel `ssh -L 7860:localhost:7860`.

## Key implementation notes / gotchas

1. **Structural VAE decode**: `model.decode()` returns ZEROED base columns —
   call `assemble_packets(recon_packets, refs)` before re-anchoring
   (leaflets attach at 0.8/0.8/1.0 × petiole length along `R_pet = R_ref @ R_rel`).
   `ref_mode="identity"` MUST pass real identity rot6d (not zeros — `rot6d_to_matrix`
   of zeros degenerates to a zero matrix).
2. **Gradio 6 removed `gr.Plotly.select()`**: 2D click uses a matplotlib image +
   `gr.Image.select` (pixel → axes-bbox inverse → nearest point). The
   `_pca2d_click_to_idx` mapping uses `ax.get_position()` fractions; with 215k
   points neighboring packets are sub-pixel apart, so the returned index is
   the *nearest* visible point — expected.
3. **gr.Model3D initial value must go through a `demo.load` event**: setting
   `model3d.value` to a raw `/tmp/opencode/...` path is **403** (gradio only
   serves files from its own cache) → blank 3D viewer. Route initial render
   through `demo.load(initial_render, ...)` so the GLB passes the queue.
4. **Env deps added to `digital-crops`**: `gradio==6.26.0`, `plotly==7.0.0`,
   `trimesh==5.1.0`; `huggingface-hub` upgraded 1.8.0 → 1.30.0 (all consumers
   allow `<2.0`). Wheels are staged in `/tmp/opencode/pipcheck` (offline install
   via `--no-index --find-links`).
5. **PCA evr changed after structural VAE**: 25.4/12.0/9.1% (relative-D, base
   predicted) → **15.5/12.3/9.3%** (structural, base stripped) — lower PC1 share
   as base co-variance left the latent. Cache must be rebuilt per checkpoint.
6. **`pkill -f` hazard**: a `pkill -f "phytomer_vae_visualizer.py"` inside a
   command whose own command line contains that string kills the calling shell
   (bash tool "timeout" with no output). Use a bracket pattern
   (`phytomer_vae_visualizer[.]py`) or kill by PID.
7. **leaf_mode "parametric" is currently broken** (`parametric_cowpea_leaf`
   module absent) — use `generic`/`obj` (highres) or `lowpoly`.
