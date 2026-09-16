# Whole-Frame Multi-Plant Reconstruction

**Date**: September 16, 2026
**Follows**: `docs/results/20260916_agml_dataset_swap_real_image_test.md` (single-crop AgML testing)
**Request**: real rover frames carry several plants at once, and that multi-plant frame — not an
isolated crop — is the input the project actually targets. Find the plants by object detection or
instance segmentation, build a rough plant architecture from the estimated DAP with Helios, estimate
the architecture with the trained model, and refine it against the input image with the
differentiable renderer.

## 1. The pipeline already existed in pieces

Two conventions were reused rather than reinvented:

- **`Image2PlantArchitecture_v2`** (the VLM work, Yun et al. 2026) established the
  image → `params.json` → Helios XML path. Its `params.json` carries `field.layout`
  (`mode: manual`, `plot_size_x/y` in metres) and `field.plots[].plants[]` as a list of
  `{x, y, z, xml}`; Helios emits one structure XML per plant plus `camera.json`, `boxes.txt` (YOLO
  format) and `masks.json`. There a VLM writes that config from a drone image; here object detection
  supplies the plant count and positions instead.
- **This repo's own synthetic-data generator** is the same binary
  (`Digital-Crops/projects/syntheticdata_generation/build/main`) that produced the training set, and
  it accepts a bare `--dap N` with `--renderer none` to grow one procedural plant with no image
  conditioning at all (~20 s per plant at DAP 60, XML only).

## 2. New code

- `real_world/dataset/helios_cold_start.py` — `generate_helios_xml(dap, seed)` (cached on disk by
  species/DAP/seed) and `helios_state_from_xml()`, which walks the training set's own conversion
  chain (XML → 14D part tensor → 26D FM nodes → 10-slot packets → `phytomer_scale` + `PhytomerVAE`
  latent) to produce exactly the `(pos, rot, scale, latent, exist, parent)` state that
  `run_approach1_cold.sample_cold()` returns. Either can therefore seed refinement unchanged.
  `PHYTOMER_TERMINAL_LAST=1` is set at import: the v9 packet cache the checkpoints train against was
  built with `terminal_leaflet_last=True`, and `build_phytomer_packets` reads that as a process-level
  env var, so the slot layout would otherwise not match the VAE the latents come from.
- `real_world/eval/run_multiplant_scene.py` — the end-to-end script: frame → detections → plot
  coordinates → `params.json` → Helios growth → per-plant state → refinement → refined XML →
  `params.json` rebuilt with those XMLs → one rendered plot.

Cold-start richness, DAP 60 procedural plant: **82 phytomers / 496 organs**, extent 0.59 x 0.47 x
0.29 m, live-organ scale 0.25–2.82 FM units. For comparison, the trained network's own
image-conditioned sample on real AgML crops returned a **single live phytomer on 14 of 20 plants**
(§4). That gap is the reason the Helios cold start is worth testing at all.

## 3. Three things that had to be fixed to make the scene correct

1. **Helios ignores a params entry's per-plant `(x, y)` when it loads a plant from XML.** In
   `main.cpp`, the grow-from-library branch places the plant at `origin + (X, Y)`, but the XML branch
   shifts the loaded plant by `shift = make_vec3(origin.x, origin.y, 0)` — the PLOT origin only.
   Every reloaded plant therefore stacked at the plot centre. Fixed on the Python side (no C++ rebuild)
   by baking each plant's plot position into the exported XML's coordinates before writing.
2. **Auto-FOV does not frame the plot.** Left to the binary, the scene camera framed roughly a
   quarter of the plot. The script now passes `--fov` computed from the geometry,
   `2·atan((plot_width/2) / camera_height)` = 46.9° for a 1.3 m plot at 1.5 m.
3. **Paths must be absolute.** The binary runs with `cwd=BUILD_DIR` and resolves `-f`, `--output`
   and every `plants[].xml` from there, so repo-relative paths silently fail to open.

Axis convention was verified empirically rather than assumed, with a single-plant render at
(x=+0.45, y=0): **Helios +x renders to image right, +y to image up**, so the pixel → plot mapping is
`x_m = (cx/W − 0.5)·plot_w`, `y_m = (0.5 − cy/H)·plot_h`.

Known limitation: the XML export runs with `stem_ik=False, leaf_ik=False`. Both IK passes index the
40D array and the 14D tensor as a row-aligned pair, and the converter is not 1:1 on a
packet-decoded plant (a 454-row part tensor came back as 456 rows), so they cannot align. The scene
render can therefore sit a few degrees off the refined 14D geometry; the refinement numbers
themselves are unaffected.

## 4. Scale field convention, and what it means for the absolute cap

Left open by the previous report ("scale0 is not simple positive metres, -2.3 to 2.6 observed").
Settled by measuring `scale0` on 20 AgML cold-start plants, split by component
(`[length, radius, unused]`, FM units, `metres = FM / SCALE_SCALE(50)`) and by whether the slot is
live:

| | length (FM / cm) | radius (FM / cm) | unused |
|---|---|---|---|
| **live nodes** (47) | 0.48–4.12 / 1.0–8.2cm, 0% negative | -0.11–0.18 / ~0 | 0.85–1.12, near-constant |
| **all slots incl. padding** (202) | -1.95–4.12, 24% negative | -1.67–1.48, 34% negative | -1.66–2.13 |

Live organs' cold-start scale is in-distribution (length's max lands on the training GT ceiling of
~4.0 FM almost exactly); the wide, sign-flipping range previously recorded came from padding slots.
Radius's live range is an order of magnitude smaller than length's, so a single scalar cap across
both was the wrong shape. `run_approach2_refine.py` now takes `--scale_abs_max_len` (5.0 FM, 10cm)
and `--scale_abs_max_rad` (0.3 FM, 0.6cm), clamped per component with a floor at
`phytomer_scale()`'s own magnitude floors (0.25 / 0.025) so refinement cannot drive a live organ's
scale negative.

Re-running the six AgML plants of the previous report with these defaults:
**6 of 6 stay small and plant-shaped through refinement, no flat-polygon inflation** — up from 1/6
with the best multiplicative-clip attempt (`scale_clip_mult=1.1`). Figure:
`docs/results/assets/20260916_agml_real_image_test_scale_abs_max_calibrated.png`.

## 5. First whole-frame run, and why its headline number is not a win

One AgML frame (`..._camA_000055.jpg`, 4 plants detected, DAP 25 assumed), both cold starts refined
against each plant's own crop. Figure: `docs/results/assets/20260916_multiplant_scene.png`.

| plant | Helios init: organs / moved / data loss | network init: organs / moved / data loss |
|---|---|---|
| 0 | 154 / 3.6 cm / 0.729 | 32→34 / 0.6 cm / 1.024 |
| 1 | 202 / 3.5 cm / 0.657 | 4→6 / 0.2 cm / 1.767 |
| 2 | 322 / 3.1 cm / 0.580 | 4→6 / 0.2 cm / 1.366 |
| 3 | 262 / 3.2 cm / 0.668 | 159→157 / 1.0 cm / 0.829 |
| **mean** | **0.658** | **1.247** |

The network's cold start collapses to 4–6 organs on 2 of 4 plants, so it has almost nothing to fit
with; the Helios plant starts with 154–322 organs and its nodes actually move (3+ cm vs 0.2–1.0 cm).
But **the data-loss gap must not be read as reconstruction quality**, because the refined Helios
plants are visibly inflated in the scene render, and measuring the exported XMLs confirms it:

| | leaf_scale mean | leaf_scale max |
|---|---|---|
| cold (Helios, 4 plants) | 0.031–0.034 | 0.0865 |
| after refinement | 0.051–0.062 | 0.208–0.221 (**2.4–2.6x**) |

So a good part of the lower data loss is the known canvas-inflation pathology: a bigger canopy covers
a loose target better. The per-component absolute cap from §4 bounds the phytomer scale `s_a`, not the
realized leaf size, and this source still has no real segmentation mask (§5) — the Dice target is a
bounding-box rectangle, exactly the condition under which the 2026-09-16 single-crop report already
found the priors insufficient. A fair cold-start comparison needs the Roboflow segmentation source,
or a bound on realized organ size rather than on `s_a`.

Two mechanisms were ruled out before landing on refinement as the cause, each worth recording:

- **The 14D → XML export is faithful.** Round-tripping a Helios plant through
  `PlantOrganArray.from_xml_file → to_part_tensor → PartTensorTo40DConverter → to_xml_string` leaves
  leaf scales bit-for-bit (0.0053–0.0865 in all three representations, ratio 1.00x) and the base
  extent within 5 mm (0.108/0.115/0.205 m → 0.113/0.120/0.206 m).
- **The packet/VAE decode is faithful.** `helios_state_from_xml → plant_from_nodes` returns the same
  base extent (0.108/0.115/0.205 m), the same max leaf scale (0.0865 → 0.0866) **and all 101 leaves**.
  (An earlier note in this report claimed the packet round trip lost two thirds of the leaves. That
  was wrong — see §7, the loss was in the XML writer.)
- **Helios's own XML-loading path is fine.** Rendering the scene from the ORIGINAL Helios-written
  XMLs at the same positions and camera produces correctly sized plants
  (`/tmp/.../xml_ab/helios_orig/cowpea/orig_xml_0000_rad.jpeg`).

## 7. Bug found and fixed: the XML export dropped two of every three leaflets

Tracing the leaf count through every stage of one Helios cowpea (34 phytomers):

| stage | leaves |
|---|---|
| original XML → 14D part tensor | 101 |
| after `helios_state_from_xml → plant_from_nodes` (packets + VAE) | **101** |
| after `PartTensorTo40DConverter.convert` | **101** |
| `<leaf>` elements in the XML we wrote | **34** |
| re-reading that XML | 34 |

The packet/VAE round trip is clean; the loss was entirely in the export. Cause chain:

1. Phytomer packets carry organs but no shoot metadata, so a `plant_from_nodes` output has no
   `ORGAN_SHOOT_META` rows.
2. `PartTensorTo40DConverter.convert` then synthesizes one shoot for the whole plant and hard-coded
   its type as unifoliate (`shoot_row[T_COL_SHOOT_TYPE] = 0.0  # unifoliate`,
   `part_tensor_to_40d.py`).
3. The XML writer caps a unifoliate shoot's petioles at one leaf
   (`max_leaves = 1 if "unifoliate" in stl_str else 3`, `plant_organ_array.py`), so every trifoliate
   phytomer emitted one leaflet instead of three. 101 → 34, one per petiole.

**The repo already had the answer, and the real mistake was bypassing it.** Step 1 above is by design,
and `emit_part_tensor_with_shoot_meta` (`diffusion_based/dataset/phytomer_packets.py`) exists
precisely to undo it — its docstring records the cost of not using it (94.1% → 16.5% foreground IoU
when the shoot-meta rows are the only thing missing), and `eval_phytomer_vae_helios_roundtrip.py`
already exported through it. Exporting `plant_from_nodes`'s output — the renderer's decode, which
deliberately flattens shoots — was the error. Two changes:

- `part_tensor_with_shoots()` added beside `emit_part_tensor_with_shoot_meta` in
  `phytomer_packets.py`: the same decode `plant_from_nodes` does, emitted through the existing
  shoot-meta function. `shoot_id`/`phytomer_idx` come from `chain_phytomers`, which callers already
  run. `refine_one` grew a `return_state=True` option so the export can re-decode the optimized state.
- The converter's hard-coded unifoliate is fixed anyway (shoot type now read from the data: more
  leaves than petioles means trifoliate), as a safety net for any caller that still exports a tensor
  without shoot metadata.

Result, per plant, exported vs. the original Helios XML: leaves 76/77, 100/101, 160/161, 130/131 and
shoots 5/5, 8/8, 12/12, 10/10 — `chain_phytomers` recovers every shoot. The round-tripped scene
renders with the same canopy density as the original XMLs loaded directly. (The one-leaf deficit per
plant is the cotyledon node's synthesized opposite petiole, documented in
`emit_part_tensor_with_shoot_meta`.)

While reusing the existing generator, `scripts/generate_helios_dataset.py` also gained `--renderer
none` support: both its completion check and its success check required a rendered `_rad.jpeg`, so an
XML-only run was never cached and always reported failure. `generate_helios_xml` now delegates to its
`render_one` instead of invoking the binary a second way (first call 14.5 s, cached call 0.00 s).

**Scope.** This only ever affected XML export of a packet-decoded plant. Everything that consumes the
14D part tensor — training, the differentiable renderer, strict-protocol IoU, test-time refinement —
was unaffected, which is why it went unnoticed. In particular the Helios round-trip evaluation
(`eval_phytomer_vae_helios_roundtrip.py`) is **not** affected: it exports via
`emit_part_tensor_with_shoot_meta`, which supplies real `ORGAN_SHOOT_META` rows, and the converter's
shoot-meta branch already assigns types correctly (`0.0 if curr_sid == 0 else 1.0` — shoot 0
unifoliate, the rest trifoliate, matching Helios's own XML: 1 unifoliate + 7 trifoliate).

**Still open.** The synthesized-meta path collapses all shoots into one, so branching topology is
flattened on export even with the fix (8 shoots → 1). `emit_part_tensor_with_shoot_meta` already
solves this for the round-trip eval — its docstring records 94.1% → 16.5% foreground IoU when those
rows are the only thing missing — so routing `plant_from_nodes` exports through it (it needs the
`shoot_id`/`phytomer_idx` that `chain_phytomers` already returns) is the proper follow-up.

## 6. Dataset note

AgML's catalog has no dataset that is real bean/cowpea imagery *and* segmentation-annotated: its only
bean segmentation set (`bean_synthetic_earlygrowth_aerial`) is Helios-synthetic, i.e. the same engine
as this project's training data, and the real ones carry bounding boxes (`gemini_plant_detection_2022`)
or whole-image class labels only. Anything needing a real segmentation mask therefore stays on the
Roboflow `t4_plant_weed_seg` export used on 2026-09-15.
