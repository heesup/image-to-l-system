---
title: "Sim-to-real assessment: where the next effort should go (2026-09-16)"
date: 2026-09-16
tags: [handover, session]
status: active
---

# Sim-to-real assessment: where the next effort should go (2026-09-16)

**Status**: plan, written after reading every document under `docs/` as of the 2026-09-16 handover
and checking the claims against the code. The first item of the real-image list is being executed
in the same session; its result will be appended as §5.

**Question asked**: how to make model training and real-image testing better.

---

## 1. What the record says, and what the code adds

The handover (`ongoing/README.md`, 2026-09-16 section) concludes that the synthetic track has
plateaued at strict-protocol P 33–39 across every training lever, that test-time refinement lifts the
same checkpoints to 67–68, and that the real-image reconstructions are not yet usable. All of that
holds. Reading the code alongside it adds four facts that change where the next effort should go.

### 1.1 The network never sees the CHM channel

`DINORayEncoder.forward` (`diffusion_based/models/dinov2_ray_encoder.py`, line 164 onward) takes
channels 0:3 of the input at every zoom level. The depth channel is only ever a render-loss target
during training and refinement. Consequences:

- The cold-start collapse on real crops (14 of 20 AgML plants with one live phytomer) is a pure RGB
  appearance gap. Depth Anything's pseudo-CHM cannot be its cause.
- Any CHM fix helps only the refinement target, not the network's prediction.

### 1.2 Training RGB is flat-shaded green on uniform tan, with no augmentation at all

`generate_cache.py` renders every training image with the PyTorch renderer (`background="ground"`,
a single tan colour, flat organ shading, no shadows, no texture). A grep of the dataset and
training code for colour jitter, blur, noise, flips, background or lighting variation finds
nothing. Meanwhile each of the 100,000 training plants has a Helios raytraced render
(`dataset/helios_data/cowpea/*_rad.jpeg`, 720 px, textured soil, shaded and specular leaves,
cast shadows) sitting next to its XML, unused. For the same DAP 40 plant the cache image and the
Helios render look like different datasets. Caveat on the Helios renders: all 100,000 use one
lighting configuration (sun elevation 45°, azimuth 180°), and each image is auto-framed on its
plant, so the ground footprint varies from 7 mm at DAP 5 to 1.3 m at DAP 90.

### 1.3 The real-image crop framing does not match the training framing

The cache's zoom pyramid is a **fixed ground window**: 1.2 m at zoom 1×, then 0.6 / 0.3 / 0.15 m,
centred on the plant's 3D bbox centre, camera 5 m above it
(`compute_focus_plant_camera`, `reference_window_size` branch, `helios_pytorch_renderer.py`
line 119). The real-image crop utility (`real_world/dataset/real_plant_crop_utils.py`,
`build_pyramid_16ch`) instead uses **1.2 × the detector bbox** as the zoom-1× window. Its comment
says this "matches generate_cache.py's zoom-1x window == 1.2x the plant's own extent", which is
not what the cache does. So on a real crop every plant fills about 80% of the zoom-1× frame,
whereas in training a 0.3 m plant occupies a quarter of it and a seedling a few pixels. The
network reads plant size and DAP from apparent size in a fixed window; this alone can explain
why Stage 1's DAP on real crops was "uncorrelated with the true DAP and higher in 7 of 8 cases".
The fix is one line once the pixel-to-metre scale is known (the multi-plant script already
assumes the 1.3 m plot width): window_px = 1.2 m × (frame_px / plot_width_m).

### 1.4 On synthetic data, Stage 3 is dead weight for the deployable path

Refinement from the DAP-spanning mean latent reaches the same 66.5 as from the sampled latent
(takeover guide §0-B.9, 17:20 entry). The network's real contribution is node positions,
existence and topology, and refinement cannot add organs. Every remaining synthetic error lives in
Stage 2 and in the refiner, not in Stage 3.

### 1.5 Nobody has measured the appearance gap separately from the architecture gap

A real cowpea may differ from a Helios cowpea in structure, or only in pixels. The real-image
runs to date cannot tell those apart, and they are the difference between "fine-tune on realistic
renders" and "re-tune the Helios cowpea parameters to the Davis field".

---

## 2. Training side, ranked

1. **Train on realistic RGB.** Crop the Helios raytraced renders to the cache's fixed windows (or
   re-render the training plants at those windows, since the existing renders are auto-framed) and
   fine-tune from `hierarchical_fm_v10_cam` ep160 on them, mixed with the flat renders. Add ordinary
   augmentation on top: colour jitter, blur, sensor noise, real soil backgrounds composited under
   the flat render using its own mask, random shadows. Largest untouched lever for real testing.
2. **Make existence a continuous variable in refinement.** The synthetic deficit (about 60 of 73
   GT nodes active) and the real-image collapse are both "organs that are not there".
   `results/20260915_real_image_first_test.md` §6.2 already sketches the mechanism: keep Stage 2's
   raw logits and use a soft compositing weight in the decode instead of the hard select, so
   refinement can switch nodes on and off. Lowering the threshold failed only because refinement
   could not switch false positives off.
3. **Stop investing in the Stage 3 latent.** Deploy Stage 2 nodes + mean latent + refinement. If
   Stage 3 stays, a deterministic regression head is enough; the flow only learns the marginal.
4. **Try a larger backbone with multizoom.** Node error sits at the token pitch (4–6 cm at 7.5 cm
   per token). The record shows only the unfrozen ViT-S run; `dinov2_vitb14` is one flag away.
5. **Use more eval plants for decisions.** Epoch-to-epoch spread is ±3 points on 20 plants, the
   size of most effects compared. Fifty to a hundred plants under the strict protocol.

## 3. Real-image side, ranked

1. **Measure the appearance gap first** (in progress, §4). Run the strict protocol on Helios
   raytraced crops of the same 20 eval plants, framed exactly as the cache frames them. If P falls
   from 38 toward the real-image behaviour, the fix is training item 1. If P holds, the gap is
   structural.
2. **Fix the crop framing** (§1.3): a fixed 1.2 m window from the known pixel-to-metre scale.
3. **Pick DAP from measured plant size**, not from Stage 1 or the timestamp: detector box in
   metres against an extent-versus-DAP table from the synthetic set. The multi-plant frame assumed
   DAP 25 for plants that are clearly seedlings, so the Helios cold start was too big before
   refinement inflated it further.
4. **Fix the refinement target.** A single blob mask plus blurred pseudo-depth cannot constrain
   leaves, which is why the optimiser inflates them. Per-leaf instance masks, and a DINOv2 feature
   loss between render and photo (robust to appearance, per-organ signal). Bound realised leaf
   size in metres rather than the phytomer scale, as the handover notes.
5. **Clean the pseudo-CHM** for the refinement target: estimate ground from the mask complement,
   zero the CHM outside the mask, drop `depth_downsample=3`.
6. **Add real-image proxies** so real runs get a number: leaf count vs instance count, mask IoU,
   plant extent, 2D leaf-centroid matching.

---

## 4. Item 1 in practice: the appearance-gap measurement

Design, chosen so that only the pixels differ between the two runs:

- Same 20 plants (`hierarchical_fm_v9/eval_set.json`), same checkpoint
  (`hierarchical_fm_v10_cam/hierarchical_fm_epoch_160_ema.pt`), same script
  (`eval_test_time_refinement.py`, strict 256 px protocol before and after refinement), same CHM
  channels and refinement targets from the cache.
- Only the 12 RGB channels of the 16-channel input are replaced by Helios raytraced pixels.
- The Helios image is re-rendered per plant rather than cropped from the auto-framed dataset
  JPEG: the plant's XML is shifted so its 3D bbox centre (the cache's camera centre) sits at the
  plot origin, the camera is placed 5 m above that centre, the FOV is set to the 1.2 m window,
  and the 0.6 / 0.3 / 0.15 m levels are centre crops of one high-resolution render (a narrower FOV
  from the same pinhole is exactly a centre crop). Lighting follows the dataset renders.
- Framing check before scoring: the silhouette of the Helios crop against the cache CHM > 0 mask
  at every zoom level. If those do not overlap at ~90%+, the framing is wrong and the P numbers
  are not an appearance measurement.

Readout: raw strict P and refined P on Helios RGB versus the flat-render baseline (ep160 EMA:
raw 38.7 / refined 68.3), per plant and by DAP bucket, plus the number of active nodes and the
DAP probe's prediction. The comparison isolates appearance from every other difference.

---

## 5. Result of item 1: pixels alone cost 10 points raw and 7 refined (2026-09-16, 18:30)

**Setup, as designed in §4.** Checkpoint `hierarchical_fm_v10_cam/hierarchical_fm_epoch_160_ema.pt`,
`eval_test_time_refinement.py` with its defaults (40 steps, input camera, four zoom targets,
scale/latent priors). The flat baseline was re-scored in the same session so both runs share code
and flags: 38.1 / 66.3 against the 38.7 / 68.3 recorded this morning, i.e. within the ODE-sampling
noise. Run folder: `slurm_scripts/logs/20260916/helios_eval_crops/` (`ttr_ep160ema_{flat,helios}.json`,
`compare.md`, `summary.json`, per-plant `panels/`, the Helios runs under `helios/`).

**Framing check passed.** All 20 plants keep the identity orientation (each flip loses 25 points or
more of mask IoU); mask centroid offsets are within 3 px at 2× and 4× for 18 plants, and the two
larger offsets (index 77889 at 2×, 16 px; 67809 at 4×, 6 px) are cast shadows that the green-pixel
heuristic counts as plant, not framing errors (panels inspected). Mean mask IoU per zoom is 72–80%,
limited by specular highlights and shadows in the heuristic, not by geometry. Render cost 30–100 s
per plant at 2048 px, 958 s for the set. Example panel:
`docs/results/assets/20260916_appearance_gap_framing_check_dap042.png`.

**Numbers** (`compare_rgb_source_readings.py`; nodes = active predicted phytomer nodes before
refinement, GT = ground-truth phytomers; DAP error = |Stage 1 probe − true DAP| in days):

| plants | n | raw P flat | raw P Helios | refined flat | refined Helios | nodes flat / Helios / GT | DAP error flat / Helios |
|---|---:|---:|---:|---:|---:|---|---|
| DAP ≤ 15 | 4 | 19.2 | 10.5 | 31.4 | 12.0 | 4.5 / 8.8 / 6.5 | 1.4 / 51.0 |
| 16–45 | 6 | 37.7 | 29.4 | 75.4 | 69.7 | 21.7 / 37.5 / 39.0 | 5.0 / 8.1 |
| 46–75 | 4 | 41.4 | 39.9 | 72.7 | 73.5 | 102.5 / 98.2 / 116.8 | 6.0 / 20.5 |
| > 75 | 6 | 48.8 | 32.4 | 76.0 | 70.6 | 107.0 / 63.2 / 122.2 | 5.5 / 25.1 |
| **all** | 20 | **38.1** | **28.6** | **66.3** | **59.2** | 60.0 / 51.6 / 73.0 | 4.6 / 24.3 |

Figure: `docs/results/assets/20260916_appearance_gap_flat_vs_helios.png` — (a) raw P per plant,
(b) refined P per plant, (c) predicted against true DAP, flat in blue and Helios in orange.

**Reading.**

1. **Pixels alone cost 9.5 points raw and 7.1 refined on identical geometry.** Raw P on Helios
   pixels (28.6) is where the latent-only baseline sat (27–29 under this protocol), so the
   appearance gap erases the whole v10 gain.
2. **The loss sits in the seedlings and the mature plants, and it is the DAP probe and the
   existence gate that fail.** Seedlings: the probe reads DAP 2 as 67, 3 as 46, 11 as 51, 12 as 68;
   two of four collapse to a single active node; refined P 31.4 → 12.0. Mature plants: DAP
   under-read by 25 days (97 → 63, 94 → 64, 89 → 59), active nodes 107 → 63 against 122 GT, raw P
   48.8 → 32.4. Mid-season plants are unchanged. Both heads estimate plant size from RGB, and with
   textured soil, shading and cast shadows both regress toward the middle of the DAP range: small
   plants are read as large, large plants as smaller. On the real crops the same two heads failed
   the same way (DAP uncorrelated with the truth, single-phytomer collapse). That failure is now
   reproduced on synthetic geometry, so the pixels account for it.
3. **Refinement recovers most of the loss for DAP > 15 (71.0 against 75.0)** because its target,
   the cache CHM, is unchanged in this experiment. On real images the target is the pseudo-CHM and a
   blob mask, so the recovery there will be smaller.
4. **Bounds.** This is a lower bound on the real gap: Helios renders are not photographs, and each
   plant carries one lighting draw. The mid-season null sits inside the ±3 epoch noise; the seedling
   and mature effects (10–19 points, 25–51 DAP days) are far outside it.

**Decision.** Training item 1 of §2 (fine-tune on Helios RGB crops with augmentation) is the right
next bet, and the structural gap cannot be measured until it is closed. Before the next real-image
test, also fix the crop framing (§1.3): the DAP probe is exactly the head that depends on apparent
size in a fixed window.
