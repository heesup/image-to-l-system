# Swapping the Real-Image Dataset Source: Roboflow → AgML

**Date**: September 16, 2026
**Follows**: `docs/results/20260915_real_image_first_test.md` (the first real-image pass, Roboflow
`gemini-breeding-eqsam/t4_plant_weed_seg`)
**Request**: replace the Roboflow source with an AgML (https://github.com/Project-AgML/AgML) bean
or cowpea bounding-box dataset and test it through the same pipeline.

## 1. Finding the dataset

AgML's public catalog (`agml.data.public_data_sources()`, 5985 named sources as of 2026-09-16) has
no dataset that is both real-field imagery *and* object-detection-annotated for bean or cowpea
specifically:

| candidate | real? | task | usable? |
|---|---|---|---|
| `bean_synthetic_earlygrowth_aerial` | no (`location='digital'`) | segmentation | not a real-image test |
| `bean_disease_uganda` | yes | whole-image classification | no localization at all |
| `iNatAg(-mini)/vigna_unguiculata`, `.../phaseolus_vulgaris` | yes | species classification | no localization |
| **`gemini_plant_detection_2022`** | **yes** | **object_detection (COCO bbox)** | **used** |

`gemini_plant_detection_2022` (402 images, classes `plant`/`weed`, from
http://gemini-breeding.github.io/) turned out to be from the *same* GEMINI cowpea breeding project
as the Roboflow export — confirmed by inspection: identical 2592x2048 resolution, the same nadir
tunnel-cart rig (metal rails, integrated LED bars) visible in the frame, and filenames carrying the
same `Davis-COWPEAMAGIC-<plot>` naming plus a human-readable capture timestamp. A different capture
batch/date range of the same field campaign, not an independent domain.

## 2. New/changed code

- `real_world/download_agml_dataset.py` (new) — downloads via `agml.data.AgMLDataLoader`, converts
  the COCO `annotations.json` to YOLO-detection format (images/labels/data.yaml, 80/15/15
  train/valid/test split by image), matching the Roboflow downloader's on-disk convention so every
  downstream script is unchanged.
- `real_world/dataset/real_plant_crop_utils.py::bbox_to_mask` (new) — `detect_plants` now falls back
  to a bounding-box-rectangle mask when the detector has no segmentation output, instead of leaving
  `PlantDetection.mask = None` (see §4).
- Detector: `real_world/detector/train_yolo_detector.py --weights yolo11n.pt` (plain detection, not
  `-seg` — this source has no polygons), 100 epochs, same script used for the Roboflow detector.
  Weights at `slurm_scripts/logs/20260916/agml_plant_detector/weights/best.pt`.

## 3. Detector result

| class | Box P | Box R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| all | 0.932 | 0.968 | **0.975** | 0.661 |

Stronger than the Roboflow-trained `-seg` detector on its own held-out set (plant: mAP50 0.844,
mAP50-95 0.633; weed: 0.612/0.361) — plausibly because plain detection is an easier task than
instance segmentation, and/or this export's boxes are cleaner.

## 4. Running Approach 1 / Approach 2 on the new images

Both `real_world/eval/run_approach1_cold.py` and `run_approach2_refine.py` ran end-to-end against
`real_world/data/agml_gemini_plant_detection_2022/test/images/*.jpg` with `--detector_weights`
pointed at the new checkpoint, unchanged otherwise — no code changes were needed for that part, only
for the mask fallback below.

**First pass, no mask at all** (`docs/results/assets/20260916_agml_real_image_test.png`): reproduced
the project's known "canvas inflation" failure — organs growing into large flat polygons that fill
the crop — on most of the 6 plants, despite the script's `scale_clip_mult=1.5` / `reg_pos=20.0`
defaults (the fix the 2026-09-15 report found and set as the new defaults). Root cause: this
detector has no segmentation output at all, so `build_mask_pyramid` fell all the way back to
thresholding the blurry Depth-Anything pseudo-CHM — an even looser Dice target than the "canvas
inflation" bug's original cause on 2026-09-15 (a real but zoom-saturated segmentation mask).

**Fix attempt: `bbox_to_mask`** — gave `detect_plants` a bounding-box-rectangle mask fallback so a
box-only detector still has *some* real silhouette bound, tighter than a depth threshold. Re-ran
(`docs/results/assets/20260916_agml_real_image_test_bboxmask.png`): **the inflation is still there**
on most plants, materially unchanged from the no-mask pass. A rectangle around a small, young,
already-undersized seedling is itself loose — most of the box is bare soil, not canopy — so filling
it still costs the optimizer little more than filling a depth blob did. The existing hard clamp
(`scale_clip_mult=1.5`) multiplies an already very small cold-start scale, and 1.5x of "very small"
can still read as "one large flat leaf" against a target this loose.

## 5. A tighter clip helps partially, not fully

Re-ran Approach 2 with `--scale_clip_mult 1.1` (vs. the 1.5 default) on the same 6 crops
(`docs/results/assets/20260916_agml_real_image_test_tightclip.png`): **1 of 6 plants** (DAP 77,
the smallest cold start) now stays a small, plausible branching structure through refinement
(nodes moved 0.8 cm, no inflation) — genuinely fixed. The other 5 still inflate into the same flat
polygons, materially unchanged from `scale_clip_mult=1.5`. So a tighter multiplicative clip *does*
help, confirming the mechanism, but is not sufficient by itself: 1.1x of an already-very-wrong
cold-start scale can still land well past the plant's true size for the more severely undersized
cases. This points toward an **absolute** scale cap (in metres, keyed to typical seedling organ
size) rather than a multiplicative one (keyed to the cold start's own, sometimes very wrong, scale)
as the more robust fix. Scaffolded as `--scale_abs_max` in `run_approach2_refine.py` (off by
default) but **not calibrated or re-tested**: a quick check of `scale0` across 4 AgML cold-start
plants found it ranging from about -2.3 to 2.6, i.e. this field is not simply "organ size in
metres" the way the name suggests -- picking a naive positive ceiling without understanding its
actual packing (`denormalize_packet_scales`) risks clamping something that was never the problem,
or missing the axis that actually is. Left for whoever next revisits this to check that convention
first.

## 6. Conclusion

The AgML swap itself works cleanly: `gemini_plant_detection_2022` is a legitimate, well-matched
real cowpea bounding-box source (same rig, same project, no licensing friction), the conversion and
detector fine-tune are a straight rerun of the existing scripts, and Approach 1/2 ran on it without
any pipeline code changes beyond the mask fallback. What the test surfaces is a **real limitation of
the current plausibility priors**: they were tuned against one specific real dataset with real
segmentation masks (2026-09-15) and do not transfer as-is to a box-only source. This is not a new
bug so much as the same root cause — Stage 1/2's cold-start canopy is undersized enough that only a
*data-driven, tight* silhouette target reliably keeps refinement geometrically honest; a bounding
box is not tight enough, and the priors would need their own re-tuning pass (stronger
`scale_clip_mult`, or a hard cap on absolute organ scale rather than a multiplicative one, given the
cold start's error is on the order of the target's own size here) to be revisited if this dataset is
used going forward. Not attempted in this pass — flagged for the next one.
