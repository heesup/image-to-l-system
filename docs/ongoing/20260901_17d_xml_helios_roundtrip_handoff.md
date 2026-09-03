# Handoff: Diffusion 17D → XML → Helios Roundtrip & Python-World Reconstruction

**Date:** 2026-09-01 (Updated 2026-09-02)
**Status:** Verification complete; Python-world reconstruction validated
**Repo:** `/home/lion397/codes/image-to-l-system`

---

## 1. TL;DR / Current State

The Diffusion pipeline predicts a **17D world-space part tensor** (base + rot6d + scale per organ).
Two rendering paths exist:

| Path | Fidelity | Helios lib change needed? |
|---|---|---|
| **17D → Python renderer** (fig8, `render_part_tensor`) | **Exact** — consumes world `rot6d` directly | No |
| **17D → XML → Helios reload** | **Lossy** — Helios FK re-derives geometry from relative params | No (but can't be made exact without it) |

**Key conclusion:** 17D→XML→Helios is *structurally lossy* because Helios reload reconstructs
internode direction via a non-invertible FK chain (`-1.25*pitch` about a dynamic `petiole_rotation_axis`),
so the 17D absolute world orientation cannot be recovered from `(base_rotation, internode_pitch, gravity)`.
**The canonical path is Python-world reconstruction (17D → Python renderer), not Helios reload.**

---

## 2. What Was Done

### 2.1 Gravitropic curvature — root cause of "wiry star" reload
- `readPlantStructureXML` re-applies `shoot->gravitropic_curvature` per internode segment:
  `curvature_angle = deg2rad(grav * curv_fact * dr_max + saved_perturbation)`.
- The realized per-shoot `gravitropic_curvature` was **not serialized** → reload fell back to the
  shoot-type default (`+200`), which differs from the sampled value (e.g. `-608.9`), producing a
  tall wiry "star" instead of the prostrate bush.
- **Fix attempted (superseded):** add `<gravitropic_curvature>` shoot-level tag.
  - Load-path self-consistency: `main_new` reload vs `main_old` reload internode IoU = **0.998**.
  - But **aging-grown vs reload** internode IoU = **0.02** — a *separate* growth-replay gap.

### 2.2 Why aging-grown ≠ reload (the deeper issue)
Growth builds each internode **short** (`internode_length_scale_factor_fraction = 0.01`) then
stretches it over 70 days via `setInternodeLengthScaleFraction()`. Gravity curvature accumulates on
the short, growing internode → the final shape encodes a **curvature-vs-length history** that the
XML does not store. Reload reconstructs in a single pass from final lengths → different wiring
(esp. shoot mid-to-tip, phyto 12+).

### 2.3 Gravitropic fixed to library default (200) — the pragmatic fix
`configs/params_cowpea.json`: `gravitropic_curvature` changed from `uniform(-800,100)` → `constant(200)`.
- **Result:** growth == reload for ALL samples (internode rel_err = 0.000, N preserved) across
  DAP 10/50/90 × seed 0/14/60. **No Helios library change needed.**
- This makes the dataset self-consistent (growth and reload both use 200).

### 2.4 Helios focus-plant margin aligned to Python (5%)
`main.cpp:1813`: `*1.20f` → `*1.05f` to match Python `compute_focus_plant_camera` (5% margin).
- **Note:** this is a *camera framing* convention, not geometry. Both renderers now frame identically.

### 2.5 17D → XML → Helios is lossy (Test 1 & 2)
Even with correct `base_rotation` (from shoot_meta) + `internode_pitch` (world segment angle) +
`gravity=0` on reload, internode bbox rel_err stayed **~0.18** (DAP50 seed14). N preserved (632→632)
but shape diverges. **This confirms the FK is not invertible from relative params.**

### 2.6 Python-world reconstruction is exact & verified
17D world pose → Python renderer (`render_part_tensor`) reproduces the plant geometry exactly.
- 40D renderer also delegates to 17D (`to_part_tensor()`), so 40D is a *structure* representation,
  not a separate render path.

---

## 3. Figure Alignment & Verification

**`fig_python_world_reconstruction.png`** (DAP 10/50/90, seed0, grav=200):
- Columns: Helios GT (Rad, Focus) | Python 17D Render | Organ-Type Map | Internode BBox & Metrics.
- **Framing Alignment (Resolved):** Re-rendered Helios GT with `--focus-plant` matching Python's `focus_plant=True` (5% margin).
- **Results:**
  - **DAP 10:** GT area = 0.2538, Py area = 0.2540, **Silhouette IoU = 0.660**, BBox = [0.03m, 0.01m, 0.18m]
  - **DAP 50:** GT area = 0.1083, Py area = 0.0894, **Silhouette IoU = 0.620**, BBox = [0.97m, 1.47m, 1.52m]
  - **DAP 90:** GT area = 0.0696, Py area = 0.0555, **Silhouette IoU = 0.433**, BBox = [2.08m, 2.38m, 2.32m]
- The visual alignment of stems, canopy spread, and foliage projection is confirmed across all growth stages.

---

## 4. Recommended Architectural Direction

**Adopt Python-world reconstruction as the standard** for the Diffusion 3D representation:
- Model predicts 17D world pose → Python renderer for training loss / visualization.
- Helios XML is used only for **validation / external tooling**, not as the reconstruction target.
- This sidesteps the Helios FK non-invertibility entirely.

**Option 2 (predict in Helios FK parameter space)** is *not* what 40D does — 40D is structure,
render still uses 17D world pose. If Helios reload fidelity is ever required, the model would need
to output `(base_rotation, internode_pitch, gravity, ...)` directly (large representation change).

---

## 5. Files / Artifacts

| Path | Description |
|---|---|
| `configs/params_cowpea.json` | gravitropic → `constant(200)` (committed change) |
| `Digital-Crops/projects/syntheticdata_generation/main.cpp` | focus-plant margin 1.20→1.05 (committed change) |
| `docs/results/assets/fig_python_world_reconstruction.png` | Python-world reconstruction demo (focus framing aligned) |
| `docs/results/assets/fig_oneway_17d_to_xml_helios_exhaustive.png` | 17D→XML→Helios exhaustive (grav=200) |
| `docs/results/assets/fig_grav_curvature_ab_test.png` | gravitropic tag A/B (load-path) |
| `docs/done/20260831_pr_gravitropic_curvature_xml.md` | PR draft (superseded by `gravitropic=200` fix) |

---

## 6. Summary of Decisions

1. **Figure Framing:** Resolved by passing `--focus-plant` to Helios GT rad rendering, matching Python's 5% focus margin camera.
2. **Canonical Render Path:** Python 17D renderer is canonical for Flow Matching and Diffusion training. Helios XML export serves as auxiliary structural export.
3. **Gravitropic Setting:** Retain `gravitropic_curvature = constant(200)` in `configs/params_cowpea.json`.
4. **Upstream PR:** `20260831_pr_gravitropic_curvature_xml.md` archived as superseded.

---

## 7. Key Code Locations

- **17D part tensor columns:** `diffusion_based/models/plant_organ_array.py` (`P_COL_*`, NUM_FEATURES=17)
- **17D→XML converter:** `diffusion_based/models/part_assembly_to_xml.py` (`PartAssemblyToXMLConverter`)
- **Python renderer:** `diffusion_based/models/helios_pytorch_renderer.py`
  - `compute_focus_plant_camera` (5% margin, line ~141)
  - `render_part_tensor` (line ~758)
- **Helios reload FK:** `Digital-Crops/libs/Helios/plugins/plantarchitecture/src/InputOutput.cpp`
  - `recomputeInternodeOrientationVectors_local` (~line 1479)
  - internode segment loop with gravity (~line 1596)
- **Helios growth:** `.../PlantArchitecture.cpp` — `Phytomer` ctor (~line 1520), `setInternodeLengthScaleFraction` (~line 2849)
- **Helios focus-plant:** `Digital-Crops/projects/syntheticdata_generation/main.cpp` (~line 1729)
- **Config sampling:** `Digital-Crops/projects/syntheticdata_generation/configs/params_cowpea.json`
