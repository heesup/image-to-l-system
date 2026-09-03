> [!NOTE]
> **Status: Superseded / Archived (2026-09-01)**
> This PR draft investigated an XML `<gravitropic_curvature>` tag to resolve reload discrepancies. It was superseded by setting `gravitropic_curvature = constant(200)` in `configs/params_cowpea.json`, which avoids non-standard XML tags and achieves exact growth==reload fidelity without Helios library modifications.

## fix(plantarchitecture): persist realized per-shoot `gravitropic_curvature` in XML for lossless reload

### Problem

`readPlantStructureXML` reconstructs internode geometry by forward kinematics that re-applies
`shoot->gravitropic_curvature` per segment:

```cpp
curvature_angle = deg2rad(gravitropic_curvature * current_curvature_fact * dr_max + saved_perturbation);
```

The per-segment `curvature_perturbations` / `yaw_perturbations` are serialized, but the **dominant
deterministic term — the realized per-shoot `gravitropic_curvature` — is not**. On reload the shoot
grabs its value from the plant's `shoot_types_snapshot` at creation time, which can differ wildly from
the value that shaped the plant (e.g. the library default `+200` vs. a value sampled via `params.json`
such as `-608.9`).

**Symptom:** a plant grown with a strongly negative sampled curvature reloads as a tall, wiry
"star" (canopy height inflated ~56%, `z`-span 1.89 m → 2.95 m) instead of the original prostrate
bush. This breaks XML round-trips and any consumer that renders from the XML alone (Helios C++ reload
*and* the Python differentiable renderer).

**Scope of this PR — the *load* path.** It makes XML reload deterministic and self-consistent: the
same XML always reconstructs the same plant regardless of which shoot-type defaults were in effect at
growth time. It does **not** change the fact that a reloaded plant's internode wiring differs in detail
from the original aging-grown plant (see "Known limitation" below).

---

### Fix

Add an optional shoot-level `<gravitropic_curvature>` tag, written from the realized per-shoot value
and restored on read (falling back to the shoot-type default when absent, so older files still load).

#### Writer — `writePlantStructureXML` (InputOutput.cpp)

```diff
 output_xml << "\t\t\t<base_rotation> " << ... << " </base_rotation>" << std::endl;
+output_xml << "\t\t\t<gravitropic_curvature> " << shoot->gravitropic_curvature << " </gravitropic_curvature>" << std::endl;
```

#### Reader — `readPlantStructureXML` (InputOutput.cpp)

Parse the optional tag after `base_rotation`:

```diff
+float shoot_gravitropic_curvature;
+bool has_gravitropic_curvature = false;
+if (shoot.child("gravitropic_curvature")) {
+    shoot_gravitropic_curvature = parse_xml_tag_float(shoot.child("gravitropic_curvature"), ...);
+    has_gravitropic_curvature = true;
+}
```

Override the snapshot-sampled value on the `Shoot` object **before** phytomer geometry is
reconstructed (the snapshot is shared by all shoots of this type on the plant, so it must not be
mutated):

```diff
 current_shoot_ID = addBaseStemShoot(...);   // or addChildShoot(...)
 shoot_ID_mapping[shootID] = current_shoot_ID;
+if (has_gravitropic_curvature) {
+    plant_instances.at(plantID).shoot_tree.at(current_shoot_ID)->gravitropic_curvature = shoot_gravitropic_curvature;
+}
```

---

### Validation

**A/B test — tag handling on the load path.** The *same* grown plant (XML carrying the tag) is
reloaded with library defaults (no `params.json`), and only whether the binary reads the tag differs:

| | reload-vs-grow plant-mask IoU | reload-vs-grow pixel MAE |
|---|---|---|
| **BEFORE fix** (`main_old`, ignores tag → `+200`) | 0.287 | 0.182 |
| **AFTER fix** (`main_new`, reads tag → `-608.9`) | 0.287 | 0.182 |

> The plant-mask/MAE numbers are dominated by leaf rendering and framing, so they barely move. The fix
> is validated at the **internode level**, where the effect is decisive.

**Internode-mask IoU, load-path self-consistency** (same XML, default reload context, only binary
differs):

| compare | internode IoU | meaning |
|---|---|---|
| `main_new` reload vs `main_old` reload (same XML) | **0.992** | tag makes load deterministic; both binaries agree |
| `main_new` reload vs original aging-grown | 0.019 | separate "growth-replay" gap, see Known limitation |

![A/B test](results/assets/fig_grav_curvature_ab_test.png)

**Roundtrip + Python renderer parity (same plant seed00, grav=`-608.9`, DAP 10/50/90).** The Python
differentiable renderer reconstructs the *same* internode geometry as C++ reload (both read the tag):
internode bbox:

| DAP | C++ reload | Python FK |
|---|---|---|
| 10 | (0.025, 0.007, 0.147) | (0.032, 0.012, 0.178) |
| 50 | (0.856, 1.404, 1.029) | (0.853, 1.427, 1.025) |
| 90 | (1.040, 1.969, 1.828) | (0.966, 1.838, 1.864) |

![Roundtrip + Python renderer, DAP 10/50/90](results/assets/fig3_roundtrip_python_dap10_50_90.png)

---

### Known limitation (out of scope, separate work)

The reload FK reconstructs internode geometry **in a single pass from the final XML lengths**, whereas
the original growth builds each internode short (`internode_length_scale_factor_fraction = 0.01`) then
stretches it over 70 days via `setInternodeLengthScaleFraction()`. Because gravity curvature accumulates
on the short, growing internode, the final shape encodes a **curvature-vs-length history** that the XML
does not store. As a result, a reloaded plant's internode wiring (esp. shoot mid-to-tip, phyto 12+) can
diverge from the original aging-grown plant even with a correct gravitropic tag. This is a model-replay
gap, not a tag bug, and is tracked separately.

---

### Notes

- The tag is **optional on read**; files written before it existed load unchanged (snapshot default).
- The tag is **shoot-level, not per-internode**: `gravitropic_curvature` is a single realized scalar
  sampled per shoot (`Shoot` ctor), shared by every internode on that shoot.
- `tortuosity` is intentionally **not** persisted: its effect is already captured by the saved
  per-segment perturbation vectors, so re-persisting the scalar would be redundant.
- The Python differentiable renderer (`helios_pytorch_geometry.py`) now reads the same tag through the
  typed 40D layout (`T_COL_RESERVED` on the `ORGAN_SHOOT_META` row) instead of hardcoding `eff_grav = 200`.
- `CanopyGenerator` is unrelated: it builds raw Context geometry (no shoot tree) and has no XML export
  for individual plants, so its output cannot be converted to PlantArchitecture or stored as architecture XML.

### Related: `main.cpp` uses the age-growing build API

`syntheticdata_generation/main.cpp` now passes the real DAP as the age to the builder, so library-built
plants grow internally in one place:

- MANUAL: `buildPlantInstanceFromLibrary(origin, true)` → `buildPlantInstanceFromLibrary(origin, plant_age_days)`
- AUTO: `buildPlantCanopyFromLibrary(..., 0)` → `buildPlantCanopyFromLibrary(..., plant_age_days)`
- Removed the redundant batch `advanceTime(plant_IDs_aging, dap)` (double-aging), keeping `plant_IDs_aging`
  only for per-plant mask labeling.
