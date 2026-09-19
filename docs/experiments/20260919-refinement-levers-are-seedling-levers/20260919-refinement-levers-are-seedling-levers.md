# Test-time refinement: what survives replication

2026-09-19. A sweep of refinement flags, the correction that followed when its weakest claim was
checked, and the noise floor that should have been measured first.

Checkpoint `outputs/checkpoints/hierarchical_fm_v10_cam/hierarchical_fm_epoch_160.pt`, eval set
`outputs/checkpoints/hierarchical_fm_v9/eval_set.json` (20 plants), `--steps 200 --sample_seed 0`.
Seedling set `outputs/eval_sets/seedlings24.json` (24 plants, DAP 1-15). Logs and per-plant JSON in
`outputs/logs/20260919/{s_*,sd_*,rep_*}.{log,json}`. Figure: `assets/refinement_levers.png`
(regenerate with `assets/plot_refinement_levers.py`).

## 0. The methodological finding, which matters more than the sweep

**Test-time refinement is nondeterministic, and on a per-plant basis it is wildly so.** Re-running a
byte-identical command reproduces the flow's starting sample bit-for-bit — `--sample_seed` keys off
the dataset index, not list position, so it is doing its job — but not the refined result:

| plant (DAP) | raw | `reg_latent 0` run A | run B |
| :--- | ---: | ---: | ---: |
| 1354 (2) | 0.0 | 41.0 | **9.7** |
| 2616 (3) | 26.8 | 17.9 | 23.2 |
| 10833 (11) | 26.3 | 64.0 | 61.8 |
| 11498 (12) | 23.9 | 54.2 | 62.1 |

Plant 1354 swings 31 points between runs that differ in nothing. The likely cause is
order-nondeterministic atomic gradient accumulation in the nvdiffrast backward pass; 200 optimiser
steps compound it, and seedlings suffer most because the plant covers few pixels and the loss
surface is poor, so a small perturbation early sends the trajectory to a different optimum.

`--sample_seed` (commit `a78299d`) removed the *sampling* confound and it was tempting to treat that
as having made runs comparable. It only fixed the starting point. Everything after it is still
stochastic.

**But the 20-plant mean is stable**, because independent per-plant noise averages down by sqrt(n).
Four runs of each of two configs:

```
default (reg_latent 0.5) : 70.6  71.4  71.5  71.5    mean 71.23   SD 0.43
reg_latent 0             : 72.5  73.1  73.3  74.2    mean 73.27   SD 0.70
```

**Single-run noise floor on the 20-plant mean: SD ~0.6 points.** A difference between two single
runs needs to clear roughly **1.6 points** before it means anything. Several conclusions in the
first version of this document did not clear it.

The rule this gives us: aggregate differences above ~2 points on 20 plants are believable from
single runs; anything smaller needs replicates; and **no per-plant claim is safe without them.**

## 1. What survives

Every config below is one run, judged against the replicated default mean of 71.23 and the 0.58
noise floor.

| config | refined | vs default | in SD | verdict |
| :--- | ---: | ---: | ---: | :--- |
| `--n_starts 8` (Gate G) | 74.3 | +3.07 | +5.3 | **real** |
| `--reg_latent 0` + `--n_starts 8` | 74.6 | +3.37 | +5.8 | **real** |
| `--reg_latent 0` | 73.27 (4 runs) | +2.04 | 4.9 SE | **real**, 95% CI [+1.2, +2.9] |
| `--refine_px 256` | 69.2 | -2.03 | -3.5 | **real** (hurts) |
| `--refine_px 384` | 69.8 | -1.43 | -2.4 | marginal (hurts) |
| no latent *or* scale prior | 72.5 | +1.27 | +2.2 | marginal |
| `--reg_latent 0.25` | 71.9 | +0.67 | +1.1 | not resolved |
| `--reg_latent 2.0` | 70.2 | -1.03 | -1.8 | not resolved |

**Gate G is the strongest single lever at +3.1.** `--reg_latent 0` is second at +2.0 and costs
nothing. The two **do not compose**: together they give +3.4, which is +0.3 over Gate G alone and
well inside the noise floor. They are two escapes from the same bottleneck — a bad latent proposal,
resampled eight times or simply not adhered to — and once escaped the second escape adds nothing.

**Raising the render resolution hurts.** A plausible mechanism is that a sharper silhouette narrows
the rasteriser's gradient support band at the mask boundary, so an organ must already overlap the
target to feel any pull; low resolution is a wider basin of attraction. That predicts the damage
should concentrate where the initialisation is worst, which is what the DAP split shows (below).
This retires the queued "GT depth pyramid re-rendered at 512" arm.

## 2. What does not survive

Three claims from the first version of this document are withdrawn.

- **"The prior ladder is cleanly monotone."** Only `reg_latent 0` separates from the default. The
  0.25 and 2.0 points sit within the noise floor, so the shape of the curve between them is not
  measured. What is established is that **zero beats the default**; "monotone in prior strength" was
  reading a trend into four unreplicated points.
- **"The scale prior is worth 1.7 points."** Dropping it alongside the latent prior scores 72.5
  against `reg_latent 0`'s replicated 73.27 — a 0.8-point gap, comfortably inside noise. The scale
  prior may well be doing its job, but this sweep does not show it. Keeping it is a reasonable
  default on the original argument (it stops leaves inflating to fill the silhouette), not on this
  evidence.
- **"`--reg_latent 0` matches Gate G at one eighth the compute."** It does not. Gate G is about a
  point better (+3.07 vs +2.04). The single runs that made them look tied (74.3 vs 74.2) differed by
  less than the noise floor. The honest version is below.

## 3. The DAP split, and the seedling replication

Splitting the original nine-run sweep by growth stage:

**DAP>15 is flat at 79.8-82.2 across all nine configs** — a 2.4-point spread against a 0.6 noise
floor, so at most one or two of those gaps are real and none is large. Nothing in this sweep moves
mature plants meaningfully. **DAP<=15 spans 26.2-48.0.** Every aggregate effect above is a seedling
effect diluted by the sixteen mature plants in the set.

That seedling band is n=4, which is far too small given the per-plant noise documented in §0. So it
was re-measured on a purpose-built 24-plant seedling set (DAP 1-15, the original four included),
identical settings, identical raw of 18.0 across configs:

| `reg_latent` | seedling IoU (n=24) |
| ---: | ---: |
| 2.0 | 29.4 |
| 0.5 (default) | 30.8 |
| 0.0 | **36.6** |

Paired over plants, `reg_latent 0` beats the default by **+5.8**, 95% CI [+1.6, +10.1], better on 16
of 24 plants and worse on 7. The direction replicates; **the magnitude is about a third of the +18
the four-plant sample suggested.** A +5.8 seedling effect on 4 of 20 plants contributes ~+1.2 to the
20-plant mean, which is consistent with the measured +2.04 once small mature-plant gains are added.

## 4. What to use

**`--n_starts 8` if the compute is available; `--reg_latent 0` if it is not.** Gate G is ~1 point
better and costs about 25 minutes per 20 plants against 4. `--reg_latent 0` recovers roughly
two-thirds of Gate G's gain for free, as a single flag. Do not bother combining them.

Left as an explicit flag rather than changed as a default, so in-flight runs stay comparable with
past numbers.

**For any future refinement comparison: replicate, or compare only differences above ~2 points on
20 plants.** Per-plant numbers need replicates regardless.

## 5. What this reframes

The refinement plateau is not an optimiser problem — L-BFGS lost badly to Adam (54.3 vs 71.1 at 7.8x
the cost, commit `917d11e`) — and not a resolution problem. On mature plants there is no plateau
worth attacking: refinement reaches ~81% and the residual is in the initialisation the flow hands
over, not in the fitting. On seedlings the fitting itself is fragile, in the strong sense that its
outcome is not reproducible run to run.

Both levers that work, work by weakening the flow's proposal. That is an odd place to be: the
generative model's per-organ belief is worth less than no belief at all. It is consistent with the
Stage 3 conditioning probes (`docs/experiments/20260918-stage3-conditioning-settled/`), which found
per-organ shape is not recoverable from a nadir view at any resolution tested — pulling toward a
latent that carries no shape information is pulling toward noise.

It also argues against score distillation, which is a *stronger* pull toward that same belief.

The open question is why the flow's seedling proposal is bad enough that ignoring it beats using it.
That is the population sitting at 18-19% raw IoU with a 19x hull ratio, and it is a training-side
problem that no amount of refinement machinery will fix.
