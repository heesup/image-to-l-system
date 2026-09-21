# Was merging the phytomer parameters into one flow state worth it? Re-measured

2026-09-20. The decision the whole current lineage rests on was made on 2026-09-14/15 and never
re-checked: should the phytomer parameters be predicted **jointly in one flow state**
(`--stage3_geometry`), or **split** — position, rotation and scale from direct Stage-2 heads, with
the flow carrying only the 128-D shape latent?

The original reading recorded geometry-in-flow at **36.0 / 33.6** against **25–29** for the
latent-only variants on 10% data. Those were *raw* deployable numbers, on the rasterised protocol,
before the 2026-09-19 finding that rasterised barely discriminates and before the 2026-09-20
rotation fix. All the checkpoints survive, so it can simply be re-run.

`sub10_base` ep095 (latent-only), `sub10_v10` ep100 and `sub10_combo` ep095 (phytomer flow), all
10 000 samples, `--steps 200 --sample_seed 0`, replicated, on the shared 20-plant set.

| | raw | refined | SD |
| :--- | ---: | ---: | ---: |
| latent-only, rasterised | 26.2 | 70.05 | 0.60 |
| latent-only, **raytraced** | 15.2 | **59.05** | 0.56 |
| phytomer flow, rasterised | 32.8 | 71.67 | 0.05 |
| phytomer flow, **raytraced** | 26.8 | **63.32** | 0.65 |
| + combo, rasterised | 32.1 | 69.17 | 0.19 |
| + combo, **raytraced** | 23.5 | **64.30** | 0.11 |

## The merge was the right call, but the number that was quoted is the raw one

    merge advantage      raw      refined
      rasterised        +6.6       +1.62
      raytraced        +11.6       +4.26

The raw re-measurement reproduces the original story: **+6.6 rasterised, +11.6 raytraced**. That is
the gap the 2026-09-14 reading saw, and it is large.

But refinement is always applied, so the figure that actually ships is the refined one, and there
the advantage is **+1.62 rasterised / +4.26 raytraced**. The merge is still clearly worth having —
+4.26 against an 0.6 noise floor is unambiguous — but "worth ten points" describes the forward pass,
not the delivered system. This is the same pattern that has recurred all week: refinement
compensates for most of an architectural difference, so any architecture claim measured before
refinement overstates what the change is worth in deployment.

## Raytraced separates the two designs about 2.5x better than rasterised

    raw       rasterised +6.6   raytraced +11.6
    refined   rasterised +1.62  raytraced +4.26

An independent confirmation of the 2026-09-19 protocol finding, on a comparison that has nothing to
do with the absolute layout: the in-domain protocol compresses a real architectural difference
toward zero at both stages. On the refined rasterised column the merge looks nearly free to skip
(+1.6); on raytraced it is plainly not (+4.3).

## Why the split design failed

Recorded at the time and not contradicted here: in the latent-only configuration the flow was doing
almost nothing. The sampled latent's error against the matched GT latent was *worse* than the
dataset-mean latent's (RMSE 5.39 vs 4.66), and substituting the mean latent outright scored 28.5
against 27.5 for the prediction. Splitting the phytomer parameters left the flow with the one
component that carries no recoverable image information; merging gave it the components that do.

## Caveat

These are ~ep95–100 checkpoints on 10 000 samples, so this settles the historical decision, not the
current lineage's standing. The live question is the layout of the merged state — relative versus
absolute — which `relative_fullaug`, `absolute_full` and `absolute_fullaug` are running now.


## Correction (2026-09-21): "merge" is the project's word, not what the code does

Asked whether the two are merged, the code says no. `loss_phytomer_pos` and its siblings are gated
only on `total_matched_nodes > 0` — never on `stage3_geometry` — so the coarse stage keeps
predicting position, roll and scale in every run, and the training log shows those losses live
alongside the flow's. At sampling, `sample_ode` stores the coarse values as `stage2_*` and then
**overwrites** `phytomer_pos / phytomer_roll / phytomer_scale` from the flow's geometry block.

So the relationship is a **cascade, not a merge**: the coarse stage predicts the phytomer
parameters, that prediction conditions the flow, and the flow predicts them again and wins. What
`--stage3_geometry` changed was giving the flow geometry to predict at all; before it, the flow
carried only the latent and the coarse geometry was final. That also explains why the relative
layout's target is a displacement from a *noised* parent — the model is being taught to correct the
scaffold's error from the image.

The comparison above is unaffected: it contrasts a flow that predicts geometry against one that does
not, which is exactly what the flag toggles.
