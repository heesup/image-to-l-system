# Data diversity, not architecture, is most of the appearance gap

2026-09-20. The 2026-09-19 protocol finding attributed the appearance gap largely to
`stage3_absolute`. That comparison was run with every model trained on the same 10 000-sample
subset. With ten times the unique data the gap shrinks by up to 15 points, which means a large part
of what looked architectural was **overfitting to a small training subset**.

## The measurement

`merged_full` and `merged_fullaug` warm-start from `v10_cam` ep160 on all 100 000 plants
(25 000 steps/epoch). At epoch 170 they have seen 10 x 100 000 = **1.0 M samples**, slightly FEWER
than `merged_abs`/`merged_aug` had seen at ep280 (120 x 10 000 = 1.2 M). So this is sample-matched
or better, and the only thing that changed is how many distinct plants those samples came from.

| variant | data | flat | Helios | gap |
| :--- | :--- | ---: | ---: | ---: |
| absolute, no aug | 10k ep280 | 73.34 | 48.36 | −24.98 |
| absolute, no aug | **full ep170** | 71.95 | **62.13** | **−9.82** |
| absolute, aug | 10k ep280 | 73.37 | 59.11 | −14.26 |
| absolute, aug | **full ep170** | 70.92 | **64.99** | **−5.93** |
| plain, aug | 10k ep230 | 72.65 | 70.16 | −2.49 |
| plain, no aug | 10k ep160 | 71.23 | 66.53 | −4.70 |

    effect of 10x unique data on the gap:  no aug +15.17    aug +8.33

That is larger than the effect of appearance augmentation itself (+2.2 on the plain lineage, +10.7
on the absolute one), and it is the biggest single factor on the gap measured so far.

## Flat goes DOWN while Helios goes UP

The full-data checkpoints score **lower** on flat (73.34 → 71.95, 73.37 → 70.92) and substantially
**higher** on Helios. That is the signature of reduced memorisation: the eval plants are in-sample
for every full-data lineage here, and with ten times the data each individual plant is seen a tenth
as often, so the memorisation-friendly score falls while the one that needs generalisation rises.

This sharpens the 2026-09-19 protocol conclusion. The flat render does not merely fail to
discriminate architectures — **it actively rewards memorisation**, and every model in yesterday's
2x2 was trained on 10 k and therefore memorising. Two separate reasons to stop quoting it as the
headline.

## What this does and does not overturn

**Revised.** "`stage3_absolute` is what makes the lineage appearance-brittle, costing 20.3 points of
gap" was measured entirely at 10 k. At full data the absolute+aug gap is −5.93, against −2.49 for
plain+aug *at 10 k*. Most of the penalty was data, not architecture.

**Still standing.** On the Helios column as it stands today, the plain lineage still leads:

    plain, aug        10k    70.16
    plain, no aug     10k    66.53
    absolute, aug     full   64.99
    absolute, no aug  full   62.13

But the comparison is now unfair in the opposite direction from yesterday: the full-data runs are
**10 epochs into a 60-epoch schedule**, while `sub10_v10_cam_aug` had 70 epochs of fine-tuning. They
are sample-matched to the 10 k runs, not to the plain lineage's training length, and they are still
improving.

**Unknown, and the reason for the new run.** Nobody has trained the plain lineage with augmentation
on full data. It is the current best configuration and the missing cell of the 2x2. `plain_fullaug`
(job 38495668) was launched today with settings byte-identical to `merged_fullaug` except that
`STAGE3_ABSOLUTE` is absent, so the pair isolates the architecture at full data.

## Decision

**Do not cancel `merged_full` / `merged_fullaug`.** Yesterday's recommendation to consider cancelling
rested on the 10 k comparison, which overstated the architectural penalty by roughly a factor of
three. The gap is closing quickly, the runs are early, and they are now the only evidence on whether
the absolute state is viable at scale.
