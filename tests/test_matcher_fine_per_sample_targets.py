"""Regression test for the Pass-3 stale-target bug (2026-09-17, organ mode).

`t_label` / `t_geom` are loop variables of the matcher's Pass 1. Pass 3 loops over the batch
again to do the fine (organ) matching; it used to read those two names without re-reading them
per sample, so every sample was matched against the LAST sample's organ labels and geometry.
When an earlier sample had more organs than the last one, indexing ran past the end of
`t_label` and the CUDA kernel raised "index out of bounds"; when the counts happened to line
up it matched silently against the wrong plant.

Three checks, each of which the buggy version fails:
  1. Matched GT organ indices stay inside this sample's organ list.
  2. Matching a sample alone (B=1) gives exactly what matching it inside a batch gives. This is
     the exact oracle: the fine matching of a sample depends on nothing outside that sample.
  3. The same holds when the last sample of the batch has no organs at all, which is the case
     that leaves the stale `t_label` empty and so breaks every earlier sample.

Note that role agreement between a slot and the organ matched to it is NOT an invariant and so
is not tested here: after the role-constrained matching, the matcher deliberately fills any
free slot of a phytomer with that phytomer's leftover organs regardless of role.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from plant_recon.training.hierarchical_hungarian_matcher import HierarchicalBotanicalMatcher

M = 8          # slots per phytomer in organ mode
K = 32         # predicted node positions
NODE_DIM = 16  # organ latent width


def _sample(n, g):
    """One plant: `n` organs with labels spanning every role."""
    labels = torch.randint(3, 8, (n,), generator=g)      # 3 internode, 4 petiole, 5 leaflet, 6 peduncle, 7 bud
    positions = torch.rand(n, 3, generator=g) * 0.3
    geoms = torch.randn(n, NODE_DIM, generator=g)
    return labels, positions, geoms


def _preds(B, g):
    pred_pos = torch.rand(B, K, 3, generator=g) * 0.3
    pred_logits = torch.randn(B, K, 1, generator=g)
    pred_fine_geom = torch.randn(B, K * M, NODE_DIM, generator=g)
    pred_fine_logits = torch.randn(B, K * M, 13, generator=g)
    return pred_pos, pred_logits, pred_fine_geom, pred_fine_logits


class TestMatcherFinePerSampleTargets(unittest.TestCase):
    # Sample 0 is the largest and the last is the smallest: this is the ordering that made the
    # stale-target bug index past the end rather than merely match the wrong plant.
    COUNTS = [70, 40, 55, 12]

    def setUp(self):
        self.matcher = HierarchicalBotanicalMatcher(slots_per_phytomer=M, phytomer_locality_radius=0.15)
        g = torch.Generator().manual_seed(20260917)
        self.labels, self.positions, self.geoms = [], [], []
        for n in self.COUNTS:
            lab, pos, geo = _sample(n, g)
            self.labels.append(lab); self.positions.append(pos); self.geoms.append(geo)
        self.pred = _preds(len(self.COUNTS), g)

    def _match_batch(self):
        pred_pos, pred_logits, pred_fine_geom, pred_fine_logits = self.pred
        return self.matcher(
            pred_pos, pred_logits, pred_fine_geom, pred_fine_logits,
            tgt_geoms=self.geoms, tgt_labels=self.labels, tgt_positions=self.positions,
            skip_fine=False,
        )

    def _match_alone(self, b):
        pred_pos, pred_logits, pred_fine_geom, pred_fine_logits = self.pred
        return self.matcher(
            pred_pos[b:b + 1], pred_logits[b:b + 1], pred_fine_geom[b:b + 1], pred_fine_logits[b:b + 1],
            tgt_geoms=[self.geoms[b]], tgt_labels=[self.labels[b]], tgt_positions=[self.positions[b]],
            skip_fine=False,
        )[0]

    def test_fine_indices_stay_inside_this_samples_targets(self):
        for b, m in enumerate(self._match_batch()):
            tgt = m["fine_tgt_idx"]
            self.assertGreater(tgt.numel(), 0, f"sample {b} matched no organs at all")
            self.assertLess(int(tgt.max()), self.COUNTS[b],
                            f"sample {b} indexed GT organ {int(tgt.max())} of {self.COUNTS[b]}")
            self.assertGreaterEqual(int(tgt.min()), 0)

    def test_batched_fine_matching_equals_matching_each_sample_alone(self):
        batched = self._match_batch()
        for b in range(len(self.COUNTS)):
            alone = self._match_alone(b)
            for key in ("fine_src_idx", "fine_tgt_idx", "phytomer_src_idx", "phytomer_tgt_idx"):
                torch.testing.assert_close(
                    batched[b][key], alone[key],
                    msg=f"sample {b}: '{key}' differs between batched and single-sample matching",
                )

    def test_an_empty_last_sample_does_not_corrupt_the_others(self):
        """A plant with no organs is a normal batch member (the matcher has a branch for it), and
        it is the worst case for the bug: it leaves the stale targets empty, so every earlier
        sample in the batch loses its fine matching."""
        matcher = HierarchicalBotanicalMatcher(slots_per_phytomer=M, phytomer_locality_radius=0.15)
        g = torch.Generator().manual_seed(4242)
        counts = [64, 30, 0]
        labels, positions, geoms = [], [], []
        for n in counts:
            if n == 0:
                labels.append(torch.empty(0, dtype=torch.long))
                positions.append(torch.empty(0, 3))
                geoms.append(torch.empty(0, NODE_DIM))
                continue
            lab, pos, geo = _sample(n, g)
            labels.append(lab); positions.append(pos); geoms.append(geo)
        pred_pos, pred_logits, pred_fine_geom, pred_fine_logits = _preds(len(counts), g)

        def match(sl):
            return matcher(
                pred_pos[sl], pred_logits[sl], pred_fine_geom[sl], pred_fine_logits[sl],
                tgt_geoms=geoms[sl], tgt_labels=labels[sl], tgt_positions=positions[sl],
                skip_fine=False,
            )

        batched = match(slice(0, 3))
        self.assertEqual(batched[2]["fine_tgt_idx"].numel(), 0, "the empty sample matched organs")
        for b in range(2):
            alone = match(slice(b, b + 1))[0]
            self.assertGreater(batched[b]["fine_tgt_idx"].numel(), 0,
                               f"sample {b} lost its fine matching next to an empty sample")
            for key in ("fine_src_idx", "fine_tgt_idx"):
                torch.testing.assert_close(
                    batched[b][key], alone[key],
                    msg=f"sample {b}: '{key}' changed because of the empty sample beside it",
                )


if __name__ == "__main__":
    unittest.main()
