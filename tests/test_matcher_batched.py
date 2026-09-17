"""The batched phytomer-level matcher pass must reproduce the per-sample pass: same clusters (as
positions), same matched (predicted node -> cluster position) pairs, same counts."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from plant_recon.training.hierarchical_hungarian_matcher import HierarchicalBotanicalMatcher


def _pairs(m):
    return {(int(s), tuple(round(float(x), 6) for x in p)) for s, p in zip(m["phytomer_src_idx"], m["phytomer_tgt_pos"])}


def _centers(m):
    return {tuple(round(float(x), 6) for x in p) for p in m["gt_node_centers"]}


class TestMatcherBatched(unittest.TestCase):
    def _batch(self, seed, B=6, K=64, no_pet_samples=(1,), no_ino_samples=(2,), empty_samples=(3,)):
        g = torch.Generator().manual_seed(seed)
        tgt_labels, tgt_positions = [], []
        for b in range(B):
            if b in empty_samples:
                tgt_labels.append(torch.empty(0, dtype=torch.long)); tgt_positions.append(torch.empty(0, 3)); continue
            n = int(torch.randint(5, 60, (1,), generator=g))
            labels = torch.randint(3, 8, (n,), generator=g)          # 3 internode, 4 petiole, 5 leaf, 6 peduncle, 7 bud
            if b in no_pet_samples:
                labels[labels == 4] = 5
            if b in no_ino_samples:
                labels[labels == 4] = 5; labels[labels == 3] = 5
            pos = torch.rand(n, 3, generator=g) * 0.3
            # some internodes right next to a petiole (not standalone), some far
            tgt_labels.append(labels); tgt_positions.append(pos)
        pred_pos = torch.rand(B, K, 3, generator=g) * 0.3
        logits = torch.randn(B, K, 1, generator=g)
        soft = torch.rand(B, K, generator=g)
        cap = torch.randint(3, 40, (B,), generator=g)
        return tgt_labels, tgt_positions, pred_pos, logits, soft, cap

    def _compare(self, matcher, seed, **kw):
        tgt_labels, tgt_positions, pred_pos, logits, soft, cap = self._batch(seed)
        args = dict(pred_phytomer_pos=pred_pos, pred_phytomer_logits=logits, pred_fine_geom=torch.zeros(6, 640, 16),
                    pred_fine_logits=torch.zeros(6, 640, 13), tgt_geoms=[torch.zeros(t.shape[0], 16) for t in tgt_labels],
                    tgt_labels=tgt_labels, tgt_positions=tgt_positions, skip_fine=True, **kw)
        if kw.get("use_soft"): args["soft_margin_weights"] = soft
        if kw.get("use_cap"): args["per_sample_max_phytomers"] = cap
        args.pop("use_soft", None); args.pop("use_cap", None)
        matcher.batched = True; a = matcher(**args)
        matcher.batched = False; b = matcher(**args)
        for i, (ma, mb) in enumerate(zip(a, b)):
            self.assertEqual(ma["num_gt_phytomers"], mb["num_gt_phytomers"], f"seed {seed} sample {i}: cluster count")
            self.assertEqual(_centers(ma), _centers(mb), f"seed {seed} sample {i}: cluster centres")
            self.assertEqual(_pairs(ma), _pairs(mb), f"seed {seed} sample {i}: matched pairs")
            self.assertEqual(len(ma["phytomer_src_idx"]), len(set(ma["phytomer_src_idx"].tolist())), "1:1 on the predicted side")

    def test_plain(self):
        m = HierarchicalBotanicalMatcher(matcher_type="greedy")
        for seed in range(5):
            self._compare(m, seed)

    def test_soft_margin_and_capacity(self):
        m = HierarchicalBotanicalMatcher(matcher_type="greedy")
        for seed in range(5):
            self._compare(m, 10 + seed, use_soft=True, use_cap=True)

    def test_padded_inputs_equal_lists(self):
        m = HierarchicalBotanicalMatcher(matcher_type="greedy")
        for seed in range(4):
            tgt_labels, tgt_positions, pred_pos, logits, soft, cap = self._batch(30 + seed)
            B = len(tgt_labels); N = 80
            lab = torch.zeros(B, N, dtype=torch.long); pos = torch.zeros(B, N, 3); valid = torch.zeros(B, N, dtype=torch.bool)
            for b in range(B):
                n = tgt_labels[b].shape[0]
                # scattered rows, in order (the padded batch keeps the organ order; the
                # no-petiole/no-internode fallback takes the FIRST max(1, n // M) organs)
                perm = torch.sort(torch.randperm(N)[:n]).values
                lab[b, perm] = tgt_labels[b]; pos[b, perm] = tgt_positions[b]; valid[b, perm] = True
            common = dict(pred_phytomer_pos=pred_pos, pred_phytomer_logits=logits, pred_fine_geom=torch.zeros(B, 640, 16),
                          pred_fine_logits=torch.zeros(B, 640, 13), tgt_geoms=[torch.zeros(t.shape[0], 16) for t in tgt_labels],
                          tgt_labels=tgt_labels, skip_fine=True, soft_margin_weights=soft, per_sample_max_phytomers=cap)
            m.batched = True
            a = m(tgt_positions=tgt_positions, **common)
            b_ = m(tgt_positions=None, tgt_labels_padded=lab, tgt_positions_padded=pos, tgt_valid=valid, **common)
            for i, (ma, mb) in enumerate(zip(a, b_)):
                self.assertEqual(ma["num_gt_phytomers"], mb["num_gt_phytomers"], f"seed {seed} sample {i}")
                self.assertEqual(_centers(ma), _centers(mb), f"seed {seed} sample {i}: centres")
                self.assertEqual(_pairs(ma), _pairs(mb), f"seed {seed} sample {i}: pairs")

    def test_locality_radius(self):
        m = HierarchicalBotanicalMatcher(matcher_type="greedy", phytomer_locality_radius=0.1)
        for seed in range(3):
            self._compare(m, 20 + seed, use_cap=True)


if __name__ == "__main__":
    unittest.main()
