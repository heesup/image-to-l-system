"""chain_phytomers' pointer-jumping cycle breaking and shoot walk (2026-09-14) must agree with the
Python loops they replaced: same parents, same phytomer index along the shoot, same partition into
shoots (numbering may differ; a parent shoot must still come before its children)."""
import glob
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from plant_recon.dataset.phytomer_topology import chain_phytomers

_XMLS = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "dataset", "helios_data", "cowpea",
                                      "cowpea_dap0[2-9]0_seed0[0-1]_*_0000_plant_0000.xml")))


def _partition(shoot, phy):
    groups = {}
    for i, (s, k) in enumerate(zip(shoot.tolist(), phy.tolist())):
        if s >= 0:
            groups.setdefault(s, []).append((k, i))
    return {tuple(i for _, i in sorted(v)) for v in groups.values()}


def _assert_same(tc, a, b, msg):
    pa, sa, ka = a; pb, sb, kb = b
    tc.assertEqual(pa.tolist(), pb.tolist(), f"{msg}: parents")
    tc.assertEqual(ka.tolist(), kb.tolist(), f"{msg}: phytomer index")
    tc.assertEqual(_partition(sa, ka), _partition(sb, kb), f"{msg}: partition")
    # a parent shoot is numbered before its children (both implementations)
    for s_, p_ in ((sa, pa), (sb, pb)):
        for c, par in enumerate(p_.tolist()):
            if par >= 0 and s_[c] != s_[par]:
                tc.assertLess(int(s_[par]), int(s_[c]), f"{msg}: shoot numbering")


class TestChainVectorized(unittest.TestCase):
    def _run_both(self, pos, **kw):
        return (chain_phytomers(pos, vectorized=True, **kw), chain_phytomers(pos, vectorized=False, **kw))

    def test_random_clouds_all_cue_combinations(self):
        torch.manual_seed(1)
        for trial in range(30):
            N = int(torch.randint(2, 40, (1,)))
            pos = torch.rand(N, 3) * 0.2
            rot = torch.nn.functional.normalize(torch.randn(N, 6), dim=-1)
            ordinal = torch.randint(0, 6, (N,)).float() + torch.rand(N) * 0.3
            is_base = (torch.rand(N) < 0.1).float()
            exist = (torch.rand(N) > 0.15).float()
            for kw in (dict(), dict(ordinal=ordinal), dict(ordinal=ordinal, is_base=is_base),
                       dict(rot6d=rot, ordinal=ordinal), dict(rot6d=rot), dict(ordinal=ordinal, root_own_shoot=True),
                       dict(rot6d=rot, ordinal=ordinal, root_own_shoot=True, exist=exist)):
                a, b = self._run_both(pos, **kw)
                _assert_same(self, a, b, f"trial {trial} {sorted(kw)}")

    @unittest.skipUnless(_XMLS, "no cowpea XML available")
    def test_ground_truth_plants(self):
        from plant_recon.models.plant_organ_array import PlantOrganArray, P_COL_ORGAN_TYPE, ORGAN_NONE
        from plant_recon.dataset.part_array_dataset import encode_fm, attach_parent_links
        from plant_recon.dataset.phytomer_packets import build_phytomer_packets
        from plant_recon.dataset.generate_cache import extract_phytomer_ids
        for path in _XMLS[:6]:
            arr = PlantOrganArray.from_xml_file(path); gt14 = arr.to_part_tensor().float()
            exist = (gt14[:, P_COL_ORGAN_TYPE] > ORGAN_NONE).float()
            packets, presence, centers, refs, keys = build_phytomer_packets(
                encode_fm(gt14), existence_mask=exist, phytomer_ids=extract_phytomer_ids(arr, gt14.shape[0]), return_keys=True)
            pkt = {"packets": packets, "presence": presence, "centers": centers, "refs": refs, "keys": keys}
            attach_parent_links(pkt)
            depth = pkt["depth"].float()
            for kw in (dict(ordinal=depth, is_base=(depth == 0).float(), root_own_shoot=True),
                       dict(rot6d=refs, ordinal=depth, is_base=(depth == 0).float(), root_own_shoot=True),
                       dict(ordinal=depth + torch.randn_like(depth) * 0.7)):
                a, b = self._run_both(centers, **kw)
                _assert_same(self, a, b, os.path.basename(path))


if __name__ == "__main__":
    unittest.main()
