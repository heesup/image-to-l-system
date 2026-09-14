"""assemble_packets places leaflets and flowers on curved petioles/peduncles for all phytomers at once
(2026-09-14). This pins it to the per-phytomer loop it replaced: same bases, same gradients."""
import glob
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from diffusion_based.dataset import phytomer_packets as pp
from diffusion_based.dataset.phytomer_packets import (
    assemble_packets, rot6d_to_matrix, _petiole_curve_points, terminal_leaflet_is_slot2,
    FM_BASE_START, FM_BASE_END, FM_ROT_START, FM_ROT_END, FM_SCALE_START, FM_CURV, FM_OT_END,
    LEAFLET_ATTACH_FRAC, REPRO_ATTACH_FRAC, REPRO_TIP_ORGAN_TYPES, DETERMINISTIC_ORGAN_TYPES, NUM_SLOTS,
    SCALE_SCALE, CURV_SCALE,
)


def _place_loop(out, base, R_ref, base_scale):
    """The per-phytomer loop assemble_packets used until 2026-09-14, verbatim in effect."""
    P = out.shape[0]
    ot = out[:, :, :FM_OT_END].argmax(dim=-1)
    det = torch.zeros_like(ot, dtype=torch.bool)
    for t in DETERMINISTIC_ORGAN_TYPES:
        det |= ot == t
    term2 = terminal_leaflet_is_slot2(out[:, :, FM_SCALE_START], ot)
    base = base.clone()
    for p in range(P):
        if not bool(det[p, 1]):
            continue
        R_pet = R_ref[p] @ rot6d_to_matrix(out[p, 1, FM_ROT_START:FM_ROT_END])
        pet_len = out[p, 1, FM_SCALE_START] / SCALE_SCALE
        pet_curv = out[p, 1, FM_CURV] / CURV_SCALE
        if pet_len < 1e-4:
            continue
        curve = _petiole_curve_points(R_pet, pet_len, pet_curv)
        fracs = LEAFLET_ATTACH_FRAC
        if bool(term2[p]):
            fracs = {2: LEAFLET_ATTACH_FRAC[4], 3: LEAFLET_ATTACH_FRAC[3], 4: LEAFLET_ATTACH_FRAC[2]}
        for s, frac in fracs.items():
            if bool(det[p, s]):
                idx_f = frac * (len(curve) - 1); i0 = int(idx_f); t = idx_f - i0; i1 = min(i0 + 1, len(curve) - 1)
                base[p, s] = (curve[i0] * (1 - t) + curve[i1] * t) * base_scale
        if bool(det[p, 5]):
            R_ped = R_ref[p] @ rot6d_to_matrix(out[p, 5, FM_ROT_START:FM_ROT_END])
            ped_len = out[p, 5, FM_SCALE_START] / SCALE_SCALE
            ped_curv = out[p, 5, FM_CURV] / CURV_SCALE
            if ped_len >= 1e-4:
                pcurve = _petiole_curve_points(R_ped, ped_len, ped_curv)
                idx_f = REPRO_ATTACH_FRAC * (len(pcurve) - 1); i0 = int(idx_f); t = idx_f - i0; i1 = min(i0 + 1, len(pcurve) - 1)
                cpt = pcurve[i0] * (1 - t) + pcurve[i1] * t
                for s in range(6, NUM_SLOTS):
                    if bool(det[p, s]) and int(ot[p, s]) in REPRO_TIP_ORGAN_TYPES:
                        base[p, s] = cpt * base_scale
    return base


def _reference(packets, refs, **kw):
    """assemble_packets with the batched placement swapped for the loop."""
    saved = (pp._curve_points_batched, pp._point_on_curve)
    out = assemble_packets(packets, refs, **kw)   # stem/scale rules identical; only the placement differs
    # recompute the placement with the loop on the same pre-placement state
    pre = out.clone()
    R_ref = rot6d_to_matrix(refs)
    ot = pre[:, :, :FM_OT_END].argmax(dim=-1)
    det = torch.zeros_like(ot, dtype=torch.bool)
    for t in DETERMINISTIC_ORGAN_TYPES:
        det |= ot == t
    base0 = pre[:, :, FM_BASE_START:FM_BASE_END].clone()
    # slots 2-4 and 6-9 are what the placement writes; restore them to the pre-placement value (zero for det slots)
    base0[:, 2:5] = torch.where(det[:, 2:5].unsqueeze(-1), torch.zeros_like(base0[:, 2:5]), base0[:, 2:5])
    base0[:, 6:] = torch.where(det[:, 6:].unsqueeze(-1), torch.zeros_like(base0[:, 6:]), base0[:, 6:])
    base_loop = _place_loop(pre, base0, R_ref, 20.0)
    return torch.cat([pre[..., :FM_BASE_START], base_loop, pre[..., FM_BASE_END:]], dim=-1)


_XMLS = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "dataset", "helios_data", "cowpea",
                                      "cowpea_dap0[4-8]0_seed00_*_0000_plant_0000.xml")))


class TestAssembleBatched(unittest.TestCase):
    def _real_packets(self, path, terminal_last):
        from diffusion_based.models.plant_organ_array import PlantOrganArray, P_COL_ORGAN_TYPE, ORGAN_NONE
        from diffusion_based.dataset.part_array_dataset import encode_fm
        from diffusion_based.dataset.generate_cache import extract_phytomer_ids
        arr = PlantOrganArray.from_xml_file(path); gt14 = arr.to_part_tensor().float()
        exist = (gt14[:, P_COL_ORGAN_TYPE] > ORGAN_NONE).float()
        packets, presence, centers, refs, keys = pp.build_phytomer_packets(
            encode_fm(gt14), existence_mask=exist, phytomer_ids=extract_phytomer_ids(arr, gt14.shape[0]),
            return_keys=True, terminal_leaflet_last=terminal_last)
        s_a = pp.phytomer_scale(packets)
        den = pp.denormalize_packet_scales(pp.strip_base(pp.normalize_packet_scales(packets, s_a)), s_a)
        return den, refs, centers

    @unittest.skipUnless(_XMLS, "no cowpea XML available")
    def test_matches_loop_on_real_plants_both_leaflet_orders(self):
        for path in _XMLS[:3]:
            for tl in (False, True):
                den, refs, centers = self._real_packets(path, tl)
                parent = centers + torch.randn_like(centers) * 0.01
                got = assemble_packets(den, refs, parent_pos=parent, centers=centers)
                ref = _reference(den, refs, parent_pos=parent, centers=centers)
                self.assertTrue(torch.allclose(got, ref, atol=1e-5), f"{os.path.basename(path)} tl={tl}: max diff {(got-ref).abs().max()}")

    def test_matches_loop_on_random_packets_with_gradients(self):
        torch.manual_seed(0)
        P = 64
        pk = torch.randn(P, NUM_SLOTS, 26)
        # organ types: slot0 internode(3) slot1 petiole(4) slots2-4 leaf(5) slot5 peduncle(6) slots6-9 mixed repro/bud
        types = torch.tensor([3, 4, 5, 5, 5, 6, 9, 10, 8, 11])
        pk[:, :, :FM_OT_END] = -5.0
        for s_, t_ in enumerate(types.tolist()):
            pk[:, s_, t_] = 5.0
        pk[:, 1, FM_SCALE_START] = torch.rand(P) * 3 + 0.5     # petiole lengths (FM units)
        pk[:, 5, FM_SCALE_START] = torch.rand(P) * 3 + 0.5
        pk[:8, 1, FM_SCALE_START] = 0.0                        # a few too-short petioles: loop skips them
        pk[:, 2:5, FM_SCALE_START] = torch.rand(P, 3) * 2 + 0.5
        pk[:, [1, 5], FM_CURV] = torch.randn(P, 2) * 0.5
        refs = torch.nn.functional.normalize(torch.randn(P, 6), dim=-1)
        pk_a = pk.clone().requires_grad_(True); pk_b = pk.clone().requires_grad_(True)
        got = assemble_packets(pk_a, refs); ref = _reference(pk_b, refs)
        # One intended difference: the loop `continue`d past a phytomer whose petiole was
        # missing or too short, so its flowers were never placed on the peduncle; the
        # batched version places them regardless. Compare those rows on slots 0-5 only.
        short = pk[:, 1, FM_SCALE_START] / SCALE_SCALE < 1e-4
        same = torch.ones_like(got, dtype=torch.bool); same[short, 6:] = False
        self.assertTrue(torch.allclose(got[same], ref[same], atol=1e-5), f"max diff {(got-ref)[same].abs().max()}")
        self.assertTrue(bool(short.any()) and not torch.allclose(got[short, 6:], ref[short, 6:]),
                        "the short-petiole rows are where the batched placement intentionally differs")
        w = torch.randn_like(got); w[short, 6:] = 0.0
        (got * w).sum().backward(); (ref * w).sum().backward()
        self.assertTrue(torch.allclose(pk_a.grad, pk_b.grad, atol=1e-4), f"grad max diff {(pk_a.grad-pk_b.grad).abs().max()}")


if __name__ == "__main__":
    unittest.main()
