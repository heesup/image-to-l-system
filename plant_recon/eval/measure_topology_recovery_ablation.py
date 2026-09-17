import glob, random, torch
from plant_recon.models.plant_organ_array import PlantOrganArray, P_COL_ORGAN_TYPE, ORGAN_NONE
from plant_recon.dataset.part_array_dataset import encode_fm
from plant_recon.dataset.generate_cache import extract_phytomer_ids
from plant_recon.dataset.phytomer_packets import build_phytomer_packets
from plant_recon.dataset.phytomer_topology import chain_phytomers

def true_parent(keys):
    """GT parent index per packet: same shoot, ordinal-1. -1 if none (shoot base)."""
    P = keys.shape[0]
    parent = torch.full((P,), -1, dtype=torch.long)
    for i in range(P):
        sid, idx = int(keys[i,0]), int(keys[i,1])
        if idx == 0:
            continue
        for j in range(P):
            if int(keys[j,0]) == sid and int(keys[j,1]) == idx - 1:
                parent[i] = j
                break
    return parent

def recovery_acc(pred_parent, gt_parent):
    has_gt = gt_parent >= 0
    if int(has_gt.sum()) == 0:
        return None
    correct = (pred_parent[has_gt] == gt_parent[has_gt]).float().mean().item() * 100
    return correct, int(has_gt.sum())

def density_weight(ctr):
    """Proxy for 'how ambiguous is this node's assignment' = inverse of
    distance to nearest neighbor (denser local packing -> lower confidence)."""
    d = torch.cdist(ctr, ctr)
    d.fill_diagonal_(float('inf'))
    nn_dist = d.min(dim=1).values.clamp(min=1e-6)
    med = nn_dist.median()
    return (med / nn_dist).clamp(max=5.0)  # >1 in dense regions, <1 in sparse

configs = {
    "A: dist+dir+ord (current default)":      dict(dir_weight=0.05, ord_weight=0.02, use_ord=True),
    "B: dist+ord only (no direction)":         dict(dir_weight=0.0,  ord_weight=0.02, use_ord=True),
    "C: dist+dir only (no ordinal)":           dict(dir_weight=0.05, ord_weight=0.0,  use_ord=False),
    "D: dist only (nothing)":                  dict(dir_weight=0.0,  ord_weight=0.0,  use_ord=False),
}

random.seed(0)
results = {dap: {name: [] for name in configs} for dap in (10, 50, 90)}
density_results = {dap: [] for dap in (10, 50, 90)}

for dap in (10, 50, 90):
    files = sorted(glob.glob(f"dataset/helios_data/cowpea/*dap{dap:03d}*_plant_*.xml"))
    picks = random.sample(files, min(8, len(files)))
    for f in picks:
        arr = PlantOrganArray.from_xml_file(f)
        gt = arr.to_part_tensor()
        nodes = encode_fm(gt)
        ex = (gt[:, P_COL_ORGAN_TYPE] > ORGAN_NONE).float()
        ids = extract_phytomer_ids(arr, gt.shape[0])
        pk, pres, ctr, refs, keys = build_phytomer_packets(nodes, existence_mask=ex, phytomer_ids=ids, return_keys=True)
        if pk.shape[0] < 3:
            continue
        gtp = true_parent(keys)
        ordinal = keys[:, 1].float()
        is_base = (keys[:, 1] == 0).float()

        for name, cfg in configs.items():
            ord_arg = ordinal if cfg["use_ord"] else None
            parent, shoot, phyidx = chain_phytomers(
                ctr, refs, ordinal=ord_arg, is_base=is_base,
                dir_weight=cfg["dir_weight"], ord_weight=cfg["ord_weight"])
            r = recovery_acc(parent, gtp)
            if r is not None:
                results[dap][name].append(r[0])

        # Density-weighted direction cost (proxy for uncertainty-aware weighting):
        # scale dir_weight per-node isn't supported by chain_phytomers directly,
        # so approximate by comparing against config B/C's FIXED weights using
        # the median density-adjusted dir_weight as a single global value tuned
        # from the local density distribution (cheap proxy, not a real per-node model).
        w = density_weight(ctr)
        adaptive_dw = float((0.05 / w.clamp(min=0.2)).median())
        parent_adapt, _, _ = chain_phytomers(ctr, refs, ordinal=ordinal, is_base=is_base,
                                              dir_weight=adaptive_dw, ord_weight=0.02)
        r2 = recovery_acc(parent_adapt, gtp)
        if r2 is not None:
            density_results[dap].append(r2[0])

print("="*90)
print("PARENT-RECOVERY ACCURACY (chain_phytomers on GT, exact positions/rotations)")
print("="*90)
print(f"{'DAP':>4} | " + " | ".join(f"{n[:20]:>20}" for n in configs))
for dap in (10, 50, 90):
    row = []
    for name in configs:
        vals = results[dap][name]
        row.append(f"{sum(vals)/len(vals):19.1f}%" if vals else "n/a".rjust(20))
    print(f"{dap:>4} | " + " | ".join(row))

print()
print("Density-adjusted dir_weight variant (proxy for uncertainty-weighted cost):")
for dap in (10, 50, 90):
    vals = density_results[dap]
    if vals:
        print(f"  DAP {dap}: {sum(vals)/len(vals):.1f}%  (n={len(vals)} plants)")
