"""Diagnostic: compare leaflet slot-identity orderings by within-slot rotation variance.

Lower within-slot angular spread => more consistent slot identity => better
canonical ordering for VAE training. Compares:
  A. base z, then absolute azimuth (current builder)
  B. tip distance from cluster center, descending (terminal-like first)
  C. tip height (z), descending
  D. scale norm, descending (largest first)
"""
import glob
import math
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from diffusion_based.dataset.phytomer_packets import cluster_organs


def Rmat(r6):
    x = F.normalize(r6[..., 0:3], dim=-1)
    z = F.normalize(torch.cross(x, r6[..., 3:6], dim=-1), dim=-1)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)


def ang(a, b):
    c = float((((a * b).sum() - 1.0) / 2.0).clamp(-1, 1))
    return math.degrees(math.acos(c))


def collect(cache_dir, n_files, seed):
    rng = random.Random(seed)
    files = rng.sample(sorted(glob.glob(os.path.join(cache_dir, "*.pt"))), n_files)
    out = []  # (rots(3,6), base(3,3), tips(3,3), center(3), scales(3,))
    for f in files:
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:
            continue
        nodes = d["nodes"]
        em = d.get("existence_mask")
        ot = nodes[:, :13].argmax(-1)
        if em is not None and em.shape[0] != nodes.shape[0]:
            active = torch.zeros(nodes.shape[0], dtype=torch.bool)
            n = min(em.shape[0], nodes.shape[0])
            active[:n] = em[:n] > 0.5
            active = active & (ot != 0)
        else:
            active = (ot != 0) & (nodes[:, 0] < 0.5)
        if active.sum() == 0:
            continue
        ai = torch.nonzero(active, as_tuple=True)[0]
        pos = nodes[ai][:, 13:16] / 20.0
        lab = ot[ai]
        scl = nodes[ai][:, 22:25] / 50.0
        centers, assign = cluster_organs(pos, lab)
        for c in range(centers.shape[0]):
            idx = (assign == c).nonzero().squeeze(-1)
            li = [i for i in idx.tolist() if int(lab[i]) == 5]
            if len(li) < 3:
                continue
            li = li[:3]
            rots = nodes[ai][li][:, 16:22]
            base = pos[li]
            tips = base + torch.einsum("nij,nj->ni", Rmat(rots),
                                       torch.tensor([0.08, 0, 0]).expand(3, -1))
            out.append((rots, base, tips, centers[c], scl[li].norm(dim=-1)))
    return out


def order_base_z(base, tips, center, rots, scales):
    key = [(base[i][2].item(), math.atan2(base[i][1].item(), base[i][0].item()), i)
           for i in range(3)]
    return [x[2] for x in sorted(key)]


def order_tip_dist(base, tips, center, rots, scales):
    key = [(-float(((tips[i] - center) ** 2).sum() ** 0.5), i) for i in range(3)]
    return [x[1] for x in sorted(key)]


def order_tip_height(base, tips, center, rots, scales):
    key = [(-tips[i][2].item(), i) for i in range(3)]
    return [x[1] for x in sorted(key)]


def order_scale(base, tips, center, rots, scales):
    key = [(-scales[i].item(), i) for i in range(3)]
    return [x[1] for x in sorted(key)]


def slot_spread(triplets, order_fn, n_sample=400):
    rng = random.Random(0)
    sample = rng.sample(triplets, min(n_sample, len(triplets)))
    ordered = []
    for rots, base, tips, center, scales in sample:
        perm = order_fn(base, tips, center, rots, scales)
        ordered.append(Rmat(rots[perm]))
    tot = [0.0, 0.0, 0.0]
    cnt = 0
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            for s in range(3):
                tot[s] += ang(ordered[i][s], ordered[j][s])
            cnt += 1
    return [t / max(cnt, 1) for t in tot]


def main():
    triplets = collect("dataset/cache/cowpea_curv26", 60, seed=0)
    print(f"leaflet triplets: {len(triplets)}")
    for name, fn in [("A base-z+azim ", order_base_z),
                     ("B tip-dist   ", order_tip_dist),
                     ("C tip-height ", order_tip_height),
                     ("D scale-desc ", order_scale)]:
        s = slot_spread(triplets, fn)
        print(f"{name}: slot mean-pairwise-ang = "
              f"[{s[0]:5.1f}, {s[1]:5.1f}, {s[2]:5.1f}] deg (mean {sum(s)/3:.1f})")


if __name__ == "__main__":
    main()
