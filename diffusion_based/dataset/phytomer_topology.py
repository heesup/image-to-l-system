"""
Recover the stem chain from predicted phytomer nodes.

The model predicts phytomers as an unordered set: nothing says which node sits
above which, or which branch a node belongs to. Two things need that structure:

1. The internode. A phytomer's node is the TOP of its internode, so the
   internode is the segment from the parent node up to this one. Its length is
   therefore the parent gap, not an independent quantity — measured on ground
   truth, the two agree to 0.000 cm over 352 parent/child pairs.

2. Helios XML export. `PartTensorTo40DConverter` splits shoots by scanning for
   `ORGAN_SHOOT_META` rows. Packets exclude meta rows by construction, so a
   reconstructed plant exports as a single shoot; stripping only those rows from
   otherwise-perfect ground truth drops DAP 50 foreground IoU from 94.1% to
   16.5%.

Both are served by the same chain.

The parent of node i is the node the internode of i points back to, so the cost
combines distance with agreement against that internode's own direction
(column 1 of its rotation, the tube forward axis used by the mesh builder).
Measured parent recovery on ground truth: 100% at DAP 10/50/90 with the
directional term, 40-52% on distance alone. Do not symmetrise the cost into an
MST — that measured worse (88-90%) even at zero noise.

The chain degrades with node position error (5-seed sweep, parent recovery):
0.5 cm -> 31/86/97%, 1.0 cm -> 13/58/70% for DAP 10/50/90. DAP 10 is the
fragile case because its nodes sit 0.47 cm apart. Predicted nodes will need a
learned ordinal to be reliable at that spacing.
"""

from typing import Optional, Tuple

import torch

from diffusion_based.dataset.phytomer_packets import rot6d_to_matrix

# Weight on the direction term, in metres per unit of (1 - cos). Large enough to
# break ties between near-equidistant candidates, small enough that it never
# overrides a clearly closer parent.
DIR_WEIGHT = 0.05
# A candidate parent further than this multiple of the median node spacing is
# treated as absent, which is what makes a node a shoot base.
MAX_EDGE_FACTOR = 3.0
# Weight on the predicted-ordinal term, in metres per step of ordinal error.
# Comparable to a typical internode so it can outvote a slightly closer but
# out-of-sequence candidate.
ORD_WEIGHT = 0.02


def chain_phytomers(
    pos: torch.Tensor,
    rot6d: torch.Tensor,
    exist: Optional[torch.Tensor] = None,
    dir_weight: float = DIR_WEIGHT,
    max_edge_factor: float = MAX_EDGE_FACTOR,
    ordinal: Optional[torch.Tensor] = None,
    ord_weight: float = ORD_WEIGHT,
    is_base: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Links phytomer nodes into shoots.

    Args:
        pos: (P, 3) node positions in metres.
        rot6d: (P, 6) node rotations; column 1 of the matrix is the internode
            forward axis, pointing from the parent up to this node.
        exist: optional (P,) bool/float mask of which nodes are real.
        ordinal: optional (P,) predicted position along the shoot (Stage 2's
            order head). A parent should be one step below its child, so this
            adds |(o_child - o_parent) - 1| to the cost. Geometry alone is not
            enough where nodes are densely spaced.
        is_base: optional (P,) bool/float, predicted shoot bases. Nodes flagged
            here are allowed to have no parent regardless of the edge gate.

    Returns:
        parent_idx: (P,) int64, index of each node's parent, -1 for shoot bases.
        shoot_id: (P,) int64, which shoot each node belongs to.
        phytomer_idx: (P,) int64, 0-based position along that shoot.
        Absent nodes get -1 in all three.
    """
    device = pos.device
    P = pos.shape[0]
    parent_idx = torch.full((P,), -1, dtype=torch.long, device=device)
    shoot_id = torch.full((P,), -1, dtype=torch.long, device=device)
    phytomer_idx = torch.full((P,), -1, dtype=torch.long, device=device)
    if P == 0:
        return parent_idx, shoot_id, phytomer_idx

    live = (torch.ones(P, dtype=torch.bool, device=device) if exist is None
            else (exist > 0.5).reshape(-1))
    idx = torch.nonzero(live, as_tuple=True)[0]
    if idx.numel() == 0:
        return parent_idx, shoot_id, phytomer_idx

    p = pos[idx]                                    # (N, 3)
    fwd = rot6d_to_matrix(rot6d[idx])[:, :, 1]      # (N, 3) internode forward
    N = p.shape[0]

    delta = p.unsqueeze(1) - p.unsqueeze(0)         # (N, N, 3) child - candidate
    dist = delta.norm(dim=-1)
    # A node's parent must sit below it: rank by height so the parent is always
    # earlier in the order. This is what guarantees a forest with no cycles.
    order = torch.argsort(p[:, 2])
    rank = torch.empty(N, dtype=torch.long, device=device)
    rank[order] = torch.arange(N, device=device)
    higher = rank.unsqueeze(1) <= rank.unsqueeze(0)  # candidate not below child

    cos = torch.nn.functional.cosine_similarity(
        delta, fwd.unsqueeze(1).expand_as(delta), dim=-1)
    cost = dist + dir_weight * (1.0 - cos)
    if ordinal is not None:
        o = ordinal.reshape(-1)[idx]
        step = o.unsqueeze(1) - o.unsqueeze(0)          # child - candidate
        cost = cost + ord_weight * (step - 1.0).abs()
    cost = cost.masked_fill(higher, float("inf"))
    cost.fill_diagonal_(float("inf"))

    best_cost, best = cost.min(dim=1)
    finite = dist[dist > 0]
    gate = (max_edge_factor * finite.median() if finite.numel() > 0
            else torch.tensor(float("inf"), device=device))
    has_parent = torch.isfinite(best_cost) & (best_cost <= gate)
    if is_base is not None:
        has_parent = has_parent & ~(is_base.reshape(-1)[idx] > 0.5)
    local_parent = torch.where(has_parent, best, torch.full_like(best, -1))

    # Walk each chain from its base so shoot ids and ordinals are consistent.
    children: list[list[int]] = [[] for _ in range(N)]
    for c in range(N):
        pa = int(local_parent[c])
        if pa >= 0:
            children[pa].append(c)

    local_shoot = torch.full((N,), -1, dtype=torch.long, device=device)
    local_ord = torch.full((N,), -1, dtype=torch.long, device=device)
    next_shoot = 0
    for base in torch.nonzero(local_parent < 0, as_tuple=True)[0].tolist():
        stack = [(base, next_shoot, 0)]
        next_shoot += 1
        while stack:
            node, sid, depth = stack.pop()
            local_shoot[node] = sid
            local_ord[node] = depth
            kids = children[node]
            if not kids:
                continue
            # One child continues this shoot; the rest start their own, which is
            # how a lateral branch becomes a separate shoot. With ordinals the
            # successor is simply the one a step further along; otherwise fall
            # back to the straightest child.
            if len(kids) == 1:
                cont = kids[0]
            elif ordinal is not None:
                o = ordinal.reshape(-1)[idx]
                want = o[node] + 1.0
                cont = kids[int(torch.stack([(o[k] - want).abs() for k in kids]).argmin())]
            else:
                ax = fwd[node]
                cos_k = torch.stack([
                    torch.nn.functional.cosine_similarity(
                        (p[k] - p[node]).unsqueeze(0), ax.unsqueeze(0), dim=-1)[0]
                    for k in kids])
                cont = kids[int(cos_k.argmax())]
            for k in kids:
                if k == cont:
                    stack.append((k, sid, depth + 1))
                else:
                    stack.append((k, next_shoot, 0))
                    next_shoot += 1

    parent_idx[idx] = torch.where(local_parent >= 0, idx[local_parent.clamp(min=0)],
                                  torch.full_like(local_parent, -1))
    shoot_id[idx] = local_shoot
    phytomer_idx[idx] = local_ord
    return parent_idx, shoot_id, phytomer_idx
