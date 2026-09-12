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
can combine distance with agreement against that internode's own direction
(column 1 of its rotation, the tube forward axis used by the mesh builder) when
a rotation is available. Measured parent recovery on ground truth: 100% at DAP
10/50/90 with the directional term, 40-52% on distance alone (no rotation, no
ordinal). Do not symmetrise the cost into an MST — that measured worse (88-90%)
even at zero noise.

The chain degrades with node position error (5-seed sweep, parent recovery):
0.5 cm -> 31/86/97%, 1.0 cm -> 13/58/70% for DAP 10/50/90. DAP 10 is the
fragile case because its nodes sit 0.47 cm apart.

2026-09-11: rotation is now OPTIONAL here (`rot6d=None` drops the directional
term entirely). Measured on ground truth (8 plants/DAP, GT is_base gate on all
legs, see diffusion_based/eval/measure_topology_recovery_ablation.py):
distance+ordinal alone reaches 97.5/94.5/97.7% at DAP 10/50/90, within 2.5
points of the full distance+direction+ordinal combination (100.0/95.4/98.9%).
This is what makes it possible for a node's rotation to be DERIVED from the
resolved chain (this node's position minus its parent's) instead of predicted
independently -- chaining no longer needs rotation as an input to produce it
as an output. See phytomer_roll.py for that derivation.
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


def gt_parent_links(
    centers: torch.Tensor,
    keys: torch.Tensor,
    max_edge_factor: float = MAX_EDGE_FACTOR,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ground-truth parent of every phytomer, under one rule with no exceptions.

    **A node's parent is one internode below it.** A node is the TOP of its
    internode, so:

    - the previous node in the same shoot, where there is one;
    - the **origin** for the main stem's first node, which sits one internode
      above it (measured: 3.0 cm from the origin at DAP 90 against a 2.9 cm
      median internode). This is a real geometric parent, not a fallback guess;
    - the **branch-point node on the parent shoot** for a lateral shoot's first
      node, resolved as the nearest node below it on a different shoot
      (measured 3.1 cm at DAP 90 -- also one internode).

    This replaces the previous convention, under which every shoot's first node
    was a parentless "base". That left 13.8% of phytomers with no parent, and
    since a lateral's first node is rarely vertical, the world-+Z axis they fell
    back on was 50.8 deg off on average. Here every node has a parent, so the
    fallback has nothing left to cover.

    `keys` carries only (shoot_id, phytomer_idx) and does not record which node a
    lateral branches from, hence the geometric lookup. It is gated by the same
    `max_edge_factor` x median-spacing rule `chain_phytomers` uses, so a lateral
    in a sparse region is left unresolved rather than linked to something absurd.

    Args:
        centers: (P, 3) ground-truth node positions in metres.
        keys: (P, 2) int64 (shoot_id, phytomer_idx), from the packet cache.
        max_edge_factor: distance gate for the branch-point lookup.

    Returns:
        parent_pos: (P, 3) parent position. The origin for the main stem's first
            node; `centers[i]` itself for any row left unresolved, so callers
            that subtract get a zero vector rather than garbage.
        parent_idx: (P,) int64 index of the parent row; **-1 where the parent is
            the origin or could not be resolved** -- use `parent_pos` for
            geometry and this only when the parent must be another predicted row.
    """
    device = centers.device
    P = centers.shape[0]
    parent_idx = torch.full((P,), -1, dtype=torch.long, device=device)
    parent_pos = centers.clone()
    if P == 0:
        return parent_pos, parent_idx

    shoot, ordi = keys[:, 0], keys[:, 1]

    # Same shoot, one step down. (P, P) is fine at P <= ~600.
    want = torch.stack([shoot, ordi - 1], dim=-1)
    hit = (keys.unsqueeze(0) == want.unsqueeze(1)).all(dim=-1)      # (child, cand)
    has_same_shoot = hit.any(dim=-1)
    same_row = hit.float().argmax(dim=-1)
    parent_idx = torch.where(has_same_shoot, same_row, parent_idx)
    parent_pos = torch.where(has_same_shoot.unsqueeze(-1), centers[same_row], parent_pos)

    # The main stem is the shoot whose first node sits lowest; its first node's
    # parent is the origin.
    firsts = torch.nonzero(ordi == 0, as_tuple=True)[0]
    if firsts.numel() == 0:
        return parent_pos, parent_idx
    root_shoot = shoot[firsts[centers[firsts, 2].argmin()]]

    dist = torch.cdist(centers, centers)
    finite = dist[dist > 0]
    gate = (max_edge_factor * finite.median() if finite.numel() > 0
            else torch.tensor(float("inf"), device=device))

    for i in firsts.tolist():
        if shoot[i] == root_shoot:
            parent_pos[i] = 0.0                 # the origin
            continue
        # Nearest node below, on another shoot: the branch point.
        cand = (centers[:, 2] < centers[i, 2]) & (shoot != shoot[i])
        if not bool(cand.any()):
            continue
        d = dist[i].masked_fill(~cand, float("inf"))
        j = int(d.argmin())
        if d[j] <= gate:
            parent_idx[i] = j
            parent_pos[i] = centers[j]

    return parent_pos, parent_idx


def chain_phytomers(
    pos: torch.Tensor,
    rot6d: Optional[torch.Tensor] = None,
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
        rot6d: optional (P, 6) node rotations; column 1 of the matrix is the
            internode forward axis, pointing from the parent up to this node.
            None drops the directional cost term entirely (see module
            docstring for the measured cost of doing so: <=2.5 points of
            parent-recovery accuracy once `ordinal` is supplied) -- this is
            the normal calling convention once rotation is itself DERIVED
            from this function's own output rather than predicted
            independently (phytomer_roll.py), since deriving it requires
            knowing the chain first.
        exist: optional (P,) bool/float mask of which nodes are real.
        ordinal: optional (P,) predicted position along the shoot (Stage 2's
            order head). A parent should be one step below its child, so this
            adds |(o_child - o_parent) - 1| to the cost. Geometry alone is not
            enough where nodes are densely spaced, and substitutes for the
            directional term almost as well as rotation did (measured).
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
    use_dir = rot6d is not None and dir_weight != 0.0
    fwd = rot6d_to_matrix(rot6d[idx])[:, :, 1] if rot6d is not None else None
    N = p.shape[0]

    delta = p.unsqueeze(1) - p.unsqueeze(0)         # (N, N, 3) child - candidate
    dist = delta.norm(dim=-1)
    # A node's parent must sit below it: rank by height so the parent is always
    # earlier in the order. This is what guarantees a forest with no cycles.
    order = torch.argsort(p[:, 2])
    rank = torch.empty(N, dtype=torch.long, device=device)
    rank[order] = torch.arange(N, device=device)
    higher = rank.unsqueeze(1) <= rank.unsqueeze(0)  # candidate not below child

    cost = dist
    if use_dir:
        cos = torch.nn.functional.cosine_similarity(
            delta, fwd.unsqueeze(1).expand_as(delta), dim=-1)
        cost = cost + dir_weight * (1.0 - cos)
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
            elif fwd is not None:
                ax = fwd[node]
                cos_k = torch.stack([
                    torch.nn.functional.cosine_similarity(
                        (p[k] - p[node]).unsqueeze(0), ax.unsqueeze(0), dim=-1)[0]
                    for k in kids])
                cont = kids[int(cos_k.argmax())]
            else:
                # Neither cue available: continue with the nearest child.
                # Only reachable with rot6d=None AND ordinal=None -- an
                # unusual call; distance alone (§ module docstring) still
                # recovers most parents, just not which child extends the
                # shoot at a branch point as reliably.
                cont = kids[int(dist[node][torch.tensor(kids)].argmin())]
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
