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
# gt_parent_links gates its branch-point lookup on the INTERNODE scale, not on
# chain_phytomers' median-of-all-pairwise-distances (which grows with the plant:
# 10.7 cm median at DAP 90, so a 3x gate is 32 cm and rejects almost nothing).
# Measured over 24 plants, a true child-to-parent distance is 1.00x the median
# internode at the median, 2.78x at p99 and 5.32x at the maximum, so 6.0 accepts
# every observed true parent while still rejecting anything absurd.
MAX_INTERNODE_FACTOR = 6.0


def gt_parent_links(
    centers: torch.Tensor,
    keys: torch.Tensor,
    max_internode_factor: float = MAX_INTERNODE_FACTOR,
    internode_base: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    lateral branches from, hence the geometric lookup. It is gated on this
    plant's own median internode (see `MAX_INTERNODE_FACTOR`), so a lateral in a
    sparse region is left unresolved rather than linked to something absurd.

    Args:
        centers: (P, 3) ground-truth node positions in metres.
        keys: (P, 2) int64 (shoot_id, phytomer_idx), from the packet cache.
        max_internode_factor: branch-point lookup gate, in multiples of this
            plant's own median internode (see MAX_INTERNODE_FACTOR).
        internode_base: optional (P, 3) world position of each node's internode
            BASE (packet slot 0's base column; the cache stores absolute
            packets). A lateral's first internode starts AT its branch point
            (0.2-0.9 cm off, against 1-3 cm to any other node), so with it the
            branch point is the node nearest that base on another shoot,
            wherever it sits. Without it the lookup falls back to the nearest
            node BELOW the first node, which is wrong for a third of the
            laterals on dataset plants (66.2% right over 272 laterals, 30
            plants: cowpea laterals droop, so the branch point is often level
            with or above the lateral's first node).

    Returns:
        parent_pos: (P, 3) parent position. The origin for the main stem's first
            node; `centers[i]` itself for any row left unresolved, so callers
            that subtract get a zero vector rather than garbage.
        parent_idx: (P,) int64 index of the parent row; **-1 where the parent is
            the origin or could not be resolved** -- use `parent_pos` for
            geometry and this only when the parent must be another predicted row.
        depth: (P,) int64 number of internodes from the plant root along the
            parent chain (root = 0). **This, not the per-shoot ordinal in
            `keys`, is what the ordinal head should predict.** chain_phytomers
            scores a candidate parent by |(o_child - o_parent) - 1|; with
            per-shoot ordinals a lateral's first node (ordinal 0) has a true
            parent at, say, ordinal 5 on the main stem, and that term charges
            it 6 x ORD_WEIGHT = 0.12 m -- more than any internode -- so the
            true branch parent is actively rejected. With depth every parent
            is exactly depth - 1, laterals included, and the same rule serves
            the chain cost, the step loss and is_base (depth == 0).
    """
    device = centers.device
    P = centers.shape[0]
    parent_idx = torch.full((P,), -1, dtype=torch.long, device=device)
    parent_pos = centers.clone()
    if P == 0:
        return parent_pos, parent_idx, torch.zeros(0, dtype=torch.long, device=device)

    shoot, ordi = keys[:, 0], keys[:, 1]

    # Same shoot, one step down. (P, P) is fine at P <= ~600.
    want = torch.stack([shoot, ordi - 1], dim=-1)
    hit = (keys.unsqueeze(0) == want.unsqueeze(1)).all(dim=-1)      # (child, cand)
    has_same_shoot = hit.any(dim=-1)
    same_row = hit.float().argmax(dim=-1)
    parent_idx = torch.where(has_same_shoot, same_row, parent_idx)
    parent_pos = torch.where(has_same_shoot.unsqueeze(-1), centers[same_row], parent_pos)

    # The internode scale comes free from the same-shoot links just resolved:
    # each of those IS one internode.
    seg = (centers - centers[same_row])[has_same_shoot].norm(dim=-1)
    internode = seg.median() if seg.numel() > 0 else centers.new_tensor(float("inf"))
    gate = max_internode_factor * internode

    is_first = ordi == 0
    first_rows = torch.nonzero(is_first, as_tuple=True)[0]
    if first_rows.numel() == 0:
        return parent_pos, parent_idx, _depth_from_root(parent_idx)

    # The main stem is the shoot whose first node sits lowest. Derived from
    # geometry rather than assuming shoot_id 0, so a relabelled cache still works.
    root_shoot = shoot[first_rows[centers[first_rows, 2].argmin()]]
    is_root = is_first & (shoot == root_shoot)

    # Branch point for every lateral's first node: nearest node below it on a
    # different shoot. Fully vectorised -- the earlier per-shoot Python loop cost
    # 4 ms per call from its .item() syncs, which at 48 samples per step was +64%
    # of step time.
    if internode_base is not None:
        dist = torch.cdist(internode_base.to(centers.dtype), centers)
        cand = shoot.unsqueeze(1) != shoot.unsqueeze(0)
    else:
        dist = torch.cdist(centers, centers)
        below = centers[:, 2].unsqueeze(1) > centers[:, 2].unsqueeze(0)
        cand = below & (shoot.unsqueeze(1) != shoot.unsqueeze(0))
    d_lat = dist.masked_fill(~cand, float("inf"))
    best = d_lat.argmin(dim=1)
    best_d = d_lat.gather(1, best.unsqueeze(1)).squeeze(1)

    use_branch = is_first & ~is_root & (best_d <= gate)
    parent_idx = torch.where(use_branch, best, parent_idx)
    parent_pos = torch.where(use_branch.unsqueeze(-1), centers[best], parent_pos)

    # The root's parent is the origin: a real geometric parent one internode
    # below it, not a row of `centers`.
    parent_pos = torch.where(is_root.unsqueeze(-1),
                             torch.zeros_like(parent_pos), parent_pos)
    parent_idx = torch.where(is_root, torch.full_like(parent_idx, -1), parent_idx)

    return parent_pos, parent_idx, _depth_from_root(parent_idx)


def _depth_from_root(parent_idx: torch.Tensor) -> torch.Tensor:
    """Edges from each node up to its root, by pointer jumping: O(log P)
    rounds of pure gathers, no host syncs (a per-level loop would cost one
    sync per level, ~50-100 on a mature plant, x48 samples per step)."""
    P = parent_idx.shape[0]
    ar = torch.arange(P, device=parent_idx.device)
    has_par = parent_idx >= 0
    jump = torch.where(has_par, parent_idx, ar)
    depth = has_par.long()
    for _ in range(max(1, P.bit_length())):
        depth = depth + depth[jump]
        jump = jump[jump]
    return depth


def _resolve_forest(
    local_parent: torch.Tensor,
    best_cost: torch.Tensor,
    p: torch.Tensor,
    dist: torch.Tensor,
    fwd: Optional[torch.Tensor],
    o: Optional[torch.Tensor],
    root_own_shoot: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cycle breaking + shoot walk of chain_phytomers as tensor ops (no per-node
    Python). Returns (parent, shoot_id, phytomer_idx) over the N live nodes.

    Every node points at its cheapest candidate, so the pointer graph is a
    forest plus, possibly, cycles (the top two nodes of a shoot are each
    other's nearest neighbour). Each cycle is cut at its most expensive edge
    (ties keep the edge whose parent is lower, where the shoot base is).
    Then one child per node continues its shoot -- the one a step further
    along the ordinal, the straightest at a tie (or nearest without a
    direction), never a root's child when root_own_shoot -- and every other
    child starts a new shoot. Pointer jumping (log N rounds of gathers) finds
    cycles, cycle labels, chain starts and positions along the chain.
    """
    device = local_parent.device
    N = local_parent.shape[0]
    ar = torch.arange(N, device=device)
    rounds = max(1, N.bit_length())

    # ---- cycles ---------------------------------------------------------
    par = local_parent.clone()
    cut_cost = best_cost.double() + 1e-6 * (p[par.clamp(min=0), 2] - p[:, 2]).double()   # float64: see the loop path
    reach_root = par < 0
    jump = torch.where(par >= 0, par, ar)
    for _ in range(rounds):
        reach_root = reach_root | reach_root[jump]
        jump = jump[jump]
    in_cycle_or_tail = ~reach_root
    if bool(in_cycle_or_tail.any()):
        far = jump[in_cycle_or_tail]                     # lands on a cycle after >= N steps
        on_cycle = torch.zeros(N, dtype=torch.bool, device=device)
        on_cycle[far] = True                             # the images cover every cycle exactly
        # label each cycle by its lowest index (min over ancestors, doubling)
        lab = torch.where(on_cycle, ar, torch.full_like(ar, N))
        jp = torch.where(on_cycle & (par >= 0), par, ar)
        for _ in range(rounds):
            lab = torch.minimum(lab, lab[jp])
            jp = jp[jp]
        cyc = torch.nonzero(on_cycle, as_tuple=True)[0]
        grp = lab[cyc]
        gmax = torch.full((N,), float("-inf"), device=device, dtype=torch.float64).scatter_reduce(0, grp, cut_cost[cyc], reduce="amax")
        is_max = cut_cost[cyc] == gmax[grp]
        pick = torch.full((N,), N, dtype=torch.long, device=device).scatter_reduce(
            0, grp[is_max], cyc[is_max], reduce="amin")       # lowest index among the ties
        worst = pick[pick < N]
        par[worst] = -1

    # ---- continuation child per node ------------------------------------
    is_child = par >= 0
    c = torch.nonzero(is_child, as_tuple=True)[0]
    pp = par[c]
    neg_inf = torch.full((N,), float("-inf"), device=device)
    if o is not None:
        gap = (o[c] - (o[pp] + 1.0)).abs()
        gmin = torch.full((N,), float("inf"), device=device).scatter_reduce(0, pp, gap, reduce="amin")
        tied = gap <= gmin[pp] + 0.5
        if fwd is not None:
            score = torch.nn.functional.cosine_similarity(p[c] - p[pp], fwd[pp], dim=-1)
        else:
            score = -dist[pp, c]
        score = torch.where(tied, score, torch.full_like(score, float("-inf")))
    elif fwd is not None:
        score = torch.nn.functional.cosine_similarity(p[c] - p[pp], fwd[pp], dim=-1)
    else:
        score = -dist[pp, c]
    smax = neg_inf.scatter_reduce(0, pp, score, reduce="amax")
    is_best = score == smax[pp]
    pick = torch.full((N,), N, dtype=torch.long, device=device).scatter_reduce(
        0, pp[is_best], c[is_best], reduce="amin")            # first child among the ties
    cont = torch.zeros(N, dtype=torch.bool, device=device)
    cont[pick[pick < N]] = True
    if root_own_shoot:
        cont = cont & (par[par.clamp(min=0)] >= 0) & is_child   # a root's children all start shoots
    cont = cont & is_child

    # ---- chains: start, position along the chain, shoot numbering -------
    cont_parent = torch.where(cont, par, torch.full_like(par, -1))
    local_ord = _depth_from_root(cont_parent)
    depth = _depth_from_root(par)
    start = cont_parent < 0
    start_idx = torch.nonzero(start, as_tuple=True)[0]
    order = torch.argsort(depth[start_idx] * N + start_idx)  # parents' shoots before their children's
    sid_of_start = torch.full((N,), -1, dtype=torch.long, device=device)
    sid_of_start[start_idx[order]] = torch.arange(start_idx.numel(), device=device)
    jump = torch.where(cont_parent >= 0, cont_parent, ar)
    for _ in range(rounds):
        jump = jump[jump]
    local_shoot = sid_of_start[jump]
    return par, local_shoot, local_ord


def chain_phytomers(
    pos: torch.Tensor,
    rot6d: Optional[torch.Tensor] = None,
    exist: Optional[torch.Tensor] = None,
    dir_weight: float = DIR_WEIGHT,
    max_edge_factor: float = MAX_EDGE_FACTOR,
    ordinal: Optional[torch.Tensor] = None,
    ord_weight: float = ORD_WEIGHT,
    is_base: Optional[torch.Tensor] = None,
    root_own_shoot: bool = False,
    vectorized: bool = True,
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
        vectorized: cycle breaking and the shoot walk as pointer-jumping tensor
            ops (default) instead of the Python loops they replaced on
            2026-09-14; the loops are kept for `tests/test_chain_vectorized.py`.
            Same parents and phytomer indices, same partition into shoots;
            shoots are numbered by (depth of their first node, index), which
            keeps a parent shoot before its children like the walk did.
        root_own_shoot: every child of a parentless node starts a new shoot,
            so the root node is a shoot of its own. Cowpea's first phytomer is
            the cotyledon node, alone on the unifoliate shoot 0, and both the
            emitter (cotyledon twin petiole) and the XML converter key on
            shoot 0 / index 0 -- under the depth-from-root ordinal the root
            has a parent-less single child that would otherwise continue
            shoot 0 through the whole main stem (measured: a DAP 15 export
            fell from 95.8% to 57.4% IoU). Off by default because the older
            per-shoot convention flags every shoot's first node as a base.

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

    cost = dist
    if use_dir:
        cos = torch.nn.functional.cosine_similarity(
            delta, fwd.unsqueeze(1).expand_as(delta), dim=-1)
        cost = cost + dir_weight * (1.0 - cos)
    if ordinal is not None:
        o = ordinal.reshape(-1)[idx]
        step = o.unsqueeze(1) - o.unsqueeze(0)          # child - candidate
        cost = cost + ord_weight * (step - 1.0).abs()
    cost.fill_diagonal_(float("inf"))

    best_cost, best = cost.min(dim=1)
    finite = dist[dist > 0]
    gate = (max_edge_factor * finite.median() if finite.numel() > 0
            else torch.tensor(float("inf"), device=device))
    has_parent = torch.isfinite(best_cost) & (best_cost <= gate)
    if is_base is not None:
        has_parent = has_parent & ~(is_base.reshape(-1)[idx] > 0.5)
    local_parent = torch.where(has_parent, best, torch.full_like(best, -1))

    if not vectorized:
        # Every node points at its cheapest candidate, so the pointer graph is a
        # forest plus, possibly, cycles (the top two nodes of a shoot are each
        # other's nearest neighbour). Cut each cycle at its most expensive edge;
        # at a tie keep the edge whose parent is lower, which is where the shoot
        # base is. Until 2026-09-14 a parent was simply required to sit BELOW its
        # child, which made cycles impossible but also cut every drooping lateral
        # (dataset cowpea shoots at DAP 40-75 fall 1-3 mm per node): 8 of 99
        # same-shoot links lost on one DAP 75 plant, and a 73 cm export error.
        # float64: the 1e-6 height tie-break is below float32 resolution next to
        # a 3 cm distance, so in float32 two nodes of a mutual-nearest pair
        # often tie exactly and the cut fell to traversal order.
        cut_cost = best_cost.double() + 1e-6 * (p[best.clamp(min=0), 2] - p[:, 2]).double()
        state = torch.zeros(N, dtype=torch.long)                 # 0 new, 1 on path, 2 done
        parent_list = local_parent.tolist()
        for start in range(N):
            if state[start] != 0:
                continue
            path = []
            node = start
            while node >= 0 and state[node] == 0:
                state[node] = 1
                path.append(node)
                node = parent_list[node]
            if node >= 0 and state[node] == 1:                  # closed a cycle
                cyc = path[path.index(node):]
                top = max(float(cut_cost[k]) for k in cyc)
                worst = min(k for k in cyc if float(cut_cost[k]) == top)   # lowest index among exact ties
                parent_list[worst] = -1
            for k in path:
                state[k] = 2
        local_parent = torch.tensor(parent_list, dtype=torch.long, device=device)

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
                if root_own_shoot and int(local_parent[node]) < 0:
                    cont = -1                                 # the root keeps to itself
                elif len(kids) == 1:
                    cont = kids[0]
                elif ordinal is not None:
                    o = ordinal.reshape(-1)[idx]
                    want = o[node] + 1.0
                    gap = torch.stack([(o[k] - want).abs() for k in kids])
                    tied = [k for k, g in zip(kids, gap.tolist()) if g <= float(gap.min()) + 0.5]
                    if len(tied) == 1:
                        cont = tied[0]
                    elif fwd is not None:
                        # Depth cannot separate the continuation from a lateral's
                        # first node (both are one step up): the straighter one
                        # continues the shoot.
                        ax = fwd[node]
                        cos_k = torch.stack([
                            torch.nn.functional.cosine_similarity(
                                (p[k] - p[node]).unsqueeze(0), ax.unsqueeze(0), dim=-1)[0]
                            for k in tied])
                        cont = tied[int(cos_k.argmax())]
                    else:
                        cont = tied[int(dist[node][torch.tensor(tied)].argmin())]
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

    else:
        local_parent, local_shoot, local_ord = _resolve_forest(
            local_parent, best_cost, p, dist, fwd,
            ordinal.reshape(-1)[idx] if ordinal is not None else None, root_own_shoot)

    parent_idx[idx] = torch.where(local_parent >= 0, idx[local_parent.clamp(min=0)],
                                  torch.full_like(local_parent, -1))
    shoot_id[idx] = local_shoot
    phytomer_idx[idx] = local_ord
    return parent_idx, shoot_id, phytomer_idx
