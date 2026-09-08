"""
Hierarchical Bipartite Matcher for Botanical Flow Matching.

Matches coarse predicted anchors to ground-truth phytomer clusters,
then performs ultra-fast local bipartite matching (M=8 slots) within each cluster.
Replaces the expensive O(4096^3) global Hungarian assignment with two lightweight stages:
  1. Coarse Anchor Matching: O(K^3) with K <= 512 (<1ms)
  2. Fine Intra-Cluster Matching: O(M^3) with M = 8 (<0.01ms)
"""

from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HierarchicalBotanicalMatcher(nn.Module):
    """Hierarchical Matcher: Coarse Phytomer Anchor Matching + Fine Organ Matching."""

    def __init__(
        self,
        cost_anchor_pos: float = 3.0,
        cost_anchor_exist: float = 1.0,
        cost_cls: float = 2.0,
        cost_geom: float = 2.0,
        slots_per_anchor: int = 8,
    ):
        super().__init__()
        self.cost_anchor_pos = cost_anchor_pos
        self.cost_anchor_exist = cost_anchor_exist
        self.cost_cls = cost_cls
        self.cost_geom = cost_geom
        self.slots_per_anchor = slots_per_anchor

    @torch.no_grad()
    def forward(
        self,
        pred_anchor_pos: torch.Tensor,       # (B, K, 3)
        pred_anchor_logits: torch.Tensor,    # (B, K, 1)
        pred_fine_geom: torch.Tensor,        # (B, N_fine, node_dim) 16D latent or 13D geom
        pred_fine_logits: torch.Tensor,      # (B, N_fine, num_classes)
        tgt_geoms: List[torch.Tensor],        # List of (N_active_i, node_dim)
        tgt_labels: List[torch.Tensor],       # List of (N_active_i,)
        tgt_positions: Optional[List[torch.Tensor]] = None,  # Optional List of (N_active_i, 3) 3D base positions
        soft_margin_weights: Optional[torch.Tensor] = None,  # Optional (B, K) soft tapering existence prior
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Performs hierarchical bipartite matching for each batch item.

        Returns:
            List of length B containing dicts with:
                'anchor_src_idx': 1D int64 tensor of matched predicted anchor indices.
                'anchor_tgt_idx': 1D int64 tensor of matched GT cluster indices.
                'anchor_tgt_pos': (M_anc, 3) GT cluster center positions for matched anchors.
                'fine_src_idx': 1D int64 tensor of matched fine slot indices in [0, N_fine-1].
                'fine_tgt_idx': 1D int64 tensor of matched GT organ indices in [0, N_active_i-1].
                'num_gt_phytomers': integer count of GT phytomer clusters.
                'gt_node_centers': (N_gt_phytomers, 3) GT phytomer insertion node coordinates.
        """
        B, K, _ = pred_anchor_pos.shape
        M = self.slots_per_anchor
        N_fine = pred_fine_geom.shape[1]
        device = pred_anchor_pos.device

        batch_matches: List[Dict[str, torch.Tensor]] = []

        for b in range(B):
            t_geom = tgt_geoms[b]      # (M_i, node_dim)
            t_label = tgt_labels[b]    # (M_i,)
            M_i = t_geom.shape[0]

            if M_i == 0:
                empty = torch.empty(0, dtype=torch.int64, device=device)
                batch_matches.append({
                    "anchor_src_idx": empty, "anchor_tgt_idx": empty,
                    "anchor_tgt_pos": torch.empty((0, 3), device=device),
                    "fine_src_idx": empty, "fine_tgt_idx": empty,
                    "num_gt_phytomers": 0,
                    "gt_node_centers": torch.empty((0, 3), device=device),
                })
                continue

            # -------------------------------------------------------------
            # 1. Cluster GT organs into Phytomer Groups (Petiole/Node Anchors)
            # -------------------------------------------------------------
            # Use explicit 3D base coordinates if provided (essential for 16D latent mode)
            if tgt_positions is not None:
                gt_positions = tgt_positions[b]  # (M_i, 3)
            else:
                gt_positions = t_geom[:, 0:3]    # (M_i, 3)

            # Extract true botanical Phytomer Node centers:
            # The petiole base (class 4) is physically the exact insertion point (Node) of the phytomer.
            petiole_indices = torch.nonzero(t_label == 4, as_tuple=True)[0]
            if len(petiole_indices) > 0:
                cluster_centers = gt_positions[petiole_indices]  # (N_phytomers, 3)
                # Also capture basal/terminal internodes that have no petiole (e.g. distance > 4cm from all petioles)
                internode_indices = torch.nonzero(t_label == 3, as_tuple=True)[0]
                if len(internode_indices) > 0:
                    dist_in_to_pet = torch.cdist(gt_positions[internode_indices], cluster_centers).min(dim=1).values
                    standalone_in = internode_indices[dist_in_to_pet > 0.04]
                    if len(standalone_in) > 0:
                        cluster_centers = torch.cat([cluster_centers, gt_positions[standalone_in]], dim=0)
            else:
                # Early seedling fallback: use internodes or all active organs
                internode_indices = torch.nonzero(t_label == 3, as_tuple=True)[0]
                if len(internode_indices) > 0:
                    cluster_centers = gt_positions[internode_indices]
                else:
                    num_clusters = min(K, max(1, M_i // M))
                    cluster_centers = gt_positions[:num_clusters]

            num_gt_clusters = cluster_centers.shape[0]

            # Assign each GT organ to its nearest cluster center
            dist_to_centers = torch.cdist(gt_positions, cluster_centers)  # (M_i, num_gt_clusters)
            gt_cluster_assignments = torch.argmin(dist_to_centers, dim=1)  # (M_i,)

            # -------------------------------------------------------------
            # 2. Stage 1: Coarse Anchor Hungarian Matching
            # -------------------------------------------------------------
            # Cost between K predicted anchors and num_gt_clusters
            p_pos = pred_anchor_pos[b]  # (K, 3)
            cost_pos = torch.cdist(p_pos, cluster_centers, p=1)  # (K, num_gt_clusters)

            p_exist_prob = torch.sigmoid(pred_anchor_logits[b]).squeeze(-1)  # (K,)
            if soft_margin_weights is not None:
                p_exist_prob = p_exist_prob * soft_margin_weights[b]
            cost_exist = -p_exist_prob.unsqueeze(1).expand(-1, num_gt_clusters)

            total_anchor_cost = self.cost_anchor_pos * cost_pos + self.cost_anchor_exist * cost_exist
            total_anchor_cost_cpu = total_anchor_cost.cpu().numpy()

            anc_src_np, anc_tgt_np = linear_sum_assignment(total_anchor_cost_cpu)
            anc_src = torch.as_tensor(anc_src_np, dtype=torch.int64, device=device)
            anc_tgt = torch.as_tensor(anc_tgt_np, dtype=torch.int64, device=device)

            # -------------------------------------------------------------
            # 3. Stage 2: Fine Intra-Cluster Matching (Role-Partitioned M=8)
            # -------------------------------------------------------------
            # Canonical Phytomer Slot Roles for M=8:
            #   Slot 0: Internode/Stem (Types 1, 2, 3)
            #   Slot 1: Petiole (Type 4)
            #   Slots 2, 3, 4: Leaflets (Type 5)
            #   Slot 5: Peduncle (Type 6)
            #   Slots 6, 7: Reproductive: Buds, Flowers, Pods (Types 7..12)
            all_fine_src = []
            all_fine_tgt = []

            has_multiclass = pred_fine_logits[b].shape[-1] > 1
            if has_multiclass:
                pred_fine_prob_b = F.softmax(pred_fine_logits[b], dim=-1)  # (N_fine, num_classes)
            pred_fine_geom_b = pred_fine_geom[b]                        # (N_fine, node_dim)

            for a_idx, c_idx in zip(anc_src, anc_tgt):
                # Child slots for predicted anchor a_idx: [a_idx*M .. (a_idx+1)*M - 1]
                slot_start = a_idx.item() * M
                slot_end = slot_start + M
                slot_indices = torch.arange(slot_start, slot_end, device=device)  # (M,)

                # GT organs assigned to cluster c_idx
                cluster_gt_indices = torch.nonzero(gt_cluster_assignments == c_idx, as_tuple=True)[0]
                if len(cluster_gt_indices) == 0:
                    continue

                cluster_labels = t_label[cluster_gt_indices]

                # Partition GT cluster organs by functional role
                gt_by_role = {
                    0: cluster_gt_indices[(cluster_labels >= 1) & (cluster_labels <= 3)],  # Stem / Internode
                    1: cluster_gt_indices[cluster_labels == 4],                           # Petiole
                    2: cluster_gt_indices[cluster_labels == 5],                           # Leaflets
                    3: cluster_gt_indices[cluster_labels == 6],                           # Peduncle
                    4: cluster_gt_indices[cluster_labels >= 7],                           # Reproductive
                }

                # Dedicated slot indices for each functional role in this anchor
                slots_by_role = {
                    0: slot_indices[0:1],  # Slot 0: Internode
                    1: slot_indices[1:2],  # Slot 1: Petiole
                    2: slot_indices[2:5],  # Slots 2, 3, 4: Leaflets
                    3: slot_indices[5:6],  # Slot 5: Peduncle
                    4: slot_indices[6:8],  # Slots 6, 7: Reproductive
                }

                matched_slots_list = []
                matched_gt_list = []

                # 1. Role-constrained bipartite matching within each functional category
                for r in range(5):
                    r_slots = slots_by_role[r]
                    r_gt = gt_by_role[r]
                    n_s = len(r_slots)
                    n_g = len(r_gt)

                    if n_s == 0 or n_g == 0:
                        continue

                    if n_s == 1 and n_g == 1:
                        # 1:1 direct match (e.g. 1 stem to Slot 0, or 1 petiole to Slot 1)
                        matched_slots_list.append(r_slots)
                        matched_gt_list.append(r_gt)
                    else:
                        # Small bipartite matching within this specific role (e.g. <= 3x3 for leaves)
                        sub_slots = r_slots
                        sub_gt = r_gt
                        if pred_fine_geom_b.shape[-1] == 16:
                            cost_geom = torch.cdist(pred_fine_geom_b[sub_slots], t_geom[sub_gt], p=1) / 16.0
                        else:
                            cost_geom = torch.cdist(pred_fine_geom_b[sub_slots, :3], t_geom[sub_gt, :3], p=1)

                        if has_multiclass:
                            sub_prob = pred_fine_prob_b[sub_slots]
                            sub_tgt_labels = t_label[sub_gt]
                            cost_cls = -sub_prob[:, sub_tgt_labels]
                        else:
                            sub_exist = torch.sigmoid(pred_fine_logits[b][sub_slots])
                            cost_cls = -sub_exist.expand(-1, len(sub_gt))

                        local_cost = self.cost_cls * cost_cls + self.cost_geom * cost_geom
                        r_src_np, r_tgt_np = linear_sum_assignment(local_cost.cpu().numpy())

                        matched_slots_list.append(sub_slots[r_src_np])
                        matched_gt_list.append(sub_gt[r_tgt_np])

                # 2. Overflow matching: if any GT organs were left over (e.g. 4th leaf or 3rd flower),
                # match them to any remaining unassigned slots in this anchor
                if matched_slots_list:
                    cur_matched_s = torch.cat(matched_slots_list)
                    cur_matched_g = torch.cat(matched_gt_list)
                    matched_s_set = set(cur_matched_s.tolist())
                    matched_g_set = set(cur_matched_g.tolist())
                else:
                    cur_matched_s = torch.empty(0, dtype=torch.int64, device=device)
                    cur_matched_g = torch.empty(0, dtype=torch.int64, device=device)
                    matched_s_set = set()
                    matched_g_set = set()

                free_slots = [s for s in slot_indices if s.item() not in matched_s_set]
                leftover_gt = [g for g in cluster_gt_indices if g.item() not in matched_g_set]

                if free_slots and leftover_gt:
                    fs_t = torch.stack(free_slots)
                    lg_t = torch.stack(leftover_gt)
                    if pred_fine_geom_b.shape[-1] == 16:
                        cost_geom_rem = torch.cdist(pred_fine_geom_b[fs_t], t_geom[lg_t], p=1) / 16.0
                    else:
                        cost_geom_rem = torch.cdist(pred_fine_geom_b[fs_t, :3], t_geom[lg_t, :3], p=1)

                    if has_multiclass:
                        cost_cls_rem = -pred_fine_prob_b[fs_t][:, t_label[lg_t]]
                    else:
                        sub_exist = torch.sigmoid(pred_fine_logits[b][fs_t])
                        cost_cls_rem = -sub_exist.expand(-1, len(lg_t))

                    rem_cost = self.cost_cls * cost_cls_rem + self.cost_geom * cost_geom_rem
                    rem_src_np, rem_tgt_np = linear_sum_assignment(rem_cost.cpu().numpy())
                    matched_slots_list.append(fs_t[rem_src_np])
                    matched_gt_list.append(lg_t[rem_tgt_np])

                if matched_slots_list:
                    all_fine_src.append(torch.cat(matched_slots_list))
                    all_fine_tgt.append(torch.cat(matched_gt_list))

            if all_fine_src:
                fine_src = torch.cat(all_fine_src, dim=0)
                fine_tgt = torch.cat(all_fine_tgt, dim=0)
            else:
                fine_src = torch.empty(0, dtype=torch.int64, device=device)
                fine_tgt = torch.empty(0, dtype=torch.int64, device=device)

            anc_tgt_pos = cluster_centers[anc_tgt] if len(anc_tgt) > 0 else torch.empty((0, 3), device=device)
            batch_matches.append({
                "anchor_src_idx": anc_src,
                "anchor_tgt_idx": anc_tgt,
                "anchor_tgt_pos": anc_tgt_pos,
                "fine_src_idx": fine_src,
                "fine_tgt_idx": fine_tgt,
                "num_gt_phytomers": num_gt_clusters,
                "gt_node_centers": cluster_centers,
            })

        return batch_matches
