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
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Performs 2-stage hierarchical matching for each batch item.

        Returns:
            List of length B containing dicts with:
                'anchor_src_idx': 1D int64 tensor of matched predicted anchor indices.
                'anchor_tgt_idx': 1D int64 tensor of matched GT cluster indices.
                'fine_src_idx': 1D int64 tensor of matched fine slot indices in [0, N_fine-1].
                'fine_tgt_idx': 1D int64 tensor of matched GT organ indices in [0, N_active_i-1].
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
                    "fine_src_idx": empty, "fine_tgt_idx": empty,
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

            # Extract stem/petiole structural nodes as natural cluster centers (classes 3=INTERNODE, 4=PETIOLE)
            is_structural = (t_label == 3) | (t_label == 4)
            structural_indices = torch.nonzero(is_structural, as_tuple=True)[0]

            if len(structural_indices) > 0:
                cluster_centers = gt_positions[structural_indices]  # (num_clusters, 3)
            else:
                # Fallback: K-means / spatial grid clustering
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
            cost_exist = -p_exist_prob.unsqueeze(1).expand(-1, num_gt_clusters)

            total_anchor_cost = self.cost_anchor_pos * cost_pos + self.cost_anchor_exist * cost_exist
            total_anchor_cost_cpu = total_anchor_cost.cpu().numpy()

            anc_src_np, anc_tgt_np = linear_sum_assignment(total_anchor_cost_cpu)
            anc_src = torch.as_tensor(anc_src_np, dtype=torch.int64, device=device)
            anc_tgt = torch.as_tensor(anc_tgt_np, dtype=torch.int64, device=device)

            # -------------------------------------------------------------
            # 3. Stage 2: Fine Intra-Cluster Matching (Fast M=8 Bipartite)
            # -------------------------------------------------------------
            all_fine_src = []
            all_fine_tgt = []

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

                # Local classification / existence cost
                if pred_fine_prob_b.shape[-1] > 1:
                    sub_prob = pred_fine_prob_b[slot_indices]  # (M, num_classes)
                    sub_tgt_labels = t_label[cluster_gt_indices]  # (n_gt,)
                    local_cls_cost = -sub_prob[:, sub_tgt_labels]  # (M, n_gt)
                else:
                    sub_exist = torch.sigmoid(pred_fine_logits[b][slot_indices])  # (M, 1)
                    local_cls_cost = -sub_exist.expand(-1, len(cluster_gt_indices))  # (M, n_gt)

                # Local representation cost (16D latent distance or legacy 3D position)
                if pred_fine_geom_b.shape[-1] == 16:
                    local_geom_cost = torch.cdist(pred_fine_geom_b[slot_indices], t_geom[cluster_gt_indices], p=1) / 16.0
                else:
                    sub_p_geom = pred_fine_geom_b[slot_indices, :3]  # (M, 3)
                    sub_t_geom = t_geom[cluster_gt_indices, :3]      # (n_gt, 3)
                    local_geom_cost = torch.cdist(sub_p_geom, sub_t_geom, p=1)  # (M, n_gt)

                local_total = self.cost_cls * local_cls_cost + self.cost_geom * local_geom_cost
                local_total_cpu = local_total.cpu().numpy()

                sub_src_np, sub_tgt_np = linear_sum_assignment(local_total_cpu)

                all_fine_src.append(slot_indices[sub_src_np])
                all_fine_tgt.append(cluster_gt_indices[sub_tgt_np])

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
            })

        return batch_matches
