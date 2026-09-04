"""
Botanical Scaffold Generator for Flow Matching Prior (x_0 ~ p_scaffold).

Replaces the degenerate point singularity x_0 = 0 with a physically grounded,
3D Fibonacci spiral canopy lattice matching empirical plant organ proportions:
  - 40.5% Leaf (distributed on outer canopy shell via Fibonacci spiral)
  - 14.1% Internode (aligned vertically along central main stem axis)
  - 14.2% Petiole (branching radially outward from internode nodes)
  - 13.2% Peduncle (inflorescence flowering stalks)
  -  9.6% Bud (dormant/active axillary and apical meristems)
  -  3.6% Bud Aborted
  -  1.6% Fruit (pods/berries at distal nodes)
  -  2.1% Flower (open and closed floral structures)
  -  1.2% Shoot/Root Meta (structural baseline anchors)
"""

import math
from typing import Optional, Dict
import numpy as np
import torch
import torch.nn.functional as F

from diffusion_based.dataset.part_array_dataset import (
    ORGAN_CATEGORIES,
    CATEGORY_TO_IDX,
    EMPTY_IDX,
    NUM_ORGAN_CATEGORIES,
    BASE_SCALE,
    SCALE_SCALE,
    FM_BASE_START,
    FM_BASE_END,
    FM_ROT_START,
    FM_ROT_END,
    FM_SCALE_START,
    FM_SCALE_END,
    FM_NODE_DIM,
    ORGAN_ROOT_META,
    ORGAN_SHOOT_META,
    ORGAN_INTERNODE,
    ORGAN_PETIOLE,
    ORGAN_LEAF,
    ORGAN_BUD,
    ORGAN_PEDUNCLE,
    ORGAN_FLOWER,
    ORGAN_FRUIT,
    ORGAN_FLOWER_CLOSED,
    ORGAN_BUD_ABORTED,
)

# Empirical organ proportions based on 214,796 organs from 15,000+ XML plants
EMPIRICAL_PROPORTIONS = {
    ORGAN_LEAF: 0.4050,
    ORGAN_PETIOLE: 0.1419,
    ORGAN_INTERNODE: 0.1408,
    ORGAN_PEDUNCLE: 0.1315,
    ORGAN_BUD: 0.0955,
    ORGAN_BUD_ABORTED: 0.0360,
    ORGAN_FRUIT: 0.0161,
    ORGAN_FLOWER_CLOSED: 0.0113,
    ORGAN_SHOOT_META: 0.0101,
    ORGAN_FLOWER: 0.0100,
    ORGAN_ROOT_META: 0.0018,
}

GOLDEN_RATIO_ANGLE = 137.50776405003785 * (math.pi / 180.0)  # 2.399963 rad


class BotanicalScaffoldGenerator:
    """Generates a 3D Botanical Canopy Scaffold Prior tensor (max_nodes, node_dim)."""

    def __init__(
        self,
        max_nodes: int = 2048,
        canopy_radius: float = 0.55,
        canopy_height: float = 0.70,
        node_dim: int = FM_NODE_DIM,
    ):
        self.max_nodes = max_nodes
        self.canopy_radius = canopy_radius
        self.canopy_height = canopy_height
        self.node_dim = node_dim
        self._canonical_scaffold = self._build_canonical_scaffold()

    def _build_canonical_scaffold(self) -> torch.Tensor:
        """Constructs the canonical 3D Botanical Scaffold tensor (max_nodes, node_dim)."""
        scaffold = torch.zeros((self.max_nodes, self.node_dim), dtype=torch.float32)

        # 1. Compute exact slot counts per organ type
        slot_counts = {}
        allocated = 0
        sorted_types = sorted(EMPIRICAL_PROPORTIONS.items(), key=lambda x: x[1], reverse=True)
        for ot, prop in sorted_types[:-1]:
            cnt = int(round(prop * self.max_nodes))
            slot_counts[ot] = cnt
            allocated += cnt
        # Remainder to the last type (ROOT_META)
        slot_counts[sorted_types[-1][0]] = max(1, self.max_nodes - allocated)

        # 2. Distribute positions and rotations based on biological organ morphology
        curr_idx = 0

        # A. Root Meta (Origin anchor)
        root_cnt = slot_counts.get(ORGAN_ROOT_META, 1)
        for _ in range(root_cnt):
            if curr_idx >= self.max_nodes: break
            scaffold[curr_idx, CATEGORY_TO_IDX[ORGAN_ROOT_META]] = 1.0
            scaffold[curr_idx, FM_BASE_START:FM_BASE_END] = torch.tensor([0.0, 0.0, 0.0]) * BASE_SCALE
            scaffold[curr_idx, FM_ROT_START:FM_ROT_END] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            scaffold[curr_idx, FM_SCALE_START:FM_SCALE_END] = torch.tensor([0.01, 0.01, 0.01]) * SCALE_SCALE
            curr_idx += 1

        # B. Shoot Meta (Stem base anchor)
        shoot_cnt = slot_counts.get(ORGAN_SHOOT_META, 1)
        for i in range(shoot_cnt):
            if curr_idx >= self.max_nodes: break
            scaffold[curr_idx, CATEGORY_TO_IDX[ORGAN_SHOOT_META]] = 1.0
            scaffold[curr_idx, FM_BASE_START:FM_BASE_END] = torch.tensor([0.0, 0.0, 0.01 * (i + 1)]) * BASE_SCALE
            scaffold[curr_idx, FM_ROT_START:FM_ROT_END] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            scaffold[curr_idx, FM_SCALE_START:FM_SCALE_END] = torch.tensor([0.02, 0.02, 0.02]) * SCALE_SCALE
            curr_idx += 1

        # C. Internodes (Vertical main stem backbone)
        internode_cnt = slot_counts.get(ORGAN_INTERNODE, 1)
        for i in range(internode_cnt):
            if curr_idx >= self.max_nodes: break
            frac = (i + 1) / max(1, internode_cnt)
            z_pos = frac * self.canopy_height
            r_jitter = 0.02 * math.sin(i * 1.5)
            theta = i * GOLDEN_RATIO_ANGLE
            scaffold[curr_idx, CATEGORY_TO_IDX[ORGAN_INTERNODE]] = 1.0
            scaffold[curr_idx, FM_BASE_START:FM_BASE_END] = torch.tensor([
                r_jitter * math.cos(theta),
                r_jitter * math.sin(theta),
                z_pos,
            ]) * BASE_SCALE
            scaffold[curr_idx, FM_ROT_START:FM_ROT_END] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            scaffold[curr_idx, FM_SCALE_START:FM_SCALE_END] = torch.tensor([0.02, 0.02, self.canopy_height / max(1, internode_cnt)]) * SCALE_SCALE
            curr_idx += 1

        # D. Petioles & Peduncles (Radial branching arms)
        for ot, cnt in [(ORGAN_PETIOLE, slot_counts.get(ORGAN_PETIOLE, 1)), (ORGAN_PEDUNCLE, slot_counts.get(ORGAN_PEDUNCLE, 1))]:
            for i in range(cnt):
                if curr_idx >= self.max_nodes: break
                frac = (i + 1) / max(1, cnt)
                z_pos = 0.05 + frac * (self.canopy_height - 0.08)
                r_pos = 0.04 + 0.6 * self.canopy_radius * math.sqrt(frac)
                theta = i * GOLDEN_RATIO_ANGLE
                scaffold[curr_idx, CATEGORY_TO_IDX[ot]] = 1.0
                scaffold[curr_idx, FM_BASE_START:FM_BASE_END] = torch.tensor([
                    r_pos * math.cos(theta),
                    r_pos * math.sin(theta),
                    z_pos,
                ]) * BASE_SCALE
                # Outward pointing orientation
                cos_t, sin_t = math.cos(theta), math.sin(theta)
                scaffold[curr_idx, FM_ROT_START:FM_ROT_END] = torch.tensor([cos_t, sin_t, 0.3, -sin_t, cos_t, 0.0])
                scaffold[curr_idx, FM_SCALE_START:FM_SCALE_END] = torch.tensor([0.015, 0.015, 0.10]) * SCALE_SCALE
                curr_idx += 1

        # E. Leaves (Fibonacci Spiral Canopy Shell)
        leaf_cnt = slot_counts.get(ORGAN_LEAF, 1)
        for i in range(leaf_cnt):
            if curr_idx >= self.max_nodes: break
            # Vogel's Fibonacci spiral distribution across conical canopy volume
            frac = (i + 0.5) / max(1, leaf_cnt)
            z_pos = 0.05 + (frac ** 0.75) * (self.canopy_height - 0.06)
            # Conical widening: radius is wider at mid-to-high canopy
            cone_factor = math.sin(frac * math.pi * 0.85)
            r_pos = 0.06 + self.canopy_radius * (0.35 + 0.65 * cone_factor) * math.sqrt(frac)
            theta = i * GOLDEN_RATIO_ANGLE

            scaffold[curr_idx, CATEGORY_TO_IDX[ORGAN_LEAF]] = 1.0
            scaffold[curr_idx, FM_BASE_START:FM_BASE_END] = torch.tensor([
                r_pos * math.cos(theta),
                r_pos * math.sin(theta),
                z_pos,
            ]) * BASE_SCALE
            # Leaf normal tilt (~45 deg upward-outward)
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            scaffold[curr_idx, FM_ROT_START:FM_ROT_END] = torch.tensor([cos_t, sin_t, 0.707, -sin_t, cos_t, 0.0])
            scaffold[curr_idx, FM_SCALE_START:FM_SCALE_END] = torch.tensor([0.12, 0.12, 0.002]) * SCALE_SCALE
            curr_idx += 1

        # F. Reproductive structures (Buds, Flowers, Fruits)
        repro_types = [ORGAN_BUD, ORGAN_BUD_ABORTED, ORGAN_FRUIT, ORGAN_FLOWER, ORGAN_FLOWER_CLOSED]
        for ot in repro_types:
            cnt = slot_counts.get(ot, 1)
            for i in range(cnt):
                if curr_idx >= self.max_nodes: break
                frac = (i + 1) / max(1, cnt)
                z_pos = 0.08 + frac * (self.canopy_height - 0.1)
                r_pos = 0.05 + 0.65 * self.canopy_radius * frac
                theta = (i * GOLDEN_RATIO_ANGLE) + 0.5
                scaffold[curr_idx, CATEGORY_TO_IDX[ot]] = 1.0
                scaffold[curr_idx, FM_BASE_START:FM_BASE_END] = torch.tensor([
                    r_pos * math.cos(theta),
                    r_pos * math.sin(theta),
                    z_pos,
                ]) * BASE_SCALE
                scaffold[curr_idx, FM_ROT_START:FM_ROT_END] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                scaffold[curr_idx, FM_SCALE_START:FM_SCALE_END] = torch.tensor([0.02, 0.02, 0.02]) * SCALE_SCALE
                curr_idx += 1

        # Fill any remaining slots with empty-prior background slots
        while curr_idx < self.max_nodes:
            scaffold[curr_idx, EMPTY_IDX] = 1.0
            curr_idx += 1

        return scaffold

    def generate_conditioned(
        self,
        radius: float,
        height: float,
        leaf_scale: float,
        active_count: int,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Dynamically constructs a 3D Botanical Scaffold tensor (max_nodes, node_dim)
        precisely scaled to (radius, height, leaf_scale) with exactly `active_count` non-empty slots.
        All remaining slots (max_nodes - active_count) are initialized as empty background slots.
        """
        active_count = max(4, min(self.max_nodes, active_count))
        scaffold = torch.zeros((self.max_nodes, self.node_dim), dtype=torch.float32)

        # 1. Compute slot counts for the active subset
        slot_counts = {}
        allocated = 0
        sorted_types = sorted(EMPIRICAL_PROPORTIONS.items(), key=lambda x: x[1], reverse=True)
        for ot, prop in sorted_types[:-1]:
            cnt = int(round(prop * active_count))
            slot_counts[ot] = cnt
            allocated += cnt
        slot_counts[sorted_types[-1][0]] = max(1, active_count - allocated)

        curr_idx = 0

        # A. Root Meta
        root_cnt = slot_counts.get(ORGAN_ROOT_META, 1)
        for _ in range(root_cnt):
            if curr_idx >= active_count: break
            scaffold[curr_idx, CATEGORY_TO_IDX[ORGAN_ROOT_META]] = 1.0
            scaffold[curr_idx, FM_BASE_START:FM_BASE_END] = torch.tensor([0.0, 0.0, 0.0]) * BASE_SCALE
            scaffold[curr_idx, FM_ROT_START:FM_ROT_END] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            scaffold[curr_idx, FM_SCALE_START:FM_SCALE_END] = torch.tensor([0.005, 0.005, 0.005]) * SCALE_SCALE
            curr_idx += 1

        # B. Shoot Meta
        shoot_cnt = slot_counts.get(ORGAN_SHOOT_META, 1)
        for i in range(shoot_cnt):
            if curr_idx >= active_count: break
            scaffold[curr_idx, CATEGORY_TO_IDX[ORGAN_SHOOT_META]] = 1.0
            scaffold[curr_idx, FM_BASE_START:FM_BASE_END] = torch.tensor([0.0, 0.0, 0.005 * (i + 1)]) * BASE_SCALE
            scaffold[curr_idx, FM_ROT_START:FM_ROT_END] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            scaffold[curr_idx, FM_SCALE_START:FM_SCALE_END] = torch.tensor([0.01, 0.01, 0.01]) * SCALE_SCALE
            curr_idx += 1

        # C. Internodes
        internode_cnt = slot_counts.get(ORGAN_INTERNODE, 1)
        for i in range(internode_cnt):
            if curr_idx >= active_count: break
            frac = (i + 1) / max(1, internode_cnt)
            z_pos = frac * height
            r_jitter = 0.01 * radius * math.sin(i * 1.5)
            theta = i * GOLDEN_RATIO_ANGLE
            scaffold[curr_idx, CATEGORY_TO_IDX[ORGAN_INTERNODE]] = 1.0
            scaffold[curr_idx, FM_BASE_START:FM_BASE_END] = torch.tensor([
                r_jitter * math.cos(theta),
                r_jitter * math.sin(theta),
                z_pos,
            ]) * BASE_SCALE
            scaffold[curr_idx, FM_ROT_START:FM_ROT_END] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            scaffold[curr_idx, FM_SCALE_START:FM_SCALE_END] = torch.tensor([0.01, 0.01, height / max(1, internode_cnt)]) * SCALE_SCALE
            curr_idx += 1

        # D. Petioles & Peduncles
        for ot, cnt in [(ORGAN_PETIOLE, slot_counts.get(ORGAN_PETIOLE, 1)), (ORGAN_PEDUNCLE, slot_counts.get(ORGAN_PEDUNCLE, 1))]:
            for i in range(cnt):
                if curr_idx >= active_count: break
                frac = (i + 1) / max(1, cnt)
                z_pos = 0.03 + frac * max(0.01, height - 0.04)
                r_pos = 0.02 + 0.6 * radius * math.sqrt(frac)
                theta = i * GOLDEN_RATIO_ANGLE
                scaffold[curr_idx, CATEGORY_TO_IDX[ot]] = 1.0
                scaffold[curr_idx, FM_BASE_START:FM_BASE_END] = torch.tensor([
                    r_pos * math.cos(theta),
                    r_pos * math.sin(theta),
                    z_pos,
                ]) * BASE_SCALE
                cos_t, sin_t = math.cos(theta), math.sin(theta)
                scaffold[curr_idx, FM_ROT_START:FM_ROT_END] = torch.tensor([cos_t, sin_t, 0.3, -sin_t, cos_t, 0.0])
                scaffold[curr_idx, FM_SCALE_START:FM_SCALE_END] = torch.tensor([0.008, 0.008, max(0.02, 0.5 * radius)]) * SCALE_SCALE
                curr_idx += 1

        # E. Leaves
        leaf_cnt = slot_counts.get(ORGAN_LEAF, 1)
        for i in range(leaf_cnt):
            if curr_idx >= active_count: break
            frac = (i + 0.5) / max(1, leaf_cnt)
            z_pos = 0.02 + (frac ** 0.75) * max(0.01, height - 0.03)
            cone_factor = math.sin(frac * math.pi * 0.85)
            r_pos = 0.03 + radius * (0.35 + 0.65 * cone_factor) * math.sqrt(frac)
            theta = i * GOLDEN_RATIO_ANGLE

            scaffold[curr_idx, CATEGORY_TO_IDX[ORGAN_LEAF]] = 1.0
            scaffold[curr_idx, FM_BASE_START:FM_BASE_END] = torch.tensor([
                r_pos * math.cos(theta),
                r_pos * math.sin(theta),
                z_pos,
            ]) * BASE_SCALE
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            scaffold[curr_idx, FM_ROT_START:FM_ROT_END] = torch.tensor([cos_t, sin_t, 0.707, -sin_t, cos_t, 0.0])
            scaffold[curr_idx, FM_SCALE_START:FM_SCALE_END] = torch.tensor([leaf_scale, leaf_scale, 0.002]) * SCALE_SCALE
            curr_idx += 1

        # F. Reproductive structures
        repro_types = [ORGAN_BUD, ORGAN_BUD_ABORTED, ORGAN_FRUIT, ORGAN_FLOWER, ORGAN_FLOWER_CLOSED]
        for ot in repro_types:
            cnt = slot_counts.get(ot, 1)
            for i in range(cnt):
                if curr_idx >= active_count: break
                frac = (i + 1) / max(1, cnt)
                z_pos = 0.03 + frac * max(0.01, height - 0.05)
                r_pos = 0.02 + 0.65 * radius * frac
                theta = (i * GOLDEN_RATIO_ANGLE) + 0.5
                scaffold[curr_idx, CATEGORY_TO_IDX[ot]] = 1.0
                scaffold[curr_idx, FM_BASE_START:FM_BASE_END] = torch.tensor([
                    r_pos * math.cos(theta),
                    r_pos * math.sin(theta),
                    z_pos,
                ]) * BASE_SCALE
                scaffold[curr_idx, FM_ROT_START:FM_ROT_END] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                scaffold[curr_idx, FM_SCALE_START:FM_SCALE_END] = torch.tensor([0.015, 0.015, 0.015]) * SCALE_SCALE
                curr_idx += 1

        # Inactive background slots
        while curr_idx < self.max_nodes:
            scaffold[curr_idx, EMPTY_IDX] = 1.0
            curr_idx += 1

        if device is not None:
            scaffold = scaffold.to(device)
        return scaffold

    def generate_from_dap(
        self,
        dap: float,
        species: str = "cowpea",
        device: torch.device = None,
    ) -> torch.Tensor:
        """Generates a stage-conditioned scaffold tensor (max_nodes, node_dim) from developmental age (DAP)."""
        dap_val = float(dap)
        frac = min(1.0, max(0.0, (dap_val - 5.0) / 85.0))
        radius = 0.10 + 0.68 * (frac ** 0.8)
        height = 0.09 + 0.64 * (frac ** 0.9)
        leaf_scale = 0.035 + 0.075 * (frac ** 0.5)
        active_count = int(16 + (self.max_nodes - 16) * (frac ** 1.1))

        return self.generate_conditioned(
            radius=radius,
            height=height,
            leaf_scale=leaf_scale,
            active_count=active_count,
            device=device,
        )

    def get_canonical_scaffold(self, device: torch.device = None) -> torch.Tensor:
        """Returns the canonical deterministic scaffold tensor (max_nodes, node_dim)."""
        t = self._canonical_scaffold.clone()
        if device is not None:
            t = t.to(device)
        return t

    def sample_prior(
        self,
        batch_size: int,
        device: torch.device = None,
        noise_std: float = 0.015,
        dap: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Samples batch of initial prior states x_0 ~ p_scaffold(x).
        If dap is given, generates developmentally conditioned prior.
        """
        if dap is not None:
            canon = self.generate_from_dap(dap, device=device)
        else:
            canon = self.get_canonical_scaffold(device=device)

        batch = canon.unsqueeze(0).repeat(batch_size, 1, 1)

        if noise_std > 0:
            base_noise = torch.randn((batch_size, self.max_nodes, 3), device=device) * noise_std
            batch[:, :, FM_BASE_START:FM_BASE_END] += base_noise
            scale_noise = torch.randn((batch_size, self.max_nodes, 3), device=device) * (noise_std * 0.5)
            batch[:, :, FM_SCALE_START:FM_SCALE_END] = (batch[:, :, FM_SCALE_START:FM_SCALE_END] + scale_noise).clamp(min=1e-4)

        return batch
