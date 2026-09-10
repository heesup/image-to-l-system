"""
Dataset for paired (rendered image, part-centric PlantOrganArray tensor) samples.

The 13D part tensor layout (per organ):
    [OrganType(0), Base(1..3), Rot6D(4..9), Scale(10..12)]

Flow matching 25D node layout:
    [One-hot Organ Type (0..12), Base*20 (13..15), Rot6D (16..21), Scale*50 (22..24)]
    - Organ category 0 is ORGAN_NONE (empty slot).
    - Real organs are 1..12.
    - Existence is recovered as 1.0 - p(ORGAN_NONE).
"""

import os
import glob
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    ORGAN_NONE,
    ORGAN_ROOT_META,
    ORGAN_SHOOT_META,
    ORGAN_INTERNODE,
    ORGAN_PETIOLE,
    ORGAN_LEAF,
    ORGAN_PEDUNCLE,
    ORGAN_BUD_DORMANT,
    ORGAN_BUD_ACTIVE,
    ORGAN_FLOWER_CLOSED,
    ORGAN_FLOWER_OPEN,
    ORGAN_FRUIT,
    ORGAN_BUD_ABORTED,
    NUM_ORGAN_TYPES,
    P_COL_ORGAN_TYPE,
    P_COL_BASE_X,
    P_COL_BASE_Z,
    P_COL_ROT_0,
    P_COL_ROT_5,
    P_COL_SCALE_X,
    P_COL_SCALE_Y,
    P_COL_SCALE_Z,
    NUM_FEATURES,
)

# Fixed normalization constants
BASE_SCALE = 20.0
SCALE_SCALE = 50.0

# Aliases re-exported for downstream modules (botanical_scaffold etc.) —
# canonical definitions live in plant_organ_array.
from diffusion_based.models.plant_organ_array import (  # noqa: E402,F401
    ORGAN_BUD,
    ORGAN_FLOWER,
)

# ---------------------------------------------------------------------------
# Flow-matching organ-category encoding.
#
# The canonical part tensor keeps a scalar organ type (0=NONE, 1..12=real organs).
# For flow matching we one-hot the organ type into 13 classes.
# Existence is recovered as 1.0 - p(ORGAN_NONE).
# ---------------------------------------------------------------------------
ORGAN_CATEGORIES = list(range(NUM_ORGAN_TYPES))
CATEGORY_TO_IDX = {ot: ot for ot in ORGAN_CATEGORIES}
EMPTY_IDX = ORGAN_NONE
NUM_ORGAN_CATEGORIES = NUM_ORGAN_TYPES

# Flow-matching node layout (per organ):
#   [one-hot organ type (13), Base(3), Rot6D(6), Scale(3), Curvature(1)] = 26D
# Curvature (14D col 13, deg/m) added 2026-09-04: exp6/exp7 verified the
# dimension is live in the renderer (AGENT_TAKEOVER_GUIDE §8.5). Empty slots
# encode curvature 0 via the same scaling.
FM_OT_START = 0
FM_OT_END = NUM_ORGAN_CATEGORIES  # 13
FM_BASE_START = FM_OT_END         # 13
FM_BASE_END = FM_OT_END + 3       # 16
FM_ROT_START = FM_BASE_END        # 16
FM_ROT_END = FM_BASE_END + 6      # 22
FM_SCALE_START = FM_ROT_END       # 22
FM_SCALE_END = FM_ROT_END + 3     # 25
FM_CURV = FM_SCALE_END            # 25
FM_NODE_DIM = FM_CURV + 1         # 26

# Two-Stage 13D Pure Geometry Layout (no one-hot block):
#   [Base(3), Rot6D(6), Scale(3), Curvature(1)] = 13D
GEOM_BASE_START = 0
GEOM_BASE_END = 3
GEOM_ROT_START = 3
GEOM_ROT_END = 9
GEOM_SCALE_START = 9
GEOM_SCALE_END = 12
GEOM_CURV = 12
GEOM_NODE_DIM = 13

# Curvature normalization: cowpea shard curvature values concentrate in
# [-60, 60] deg/m with std ~18.3; scale by 1/60 to keep ODE magnitudes ~O(1)
# comparable to the other normalized blocks.
CURV_SCALE = 1.0 / 60.0


def encode_fm(part: torch.Tensor) -> torch.Tensor:
    """Convert a canonical (N, 13) part tensor to the 26D FM node layout.

    One-hot organ type (13) + Base(3) + Rot6D(6) + Scale(3) + Curvature(1) = 26D.
    """
    N = part.shape[0]
    out = torch.zeros((N, FM_NODE_DIM), dtype=part.dtype, device=part.device)
    ot = part[:, P_COL_ORGAN_TYPE].long().clamp(0, NUM_ORGAN_CATEGORIES - 1)

    # 1. One-hot organ category (0 = ORGAN_NONE, 1..12 = organs)
    out.scatter_(1, ot.unsqueeze(1), 1.0)

    # 2. Continuous geometric regression targets
    active_mask = ot > ORGAN_NONE
    out[active_mask, FM_BASE_START:FM_BASE_END] = part[active_mask, P_COL_BASE_X:P_COL_BASE_Z + 1] * BASE_SCALE
    out[active_mask, FM_ROT_START:FM_ROT_END] = part[active_mask, P_COL_ROT_0:P_COL_ROT_5 + 1]
    out[active_mask, FM_SCALE_START:FM_SCALE_END] = part[active_mask, P_COL_SCALE_X:P_COL_SCALE_Z + 1] * SCALE_SCALE
    if part.shape[1] > 13:
        # Curvature (col 13, deg/m) -> normalized; zeros stay zeros (straight).
        out[active_mask, FM_CURV] = part[active_mask, 13] * CURV_SCALE
    return out


def encode_fm_geom(part: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a canonical (N, 14) part tensor to (geom_13d, organ_type_labels).

    Returns:
        geom: (N, 13) continuous geometry targets [Base*20, Rot6D, Scale*50, Curv*CURV_SCALE].
        labels: (N,) int64 categorical labels in [0, 12].
    """
    N = part.shape[0]
    labels = part[:, P_COL_ORGAN_TYPE].long().clamp(0, NUM_ORGAN_CATEGORIES - 1)
    geom = torch.zeros((N, GEOM_NODE_DIM), dtype=part.dtype, device=part.device)

    active_mask = labels > ORGAN_NONE
    geom[active_mask, GEOM_BASE_START:GEOM_BASE_END] = part[active_mask, P_COL_BASE_X:P_COL_BASE_Z + 1] * BASE_SCALE
    geom[active_mask, GEOM_ROT_START:GEOM_ROT_END] = part[active_mask, P_COL_ROT_0:P_COL_ROT_5 + 1]
    geom[active_mask, GEOM_SCALE_START:GEOM_SCALE_END] = part[active_mask, P_COL_SCALE_X:P_COL_SCALE_Z + 1] * SCALE_SCALE
    if part.shape[1] > 13:
        geom[active_mask, GEOM_CURV] = part[active_mask, 13] * CURV_SCALE
    return geom, labels


def decode_fm(fm: torch.Tensor) -> torch.Tensor:
    """Convert a 26D FM node tensor back to a canonical (N, 14) part tensor.

    Organ type = argmax over the 13 categories (0 = NONE).
    Returns 14D: [organ_type, base(3), rot6d(6), scale(3), curvature(1)].
    """
    N = fm.shape[0]
    out = torch.zeros((N, 14), dtype=fm.dtype, device=fm.device)
    ot_probs = fm[:, :NUM_ORGAN_CATEGORIES]
    ot = ot_probs.argmax(dim=1)
    out[:, P_COL_ORGAN_TYPE] = ot.float()
    out[:, P_COL_BASE_X:P_COL_BASE_Z + 1] = fm[:, FM_BASE_START:FM_BASE_END] / BASE_SCALE
    out[:, P_COL_ROT_0:P_COL_ROT_5 + 1] = fm[:, FM_ROT_START:FM_ROT_END]
    out[:, P_COL_SCALE_X:P_COL_SCALE_Z + 1] = fm[:, FM_SCALE_START:FM_SCALE_END] / SCALE_SCALE
    out[:, 13] = fm[:, FM_CURV] / CURV_SCALE
    return out


def decode_fm_twostage(geom: torch.Tensor, type_logits: torch.Tensor) -> torch.Tensor:
    """Convert 13D geometry + 13-class logits to a canonical (N, 14) part tensor.

    Args:
        geom: (N, 13) [base*20(3), rot6d(6), scale*50(3), curv*CURV_SCALE(1)].
        type_logits: (N, 13) discrete organ-type logits.

    Returns:
        (N, 14) canonical part tensor: [organ_type, base(3), rot6d(6), scale(3), curvature(1)].
    """
    N = geom.shape[0]
    out = torch.zeros((N, 14), dtype=geom.dtype, device=geom.device)
    ot = type_logits.argmax(dim=-1)
    out[:, P_COL_ORGAN_TYPE] = ot.float()
    out[:, P_COL_BASE_X:P_COL_BASE_Z + 1] = geom[:, GEOM_BASE_START:GEOM_BASE_END] / BASE_SCALE
    out[:, P_COL_ROT_0:P_COL_ROT_5 + 1] = geom[:, GEOM_ROT_START:GEOM_ROT_END]
    out[:, P_COL_SCALE_X:P_COL_SCALE_Z + 1] = geom[:, GEOM_SCALE_START:GEOM_SCALE_END] / SCALE_SCALE
    out[:, 13] = geom[:, GEOM_CURV] / CURV_SCALE
    return out


def canonical_sort_nodes(
    nodes: torch.Tensor,
    existence_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sorts 25D plant organ node slots into a deterministic canonical botanical order.
    
    Order:
      1. Active vs Empty (Active nodes first, Empty padding slots last)
      2. Organ Type Botanical Hierarchy:
         - Root Meta / Shoot Meta (anchors)
         - Internodes (Stem skeleton, sorted bottom -> top along base_z)
         - Petioles (Branch / leaf stalks, sorted bottom -> top along base_z)
         - Leaves (Photosynthetic blades, sorted bottom -> top, then azimuth angle)
         - Peduncles / Buds (Axillary reproductive nodes)
         - Flowers / Fruits (Flowers / Pods)
      3. Z-height (base_z: height above soil)
      4. Azimuth angle (atan2(base_y, base_x) in [-pi, pi])
    """
    if nodes.ndim != 2 or nodes.shape[0] <= 1:
        if existence_mask is None:
            existence_mask = nodes[:, EMPTY_IDX] < 0.5
        return nodes, existence_mask

    N = nodes.shape[0]
    device = nodes.device

    ot_onehot = nodes[:, :FM_OT_END]  # (N, 13)
    ot_idx = ot_onehot.argmax(dim=-1)  # (N,)

    is_active = (ot_idx != EMPTY_IDX) & (ot_onehot[:, EMPTY_IDX] < 0.5)
    if existence_mask is not None and existence_mask.shape[0] == N:
        is_active = is_active & existence_mask

    rank_map = torch.tensor([
        999, # 0: NONE
        0,   # 1: Root Meta
        1,   # 2: Shoot Meta
        2,   # 3: Internode
        3,   # 4: Petiole
        4,   # 5: Leaf
        6,   # 6: Peduncle
        5,   # 7: Bud Dormant
        5,   # 8: Bud Active
        7,   # 9: Flower Closed
        7,   # 10: Flower Open
        8,   # 11: Fruit
        5,   # 12: Bud Aborted
    ], dtype=torch.float32, device=device)

    cat_ranks = rank_map[ot_idx]
    cat_ranks = torch.where(is_active, cat_ranks, torch.tensor(999.0, device=device))

    base_x = nodes[:, FM_BASE_START]
    base_y = nodes[:, FM_BASE_START + 1]
    base_z = nodes[:, FM_BASE_START + 2]
    azimuth = torch.atan2(base_y, base_x)

    import math as _math
    sort_keys = cat_ranks * 10000.0 + base_z * 10.0 + (azimuth / (2.0 * _math.pi))
    sort_indices = torch.argsort(sort_keys)

    sorted_nodes = nodes[sort_indices]
    sorted_mask = is_active[sort_indices]

    return sorted_nodes, sorted_mask


class PartArrayDataset(Dataset):
    """Loads Helios XML -> part tensor + rendered image."""

    def __init__(
        self,
        data_root: str,
        max_nodes: int = 2048,
        image_size: int = 128,
        device: torch.device = None,
        use_gt_renderer_image: bool = True,
        exclude_globs: List[str] = None,
        include_globs: List[str] = None,
        cache_dir: str = None,
        species: Optional[str] = "cowpea",
        pkt_cache_dir: str = None,
    ):
        self.data_root = os.path.abspath(data_root)
        self.max_nodes = max_nodes
        self.image_size = image_size
        self.use_gt_renderer_image = use_gt_renderer_image
        self.node_dim = FM_NODE_DIM
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.pkt_cache_dir = pkt_cache_dir
        self.species = species
        self._cached_renderer = None
        self._image_cache = {}
        self._tensor_cache = {}
        self._pkt_cache = {}

        self._transform_tensor = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        import fnmatch as _fnmatch
        xml_paths = sorted(glob.glob(os.path.join(self.data_root, "**", "*_plant_*.xml"), recursive=True))
        if not xml_paths:
            xml_paths = sorted(glob.glob(os.path.join(self.data_root, "*_plant_*.xml")))
        xml_paths = [
            p for p in xml_paths 
            if "/_tmp_" not in p and "_tmp_" not in os.path.basename(p) and f"{os.sep}tmp{os.sep}" not in p
        ]
        # Species filter: strictly isolate requested species (e.g. cowpea)
        target_species = self.species
        if target_species is None and self.cache_dir is not None:
            c = os.path.basename(os.path.normpath(self.cache_dir)).split("_")[0]
            if c in ("cowpea", "bean", "sorghum", "soybean", "maize"):
                target_species = c
        if target_species:
            xml_paths = [
                p for p in xml_paths 
                if f"/{target_species}/" in p or f"{os.sep}{target_species}{os.sep}" in p or os.path.basename(p).startswith(f"{target_species}_")
            ]
        if include_globs:
            xml_paths = [p for p in xml_paths if any(_fnmatch.fnmatch(os.path.basename(p), pat) for pat in include_globs)]
        elif exclude_globs:
            xml_paths = [p for p in xml_paths if not any(_fnmatch.fnmatch(os.path.basename(p), pat) for pat in exclude_globs)]
        if self.cache_dir and os.path.isdir(self.cache_dir):
            cached_files = {f[:-3] for f in os.listdir(self.cache_dir) if f.endswith(".pt")}
            xml_paths = [p for p in xml_paths if os.path.basename(p).split("_plant_")[0] in cached_files]
        resolved = [self._resolve_pair(p) for p in xml_paths]
        self.samples = [p for p in resolved if p["jpeg"] and os.path.exists(p["jpeg"])]

    def _resolve_pair(self, xml_path: str) -> Dict[str, str]:
        prefix = os.path.basename(xml_path).split("_plant_")[0]
        xml_dir = os.path.dirname(os.path.abspath(xml_path))
        jpeg_path = ""
        for suffix in ("_vis.jpeg", "_rad.jpeg"):
            candidate = os.path.join(xml_dir, f"{prefix}{suffix}")
            if os.path.exists(candidate):
                jpeg_path = candidate
                break
        return {"xml": xml_path, "jpeg": jpeg_path, "prefix": prefix}

    def encode_fm(self, part: torch.Tensor) -> torch.Tensor:
        """Convert a canonical (N, 13) part tensor to the 25D FM node layout."""
        return encode_fm(part)

    def decode_fm(self, fm: torch.Tensor) -> torch.Tensor:
        """Convert a 25D FM node vector back to a canonical (N, 13) part tensor."""
        return decode_fm(fm)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        cache_key = sample["xml"]

        # Fast path: Load complete pre-cached sample dict directly
        if self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, f"{sample['prefix']}.pt")
            if os.path.exists(cache_path):
                try:
                    data = torch.load(cache_path, map_location="cpu", weights_only=False)
                    if isinstance(data, dict) and "image" in data and "nodes" in data:
                        # Pyramid-concat cache: (16, H0, W0) half -> (16, image_size, image_size) float
                        img = data["image"].float()
                        if img.shape[-1] != self.image_size:
                            img = F.interpolate(img.unsqueeze(0), size=(self.image_size, self.image_size),
                                                mode="bilinear", align_corners=False).squeeze(0)
                        data["image"] = img
                        if "dap" not in data:
                            dap = 30.0
                            if "dap" in sample["prefix"]:
                                try:
                                    import re
                                    m = re.search(r"dap(\d+)", sample["prefix"])
                                    if m:
                                        dap = float(m.group(1))
                                except Exception:
                                    pass
                            data["dap"] = torch.tensor(dap, dtype=torch.float32)
                        # nodes padding/crop to max_nodes
                        if data["nodes"].shape[0] > self.max_nodes:
                            data["nodes"] = data["nodes"][: self.max_nodes]
                        elif data["nodes"].shape[0] < self.max_nodes:
                            pad_n = self.max_nodes - data["nodes"].shape[0]
                            data["nodes"] = F.pad(data["nodes"], (0, 0, 0, pad_n))
                        # phytomer_ids (N, 2) must align with the padded nodes.
                        if "phytomer_ids" in data:
                            ids = data["phytomer_ids"]
                            if ids.shape[0] > self.max_nodes:
                                data["phytomer_ids"] = ids[: self.max_nodes]
                            elif ids.shape[0] < self.max_nodes:
                                pad_n = self.max_nodes - ids.shape[0]
                                data["phytomer_ids"] = F.pad(
                                    ids, (0, 0, 0, pad_n), value=-1)
                        data["existence_mask"] = (data["nodes"][:, EMPTY_IDX] < 0.5).float()
                        data["prefix"] = sample["prefix"]
                        data["jpeg"] = sample["jpeg"]
                        # Precomputed phytomer packet targets: embedded in the main
                        # cache (new generate_cache.py pipeline) or, for legacy
                        # datasets, in a separate pkt cache dir.
                        if "pkt" not in data and self.pkt_cache_dir:
                            pkt_path = os.path.join(self.pkt_cache_dir, f"{sample['prefix']}.pt")
                            if os.path.exists(pkt_path):
                                _pkt = torch.load(pkt_path, map_location="cpu", weights_only=True)
                                # v3 gate: 10 slots + normalized-space latent required.
                                # Stale v1 (8-slot) / v2 (absolute latent) files fall
                                # through to the on-the-fly fallback below.
                                if isinstance(_pkt, dict) and _pkt.get("pkt_version", 0) >= 3:
                                    data["pkt"] = _pkt
                        return data
                except Exception:
                    pass

        if not os.path.exists(sample["xml"]):
            return self.__getitem__((idx + 1) % len(self.samples))

        # Fallback path: Load XML and compute on the fly
        part = None
        try:
            gt_array = PlantOrganArray.from_xml_file(sample["xml"])
            part = gt_array.to_part_tensor(device=torch.device("cpu"))
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))

        # Image
        if cache_key in self._image_cache:
            image_tensor = self._image_cache[cache_key]
        else:
            image_tensor = None
            if sample.get("jpeg") and os.path.exists(sample["jpeg"]):
                try:
                    with Image.open(sample["jpeg"]) as pil_img:
                        pil_rgb = pil_img.convert("RGB")
                        pil_rgb = pil_rgb.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                        arr_img = np.array(pil_rgb, dtype=np.float32) / 255.0
                        rgb_t = torch.from_numpy(arr_img).permute(2, 0, 1)
                        image_tensor = transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        )(rgb_t)
                except Exception:
                    image_tensor = None

            if image_tensor is None:
                return self.__getitem__((idx + 1) % len(self.samples))
            self._image_cache[cache_key] = image_tensor

        # Part tensor (padded + FM-encoded)
        if cache_key in self._tensor_cache:
            nodes, existence_mask, num_nodes = self._tensor_cache[cache_key]
        else:
            N = min(part.shape[0], self.max_nodes)
            nodes = torch.zeros((self.max_nodes, FM_NODE_DIM), dtype=torch.float32)
            nodes[:N] = self.encode_fm(part[:N])

            # Active-slot mask = not the empty category.
            existence_mask = (nodes[:, EMPTY_IDX] < 0.5).float()

            num_nodes = torch.tensor(N, dtype=torch.long)
        dap = 30.0
        if "dap" in sample["prefix"]:
            try:
                import re
                m = re.search(r"dap(\d+)", sample["prefix"])
                if m:
                    dap = float(m.group(1))
            except Exception:
                pass

        return {
            "image": image_tensor,
            "nodes": nodes,
            "existence_mask": existence_mask,
            "num_nodes": num_nodes,
            "dap": torch.tensor(dap, dtype=torch.float32),
            "xml_path": sample["xml"],
            "prefix": sample["prefix"],
        }
