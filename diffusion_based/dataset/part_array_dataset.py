"""
Dataset for paired (rendered image, part-centric PlantOrganArray tensor) samples.

The part tensor layout (per organ):
    [OrganType(0), Base(1..3), Rot6D(4..9), Scale(10..12), Existence(13),
     Curvature(14), PhyllotacticAngle(15)]

Normalization (fixed, hand-tuned to unit-ish scale for flow matching):
    - organ type (col 0):  / 10.0 -> [0, 1]  (categorical, rounded at inference)
    - base (cols 1..3):    * 100.0 -> ~[-1, 1] (world coords are ~cm scale)
    - rot6d (cols 4..9):   unchanged (already [-1, 1])
    - scale (cols 10..12): unchanged (already [0, 1])
    - existence (col 13):  unchanged (already [0, 1])
    - curvature (col 14):  / 100.0 -> ~[-1, 1] (degrees)
    - phyllotactic (col 15): / 180.0 -> ~[0, 1] (degrees)
"""

import os
import glob
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    P_COL_ORGAN_TYPE,
    P_COL_BASE_X,
    P_COL_BASE_Z,
    P_COL_ROT_0,
    P_COL_ROT_5,
    P_COL_SCALE_X,
    P_COL_SCALE_Z,
    P_COL_EXISTENCE,
    P_COL_CURVATURE,
    P_COL_PHYLLOTACTIC_ANGLE,
    NUM_FEATURES,
    ORGAN_ROOT_META, ORGAN_SHOOT_META, ORGAN_INTERNODE, ORGAN_PETIOLE, ORGAN_LEAF,
    ORGAN_BUD, ORGAN_PEDUNCLE, ORGAN_FLOWER, ORGAN_FRUIT, ORGAN_FLOWER_CLOSED,
    ORGAN_BUD_ABORTED,
)

# Fixed normalization constants (see module docstring)
ORGAN_TYPE_SCALE = 10.0
BASE_SCALE = 20.0
SCALE_SCALE = 50.0
CURVATURE_SCALE = 100.0
PHYLLOTACTIC_SCALE = 180.0

# ---------------------------------------------------------------------------
# Flow-matching organ-category encoding.
#
# The canonical part tensor keeps a scalar organ type + existence column. For
# flow matching we instead one-hot the organ type and add an "empty" category,
# dropping the separate existence column. Existence is recovered as
# 1 - p(empty). This makes the categorical organ type a proper probability
# distribution (softmax) and lets the model grow organs from an empty plant.
# ---------------------------------------------------------------------------
ORGAN_CATEGORIES = [
    ORGAN_ROOT_META, ORGAN_SHOOT_META, ORGAN_INTERNODE, ORGAN_PETIOLE, ORGAN_LEAF,
    ORGAN_BUD, ORGAN_PEDUNCLE, ORGAN_FLOWER, ORGAN_FRUIT, ORGAN_FLOWER_CLOSED,
    ORGAN_BUD_ABORTED,
]
CATEGORY_TO_IDX = {ot: i for i, ot in enumerate(ORGAN_CATEGORIES)}
EMPTY_IDX = len(ORGAN_CATEGORIES)          # index of the "empty" category
NUM_ORGAN_CATEGORIES = len(ORGAN_CATEGORIES) + 1  # 11 real + 1 empty

# Flow-matching node layout (per organ):
#   [one-hot organ type (NUM_ORGAN_CATEGORIES), Base(3), Rot6D(6), Scale(3),
#    Curvature(1), Phyllotactic(1)]
FM_OT_END = NUM_ORGAN_CATEGORIES
FM_BASE_START = FM_OT_END
FM_BASE_END = FM_OT_END + 3
FM_ROT_START = FM_BASE_END
FM_ROT_END = FM_BASE_END + 6
FM_SCALE_START = FM_ROT_END
FM_SCALE_END = FM_ROT_END + 3
FM_CURV_IDX = FM_SCALE_END
FM_PHYLLO_IDX = FM_SCALE_END + 1
FM_NODE_DIM = FM_SCALE_END + 2


def decode_fm(fm: torch.Tensor) -> torch.Tensor:
    """Convert a 26D FM node tensor back to a canonical (N, 17) part tensor.

    Organ type = argmax over the real organ categories; existence = 1 - p(empty).
    This is the standalone (module-level) version of PartArrayDataset.decode_fm,
    usable directly on model output (x_gen / x_1_hat) for the 17D->XML->Helios
    round-trip without instantiating a dataset.
    """
    N = fm.shape[0]
    out = torch.zeros((N, NUM_FEATURES), dtype=fm.dtype, device=fm.device)
    ot_probs_real = fm[:, :EMPTY_IDX]
    empty_prob = fm[:, EMPTY_IDX]
    ot_idx = ot_probs_real.argmax(dim=1)
    for i, cat in enumerate(ORGAN_CATEGORIES):
        out[ot_idx == i, P_COL_ORGAN_TYPE] = cat
    out[:, P_COL_EXISTENCE] = (1.0 - empty_prob).clamp(0.0, 1.0)
    out[:, P_COL_BASE_X:P_COL_BASE_Z + 1] = fm[:, FM_BASE_START:FM_BASE_END] / BASE_SCALE
    out[:, P_COL_ROT_0:P_COL_ROT_5 + 1] = fm[:, FM_ROT_START:FM_ROT_END]
    out[:, P_COL_SCALE_X:P_COL_SCALE_Z + 1] = F.softplus(fm[:, FM_SCALE_START:FM_SCALE_END]).clamp(min=0.005) / SCALE_SCALE
    out[:, P_COL_CURVATURE] = fm[:, FM_CURV_IDX] * CURVATURE_SCALE
    out[:, P_COL_PHYLLOTACTIC_ANGLE] = fm[:, FM_PHYLLO_IDX] * PHYLLOTACTIC_SCALE
    return out


def canonical_sort_nodes(
    nodes: torch.Tensor,
    existence_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sorts 26D plant organ node slots into a deterministic canonical botanical order.
    
    Order:
      1. Active vs Empty (Active nodes first, Empty padding slots last)
      2. Organ Type Botanical Hierarchy:
         - Root / Shoot Meta (anchors)
         - Internodes (Stem skeleton, sorted bottom -> top along base_z)
         - Petioles (Branch / leaf stalks, sorted bottom -> top along base_z)
         - Leaves (Photosynthetic blades, sorted bottom -> top, then azimuth angle)
         - Peduncles / Buds (Axillary reproductive nodes)
         - Flowers / Fruits (Flowers / Pods)
      3. Z-height (base_z: height above soil)
      4. Azimuth angle (atan2(base_y, base_x) in [-pi, pi])
    
    Args:
        nodes: (N, 26) tensor of organ nodes
        existence_mask: optional (N,) boolean tensor
    Returns:
        sorted_nodes: (N, 26) canonically ordered tensor
        sorted_mask: (N,) canonically ordered boolean mask
    """
    if nodes.ndim != 2 or nodes.shape[0] <= 1:
        if existence_mask is None:
            existence_mask = nodes[:, EMPTY_IDX] < 0.5
        return nodes, existence_mask

    N = nodes.shape[0]
    device = nodes.device

    ot_onehot = nodes[:, :FM_OT_END]  # (N, 12)
    ot_idx = ot_onehot.argmax(dim=-1)  # (N,)

    # Determine active vs empty slots directly from nodes (and match mask if same shape)
    is_active = (ot_idx != EMPTY_IDX) & (ot_onehot[:, EMPTY_IDX] < 0.5)
    if existence_mask is not None and existence_mask.shape[0] == N:
        is_active = is_active & existence_mask

    rank_map = torch.tensor([
        0,   # 0: Root Meta
        1,   # 1: Shoot Meta
        2,   # 2: Internode
        3,   # 3: Petiole
        4,   # 4: Leaf
        5,   # 5: Bud
        6,   # 6: Peduncle
        7,   # 7: Flower
        8,   # 8: Fruit
        7,   # 9: Flower Closed
        5,   # 10: Bud Aborted
        999  # 11: Empty
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
    ):
        self.data_root = os.path.abspath(data_root)
        self.max_nodes = max_nodes
        self.image_size = image_size
        self.use_gt_renderer_image = use_gt_renderer_image
        self.node_dim = FM_NODE_DIM
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self._cached_renderer = None
        self._image_cache = {}
        self._tensor_cache = {}

        self._transform_tensor = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        import fnmatch as _fnmatch
        xml_paths = sorted(glob.glob(os.path.join(self.data_root, "**", "*_plant_*.xml"), recursive=True))
        if not xml_paths:
            xml_paths = sorted(glob.glob(os.path.join(self.data_root, "*_plant_*.xml")))
        # Filter out transient / temporary SLURM worker directories
        xml_paths = [p for p in xml_paths if "/_tmp_" not in p and "_tmp_" not in os.path.basename(p)]
        if include_globs:
            xml_paths = [p for p in xml_paths if any(_fnmatch.fnmatch(os.path.basename(p), pat) for pat in include_globs)]
        elif exclude_globs:
            xml_paths = [p for p in xml_paths if not any(_fnmatch.fnmatch(os.path.basename(p), pat) for pat in exclude_globs)]
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

    def normalize(self, part: torch.Tensor) -> torch.Tensor:
        """Normalize a (N, D) part tensor to unit-ish scale."""
        out = part.clone()
        out[:, P_COL_ORGAN_TYPE] = out[:, P_COL_ORGAN_TYPE] / ORGAN_TYPE_SCALE
        out[:, P_COL_BASE_X:P_COL_BASE_Z + 1] = out[:, P_COL_BASE_X:P_COL_BASE_Z + 1] * BASE_SCALE
        out[:, P_COL_CURVATURE] = out[:, P_COL_CURVATURE] / CURVATURE_SCALE
        out[:, P_COL_PHYLLOTACTIC_ANGLE] = out[:, P_COL_PHYLLOTACTIC_ANGLE] / PHYLLOTACTIC_SCALE
        return out

    def denormalize(self, part: torch.Tensor) -> torch.Tensor:
        """Undo normalization."""
        out = part.clone()
        out[:, P_COL_ORGAN_TYPE] = out[:, P_COL_ORGAN_TYPE] * ORGAN_TYPE_SCALE
        out[:, P_COL_BASE_X:P_COL_BASE_Z + 1] = out[:, P_COL_BASE_X:P_COL_BASE_Z + 1] / BASE_SCALE
        out[:, P_COL_CURVATURE] = out[:, P_COL_CURVATURE] * CURVATURE_SCALE
        out[:, P_COL_PHYLLOTACTIC_ANGLE] = out[:, P_COL_PHYLLOTACTIC_ANGLE] * PHYLLOTACTIC_SCALE
        return out

    # ------------------------------------------------------------------
    # Flow-matching encoding: canonical 16D part tensor <-> FM node vector
    # ------------------------------------------------------------------
    def encode_fm(self, part: torch.Tensor) -> torch.Tensor:
        """Convert a canonical (N, 16) part tensor to the FM node layout.

        Organ type is one-hot (with an extra 'empty' category), existence is
        dropped (recovered as 1 - p(empty)). Continuous columns are normalized.
        """
        N = part.shape[0]
        out = torch.zeros((N, FM_NODE_DIM), dtype=part.dtype, device=part.device)
        ot = part[:, P_COL_ORGAN_TYPE].long()
        exist = part[:, P_COL_EXISTENCE]
        for i, cat in enumerate(ORGAN_CATEGORIES):
            mask = (ot == cat) & (exist > 0.5)
            out[mask, i] = 1.0
        # Empty category = slots that are inactive (existence <= 0.5).
        out[exist <= 0.5, EMPTY_IDX] = 1.0
        active_mask = (exist > 0.5)
        out[active_mask, FM_BASE_START:FM_BASE_END] = part[active_mask, P_COL_BASE_X:P_COL_BASE_Z + 1] * BASE_SCALE
        out[active_mask, FM_ROT_START:FM_ROT_END] = part[active_mask, P_COL_ROT_0:P_COL_ROT_5 + 1]
        out[active_mask, FM_SCALE_START:FM_SCALE_END] = part[active_mask, P_COL_SCALE_X:P_COL_SCALE_Z + 1] * SCALE_SCALE
        out[active_mask, FM_CURV_IDX] = part[active_mask, P_COL_CURVATURE] / CURVATURE_SCALE
        out[active_mask, FM_PHYLLO_IDX] = part[active_mask, P_COL_PHYLLOTACTIC_ANGLE] / PHYLLOTACTIC_SCALE
        return out

    def decode_fm(self, fm: torch.Tensor) -> torch.Tensor:
        """Convert an FM node vector back to a canonical (N, 16) part tensor.

        Organ type = argmax over the 11 real organ categories; existence = 1 - p(empty).
        """
        N = fm.shape[0]
        out = torch.zeros((N, NUM_FEATURES), dtype=fm.dtype, device=fm.device)
        ot_probs_real = fm[:, :EMPTY_IDX]
        empty_prob = fm[:, EMPTY_IDX]
        ot_idx = ot_probs_real.argmax(dim=1)
        for i, cat in enumerate(ORGAN_CATEGORIES):
            out[ot_idx == i, P_COL_ORGAN_TYPE] = cat
        out[:, P_COL_EXISTENCE] = (1.0 - empty_prob).clamp(0.0, 1.0)
        out[:, P_COL_BASE_X:P_COL_BASE_Z + 1] = fm[:, FM_BASE_START:FM_BASE_END] / BASE_SCALE
        out[:, P_COL_ROT_0:P_COL_ROT_5 + 1] = fm[:, FM_ROT_START:FM_ROT_END]
        out[:, P_COL_SCALE_X:P_COL_SCALE_Z + 1] = F.softplus(fm[:, FM_SCALE_START:FM_SCALE_END]).clamp(min=0.005) / SCALE_SCALE
        out[:, P_COL_CURVATURE] = fm[:, FM_CURV_IDX] * CURVATURE_SCALE
        out[:, P_COL_PHYLLOTACTIC_ANGLE] = fm[:, FM_PHYLLO_IDX] * PHYLLOTACTIC_SCALE
        return out

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
                        # resize to self.image_size
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
