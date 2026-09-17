"""
Evaluation Metrics for 14D Plant Part Reconstruction.

Replaces raw full-image SSIM with metrics robust to background-dominated images:
  - masked_ssim():                SSIM computed only over foreground union mask
  - foreground_iou():             Silhouette IoU (plant vs background)
  - organ_type_iou():             Per-organ-type IoU from organ masks
  - affine_invariant_depth_loss(): Depth supervision (Phase 2: DepthAnythingV2)
  - evaluate_reconstruction():    Combined report
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# Foreground Mask Extraction
# ---------------------------------------------------------------------------

def get_foreground_mask(
    rgb: torch.Tensor,
    bg_color: Tuple[float, float, float] = (0.72, 0.62, 0.50),
    threshold: float = 0.08,
) -> torch.Tensor:
    """
    Extract foreground mask from a rendered image by comparing to the known
    background color. Designed for 14D renderer outputs (Helios ground color).

    Args:
        rgb:       (3, H, W) float [0, 1]
        bg_color:  Expected background RGB (Helios ground default)
        threshold: Per-pixel mean L1 distance from background to count as foreground

    Returns:
        mask: (H, W) bool
    """
    bg = torch.tensor(bg_color, dtype=rgb.dtype, device=rgb.device).view(3, 1, 1)
    dist = (rgb - bg).abs().mean(dim=0)  # (H, W)
    return dist > threshold


# ---------------------------------------------------------------------------
# Masked SSIM
# ---------------------------------------------------------------------------

def _gaussian_kernel(size: int = 11, sigma: float = 1.5, channels: int = 1,
                     device=None, dtype=None) -> torch.Tensor:
    coords = torch.arange(size, dtype=dtype, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k2d = g.unsqueeze(0) * g.unsqueeze(1)
    k2d = k2d / k2d.sum()
    return k2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)  # (C,1,s,s)


def masked_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_mask: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
    bg_color: Tuple[float, float, float] = (0.72, 0.62, 0.50),
    bg_threshold: float = 0.08,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """
    SSIM restricted to the union of foreground regions in pred and target.
    A blank prediction cannot score high by matching a uniform background.

    Args:
        pred, target: (3, H, W) float [0, 1]
        pred_mask, target_mask: optional (H, W) bool explicit foreground masks.
            If None, derived via color threshold.
    Returns:
        Masked SSIM scalar in [-1, 1]. Returns 0.0 if no foreground pixels.
    """
    C, H, W = pred.shape
    device, dtype = pred.device, pred.dtype

    if pred_mask is None:
        pred_mask = get_foreground_mask(pred, bg_color, bg_threshold)
    if target_mask is None:
        target_mask = get_foreground_mask(target, bg_color, bg_threshold)
    union_mask = (pred_mask | target_mask).float()  # (H, W)

    if union_mask.sum() < 1.0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    pad = window_size // 2
    pred_p = F.pad(pred.unsqueeze(0), [pad]*4, mode='reflect')
    tgt_p  = F.pad(target.unsqueeze(0), [pad]*4, mode='reflect')
    mask_p = F.pad(union_mask.unsqueeze(0).unsqueeze(0), [pad]*4, mode='reflect')

    kernel = _gaussian_kernel(window_size, sigma, channels=C, device=device, dtype=dtype)
    mk1    = _gaussian_kernel(window_size, sigma, channels=1, device=device, dtype=dtype)

    mu_p  = F.conv2d(pred_p, kernel, groups=C)
    mu_t  = F.conv2d(tgt_p,  kernel, groups=C)
    sig_p2 = F.conv2d(pred_p*pred_p, kernel, groups=C) - mu_p*mu_p
    sig_t2 = F.conv2d(tgt_p*tgt_p,  kernel, groups=C) - mu_t*mu_t
    sig_pt = F.conv2d(pred_p*tgt_p,  kernel, groups=C) - mu_p*mu_t

    ssim_map = ((2*mu_p*mu_t + C1)*(2*sig_pt + C2)) / \
               ((mu_p**2 + mu_t**2 + C1)*(sig_p2 + sig_t2 + C2))
    ssim_map = ssim_map.squeeze(0).mean(dim=0)  # (H, W)

    mask_smooth = F.conv2d(mask_p, mk1).squeeze(0).squeeze(0).clamp(min=0.0)
    denom = mask_smooth.sum()
    if denom < 1e-6:
        return torch.tensor(0.0, device=device, dtype=dtype)
    return (ssim_map * mask_smooth).sum() / denom


# ---------------------------------------------------------------------------
# Foreground IoU
# ---------------------------------------------------------------------------

def foreground_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_mask: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
    bg_color: Tuple[float, float, float] = (0.72, 0.62, 0.50),
    bg_threshold: float = 0.08,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    IoU of plant silhouettes. Blank prediction → IoU = 0.

    Args:
        pred, target: (3, H, W) float [0, 1]
        pred_mask, target_mask: optional (H, W) bool
    Returns:
        IoU scalar in [0, 1].
    """
    if pred_mask is None:
        pred_mask = get_foreground_mask(pred, bg_color, bg_threshold)
    if target_mask is None:
        target_mask = get_foreground_mask(target, bg_color, bg_threshold)

    pf = pred_mask.float()
    tf = target_mask.float()
    inter = (pf * tf).sum()
    union = (pf + tf - pf * tf).sum()
    return (inter + smooth) / (union + smooth)


# ---------------------------------------------------------------------------
# Per-Organ-Type IoU
# ---------------------------------------------------------------------------

def organ_type_iou(
    pred_organ_masks: Dict[int, torch.Tensor],
    target_organ_masks: Dict[int, torch.Tensor],
    smooth: float = 1e-6,
) -> Dict:
    """
    Per-organ-type IoU using masks from render_multimodal()['organ_masks'].

    Args:
        pred_organ_masks:   Dict[int → (H,W) bool]
        target_organ_masks: Dict[int → (H,W) bool]
    Returns:
        Dict[int → float] per-organ IoU, plus 'mean' key.
    """
    all_types = set(pred_organ_masks.keys()) | set(target_organ_masks.keys())
    results = {}
    for ot in sorted(all_types):
        p = pred_organ_masks.get(ot)
        t = target_organ_masks.get(ot)
        if p is None or t is None:
            results[ot] = 0.0
            continue
        pf = p.float(); tf = t.float()
        inter = (pf * tf).sum()
        union = (pf + tf - pf * tf).sum()
        results[ot] = float((inter + smooth) / (union + smooth))
    if results:
        results['mean'] = sum(v for k, v in results.items() if k != 'mean') / len(results)
    return results


# ---------------------------------------------------------------------------
# Affine-Invariant Depth Loss  (Phase 2: DepthAnythingV2)
# ---------------------------------------------------------------------------

def affine_invariant_depth_loss(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Affine-invariant L1 depth loss. Normalises both maps to zero-mean unit-std
    before comparing, so relative depth from DepthAnythingV2 can be used directly.

    Args:
        pred_depth:   (H, W) float — rendered depth from render_multimodal()['depth']
        target_depth: (H, W) float — GT depth (rendered) or DepthAnythingV2 output
        mask:         (H, W) bool  — restrict to foreground
        eps:          Numerical stability
    Returns:
        Scalar depth loss.
    """
    if mask is not None:
        p = pred_depth[mask]
        t = target_depth[mask]
    else:
        p = pred_depth.reshape(-1)
        t = target_depth.reshape(-1)

    if p.numel() < 2:
        return torch.tensor(0.0, device=pred_depth.device, dtype=pred_depth.dtype)

    p_n = (p - p.mean()) / (p.std() + eps)
    t_n = (t - t.mean()) / (t.std() + eps)
    return F.l1_loss(p_n, t_n)


# ---------------------------------------------------------------------------
# Combined Evaluation
# ---------------------------------------------------------------------------

def _raw_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Legacy full-image SSIM. Background-biased. Use only for comparison."""
    C1, C2 = 0.01**2, 0.03**2
    C, H, W = pred.shape
    device, dtype = pred.device, pred.dtype
    kernel = _gaussian_kernel(window_size, 1.5, channels=C, device=device, dtype=dtype)
    pad = window_size // 2
    p = F.pad(pred.unsqueeze(0), [pad]*4, mode='reflect')
    t = F.pad(target.unsqueeze(0), [pad]*4, mode='reflect')
    mu_p = F.conv2d(p, kernel, groups=C)
    mu_t = F.conv2d(t, kernel, groups=C)
    s_p2 = F.conv2d(p*p, kernel, groups=C) - mu_p*mu_p
    s_t2 = F.conv2d(t*t, kernel, groups=C) - mu_t*mu_t
    s_pt = F.conv2d(p*t, kernel, groups=C) - mu_p*mu_t
    ssim = ((2*mu_p*mu_t + C1)*(2*s_pt + C2)) / ((mu_p**2 + mu_t**2 + C1)*(s_p2 + s_t2 + C2))
    return ssim.mean()


def evaluate_reconstruction(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    pred_mask: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
    pred_depth: Optional[torch.Tensor] = None,
    target_depth: Optional[torch.Tensor] = None,
    pred_organ_masks: Optional[Dict] = None,
    target_organ_masks: Optional[Dict] = None,
    bg_color: Tuple[float, float, float] = (0.72, 0.62, 0.50),
) -> Dict:
    """
    Full evaluation report combining all metrics.

    Returns dict with:
        'masked_ssim' : float — primary quality metric (replaces raw SSIM)
        'fg_iou'      : float — silhouette accuracy
        'raw_ssim'    : float — legacy full-image SSIM (for comparison)
        'depth_loss'  : float — if depth maps provided
        'organ_iou'   : dict  — if organ masks provided
    """
    results = {}
    results['masked_ssim'] = float(masked_ssim(pred_rgb, target_rgb, pred_mask, target_mask, bg_color=bg_color))
    results['fg_iou']      = float(foreground_iou(pred_rgb, target_rgb, pred_mask, target_mask, bg_color=bg_color))
    results['raw_ssim']    = float(_raw_ssim(pred_rgb, target_rgb))

    if pred_depth is not None and target_depth is not None:
        fg_union = (pred_mask | target_mask) if (pred_mask is not None and target_mask is not None) else None
        results['depth_loss'] = float(affine_invariant_depth_loss(pred_depth, target_depth, fg_union))

    if pred_organ_masks is not None and target_organ_masks is not None:
        results['organ_iou'] = organ_type_iou(pred_organ_masks, target_organ_masks)

    return results
