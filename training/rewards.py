import json
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional
from dataset.lsystem import LSystem
from dataset.renderer import TurtleRenderer

def get_binary_mask(img: Image.Image, threshold: int = 240) -> np.ndarray:
    """Convert RGB PIL Image into binary mask where True = plant pixel, False = background."""
    arr = np.array(img.convert("RGB"))
    # Plant pixels are non-white (any channel below threshold)
    mask = (arr[:, :, 0] < threshold) | (arr[:, :, 1] < threshold) | (arr[:, :, 2] < threshold)
    return mask

def calculate_mask_iou(img1: Image.Image, img2: Image.Image) -> float:
    """Calculate Intersection-over-Union (IoU) of plant masks between two rendered images."""
    m1 = get_binary_mask(img1)
    m2 = get_binary_mask(img2)

    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)

def validate_lsystem_dict(pred_dict: Dict[str, Any]) -> Tuple[bool, Optional[LSystem]]:
    """Verify if prediction dictionary contains required keys, balanced brackets, and valid fields."""
    try:
        required_keys = ["axiom", "rules", "angle", "iterations"]
        if not all(k in pred_dict for k in required_keys):
            return False, None
        
        # Check rule bracket balance
        axiom = str(pred_dict["axiom"])
        if not LSystem.validate_brackets(axiom):
            return False, None

        rules = pred_dict["rules"]
        if not isinstance(rules, dict):
            return False, None

        for k, v in rules.items():
            if not LSystem.validate_brackets(str(v)):
                return False, None

        lsystem = LSystem.from_dict(pred_dict)
        return True, lsystem
    except Exception:
        return False, None

def compute_render_reward(pred_json_str: str, gt_image: Image.Image, renderer: Optional[TurtleRenderer] = None) -> Dict[str, float]:
    """Compute Render-in-the-Loop reward given predicted JSON string and ground truth image.
    Returns dict with keys: total_reward, iou_reward, syntax_reward.
    """
    renderer = renderer or TurtleRenderer(image_size=gt_image.size)
    
    try:
        pred_dict = json.loads(pred_json_str)
    except Exception:
        return {"total_reward": 0.0, "iou_reward": 0.0, "syntax_reward": 0.0}

    is_valid, lsystem = validate_lsystem_dict(pred_dict)
    if not is_valid or lsystem is None:
        return {"total_reward": 0.1, "iou_reward": 0.0, "syntax_reward": 0.1}

    # Render candidate plant image
    try:
        pred_img = renderer.render(lsystem)
        iou = calculate_mask_iou(gt_image, pred_img)
        syntax_reward = 1.0
        total_reward = 0.3 * syntax_reward + 0.7 * iou
        return {
            "total_reward": round(total_reward, 4),
            "iou_reward": round(iou, 4),
            "syntax_reward": syntax_reward
        }
    except Exception:
        return {"total_reward": 0.1, "iou_reward": 0.0, "syntax_reward": 0.1}
