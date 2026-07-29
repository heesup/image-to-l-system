import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from typing import Dict, Any, Optional
from dataset.generator import PRESET_GRAMMARS
from dataset.lsystem import LSystem

def get_device() -> torch.device:
    """Return Apple Silicon MPS device if available, otherwise CUDA or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class LSystemPredictor(nn.Module):
    """End-to-End Vision Model predicting L-System Topology Class and Geometric Parameters."""

    def __init__(self, num_templates: int = len(PRESET_GRAMMARS)):
        super().__init__()
        # Vision Backbone: Pre-trained ResNet-18
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Feature Projection
        self.feature_layer = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Head 1: Topology / Grammar Rule Classifier
        self.grammar_head = nn.Linear(256, num_templates)

        # Head 2: Continuous Parameter Regressor [angle, iterations, step_size, line_width]
        self.param_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(images)
        proj = self.feature_layer(feats)

        grammar_logits = self.grammar_head(proj)
        params = self.param_head(proj)

        return {
            "grammar_logits": grammar_logits,
            "pred_params": params
        }

    @torch.no_grad()
    def predict_lsystem(self, image: torch.Tensor) -> LSystem:
        """Predict LSystem specification object from a single preprocessed image tensor."""
        self.eval()
        out = self.forward(image)

        grammar_idx = torch.argmax(out["grammar_logits"], dim=-1).item()
        preset = PRESET_GRAMMARS[grammar_idx % len(PRESET_GRAMMARS)]

        norm_params = torch.sigmoid(out["pred_params"][0]).cpu().numpy()
        angle = float(round(max(12.0, min(65.0, norm_params[0] * 65.0)), 1))
        iterations = int(max(2, min(5, round(norm_params[1] * 5.0))))
        step_size = float(round(max(0.6, min(2.5, norm_params[2] * 2.0)), 2))
        line_width = float(round(max(1.0, min(4.0, norm_params[3] * 3.0)), 1))

        return LSystem(
            axiom=preset["axiom"],
            rules=preset["rules"],
            angle=angle,
            iterations=iterations,
            step_size=step_size,
            line_width=line_width
        )


class LSystemVLM:
    """Wrapper class for LSystemPredictor on Apple Silicon MPS."""

    def __init__(self, model_name: str = "standalone", device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.model = LSystemPredictor().to(self.device)

    def to(self, device: torch.device):
        self.device = device
        self.model = self.model.to(device)
        return self
