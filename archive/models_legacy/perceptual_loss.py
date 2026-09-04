"""
Multi-Scale Perceptual & Conceptual Loss module for 3D Plant Inverse Rendering.

Computes multi-layer VGG feature matching loss and structural silhouette matching
between PyTorch forward renders and target observations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Dict


class VGGPerceptualLoss(nn.Module):
    """
    Multi-layer VGG16 Perceptual / Conceptual Feature Loss.
    """

    def __init__(self, layer_weights: Dict[str, float] = None):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False

        # Extract features up to relu4_3 (slice 23)
        self.slice1 = vgg[:4]    # relu1_2
        self.slice2 = vgg[4:9]   # relu2_2
        self.slice3 = vgg[9:16]  # relu3_3
        self.slice4 = vgg[16:23] # relu4_3

        self.weights = layer_weights or {
            "relu1_2": 1.0,
            "relu2_2": 1.0,
            "relu3_3": 1.0,
            "relu4_3": 0.5,
        }

        # ImageNet normalization constants
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, 3, H, W) in [0, 1]
        return (x - self.mean) / self.std

    def forward(self, pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted L1 feature distance across VGG representation layers.
        """
        p = self._normalize(pred_rgb.clamp(0.0, 1.0))
        t = self._normalize(target_rgb.clamp(0.0, 1.0))

        loss = 0.0

        # relu1_2
        p1, t1 = self.slice1(p), self.slice1(t)
        loss += self.weights["relu1_2"] * F.l1_loss(p1, t1)

        # relu2_2
        p2, t2 = self.slice2(p1), self.slice2(t1)
        loss += self.weights["relu2_2"] * F.l1_loss(p2, t2)

        # relu3_3
        p3, t3 = self.slice3(p2), self.slice3(t2)
        loss += self.weights["relu3_3"] * F.l1_loss(p3, t3)

        # relu4_3
        p4, t4 = self.slice4(p3), self.slice4(t3)
        loss += self.weights["relu4_3"] * F.l1_loss(p4, t4)

        return loss
