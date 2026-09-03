"""
Flow Matching (Rectified Flow) scheduler for Part-Centric PlantOrganArray generation.

Implements the conditional flow-matching objective (Lipman et al. 2023, Liu et al.
2023 "Rectified Flow"): instead of DDPM's noise-prediction, we learn a velocity
field v_theta(x_t, t) that transports samples along straight-line paths from a
Gaussian prior x_0 ~ N(0, I) to the data x_1 (the part tensor).

    x_t = (1 - t) * x_0 + t * x_1          (linear interpolation)
    v_target = x_1 - x_0                    (constant velocity along the path)

The model predicts v_theta(x_t, t, image) and is trained with MSE against v_target.
Sampling is a simple ODE integration (Euler) from t=0 to t=1.

The part tensor layout (per organ):
    [OrganType(0), Base(1..3), Rot6D(4..9), Scale(10..12), Existence(13)]

This is simpler and often faster to converge than DDPM because:
  - the target is a constant velocity (not a time-varying score)
  - straight paths require fewer integration steps
  - no noise schedule / beta hyperparameters to tune
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Flow-matching node layout — single source of truth is part_array_dataset (27D).
from diffusion_based.dataset.part_array_dataset import (
    FM_OT_END, FM_BASE_START, FM_BASE_END,
    FM_ROT_START, FM_ROT_END,
    FM_SCALE_START, FM_SCALE_END,
    FM_NODE_DIM,
)


class FlowMatchingScheduler:
    """Rectified-flow scheduler: linear interpolation + constant velocity target."""

    def __init__(self, sigma_min: float = 0.0):
        self.sigma_min = sigma_min

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample t ~ U[0, 1]."""
        return torch.rand(batch_size, device=device)

    def sample_xt(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate x_t = (1-t) x0 + t x1 along the straight path."""
        t = t.view(-1, 1, 1)
        return (1.0 - t) * x0 + t * x1

    def velocity_target(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Constant velocity v = x1 - x0."""
        return x1 - x0

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        images: torch.Tensor,
        num_steps: int = 50,
        node_dim: int = 16,
        max_nodes: int = 2048,
        device: torch.device = None,
        x0: Optional[torch.Tensor] = None,
        guidance_fn: Optional[callable] = None,
        guidance_weight: float = 0.0,
        denormalize_fn: Optional[callable] = None,
    ) -> torch.Tensor:
        """
        Euler ODE integration from t=0 to t=1.

        Args:
            model: denoiser with forward(noisy_nodes, timesteps, images) -> {'pred_velocity'}
                   where 'pred_velocity' is the predicted velocity field v_theta.
            images: (B, 3, H, W) condition image.
            num_steps: number of Euler steps.
            x0: optional initial sample (defaults to N(0, I)).
            guidance_fn: optional callable(x_t_denorm) -> scalar loss. Its gradient
                w.r.t. x_t is added to the velocity to steer sampling toward a
                target render (render-loss-guided flow matching).
            guidance_weight: scale of the guidance gradient.
            denormalize_fn: optional callable to map normalized x_t back to world
                part-tensor space before calling guidance_fn.
        Returns:
            x1: (B, N, node_dim) generated 16D part tensor.
        """
        B = images.shape[0]
        if device is None:
            device = images.device
        if x0 is None:
            x0 = torch.randn((B, max_nodes, node_dim), device=device)

        x_t = x0
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)
            with torch.no_grad():
                out = model(x_t, t, images)
                v = out["pred_velocity"]  # predicted velocity

            # Render-loss guidance: steer the ODE toward a target render.
            if guidance_fn is not None and guidance_weight > 0.0:
                x_t_g = x_t.clone().requires_grad_(True)
                x_denorm = denormalize_fn(x_t_g) if denormalize_fn is not None else x_t_g
                g_loss = guidance_fn(x_denorm)
                g_grad = torch.autograd.grad(g_loss, x_t_g)[0]
                v = v + guidance_weight * g_grad

            x_t = x_t + v * dt
            # Keep the organ-type block on the valid probability simplex without softmax distortion
            ot_block = x_t[..., :FM_OT_END].clamp(min=0.0)
            ot_sum = ot_block.sum(dim=-1, keepdim=True) + 1e-8
            x_t[..., :FM_OT_END] = ot_block / ot_sum
        return x_t
