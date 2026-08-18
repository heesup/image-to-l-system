"""
Flow Matching (Rectified Flow) scheduler for 14D Part-Centric PlantOrganArray generation.

Implements the conditional flow-matching objective (Lipman et al. 2023, Liu et al.
2023 "Rectified Flow"): instead of DDPM's noise-prediction, we learn a velocity
field v_theta(x_t, t) that transports samples along straight-line paths from a
Gaussian prior x_0 ~ N(0, I) to the data x_1 (the 14D part tensor).

    x_t = (1 - t) * x_0 + t * x_1          (linear interpolation)
    v_target = x_1 - x_0                    (constant velocity along the path)

The model predicts v_theta(x_t, t, image) and is trained with MSE against v_target.
Sampling is a simple ODE integration (Euler) from t=0 to t=1.

The 14D part tensor layout (per organ):
    [OrganType(0), Base(1..3), Rot6D(4..9), Scale(10..12), Existence(13)]

This is simpler and often faster to converge than DDPM because:
  - the target is a constant velocity (not a time-varying score)
  - straight paths require fewer integration steps
  - no noise schedule / beta hyperparameters to tune
"""

import torch
import torch.nn as nn
from typing import Optional


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
        node_dim: int = 17,
        max_nodes: int = 2048,
        device: torch.device = None,
        x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Euler ODE integration from t=0 to t=1.

        Args:
            model: denoiser with forward(noisy_nodes, timesteps, images) -> {'pred_velocity'}
                   where 'pred_velocity' is the predicted velocity field v_theta.
            images: (B, 3, H, W) condition image.
            num_steps: number of Euler steps.
            x0: optional initial sample (defaults to N(0, I)).
        Returns:
            x1: (B, N, node_dim) generated 14D part tensor.
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
            x_t = x_t + v * dt
        return x_t
