import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableLineRenderer(nn.Module):
    """Fast PyTorch Differentiable Soft Line Renderer (Soft Rasterizer).
    Renders predicted plant tree graph (V, E) into continuous grayscale pixel images.
    Fully compatible with PyTorch Autograd for backpropagation.
    """

    def __init__(self, image_size: int = 128, sigma: float = 0.012):
        super().__init__()
        self.image_size = image_size
        self.sigma = sigma

        # Precompute 2D grid coordinates [0, 1] x [0, 1] of shape (1, 1, H, W, 2)
        y_coords = torch.linspace(0, 1, image_size)
        x_coords = torch.linspace(0, 1, image_size)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1) # (H, W, 2)
        self.register_buffer("grid", grid.unsqueeze(0).unsqueeze(0)) # (1, 1, H, W, 2)

    def forward(self, nodes: torch.Tensor, parent_indices: torch.Tensor, existence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            nodes: (B, N, 5) -> [norm_x, norm_y, norm_theta, norm_length, norm_width]
            parent_indices: (B, N) -> parent vertex index for each child node
            existence: (B, N) -> node existence confidence [0, 1]
        Returns:
            rendered_image: (B, 1, H, W) soft rendered pixel intensity image
        """
        B, N, _ = nodes.shape
        device = nodes.device

        p2 = nodes[:, :, :2] # Child node base positions (B, N, 2)
        widths = nodes[:, :, 4] * 0.05 + 0.005 # Normalized stem width (B, N)

        # Gather parent positions for each child
        batch_idx = torch.arange(B, device=device).unsqueeze(-1).expand(B, N)
        p1 = nodes[batch_idx, parent_indices, :2] # Parent node base positions (B, N, 2)

        # Vector p1 -> p2 (B, N, 2)
        v = p2 - p1
        v_sq = (v**2).sum(dim=-1, keepdim=True).unsqueeze(2).unsqueeze(2) + 1e-6

        # Grid points relative to p1: w = grid - p1
        w = self.grid - p1.unsqueeze(2).unsqueeze(2) # (B, N, H, W, 2)

        # Projection factor c = clamp((w . v) / |v|^2, 0, 1)
        v_expanded = v.unsqueeze(2).unsqueeze(2)
        v_dot_w = (w * v_expanded).sum(dim=-1, keepdim=True)
        c = torch.clamp(v_dot_w / v_sq, 0.0, 1.0)

        # Nearest point on line segment
        proj = p1.unsqueeze(2).unsqueeze(2) + c * v_expanded
        dist = torch.norm(self.grid - proj, dim=-1) # (B, N, H, W)

        # Soft Sigmoid line intensity falloff
        half_w = (widths / 2.0).unsqueeze(-1).unsqueeze(-1)
        line_intensity = torch.sigmoid((half_w - dist) / self.sigma) # (B, N, H, W)

        # Weight by existence confidence
        exist_mask = existence.unsqueeze(-1).unsqueeze(-1) # (B, N, 1, 1)
        active_lines = line_intensity * exist_mask

        # Composite lines: Soft Maximum intensity
        # White background (1.0) with dark stem lines (0.0)
        line_presence = 1.0 - active_lines
        rendered = torch.prod(line_presence, dim=1, keepdim=True) # (B, 1, H, W)

        return rendered
