"""Texture-free differentiable 2D rasterizer for explicit Helios geometry.

Given explicit 3D geometry (tubes + leaf meshes + ellipsoids) produced by
helios_geometry.py, this module renders a flat-shaded image using the same camera
model as Helios's OpenGL visualizer. It is intentionally simple (no textures,
no ground) so gradients flow while the structure matches Helios.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_based.models.helios_geometry import HeliosEllipsoid, HeliosLeaflet, HeliosTube


class HeliosGeometryRasterizer(nn.Module):
    """Differentiable rasterizer for explicit Helios-style plant geometry."""

    def __init__(
        self,
        image_size: int = 512,
        fov_deg: Optional[float] = None,
        near_plane: float = 1e-4,
        sigma: float = 0.0015,
        leaf_sigma: float = 0.002,
    ):
        super().__init__()
        self.image_size = image_size
        self.fov_deg = fov_deg
        self.near_plane = near_plane
        self.sigma = sigma
        self.leaf_sigma = leaf_sigma

        y = torch.linspace(0, 1, image_size)
        x = torch.linspace(0, 1, image_size)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        self.register_buffer("grid", torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0))

        self.register_buffer("stem_color", torch.tensor([0.20, 0.30, 0.10], dtype=torch.float32))
        self.register_buffer("petiole_color", torch.tensor([0.22, 0.32, 0.08], dtype=torch.float32))
        self.register_buffer("leaf_color", torch.tensor([0.30, 0.50, 0.18], dtype=torch.float32))
        self.register_buffer("leaf_top_color", torch.tensor([0.38, 0.58, 0.25], dtype=torch.float32))
        self.register_buffer("bud_color", torch.tensor([0.80, 0.70, 0.15], dtype=torch.float32))
        self.register_buffer("bg_color", torch.tensor([0.12, 0.12, 0.10], dtype=torch.float32))
        self.register_buffer("sun_dir", torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))

    def _compute_camera(
        self,
        camera_height: float,
        distance_from_center: float,
        azimuth_deg: float,
        lookat: torch.Tensor,  # (B, 3)
        hfov_deg: Optional[float] = None,
    ) -> Dict:
        """Build camera parameters matching Helios visualizer."""
        if hfov_deg is None:
            hfov_deg = self.fov_deg if self.fov_deg is not None else 45.0
        return {
            "camera_height": camera_height,
            "distance_from_center": distance_from_center,
            "azimuth_deg": azimuth_deg,
            "lookat": lookat,
            "hfov_deg": hfov_deg,
            "image_width": self.image_size,
            "image_height": self.image_size,
        }

    def recompute_focus_plant_hfov(
        self,
        all_points: torch.Tensor,  # (B, N, 3)
        camera_height: torch.Tensor,  # (B,) or float
        margin: float = 1.05,
    ) -> torch.Tensor:
        """Recompute HFOV to fit the XY bounding box of plant points (Helios --focus-plant).

        Matches C++ calculateFOV(span, camera_height) = 2*atan(span/(2*h)).
        Returns HFOV in degrees as a (B,) tensor.
        """
        B = all_points.shape[0]
        device = all_points.device
        if isinstance(camera_height, (int, float, np.floating, np.integer)):
            cam_h = torch.full((B,), float(camera_height), device=device)
        else:
            cam_h = camera_height.to(device).view(B)

        min_xy = all_points[..., :2].min(dim=1)[0]  # (B, 2)
        max_xy = all_points[..., :2].max(dim=1)[0]
        span_xy = (max_xy - min_xy) * margin  # (B, 2)
        max_span = span_xy.max(dim=1)[0]  # (B,)
        hfov_rad = 2.0 * torch.atan(max_span / (2.0 * cam_h.clamp(min=1e-6)))
        return hfov_rad * 180.0 / math.pi

    def project(
        self,
        points: torch.Tensor,  # (B, N, 3)
        cam: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Project 3D points to normalized screen coords [0,1] and return camera-depth."""
        B, N, _ = points.shape
        device = points.device
        az = math.radians(cam["azimuth_deg"])
        dist = cam["distance_from_center"]
        cam_h = cam["camera_height"]
        lookat = cam["lookat"]

        cam_pos = torch.zeros(B, 3, device=device)
        cam_pos[:, 0] = lookat[:, 0] + dist * math.sin(az)
        cam_pos[:, 1] = lookat[:, 1] - dist * math.cos(az)
        cam_pos[:, 2] = lookat[:, 2] + cam_h

        fwd = F.normalize(lookat - cam_pos, dim=-1)
        world_up = torch.tensor([0.0, 0.0, 1.0], device=device).expand(B, 3)
        right = torch.cross(fwd, world_up, dim=-1)
        rn = torch.norm(right, dim=-1, keepdim=True)
        fallback = torch.tensor([1.0, 0.0, 0.0], device=device).expand(B, 3)
        right = torch.where(rn < 1e-6, fallback, right / (rn + 1e-8))
        up = F.normalize(torch.cross(right, fwd, dim=-1), dim=-1)

        rel = points - cam_pos.unsqueeze(1)
        Xc = (rel * right.unsqueeze(1)).sum(dim=-1)
        Yc = (rel * up.unsqueeze(1)).sum(dim=-1)
        Zc = (rel * (-fwd.unsqueeze(1))).sum(dim=-1)

        fov_y = math.radians(cam["hfov_deg"])
        focal_y = 1.0 / math.tan(fov_y / 2.0)
        aspect = cam["image_width"] / max(cam["image_height"], 1)
        focal_x = focal_y / aspect

        Zc_clip = torch.clamp(Zc, max=-self.near_plane)
        x_ndc = (focal_x * Xc) / (-Zc_clip)
        y_ndc = (focal_y * Yc) / (-Zc_clip)
        px = x_ndc * 0.5 + 0.5
        py = y_ndc * 0.5 + 0.5
        return torch.stack([px, py], dim=-1), Zc, focal_y * 0.5

    def _soft_line(self, p1: torch.Tensor, p2: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
        """(B, N, H, W) soft line segment mask. width in normalized units."""
        v = p2 - p1
        v_sq = (v ** 2).sum(dim=-1, keepdim=True).unsqueeze(2).unsqueeze(2) + 1e-8
        w = self.grid - p1.unsqueeze(2).unsqueeze(2)
        v_exp = v.unsqueeze(2).unsqueeze(2)
        c = torch.clamp((w * v_exp).sum(dim=-1, keepdim=True) / v_sq, 0.0, 1.0)
        proj = p1.unsqueeze(2).unsqueeze(2) + c * v_exp
        dist = torch.norm(self.grid - proj, dim=-1)
        half_w = (width / 2.0).unsqueeze(-1).unsqueeze(-1)
        return torch.sigmoid((half_w - dist) / self.sigma)

    def _fill_triangle(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        """Soft edge-function triangle mask for a batch of triangles (B, N, 2).

        Args:
            a, b, c: vertex screen positions of shape (B, N, 2).

        Returns:
            Alpha mask of shape (B, N, H, W).
        """
        # self.grid is (1, 1, H, W, 2)
        gx = self.grid[..., 0]  # (1, 1, H, W)
        gy = self.grid[..., 1]

        # Broadcast triangle vertices to (B, N, 1, 1)
        ax = a[..., 0:1].unsqueeze(-1).unsqueeze(-1)
        ay = a[..., 1:2].unsqueeze(-1).unsqueeze(-1)
        bx = b[..., 0:1].unsqueeze(-1).unsqueeze(-1)
        by = b[..., 1:2].unsqueeze(-1).unsqueeze(-1)
        cx = c[..., 0:1].unsqueeze(-1).unsqueeze(-1)
        cy = c[..., 1:2].unsqueeze(-1).unsqueeze(-1)

        def edge(p0x, p0y, p1x, p1y):
            return (p1x - p0x) * (gy - p0y) - (p1y - p0y) * (gx - p0x)

        e0 = edge(ax, ay, bx, by)
        e1 = edge(bx, by, cx, cy)
        e2 = edge(cx, cy, ax, ay)

        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        sign = torch.where(area >= 0, 1.0, -1.0)
        inside = torch.minimum(torch.minimum(e0 * sign, e1 * sign), e2 * sign)
        # Normalize by triangle area so softness is in barycentric/distance units
        inside = inside / (area.abs() + 1e-8)
        # Squeeze the broadcasted singleton dimension added by vertex coordinate slicing
        return torch.sigmoid(inside / self.leaf_sigma).squeeze(2)

    def render_numpy_geometry(
        self,
        tubes: List[HeliosTube],
        leaflets: List[HeliosLeaflet],
        ellipsoids: List[HeliosEllipsoid],
        camera_height: float = 1.0,
        distance_from_center: float = 0.0,
        azimuth_deg: float = 0.0,
        hfov_deg: Optional[float] = None,
        target_center: Optional[np.ndarray] = None,
        sun_dir: Optional[np.ndarray] = None,
        focus_plant: bool = False,
    ) -> np.ndarray:
        """Render explicit numpy geometry and return an RGB numpy image."""
        device = next(self.buffers()).device

        # Gather all 3D points to estimate target center
        all_pts = []
        for t in tubes:
            all_pts.append(t.vertices)
        for lf in leaflets:
            all_pts.append(lf.vertices)
        for e in ellipsoids:
            all_pts.append(e.center.reshape(1, 3))
        if all_pts:
            all_pts = np.concatenate(all_pts, axis=0)
        else:
            all_pts = np.zeros((1, 3))

        if target_center is None:
            target_center = (all_pts.min(0) + all_pts.max(0)) / 2.0
        target_center_t = torch.from_numpy(target_center).float().unsqueeze(0).to(device)

        if focus_plant:
            all_pts_t = torch.from_numpy(all_pts).float().unsqueeze(0).to(device)
            hfov_deg = float(self.recompute_focus_plant_hfov(all_pts_t, camera_height, margin=1.05)[0])

        cam = self._compute_camera(camera_height, distance_from_center, azimuth_deg, target_center_t, hfov_deg)

        # Prepare buffers
        tube_p1_list, tube_p2_list, tube_w_list, tube_d_list, tube_c_list = [], [], [], [], []
        for tube in tubes:
            if tube.vertices.shape[0] < 2:
                continue
            for i in range(tube.vertices.shape[0] - 1):
                p0 = tube.vertices[i]
                p1 = tube.vertices[i + 1]
                r0 = tube.radii[i]
                r1 = tube.radii[i + 1]
                mid = (p0 + p1) / 2.0
                # project mid to get depth and ppm
                tube_p1_list.append(p0)
                tube_p2_list.append(p1)
                tube_w_list.append((r0 + r1) / 2.0)
                tube_d_list.append(mid)
                tube_c_list.append(tube.organ)

        leaf_tri_a, leaf_tri_b, leaf_tri_c, leaf_tri_d, leaf_tri_c_list = [], [], [], [], []
        for lf in leaflets:
            if lf.faces.shape[0] == 0:
                continue
            for f in lf.faces:
                a = lf.vertices[f[0]]
                b = lf.vertices[f[1]]
                c = lf.vertices[f[2]]
                tri_center = (a + b + c) / 3.0
                leaf_tri_a.append(a)
                leaf_tri_b.append(b)
                leaf_tri_c.append(c)
                leaf_tri_d.append(tri_center)
                leaf_tri_c_list.append(lf.organ)

        bud_center_list, bud_radius_list, bud_depth_list = [], [], []
        for e in ellipsoids:
            bud_center_list.append(e.center)
            bud_radius_list.append(e.radius)
            bud_depth_list.append(e.center)

        # Render in chunks to keep memory reasonable
        image = None
        B = 1

        sun = self.sun_dir.to(device)
        if sun_dir is not None:
            sun = torch.from_numpy(np.array(sun_dir, dtype=np.float32)).to(device)
            sun = F.normalize(sun, dim=-1)

        if tube_p1_list:
            p1 = torch.from_numpy(np.array(tube_p1_list, dtype=np.float32)).unsqueeze(0).to(device)
            p2 = torch.from_numpy(np.array(tube_p2_list, dtype=np.float32)).unsqueeze(0).to(device)
            w_m = torch.from_numpy(np.array(tube_w_list, dtype=np.float32)).unsqueeze(0).to(device)
            depths = torch.from_numpy(np.array(tube_d_list, dtype=np.float32)).unsqueeze(0).to(device)
            organs = np.array(tube_c_list, dtype=np.int64)
            p1_2d, _, ppm = self.project(p1, cam)
            p2_2d, _, _ = self.project(p2, cam)
            _, Zc_mid, _ = self.project(depths, cam)
            # width in normalized units at segment mid depth
            z_abs = Zc_mid.abs().clamp(min=self.near_plane)
            widths = (w_m * 2.0 * ppm / z_abs).clamp(min=0.0005, max=0.08)
            alpha = self._soft_line(p1_2d, p2_2d, widths)  # (1, N, H, W)
            # tube shading: segments approximate cylinders, use axis as normal proxy
            axis_norm = F.normalize(p2 - p1, dim=-1)
            ndotl = (axis_norm * sun).sum(dim=-1).abs()
            shade = 0.5 + 0.5 * ndotl.clamp(0, 1)
            image = self._composite_by_depth(alpha, organs, Zc_mid, shade=shade)

        if leaf_tri_a:
            a = torch.from_numpy(np.array(leaf_tri_a, dtype=np.float32)).unsqueeze(0).to(device)
            b = torch.from_numpy(np.array(leaf_tri_b, dtype=np.float32)).unsqueeze(0).to(device)
            c = torch.from_numpy(np.array(leaf_tri_c, dtype=np.float32)).unsqueeze(0).to(device)
            d = torch.from_numpy(np.array(leaf_tri_d, dtype=np.float32)).unsqueeze(0).to(device)
            organs = np.array(leaf_tri_c_list, dtype=np.int64)
            a_2d, _, _ = self.project(a, cam)
            b_2d, _, _ = self.project(b, cam)
            c_2d, _, _ = self.project(c, cam)
            _, Zc_tri, _ = self.project(d, cam)
            alpha = self._fill_triangle(a_2d, b_2d, c_2d)
            # leaf diffuse shading from face normal (double-sided leaves)
            n = torch.cross(b - a, c - a, dim=-1)
            n = F.normalize(n, dim=-1)
            ndotl = (n * sun).sum(dim=-1).abs()
            shade = 0.35 + 0.65 * ndotl.clamp(0, 1)
            leaf_img = self._composite_by_depth(
                alpha, organs, Zc_tri,
                organ_colors=[self.stem_color, self.petiole_color, self.leaf_color, self.bud_color],
                shade=shade,
            )
            if image is None:
                image = leaf_img
            else:
                # composite leaf over tubes by depth
                image = self._composite_images(image, leaf_img, Zc_tri.min(dim=1, keepdim=True)[0])

        if bud_center_list:
            center = torch.from_numpy(np.array(bud_center_list, dtype=np.float32)).unsqueeze(0).to(device)
            radius = torch.from_numpy(np.array(bud_radius_list, dtype=np.float32)).unsqueeze(0).to(device)
            center_2d, Zc_bud, ppm = self.project(center, cam)
            z_abs = Zc_bud.abs().clamp(min=self.near_plane)
            r_norm = (radius * ppm / z_abs).clamp(min=0.001, max=0.05)
            dx = self.grid[..., 0] - center_2d[:, :, 0].unsqueeze(-1).unsqueeze(-1)
            dy = self.grid[..., 1] - center_2d[:, :, 1].unsqueeze(-1).unsqueeze(-1)
            dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
            alpha = torch.sigmoid((r_norm - dist) / self.sigma)
            bud_img = self._composite_by_depth(alpha, np.full(len(bud_center_list), 3, dtype=np.int64), Zc_bud, organ_colors=[self.bud_color])
            if image is None:
                image = bud_img
            else:
                image = self._composite_images(image, bud_img, Zc_bud.min(dim=1, keepdim=True)[0])

        if image is None:
            image = self.bg_color.view(1, 3, 1, 1).expand(1, 3, self.image_size, self.image_size).to(device)
        else:
            # blend with background
            covered = image[:, 3:4]
            rgb = image[:, :3]
            bg = self.bg_color.view(1, 3, 1, 1).to(device)
            image = rgb * covered + bg * (1.0 - covered)

        img_np = image[0].permute(1, 2, 0).detach().cpu().numpy()
        return np.clip(img_np, 0.0, 1.0)

    def _composite_by_depth(
        self,
        alpha: torch.Tensor,  # (B, N, H, W)
        organs: np.ndarray,     # (N,)
        depth: torch.Tensor,    # (B, N)
        organ_colors: Optional[List[torch.Tensor]] = None,
        shade: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Composite alpha masks back-to-front by depth, return (B, 4, H, W) RGBA."""
        if organ_colors is None:
            organ_colors = [self.stem_color, self.petiole_color, self.leaf_color, self.bud_color]
        B, N, H, W = alpha.shape
        device = alpha.device
        sorted_depth, sort_idx = torch.sort(depth, dim=1, descending=False)  # back to front
        alpha_sorted = torch.gather(alpha, 1,
            sort_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W))
        organ_sorted = organs[sort_idx[0].cpu().numpy()]

        rgb = torch.zeros(B, 3, H, W, device=device)
        acc_alpha = torch.zeros(B, 1, H, W, device=device)

        for i in range(N):
            a = alpha_sorted[:, i:i+1]
            color = organ_colors[int(organ_sorted[i])]
            if shade is not None:
                s = shade[0, sort_idx[0, i]].item()
                shaded = color * s
            else:
                shaded = color
            # front-to-back compositing with accumulated alpha
            rgb = rgb + a * (1.0 - acc_alpha) * shaded.view(1, 3, 1, 1)
            acc_alpha = acc_alpha + a * (1.0 - acc_alpha)
            acc_alpha = torch.clamp(acc_alpha, 0.0, 1.0)

        return torch.cat([rgb, acc_alpha], dim=1)

    def _composite_images(self, base: torch.Tensor, overlay: torch.Tensor, overlay_depth: torch.Tensor) -> torch.Tensor:
        """Composite overlay RGBA over base RGBA by per-pixel depth."""
        # base: (B, 4, H, W), overlay: (B, 4, H, W)
        base_rgb = base[:, :3]
        base_a = base[:, 3:4]
        over_rgb = overlay[:, :3]
        over_a = overlay[:, 3:4]
        out_a = over_a + base_a * (1.0 - over_a)
        out_rgb = over_rgb * over_a + base_rgb * base_a * (1.0 - over_a)
        out_rgb = torch.where(out_a > 1e-6, out_rgb / out_a, out_rgb)
        return torch.cat([out_rgb, out_a], dim=1)
