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
from diffusion_based.models.helios_xml_parser import OrganNode3D


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

        y = torch.linspace(1, 0, image_size)  # flip Y so 0=bottom, 1=top matches OpenGL NDC
        x = torch.linspace(0, 1, image_size)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        self.register_buffer("grid", torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0))

        self.register_buffer("stem_color", torch.tensor([0.20, 0.30, 0.10], dtype=torch.float32))
        self.register_buffer("petiole_color", torch.tensor([0.22, 0.32, 0.08], dtype=torch.float32))
        self.register_buffer("leaf_color", torch.tensor([0.30, 0.50, 0.18], dtype=torch.float32))
        self.register_buffer("leaf_top_color", torch.tensor([0.38, 0.58, 0.25], dtype=torch.float32))
        self.register_buffer("bud_color", torch.tensor([0.80, 0.70, 0.15], dtype=torch.float32))
        self.register_buffer("bg_color", torch.tensor([0.12, 0.12, 0.10], dtype=torch.float32))
        # Helios-like soil/ground color (tan/brown)
        self.register_buffer("ground_color", torch.tensor([0.74, 0.67, 0.57], dtype=torch.float32))
        self.register_buffer("sun_dir", torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))
        # Fixed leaf mesh topology (matches _leaflet_local_mesh_torch in helios_geometry.py)
        leaf_faces = []
        Nx, Ny = 8, 6
        for j in range(Ny):
            for i in range(Nx):
                v0 = j * (Nx + 1) + i
                v1 = v0 + 1
                v2 = v0 + (Nx + 1) + 1
                v3 = v0 + (Nx + 1)
                leaf_faces.append([v0, v1, v2])
                leaf_faces.append([v0, v2, v3])
        self.register_buffer("leaf_faces", torch.tensor(leaf_faces, dtype=torch.int64))

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

    def _render_leaf_triangles_chunked(
        self,
        a_2d: torch.Tensor,
        b_2d: torch.Tensor,
        c_2d: torch.Tensor,
        Zc_tri: torch.Tensor,
        organs: np.ndarray,
        shade: torch.Tensor,
        chunk: int = 128,
    ) -> torch.Tensor:
        """Render all leaf triangles in small chunks to avoid OOM."""
        B, N, _ = a_2d.shape
        device = a_2d.device
        H = W = self.image_size
        rgb = torch.zeros(B, 3, H, W, device=device)
        acc_alpha = torch.zeros(B, 1, H, W, device=device)

        # Sort all triangles by depth once
        sorted_depth, sort_idx = torch.sort(Zc_tri, dim=1, descending=False)
        a_sorted = torch.gather(a_2d, 1, sort_idx.unsqueeze(-1).expand(-1, -1, 2))
        b_sorted = torch.gather(b_2d, 1, sort_idx.unsqueeze(-1).expand(-1, -1, 2))
        c_sorted = torch.gather(c_2d, 1, sort_idx.unsqueeze(-1).expand(-1, -1, 2))
        shade_sorted = torch.gather(shade, 1, sort_idx)
        organ_sorted = organs[sort_idx[0].cpu().numpy()]

        colors = [self.stem_color, self.petiole_color, self.leaf_color, self.bud_color]
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            alpha_chunk = self._fill_triangle(
                a_sorted[:, start:end],
                b_sorted[:, start:end],
                c_sorted[:, start:end],
            )
            for i in range(end - start):
                global_i = start + i
                a = alpha_chunk[:, i:i+1]
                color = colors[int(organ_sorted[global_i])]
                s = shade_sorted[0, global_i].item()
                shaded = color * s
                rgb = rgb + a * (1.0 - acc_alpha) * shaded.view(1, 3, 1, 1)
                acc_alpha = acc_alpha + a * (1.0 - acc_alpha)
                acc_alpha = torch.clamp(acc_alpha, 0.0, 1.0)
        return torch.cat([rgb, acc_alpha], dim=1)

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
        background: Optional[str] = None,
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
            n = torch.cross(b - a, c - a, dim=-1)
            n = F.normalize(n, dim=-1)
            ndotl = (n * sun).sum(dim=-1).abs()
            shade = 0.35 + 0.65 * ndotl.clamp(0, 1)
            leaf_img = self._render_leaf_triangles_chunked(
                a_2d, b_2d, c_2d, Zc_tri, organs, shade, chunk=128
            )
            if image is None:
                image = leaf_img
            else:
                image = self._composite_images(image, leaf_img, Zc_tri.min(dim=1, keepdim=True)[0])

        if bud_center_list:
            center = torch.from_numpy(np.array(bud_center_list, dtype=np.float32)).unsqueeze(0).to(device)
            radius = torch.from_numpy(np.array(bud_radius_list, dtype=np.float32)).unsqueeze(0).to(device)
            center_2d, Zc_bud, ppm = self.project(center, cam)
            z_abs = Zc_bud.abs().clamp(min=self.near_plane)
            r_norm = (radius * ppm / z_abs).clamp(min=0.001, max=0.05)
            # r_norm: (B, N) -> broadcast to (B, N, H, W)
            r_norm = r_norm.unsqueeze(-1).unsqueeze(-1)
            dx = self.grid[..., 0] - center_2d[:, :, 0].unsqueeze(-1).unsqueeze(-1)
            dy = self.grid[..., 1] - center_2d[:, :, 1].unsqueeze(-1).unsqueeze(-1)
            dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
            alpha = torch.sigmoid((r_norm - dist) / self.sigma)
            bud_img = self._composite_by_depth(alpha, np.full(len(bud_center_list), 3, dtype=np.int64), Zc_bud)
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
            bg = self.ground_color if background == "ground" else self.bg_color
            bg = bg.view(1, 3, 1, 1).to(device)
            rgb = rgb * covered + bg * (1.0 - covered)
            # soft cast shadow on the ground, matching Helios's shadowed look
            if background == "ground":
                all_pts_t = torch.from_numpy(all_pts).float().to(device)
                ground_z = float(all_pts_t[:, 2].min().item())
                shadow_mask = self._cast_ground_shadow(all_pts_t, sun, cam, ground_z)
                rgb = rgb * (1.0 - 0.55 * shadow_mask * (1.0 - covered)) + bg * (0.55 * shadow_mask * (1.0 - covered))
            image = torch.cat([rgb, covered], dim=1)

        img_np = image[0].permute(1, 2, 0).detach().cpu().numpy()
        return np.clip(img_np, 0.0, 1.0)

    def _cast_ground_shadow(
        self,
        points: torch.Tensor,
        sun: torch.Tensor,
        cam: Dict,
        ground_z: float,
    ) -> torch.Tensor:
        """Return a soft (1,1,H,W) shadow mask cast onto the z=ground_z plane.

        A plant point P shadows the ground where the ray from P along -sun first
        hits the ground plane. Points are only cast onto the ground (never occlude
        the plant). Vertical sun -> no horizontal offset, matching Helios.

        Uses chunked processing to avoid OOM when N is large (e.g. 5k-50k points).
        """
        n = torch.tensor([0.0, 0.0, 1.0], device=points.device, dtype=points.dtype)
        s_n = (sun * n).sum()
        if s_n <= 1e-6:
            return self.grid.new_zeros(1, 1, self.image_size, self.image_size)
        s_h = sun - n * s_n  # horizontal sun component
        # height above ground plane
        heights = (points @ n) - ground_z  # (N,)
        r = heights.clamp(min=0.0)
        t = r / s_n  # (N,)
        G = points - t.unsqueeze(1) * sun.unsqueeze(0)
        keep = r > 1e-4
        if not keep.any():
            return self.grid.new_zeros(1, 1, self.image_size, self.image_size)
        G = G[keep]

        # recenter at same target center as the camera used
        G_2d, Zc_g, ppm = self.project(G.unsqueeze(0), cam)
        z_abs = Zc_g.abs().clamp(min=self.near_plane)
        r_norm = (0.6 * ppm / z_abs).unsqueeze(-1).unsqueeze(-1)

        # Chunked shadow accumulation to avoid OOM from broadcasting (N,H,W)
        # At 256x256, each (N,1,H,W) tensor costs ~N*64KB. Chunk at 256 -> ~16MB per chunk.
        chunk = 256
        mask = self.grid.new_zeros(1, 1, self.image_size, self.image_size)
        N = G_2d.shape[1]
        for i in range(0, N, chunk):
            g2d_chunk = G_2d[:, i:i+chunk, :]
            rn_chunk = r_norm[:, i:i+chunk, :, :]
            gx = self.grid[..., 0] - g2d_chunk[:, :, 0].unsqueeze(-1).unsqueeze(-1)
            gy = self.grid[..., 1] - g2d_chunk[:, :, 1].unsqueeze(-1).unsqueeze(-1)
            dist = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
            alpha = torch.sigmoid((rn_chunk - dist) / self.sigma)
            mask = torch.maximum(mask, alpha.amax(dim=1, keepdim=True))
        return mask.clamp(0.0, 1.0)

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

    def _composite_by_depth_torch(
        self,
        alpha: torch.Tensor,      # (B, N, H, W)
        organs: torch.Tensor,     # (N,) or (B, N) int64
        depth: torch.Tensor,      # (B, N)
        organ_colors: Optional[torch.Tensor] = None,  # (4, 3)
        shade: Optional[torch.Tensor] = None,          # (B, N)
    ) -> torch.Tensor:
        """Composite alpha masks back-to-front by depth, fully differentiable."""
        if organ_colors is None:
            organ_colors = torch.stack([self.stem_color, self.petiole_color,
                                        self.leaf_color, self.bud_color], dim=0)  # (4, 3)
        B, N, H, W = alpha.shape
        device = alpha.device
        sorted_depth, sort_idx = torch.sort(depth, dim=1, descending=False)
        alpha_sorted = torch.gather(alpha, 1, sort_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W))

        # Differentiable organ color lookup
        if organs.dim() == 1:
            organ_sorted = torch.gather(
                organs.unsqueeze(0).expand(B, -1), 1, sort_idx
            )  # (B, N)
        else:
            organ_sorted = torch.gather(organs, 1, sort_idx)  # (B, N)
        colors_sorted = organ_colors[organ_sorted]  # (B, N, 3)

        rgb = torch.zeros(B, 3, H, W, device=device)
        acc_alpha = torch.zeros(B, 1, H, W, device=device)

        for i in range(N):
            a = alpha_sorted[:, i:i+1]  # (B, 1, H, W)
            color = colors_sorted[:, i]    # (B, 3)
            if shade is not None:
                s = shade.gather(1, sort_idx[:, i:i+1])  # (B, 1)
                shaded = color.unsqueeze(-1).unsqueeze(-1) * s.view(-1, 1, 1, 1)
            else:
                shaded = color.unsqueeze(-1).unsqueeze(-1)
            rgb = rgb + a * (1.0 - acc_alpha) * shaded
            acc_alpha = acc_alpha + a * (1.0 - acc_alpha)
            acc_alpha = torch.clamp(acc_alpha, 0.0, 1.0)

        return torch.cat([rgb, acc_alpha], dim=1)

    def _render_leaf_triangles_torch(
        self,
        a_2d: torch.Tensor,       # (B, N_tris, 2)
        b_2d: torch.Tensor,
        c_2d: torch.Tensor,
        Zc_tri: torch.Tensor,     # (B, N_tris)
        organs: torch.Tensor,     # (B, N_tris) int64
        shade: torch.Tensor,      # (B, N_tris)
        chunk: int = 128,
    ) -> torch.Tensor:
        """Render leaf triangles fully differentiable."""
        B, N, _ = a_2d.shape
        device = a_2d.device
        H = W = self.image_size

        sorted_depth, sort_idx = torch.sort(Zc_tri, dim=1, descending=False)
        a_sorted = torch.gather(a_2d, 1, sort_idx.unsqueeze(-1).expand(-1, -1, 2))
        b_sorted = torch.gather(b_2d, 1, sort_idx.unsqueeze(-1).expand(-1, -1, 2))
        c_sorted = torch.gather(c_2d, 1, sort_idx.unsqueeze(-1).expand(-1, -1, 2))
        shade_sorted = torch.gather(shade, 1, sort_idx)
        organ_sorted = torch.gather(organs, 1, sort_idx)  # (B, N)

        organ_colors = torch.stack([self.stem_color, self.petiole_color,
                                    self.leaf_color, self.bud_color], dim=0)  # (4, 3)
        colors_sorted = organ_colors[organ_sorted]  # (B, N, 3)

        rgb = torch.zeros(B, 3, H, W, device=device)
        acc_alpha = torch.zeros(B, 1, H, W, device=device)

        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            alpha_chunk = self._fill_triangle(
                a_sorted[:, start:end],
                b_sorted[:, start:end],
                c_sorted[:, start:end],
            )  # (B, chunk, H, W)
            for i in range(end - start):
                global_i = start + i
                a = alpha_chunk[:, i:i+1]
                color = colors_sorted[:, global_i]  # (B, 3)
                s = shade_sorted[:, global_i:global_i+1]  # (B, 1)
                shaded = color.unsqueeze(-1).unsqueeze(-1) * s.view(-1, 1, 1, 1)
                rgb = rgb + a * (1.0 - acc_alpha) * shaded
                acc_alpha = acc_alpha + a * (1.0 - acc_alpha)
                acc_alpha = torch.clamp(acc_alpha, 0.0, 1.0)
        return torch.cat([rgb, acc_alpha], dim=1)

    def render_torch_geometry(
        self,
        tube_verts: torch.Tensor,      # (B, N_tubes, 2, 3)
        tube_radii: torch.Tensor,      # (B, N_tubes, 2)
        tube_organs: torch.Tensor,     # (B, N_tubes) int64
        leaf_verts: torch.Tensor,      # (B, N_leaflets, V, 3)
        leaf_faces: torch.Tensor,      # (F, 3)
        leaf_organs: torch.Tensor,     # (B, N_leaflets) int64
        bud_centers: torch.Tensor,     # (B, N_buds, 3)
        bud_radii: torch.Tensor,       # (B, N_buds)
        bud_lengths: torch.Tensor,     # (B, N_buds)
        bud_organs: torch.Tensor,      # (B, N_buds) int64
        camera_height: float = 1.0,
        distance_from_center: float = 0.0,
        azimuth_deg: float = 0.0,
        hfov_deg: Optional[float] = None,
        target_center: Optional[torch.Tensor] = None,
        sun_dir: Optional[torch.Tensor] = None,
        focus_plant: bool = False,
        background: Optional[str] = None,
    ) -> torch.Tensor:
        """Render explicit torch geometry and return RGBA (B, 4, H, W)."""
        device = next(self.buffers()).device
        B = tube_verts.shape[0]

        # Collect all points for camera centering
        all_pts = []
        if tube_verts.numel():
            all_pts.append(tube_verts.reshape(B, -1, 3))
        if leaf_verts.numel():
            all_pts.append(leaf_verts.reshape(B, -1, 3))
        if bud_centers.numel():
            all_pts.append(bud_centers)
        if all_pts:
            all_pts = torch.cat(all_pts, dim=1)
        else:
            all_pts = torch.zeros(B, 1, 3, device=device)

        if target_center is None:
            target_center = (all_pts.min(dim=1)[0] + all_pts.max(dim=1)[0]) / 2.0
        target_center = target_center.to(device).view(B, 3)

        if focus_plant:
            hfov_deg = float(self.recompute_focus_plant_hfov(all_pts, camera_height, margin=1.05)[0])

        cam = self._compute_camera(camera_height, distance_from_center, azimuth_deg, target_center, hfov_deg)

        sun = self.sun_dir.to(device)
        if sun_dir is not None:
            sun = F.normalize(sun_dir.to(device), dim=-1)

        image: Optional[torch.Tensor] = None

        # ------------------------------------------------------------------
        # Tubes - filter non-tube segments
        # ------------------------------------------------------------------
        if tube_verts.numel():
            tube_mask = (tube_organs == 0) | (tube_organs == 1)  # INTERNODE or PETIOLE
            keep_mask = tube_mask[0]  # (N,)
            if keep_mask.any():
                p1 = tube_verts[:, keep_mask, 0, :]  # (B, N_tubes, 3)
                p2 = tube_verts[:, keep_mask, 1, :]
                mid = (p1 + p2) / 2.0
                w_m = (tube_radii[:, keep_mask, 0] + tube_radii[:, keep_mask, 1]) / 2.0
                organs_k = tube_organs[:, keep_mask]

                p1_2d, _, ppm = self.project(p1, cam)
                p2_2d, _, _ = self.project(p2, cam)
                _, Zc_mid, _ = self.project(mid, cam)

                z_abs = Zc_mid.abs().clamp(min=self.near_plane)
                widths = (w_m * 2.0 * ppm / z_abs).clamp(min=0.0005, max=0.08)
                alpha = self._soft_line(p1_2d, p2_2d, widths)  # (B, N, H, W)

                axis_norm = F.normalize(p2 - p1, dim=-1)
                ndotl = (axis_norm * sun.view(1, 1, 3)).sum(dim=-1).abs()
                shade = 0.5 + 0.5 * ndotl.clamp(0, 1)
                image = self._composite_by_depth_torch(alpha, organs_k, Zc_mid, shade=shade)

        # ------------------------------------------------------------------
        # Leaves (triangulate) - filter non-leaf triangles
        # ------------------------------------------------------------------
        if leaf_verts.numel():
            Fm = leaf_faces.shape[0]
            # Expand leaf_faces to all leaflets
            a = leaf_verts[:, :, leaf_faces[:, 0], :]  # (B, N_leaflets, F, 3)
            b = leaf_verts[:, :, leaf_faces[:, 1], :]
            c = leaf_verts[:, :, leaf_faces[:, 2], :]

            # Flatten batch+leaflets+faces
            a = a.reshape(B, -1, 3)  # (B, N_leaflets*F, 3)
            b = b.reshape(B, -1, 3)
            c = c.reshape(B, -1, 3)
            tri_organs = leaf_organs.unsqueeze(2).expand(-1, -1, Fm).reshape(B, -1)

            # Filter: keep only LEAF organ triangles
            leaf_mask = (tri_organs == 2)  # (B, N*F)
            # Flatten mask to 1D and gather
            keep_mask = leaf_mask[0]  # (N*F,)
            if keep_mask.any():
                a = a[:, keep_mask]
                b = b[:, keep_mask]
                c = c[:, keep_mask]
                tri_organs = tri_organs[:, keep_mask]

                tri_center = (a + b + c) / 3.0
                a_2d, _, _ = self.project(a, cam)
                b_2d, _, _ = self.project(b, cam)
                c_2d, _, _ = self.project(c, cam)
                _, Zc_tri, _ = self.project(tri_center, cam)

                n = torch.cross(b - a, c - a, dim=-1)
                n = F.normalize(n, dim=-1)
                ndotl = (n * sun.view(1, 1, 3)).sum(dim=-1).abs()
                shade = 0.35 + 0.65 * ndotl.clamp(0, 1)

                leaf_img = self._render_leaf_triangles_torch(
                    a_2d, b_2d, c_2d, Zc_tri, tri_organs, shade, chunk=128
                )
                if image is None:
                    image = leaf_img
                else:
                    image = self._composite_images(image, leaf_img, Zc_tri.min(dim=1, keepdim=True)[0])

        # ------------------------------------------------------------------
        # Buds (ellipsoid approximated as sphere in screen space) - filter non-buds
        # ------------------------------------------------------------------
        if bud_centers.numel():
            bud_mask = (bud_organs == 3) & (bud_radii > 1e-4)  # (B, N)
            keep_mask = bud_mask[0]  # (N,)
            if keep_mask.any():
                centers_k = bud_centers[:, keep_mask]
                radii_k = bud_radii[:, keep_mask]
                organs_k = bud_organs[:, keep_mask]
                
                center_2d, Zc_bud, ppm = self.project(centers_k, cam)
                z_abs = Zc_bud.abs().clamp(min=self.near_plane)
                r_norm = (radii_k * ppm / z_abs).clamp(min=0.001, max=0.05)
                r_norm = r_norm.unsqueeze(-1).unsqueeze(-1)
                dx = self.grid[..., 0] - center_2d[:, :, 0].unsqueeze(-1).unsqueeze(-1)
                dy = self.grid[..., 1] - center_2d[:, :, 1].unsqueeze(-1).unsqueeze(-1)
                dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
                alpha = torch.sigmoid((r_norm - dist) / self.sigma)
                bud_img = self._composite_by_depth_torch(alpha, organs_k, Zc_bud)
                if image is None:
                    image = bud_img
                else:
                    image = self._composite_images(image, bud_img, Zc_bud.min(dim=1, keepdim=True)[0])

        # ------------------------------------------------------------------
        # Background
        # ------------------------------------------------------------------
        if image is None:
            image = self.bg_color.view(1, 3, 1, 1).expand(B, 3, self.image_size, self.image_size).to(device)
            image = torch.cat([image, torch.zeros(B, 1, self.image_size, self.image_size, device=device)], dim=1)
        else:
            covered = image[:, 3:4]
            rgb = image[:, :3]
            bg = self.ground_color if background == "ground" else self.bg_color
            bg = bg.view(1, 3, 1, 1).to(device)
            rgb = rgb * covered + bg * (1.0 - covered)
            if background == "ground":
                ground_z = float(all_pts[..., 2].min().item())
                shadow_mask = self._cast_ground_shadow(all_pts.reshape(-1, 3), sun, cam, ground_z)
                rgb = rgb * (1.0 - 0.55 * shadow_mask * (1.0 - covered)) + bg * (0.55 * shadow_mask * (1.0 - covered))
            image = torch.cat([rgb, covered], dim=1)

        return image

    def _composite_images(self, base: torch.Tensor, overlay: torch.Tensor, overlay_depth: torch.Tensor) -> torch.Tensor:
        """Composite overlay RGBA over base RGBA by per-pixel depth."""
        # base: (B, 4, H, W), overlay: (B, 4, H, W)
        base_rgb = base[:, :3]
        base_a = base[:, 3:4]
        over_rgb = overlay[:, :3]
        over_a = overlay[:, 3:4]
        out_a = over_a + base_a * (1.0 - over_a)
        out_rgb = over_rgb * over_a + base_rgb * base_a * (1.0 - over_a)
        # Avoid 0/0 NaN in backward: clamp denominator, keep 0 rgb where out_a is 0.
        out_rgb = out_rgb / out_a.clamp(min=1e-6)
        out_rgb = torch.where(out_a > 1e-6, out_rgb, torch.zeros_like(out_rgb))
        return torch.cat([out_rgb, out_a], dim=1)
