"""Differentiable 3D Plant Renderer matching Helios PlantArchitecture output.

Key improvements for Helios alignment:
  1. Perspective projection identical to Helios (glm::infinitePerspective + glm::lookAt).
  2. Camera parameters loaded from Helios camera/params JSON.
  3. Fixed camera mode and focus-plant mode both supported.
  4. Trifoliate compound leaf geometry matching Helios C++ PlantArchitecture.
  5. Organ-aware colors with sun-direction Phong shading.
  6. Depth-aware alpha compositing for correct occlusion.
  7. Optional ground/soil background matching Helios scene.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict


class DifferentiablePlantRenderer3D(nn.Module):
    """Organ-typed differentiable 3D plant renderer matching Helios output."""

    INTERNODE = 0
    PETIOLE = 1
    LEAF = 2
    FLORAL_BUD = 3

    def __init__(
        self,
        image_size: int = 256,
        fov_deg: Optional[float] = None,       # If None, computed from ground_width & camera_height
        cam_height: float = 1.0,
        ground_width: float = 1.5,
        ground_height: Optional[float] = None,
        sigma: float = 0.002,
        leaf_sigma: float = 0.0003,
        bud_sigma: float = 0.002,
        near_plane: float = 0.001,
    ):
        super().__init__()
        self.image_size = image_size
        self.fov_deg = fov_deg
        self.cam_height = cam_height
        self.ground_width = ground_width
        self.ground_height = ground_height if ground_height is not None else ground_width
        self.sigma = sigma
        self.leaf_sigma = leaf_sigma
        self.bud_sigma = bud_sigma
        self.near_plane = near_plane

        # Precompute normalized grid [0, 1] x [0, 1]
        y = torch.linspace(0, 1, image_size)
        x = torch.linspace(0, 1, image_size)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)
        self.register_buffer("grid", grid.unsqueeze(0).unsqueeze(0))  # (1, 1, H, W, 2)

        # Helios-matched organ colors (sRGB, tuned to cowpea DAP10 reference)
        self.register_buffer("stem_color", torch.tensor([0.15, 0.20, 0.10], dtype=torch.float32))
        self.register_buffer("petiole_color", torch.tensor([0.20, 0.25, 0.06], dtype=torch.float32))
        self.register_buffer("leaf_color", torch.tensor([0.35, 0.55, 0.22], dtype=torch.float32))
        self.register_buffer("vein_color", torch.tensor([0.45, 0.65, 0.28], dtype=torch.float32))
        self.register_buffer("leaf_border", torch.tensor([0.18, 0.35, 0.10], dtype=torch.float32))
        self.register_buffer("bud_color", torch.tensor([0.82, 0.75, 0.20], dtype=torch.float32))

        # Default soil/ground color (approximate from Helios dirt texture)
        self.register_buffer("ground_color", torch.tensor([0.74, 0.67, 0.57], dtype=torch.float32))

    # ── Camera / Projection ───────────────────────────────────────────────

    def _compute_camera_params(
        self,
        camera_params: Optional[Dict],
        focus_plant: bool,
        plant_extent: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Build final camera parameters matching Helios."""
        cp = camera_params or {}
        out = {
            'camera_height': float(cp.get('camera_height', self.cam_height)),
            'ground_width': float(cp.get('ground_width', self.ground_width)),
            'ground_height': float(cp.get('ground_height', cp.get('ground_width', self.ground_height))),
            'distance_from_center': float(cp.get('distance_from_center', 0.01)),
            'azimuth_deg': float(cp.get('azimuth_deg', 0.0)),
            'lookat_offset_x': float(cp.get('lookat_offset_x', 0.0)),
            'lookat_offset_y': float(cp.get('lookat_offset_y', 0.0)),
            'lookat_offset_z': float(cp.get('lookat_offset_z', 0.0)),
            'image_width': int(cp.get('image_width', self.image_size)),
            'image_height': int(cp.get('image_height', self.image_size)),
            'sun_elevation_deg': float(cp.get('sun_elevation_deg', 45.0)),
            'sun_azimuth_deg': float(cp.get('sun_azimuth_deg', 180.0)),
            'canopy_center': cp.get('canopy_center', None),
        }

        # Default canopy center from active plant if not provided
        if out['canopy_center'] is None:
            out['canopy_center'] = torch.zeros(3)

        # Helios focus-plant mode: recenter on plant bbox and reduce HFOV to fit max_span
        if focus_plant and plant_extent is not None:
            margin = 1.25  # Helios uses 25% margin
            max_span = plant_extent * margin
            cam_h = max(out['camera_height'] - out['lookat_offset_z'], 0.1)
            # Match Helios calculateFOV (per-batch)
            out['hfov_deg'] = torch.where(
                max_span > 0,
                (180.0 / math.pi) * (2.0 * torch.atan(max_span / (2.0 * cam_h))),
                torch.full_like(max_span, 45.0)
            )
            # Helios clamps FOV between 8 and 90 degrees
            out['hfov_deg'] = torch.clamp(out['hfov_deg'], 8.0, 90.0)
            out['distance_from_center'] = 0.0
            out['lookat_offset_x'] = 0.0
            out['lookat_offset_y'] = 0.0
            out['lookat_offset_z'] = 0.0
        else:
            if self.fov_deg is not None:
                out['hfov_deg'] = float(self.fov_deg)
            else:
                cam_h = max(out['camera_height'], 0.1)
                out['hfov_deg'] = math.degrees(2.0 * math.atan(out['ground_width'] / (2.0 * cam_h)))

        return out

    def project_3d_points(
        self,
        points_3d: torch.Tensor,                # (B, N, 3) or (B, N, K, 3)
        cam: Dict,
        target_center: torch.Tensor,            # (B, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project 3D points to normalized 2D screen coordinates [0,1] matching Helios OpenGL.

        Returns:
            points_2d: (B, N, 2) or (B, N, K, 2) normalized in [0, 1]
            Zc:        (B, N) or (B, N, K) camera-space depth (negative = in front)
            ppm:       normalized units per meter at unit camera-space depth
        """
        orig_shape = points_3d.shape
        if len(orig_shape) == 4:
            B, N, K, _ = orig_shape
            pts_flat = points_3d.view(B, N * K, 3)
        else:
            B, N, _ = orig_shape
            K = 1
            pts_flat = points_3d

        device = points_3d.device
        az_rad = math.radians(cam['azimuth_deg'])
        dist = cam['distance_from_center']
        cam_h = cam['camera_height']

        # Camera eye position (Helios main.cpp formula)
        cam_pos = torch.zeros(B, 3, device=device)
        cam_pos[:, 0] = target_center[:, 0] + dist * math.sin(az_rad)
        cam_pos[:, 1] = target_center[:, 1] - dist * math.cos(az_rad)
        cam_pos[:, 2] = target_center[:, 2] + cam_h

        # LookAt point with offset
        lookat = target_center.clone()
        lookat[:, 0] += cam['lookat_offset_x']
        lookat[:, 1] += cam['lookat_offset_y']
        lookat[:, 2] += cam['lookat_offset_z']

        # Forward (camera looks down -Z in camera space)
        fwd = F.normalize(lookat - cam_pos, dim=-1)

        # World up is +Z; right = fwd x up, then up_cam = right x fwd
        world_up = torch.tensor([0.0, 0.0, 1.0], device=device).expand(B, 3)
        right = torch.cross(fwd, world_up, dim=-1)
        right_norm = torch.norm(right, dim=-1, keepdim=True)
        fallback_right = torch.tensor([1.0, 0.0, 0.0], device=device).expand(B, 3)
        right = torch.where(right_norm < 1e-6, fallback_right, right / (right_norm + 1e-8))
        up = torch.cross(right, fwd, dim=-1)
        up = F.normalize(up, dim=-1)

        # Transform to camera space: P_cam = [dot(P-eye, right), dot(P-eye, up), dot(P-eye, -fwd)]
        pts_rel = pts_flat - cam_pos.unsqueeze(1)
        Xc = (pts_rel * right.unsqueeze(1)).sum(dim=-1)
        Yc = (pts_rel * up.unsqueeze(1)).sum(dim=-1)
        Zc = (pts_rel * (-fwd.unsqueeze(1))).sum(dim=-1)

        # Infinite perspective projection matching glm::infinitePerspective
        if isinstance(cam['hfov_deg'], torch.Tensor):
            fov_y = (math.pi / 180.0) * cam['hfov_deg']
            focal_y = (1.0 / torch.tan(fov_y / 2.0)).unsqueeze(-1)  # (B, 1)
        else:
            fov_y = math.radians(cam['hfov_deg'])
            focal_y = 1.0 / math.tan(fov_y / 2.0)
        aspect = cam['image_width'] / max(cam['image_height'], 1)
        focal_x = focal_y / aspect

        # Points in front of camera have Zc < 0; clip to near plane
        Zc_clip = torch.clamp(Zc, max=-self.near_plane)

        x_ndc = (focal_x * Xc) / (-Zc_clip)
        y_ndc = (focal_y * Yc) / (-Zc_clip)

        # NDC [-1, 1] -> normalized [0, 1] screen coordinates
        px = x_ndc * 0.5 + 0.5
        py = y_ndc * 0.5 + 0.5

        # Normalized units per meter at unit camera-space depth (used to scale world widths)
        ppm = focal_y * 0.5

        if len(orig_shape) == 4:
            px = px.view(B, N, K)
            py = py.view(B, N, K)
            Zc = Zc.view(B, N, K)

        points_2d = torch.stack([px, py], dim=-1)
        return points_2d, Zc, ppm

    # ── 3D Leaf Quad Projection & Rasterization ─────────────────────────────

    def render_3d_leaf_quads(
        self,
        leaf_base_3d: torch.Tensor,     # (B, N, 3)
        lengths: torch.Tensor,          # (B, N)
        pitches: torch.Tensor,          # (B, N) degrees
        yaws: torch.Tensor,             # (B, N) degrees
        rolls: torch.Tensor,            # (B, N) degrees
        internode_dirs: torch.Tensor,   # (B, N, 3) parent internode direction for lateral leaflet rotation
        cam: Dict,
        target_center: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render trifoliate compound leaf matching Helios C++ geometry."""
        B, N = lengths.shape
        device = leaf_base_3d.device

        pitch_rad = pitches * math.pi / 180.0
        yaw_rad = yaws * math.pi / 180.0
        roll_rad = rolls * math.pi / 180.0

        # Terminal leaflet direction from pitch/yaw/roll
        cos_p = torch.cos(pitch_rad)
        sin_p = torch.sin(pitch_rad)
        cos_y = torch.cos(yaw_rad)
        sin_y = torch.sin(yaw_rad)
        cos_r = torch.cos(roll_rad)
        sin_r = torch.sin(roll_rad)

        # Base direction vector (midrib) in world space
        vl_x = cos_p * cos_y
        vl_y = cos_p * sin_y
        vl_z = sin_p
        v_length = torch.stack([vl_x, vl_y, vl_z], dim=-1)
        v_length = v_length / (torch.norm(v_length, dim=-1, keepdim=True) + 1e-8)

        # Width vector perpendicular to midrib and horizontal plane normal
        vw_x = -sin_y
        vw_y = cos_y
        vw_z = torch.zeros_like(yaw_rad)
        v_width = torch.stack([vw_x, vw_y, vw_z], dim=-1)
        # Remove component along v_length and normalize
        v_width = v_width - (v_width * v_length).sum(dim=-1, keepdim=True) * v_length
        v_width = v_width / (torch.norm(v_width, dim=-1, keepdim=True) + 1e-8)

        # Apply roll around midrib
        v_normal = torch.cross(v_length, v_width, dim=-1)
        v_width = v_width * cos_r.unsqueeze(-1) + v_normal * sin_r.unsqueeze(-1)
        v_width = v_width / (torch.norm(v_width, dim=-1, keepdim=True) + 1e-8)

        # Recalculate normal after roll
        v_normal = torch.cross(v_length, v_width, dim=-1)
        v_normal = v_normal / (torch.norm(v_normal, dim=-1, keepdim=True) + 1e-8)

        # Physical dimensions in meters. Helios cowpea leaflets are broad and
        # roughly elliptical, so use a high width/length ratio.
        L = lengths.unsqueeze(-1)
        W = (lengths * 0.85).unsqueeze(-1)

        # Lateral leaflet rotation axis = vertical shoot axis (matches Helios C++
        # compound leaf rotation around internode_axis). Petiole direction is not used
        # here because Helios rotates leaflets around the shoot, not the petiole.
        shoot_axis = torch.tensor([0.0, 0.0, 1.0], device=device).expand(B, N, 3)

        # Build 3 leaflets: terminal + left lateral (-90°) + right lateral (+90°)
        leaflet_scales = torch.tensor([1.0, 0.90, 0.90], device=device)
        leaflet_angles = torch.tensor([0.0, -0.5 * math.pi, 0.5 * math.pi], device=device)

        leaflet_quads = []
        for i, (scale_fac, angle) in enumerate(zip(leaflet_scales, leaflet_angles)):
            # Rotate terminal direction around shoot axis by ±90°
            if i == 0:
                leaflet_dir = v_length
            else:
                # Rodrigues rotation
                axis = shoot_axis
                cos_a = torch.cos(angle)
                sin_a = torch.sin(angle)
                leaflet_dir = v_length * cos_a + torch.cross(axis, v_length, dim=-1) * sin_a + \
                              axis * (axis * v_length).sum(dim=-1, keepdim=True) * (1.0 - cos_a)
                leaflet_dir = leaflet_dir / (torch.norm(leaflet_dir, dim=-1, keepdim=True) + 1e-8)

            # Width vector perpendicular to leaflet_dir and shoot axis
            leaflet_width = torch.cross(shoot_axis, leaflet_dir, dim=-1)
            leaflet_width = leaflet_width / (torch.norm(leaflet_width, dim=-1, keepdim=True) + 1e-8)

            # Leaflet length/width scaled
            l_len = L * scale_fac
            l_wid = W * scale_fac

            # Broad quad tapering to a narrow tip for a leaf-like silhouette
            p0 = leaf_base_3d
            p1 = leaf_base_3d + 0.30 * leaflet_dir * l_len + 0.55 * leaflet_width * l_wid
            p2 = leaf_base_3d + 1.00 * leaflet_dir * l_len + 0.18 * leaflet_width * l_wid
            p3 = leaf_base_3d + 0.30 * leaflet_dir * l_len - 0.55 * leaflet_width * l_wid

            leaflet_quads.append(torch.stack([p0, p1, p2, p3], dim=2))

        # (B, N, 3, 4, 3)
        all_quads_3d = torch.stack(leaflet_quads, dim=2)
        B, N, Lf, Q, _ = all_quads_3d.shape
        all_quads_3d_flat = all_quads_3d.view(B, N * Lf, Q, 3)

        quads_2d, Zc, ppm = self.project_3d_points(all_quads_3d_flat, cam, target_center)
        quads_2d = quads_2d.view(B, N, Lf, Q, 2)
        Zc = Zc.view(B, N, Lf, Q)

        # Sun direction from camera params
        sun_el = math.radians(cam['sun_elevation_deg'])
        sun_az = math.radians(cam['sun_azimuth_deg'])
        sun_dir = torch.tensor([
            math.cos(sun_el) * math.sin(sun_az),
            -math.cos(sun_el) * math.cos(sun_az),
            math.sin(sun_el)
        ], device=device).view(1, 1, 3)

        leaf_body_list, leaf_vein_list, leaf_border_list, leaf_depth_list = [], [], [], []

        for lf_idx in range(Lf):
            q0 = quads_2d[:, :, lf_idx, 0]
            q1 = quads_2d[:, :, lf_idx, 1]
            q2 = quads_2d[:, :, lf_idx, 2]
            q3 = quads_2d[:, :, lf_idx, 3]

            # Approximate depth of leaflet as mean of corner depths
            lf_depth = Zc[:, :, lf_idx].mean(dim=-1)  # (B, N)

            # Shading: diffuse + small specular
            lf_normal = v_normal
            dot_nl = (lf_normal * sun_dir).sum(dim=-1).abs().clamp(0.0, 1.0).unsqueeze(-1).unsqueeze(-1)
            spec = (dot_nl ** 8) * 0.15
            shading = (0.40 + 0.60 * dot_nl + spec).clamp(0.30, 1.15)

            # Standard edge functions (q_{i+1} - q_i) x (P - q_i)
            gx = self.grid[:, :, :, :, 0]
            gy = self.grid[:, :, :, :, 1]

            q0_x = q0[:, :, 0:1].unsqueeze(-1)
            q0_y = q0[:, :, 1:2].unsqueeze(-1)
            q1_x = q1[:, :, 0:1].unsqueeze(-1)
            q1_y = q1[:, :, 1:2].unsqueeze(-1)
            q2_x = q2[:, :, 0:1].unsqueeze(-1)
            q2_y = q2[:, :, 1:2].unsqueeze(-1)
            q3_x = q3[:, :, 0:1].unsqueeze(-1)
            q3_y = q3[:, :, 1:2].unsqueeze(-1)

            d0 = (q1_x - q0_x) * (gy - q0_y) - (q1_y - q0_y) * (gx - q0_x)
            d1 = (q2_x - q1_x) * (gy - q1_y) - (q2_y - q1_y) * (gx - q1_x)
            d2 = (q3_x - q2_x) * (gy - q2_y) - (q3_y - q2_y) * (gx - q2_x)
            d3 = (q0_x - q3_x) * (gy - q3_y) - (q0_y - q3_y) * (gx - q3_x)

            # Winding sign: CCW (area > 0) needs positive d_i inside, CW needs negative flipped
            area = (q1[:, :, 0] - q0[:, :, 0]) * (q2[:, :, 1] - q0[:, :, 1]) - \
                   (q1[:, :, 1] - q0[:, :, 1]) * (q2[:, :, 0] - q0[:, :, 0])
            sign = torch.where(area.unsqueeze(-1).unsqueeze(-1) >= 0, 1.0, -1.0)
            d0 = d0 * sign
            d1 = d1 * sign
            d2 = d2 * sign
            d3 = d3 * sign

            e0_len = torch.norm(q1 - q0, dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-4)
            e1_len = torch.norm(q2 - q1, dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-4)
            e2_len = torch.norm(q3 - q2, dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-4)
            e3_len = torch.norm(q0 - q3, dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-4)

            min_edge_dist = torch.minimum(torch.minimum(d0 / e0_len, d1 / e1_len),
                                          torch.minimum(d2 / e2_len, d3 / e3_len))

            # Solid leaf body
            l_mask = torch.sigmoid(min_edge_dist / self.leaf_sigma) * shading

            # Midrib vein (convert meters to normalized width using mean leaflet depth)
            lf_z_abs = lf_depth.abs().clamp(min=self.near_plane)
            vein_w_norm = (0.04 * lengths * ppm / lf_z_abs).clamp(min=0.001, max=0.05)
            v_mask = self._soft_line_segment(q0, q2, vein_w_norm) * l_mask

            # Dark border
            border_dist = min_edge_dist.abs()
            b_mask = torch.sigmoid((0.05 - border_dist) / 0.015) * l_mask

            leaf_body_list.append(l_mask)
            leaf_vein_list.append(v_mask)
            leaf_border_list.append(b_mask)
            leaf_depth_list.append(lf_depth)

        leaf_body = torch.stack(leaf_body_list, dim=2).max(dim=2)[0]
        leaf_vein = torch.stack(leaf_vein_list, dim=2).max(dim=2)[0]
        leaf_border = torch.stack(leaf_border_list, dim=2).max(dim=2)[0]
        leaf_depth = torch.stack(leaf_depth_list, dim=2).min(dim=2)[0]  # front-most leaflet depth

        return leaf_body, leaf_vein, leaf_border, leaf_depth

    # ── Chunked Primitive Rendering ───────────────────────────────────────

    def _render_chunks(
        self,
        render_fn,
        p1: torch.Tensor,
        p2: torch.Tensor,
        width: torch.Tensor,
        mask: torch.Tensor,
        chunk: int = 64,
    ) -> torch.Tensor:
        """Call render_fn on chunks of nodes and concatenate results."""
        B, N = p1.shape[:2]
        chunks = []
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            rendered = render_fn(p1[:, start:end], p2[:, start:end], width[:, start:end])
            rendered = rendered * mask[:, start:end].unsqueeze(-1).unsqueeze(-1)
            chunks.append(rendered)
        return torch.cat(chunks, dim=1)

    def _render_leaf_chunks(
        self,
        leaf_base_3d: torch.Tensor,
        lengths: torch.Tensor,
        pitches: torch.Tensor,
        yaws: torch.Tensor,
        rolls: torch.Tensor,
        internode_dirs: torch.Tensor,
        cam: Dict,
        target_center: torch.Tensor,
        mask: torch.Tensor,
        chunk: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render 3D leaf quads in node chunks to limit memory."""
        B, N = lengths.shape
        device = leaf_base_3d.device
        body_chunks, vein_chunks, border_chunks, depth_chunks = [], [], [], []
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            b, v, bo, d = self.render_3d_leaf_quads(
                leaf_base_3d[:, start:end], lengths[:, start:end], pitches[:, start:end],
                yaws[:, start:end], rolls[:, start:end], internode_dirs[:, start:end],
                cam, target_center)
            m = mask[:, start:end].unsqueeze(-1).unsqueeze(-1)
            body_chunks.append(b * m)
            vein_chunks.append(v * m)
            border_chunks.append(bo * m)
            depth_chunks.append(d)
        return (torch.cat(body_chunks, dim=1),
                torch.cat(vein_chunks, dim=1),
                torch.cat(border_chunks, dim=1),
                torch.cat(depth_chunks, dim=1))

    def _render_circle_chunks(
        self,
        center: torch.Tensor,
        radius: torch.Tensor,
        mask: torch.Tensor,
        chunk: int = 64,
    ) -> torch.Tensor:
        B, N = center.shape[:2]
        chunks = []
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            rendered = self._soft_circle(center[:, start:end], radius[:, start:end])
            rendered = rendered * mask[:, start:end].unsqueeze(-1).unsqueeze(-1)
            chunks.append(rendered)
        return torch.cat(chunks, dim=1)

    # ── Soft Primitives ───────────────────────────────────────────────────

    def _soft_line_segment(
        self,
        p1: torch.Tensor,       # (B, N, 2)
        p2: torch.Tensor,       # (B, N, 2)
        width: torch.Tensor,    # (B, N) in meters
    ) -> torch.Tensor:
        v = p2 - p1
        v_sq = (v ** 2).sum(dim=-1, keepdim=True).unsqueeze(2).unsqueeze(2) + 1e-8

        w = self.grid - p1.unsqueeze(2).unsqueeze(2)
        v_exp = v.unsqueeze(2).unsqueeze(2)
        v_dot_w = (w * v_exp).sum(dim=-1, keepdim=True)
        c = torch.clamp(v_dot_w / v_sq, 0.0, 1.0)

        proj = p1.unsqueeze(2).unsqueeze(2) + c * v_exp
        dist = torch.norm(self.grid - proj, dim=-1)

        half_w = (width / 2.0).unsqueeze(-1).unsqueeze(-1)
        return torch.sigmoid((half_w - dist) / self.sigma)

    def _soft_circle(
        self,
        center: torch.Tensor,   # (B, N, 2)
        radius: torch.Tensor,     # (B, N) in meters
    ) -> torch.Tensor:
        dx = self.grid[:, :, :, :, 0] - center[:, :, 0].unsqueeze(-1).unsqueeze(-1)
        dy = self.grid[:, :, :, :, 1] - center[:, :, 1].unsqueeze(-1).unsqueeze(-1)
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        r = radius.unsqueeze(-1).unsqueeze(-1).clamp(min=1e-6)
        return torch.sigmoid((r - dist) / self.bud_sigma)

    # ── Depth-Aware Compositing ────────────────────────────────────────────

    def _composite_alpha(self, rgb: torch.Tensor, alpha: torch.Tensor,
                         color: torch.Tensor, depth: torch.Tensor,
                         color_weight: float = 1.0) -> torch.Tensor:
        """Composite a colored alpha mask over rgb using depth ordering.

        Args:
            rgb: (B, 3, H, W) current canvas
            alpha: (B, N, H, W) per-organ alpha
            color: (3,) organ color
            depth: (B, N) per-organ depth (smaller = closer in front)
            color_weight: multiplier for color contribution
        Returns:
            rgb: updated canvas
        """
        B = rgb.shape[0]
        device = rgb.device

        # Sort organs by depth: back (large depth) -> front (small depth)
        # depth from project_3d_points: Zc (negative values, closer = more negative = smaller)
        # So sort ascending: most negative (closest) last
        sorted_depth, sort_idx = torch.sort(depth, dim=1, descending=False)  # back to front

        # Gather alphas in depth order
        alpha_sorted = torch.gather(
            alpha, 1,
            sort_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, alpha.shape[-2], alpha.shape[-1])
        )

        # Composite front-to-back iteratively (iterate over organs)
        # For efficiency: use alpha accumulation
        for i in range(alpha_sorted.shape[1]):
            a = alpha_sorted[:, i:i+1] * color_weight
            a = a.clamp(0, 1)
            rgb_out = []
            for c in range(3):
                rgb_out.append(rgb[:, c:c+1] * (1.0 - a) + color[c] * a)
            rgb = torch.cat(rgb_out, dim=1)

        return rgb

    # ── Main Forward ──────────────────────────────────────────────────────

    def forward(
        self,
        nodes: torch.Tensor,                # (B, N, 15)
        parent_indices: Optional[torch.Tensor] = None,  # (B, N)
        cam_azimuth_deg: float = 0.0,
        render_mode: str = "rgb",            # "rgb" or "gray"
        focus_plant: bool = False,           # Auto-scale camera to plant bounding box
        camera_params: Optional[Dict] = None,
        background: Optional[str] = None,    # None or "ground"
    ) -> torch.Tensor:
        """Render complete plant image matching Helios output."""
        B, N, D = nodes.shape
        device = nodes.device
        assert D == 15, f"Expected 15D nodes, got {D}D"

        # Extract features
        positions_3d = nodes[:, :, :3]
        lengths = nodes[:, :, 3]
        radii = nodes[:, :, 4]
        pitches = nodes[:, :, 5]
        yaws = nodes[:, :, 6]
        rolls = nodes[:, :, 7]
        organ_types = nodes[:, :, 8:12]
        shoot_ids = nodes[:, :, 12]
        phytomer_idxs = nodes[:, :, 13]
        existence = nodes[:, :, 14]

        # Organ masks
        internode_mask = organ_types[:, :, self.INTERNODE]
        petiole_mask = organ_types[:, :, self.PETIOLE]
        leaf_mask = organ_types[:, :, self.LEAF]
        bud_mask = organ_types[:, :, self.FLORAL_BUD]

        # 3D tip positions for stems/petioles
        pitch_rad = pitches * math.pi / 180.0
        yaw_rad = yaws * math.pi / 180.0
        dir_x = torch.cos(pitch_rad) * torch.cos(yaw_rad)
        dir_y = torch.cos(pitch_rad) * torch.sin(yaw_rad)
        dir_z = torch.sin(pitch_rad)
        dir_vectors = torch.stack([dir_x, dir_y, dir_z], dim=-1)
        tip_positions_3d = positions_3d + dir_vectors * lengths.unsqueeze(-1)

        # Parent directions for leaflet rotation (internode axis approximation)
        if parent_indices is not None:
            batch_idx = torch.arange(B, device=device).unsqueeze(-1).expand(B, N)
            parent_pos = positions_3d[batch_idx, parent_indices]
            internode_dirs = positions_3d - parent_pos
            # For root self-parent, use world Z
            root_mask = (parent_indices == torch.arange(N, device=device).unsqueeze(0)).float().unsqueeze(-1)
            internode_dirs = internode_dirs * (1.0 - root_mask) + torch.tensor([0.0, 0.0, 1.0], device=device) * root_mask
        else:
            internode_dirs = torch.tensor([0.0, 0.0, 1.0], device=device).expand(B, N, 3)

        # Camera target center and plant extent (include tips and full leaf reach)
        active_bool = (existence > 0.45)
        target_center_list = []
        plant_extent_list = []
        for b in range(B):
            act_idx = active_bool[b]
            if act_idx.sum() > 0:
                act_pos = torch.cat([positions_3d[b, act_idx], tip_positions_3d[b, act_idx]], dim=0)
                min_xyz = act_pos.min(dim=0)[0]
                max_xyz = act_pos.max(dim=0)[0]
                center_b = (min_xyz + max_xyz) / 2.0
                span_xyz = max_xyz - min_xyz
                span_xy = torch.sqrt(span_xyz[0]**2 + span_xyz[1]**2).clamp(min=0.05)
                # Leaves extend roughly length/2 beyond node tips in XY; add full leaf reach
                act_len = lengths[b, act_idx]
                leaf_reach = (act_len * 0.6).max().clamp(min=0.0)
                extent_b = (span_xy + 2.0 * leaf_reach) * 1.25
            else:
                center_b = torch.tensor([0.0, 0.0, 0.1], device=device)
                extent_b = torch.tensor(0.3, device=device)
            target_center_list.append(center_b)
            plant_extent_list.append(extent_b)

        target_center = torch.stack(target_center_list, dim=0)
        plant_extent = torch.stack(plant_extent_list, dim=0)

        # Build camera params
        cam = self._compute_camera_params(camera_params, focus_plant, plant_extent)
        # Override azimuth if provided directly
        cam['azimuth_deg'] = cam_azimuth_deg
        # Merge canopy_center from target
        if isinstance(cam['canopy_center'], np.ndarray):
            cam['canopy_center'] = torch.from_numpy(cam['canopy_center']).float().to(device)
        if cam['canopy_center'] is None:
            cam['canopy_center'] = target_center.clone()
        else:
            tmp = cam['canopy_center']
            if tmp.dim() == 1:
                tmp = tmp.unsqueeze(0).expand(B, 3)
            cam['canopy_center'] = tmp.to(device)

        # In focus mode, target_center should be plant center, not canopy center
        if focus_plant:
            projection_center = target_center
        else:
            projection_center = cam['canopy_center']

        # Project 3D points
        base_2d, base_Zc, ppm = self.project_3d_points(positions_3d, cam, projection_center)
        tip_2d, tip_Zc, _ = self.project_3d_points(tip_positions_3d, cam, projection_center)

        # Per-organ normalized widths. project_3d_points returns normalized [0,1]
        # coordinates and ppm = focal_y*0.5 (normalized units per meter at |Zc|=1).
        # At actual depth |Zc|, scale is ppm / |Zc|.
        z_abs = base_Zc.abs().clamp(min=self.near_plane)
        norm_ppm = ppm / z_abs

        stem_widths = (radii * 2.0 * norm_ppm).clamp(min=0.001, max=0.15)
        petiole_widths = (radii * 2.0 * norm_ppm * 0.7).clamp(min=0.001, max=0.10)
        bud_radii = (radii * norm_ppm * 1.5).clamp(min=0.002, max=0.15)

        # Render stems and petioles as soft line segments (chunked by node to limit memory)
        chunk = 32
        stem_alpha = self._render_chunks(self._soft_line_segment, base_2d, tip_2d, stem_widths,
                                         internode_mask * existence, chunk)
        stem_depth = (base_Zc + tip_Zc) / 2.0

        petiole_alpha = self._render_chunks(self._soft_line_segment, base_2d, tip_2d, petiole_widths,
                                            petiole_mask * existence, chunk)
        petiole_depth = (base_Zc + tip_Zc) / 2.0

        # Render 3D leaf quads (chunked by node)
        leaf_body, leaf_vein, leaf_border, leaf_depth = self._render_leaf_chunks(
            positions_3d, lengths, pitches, yaws, rolls, internode_dirs, cam, projection_center,
            leaf_mask * existence, chunk)
        leaf_body_alpha = leaf_body
        leaf_vein_alpha = leaf_vein
        leaf_border_alpha = leaf_border

        # Render floral buds
        bud_alpha = self._render_circle_chunks(base_2d, bud_radii,
                                               bud_mask * existence, chunk)
        bud_depth = base_Zc

        if render_mode == "gray":
            stem_img = stem_alpha.max(dim=1, keepdim=True)[0] * 0.35
            pet_img = petiole_alpha.max(dim=1, keepdim=True)[0] * 0.35
            leaf_img = leaf_body_alpha.max(dim=1, keepdim=True)[0] * 0.85
            bud_img = bud_alpha.max(dim=1, keepdim=True)[0] * 0.60
            return torch.clamp(stem_img + pet_img + leaf_img + bud_img, 0.0, 1.0)

        # ── RGB Alpha Compositing with depth ordering ──
        H = W = self.image_size
        rgb = torch.zeros(B, 3, H, W, device=device)

        # Optional ground background
        if background == "ground":
            for c in range(3):
                rgb[:, c] = self.ground_color[c]

        # Composite from back to front by depth
        # Order: stems -> petioles -> leaf body -> leaf veins -> leaf borders -> buds
        # Within each type, use depth sorting via _composite_alpha
        rgb = self._composite_alpha(rgb, stem_alpha, self.stem_color, stem_depth, 1.2)
        rgb = self._composite_alpha(rgb, petiole_alpha, self.petiole_color, petiole_depth, 1.1)
        rgb = self._composite_alpha(rgb, leaf_body_alpha, self.leaf_color, leaf_depth, 1.0)
        rgb = self._composite_alpha(rgb, leaf_vein_alpha, self.vein_color, leaf_depth, 0.6)
        rgb = self._composite_alpha(rgb, leaf_border_alpha, self.leaf_border, leaf_depth, 0.5)
        rgb = self._composite_alpha(rgb, bud_alpha, self.bud_color, bud_depth, 1.0)

        return torch.clamp(rgb, 0.0, 1.0)


class DifferentiablePlantRenderer3DRGB(nn.Module):
    """Convenience wrapper for RGB rendering."""

    def __init__(self, image_size: int = 256, **kwargs):
        super().__init__()
        self.renderer = DifferentiablePlantRenderer3D(image_size, **kwargs)

    def forward(
        self,
        nodes: torch.Tensor,
        parent_indices: Optional[torch.Tensor] = None,
        cam_azimuth_deg: float = 0.0,
        focus_plant: bool = False,
        camera_params: Optional[Dict] = None,
        background: Optional[str] = None,
    ) -> torch.Tensor:
        return self.renderer(nodes, parent_indices, cam_azimuth_deg, render_mode="rgb",
                             focus_plant=focus_plant, camera_params=camera_params,
                             background=background)
