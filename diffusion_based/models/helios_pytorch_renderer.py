"""
Differentiable PyTorch Renderer for Helios Plant Architecture.
Matches Helios C++ camera positioning, spherical azimuth/elevation rotation, and --focus-plant HFOV auto-fitting.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from diffusion_based.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder

try:
    import nvdiffrast.torch as dr
    _HAS_NVDIFFRAST = True
except Exception:
    _HAS_NVDIFFRAST = False


def compute_focus_plant_camera(
    verts: torch.Tensor,
    organ_types: Optional[torch.Tensor],
    azimuth_deg: float = 0.0,
    elevation_deg: float = 90.0,
    camera_height: float = 5.0,
    aspect_ratio: float = 1.0,
    near: float = 0.01,
    far: float = 100.0,
    focus_plant: bool = True,
    hfov_override_deg: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Computes camera view matrix and projection matrix matching Helios C++ init_camera & --focus-plant math.
    """
    device = verts.device

    # Bounding Box of plant vertices matching Helios C++ main.cpp
    if verts.shape[0] > 0:
        bb_min_x = float(verts[:, 0].min().item())
        bb_max_x = float(verts[:, 0].max().item())
        bb_min_y = float(verts[:, 1].min().item())
        bb_max_y = float(verts[:, 1].max().item())
        bb_min_z = float(verts[:, 2].min().item())
        bb_max_z = float(verts[:, 2].max().item())

        canopy_center = torch.tensor([
            (bb_min_x + bb_max_x) * 0.5,
            (bb_min_y + bb_max_y) * 0.5,
            (bb_min_z + bb_max_z) * 0.5
        ], device=device, dtype=torch.float32)

        span_x = (bb_max_x - bb_min_x) * 1.05
        span_y = (bb_max_y - bb_min_y) * 1.05
        max_span = max(span_x, span_y, 0.05)
    else:
        canopy_center = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)
        max_span = 1.0

    # Helios C++ Camera Spherical Positioning Math (main.cpp line 265-270)
    az_rad = math.radians(azimuth_deg)
    el_rad = math.radians(elevation_deg)

    if hfov_override_deg is not None:
        hfov_rad = math.radians(hfov_override_deg)
    elif focus_plant:
        # Auto-fit FOV to plant bounding box + 5% margin matching Helios C++ calculateFOV(max_span, cam_h)
        half_span = max_span * 0.5
        hfov_rad = 2.0 * math.atan(half_span / max(camera_height, 1e-3))
        hfov_rad = max(hfov_rad, math.radians(2.0))
    else:
        # Default fixed HFOV
        hfov_rad = math.radians(45.0)

    dist = camera_height / max(math.sin(el_rad), 1e-3)

    # Helios C++ camera position: (center_x + dist*sin(az), center_y - dist*cos(az), cam_z)
    cam_x = canopy_center[0] + dist * math.cos(el_rad) * math.sin(az_rad)
    cam_y = canopy_center[1] - dist * math.cos(el_rad) * math.cos(az_rad)
    cam_z = canopy_center[2] + dist * math.sin(el_rad)

    eye = torch.tensor([cam_x, cam_y, cam_z], device=device, dtype=torch.float32)
    target = torch.tensor([canopy_center[0], canopy_center[1], 0.0], device=device, dtype=torch.float32)

    # Camera Up Vector matching Helios Top View vs Angled View
    if abs(elevation_deg - 90.0) < 1e-2:
        up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32) # Top View UP is +Y
    else:
        up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)

    z_axis = eye - target
    z_axis = z_axis / (torch.linalg.norm(z_axis) + 1e-8)

    x_axis = torch.linalg.cross(up, z_axis)
    if torch.linalg.norm(x_axis) < 1e-6:
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=device)
    else:
        x_axis = x_axis / torch.linalg.norm(x_axis)

    y_axis = torch.linalg.cross(z_axis, x_axis)

    R_view = torch.stack([x_axis, y_axis, z_axis])
    t_view = -R_view @ eye

    view_mat = torch.eye(4, device=device, dtype=torch.float32)
    view_mat[:3, :3] = R_view
    view_mat[:3, 3] = t_view

    tan_half_fov = math.tan(hfov_rad / 2.0)
    f_x = 1.0 / tan_half_fov
    f_y = 1.0 / (tan_half_fov / aspect_ratio)

    proj_mat = torch.zeros((4, 4), device=device, dtype=torch.float32)
    proj_mat[0, 0] = f_x
    proj_mat[1, 1] = f_y
    proj_mat[2, 2] = -(far + near) / (far - near)
    proj_mat[2, 3] = -(2.0 * far * near) / (far - near)
    proj_mat[3, 2] = -1.0

    return view_mat, proj_mat, hfov_rad


class HeliosPyTorchRenderer(nn.Module):
    """
    Differentiable PyTorch renderer for plant organ 3D meshes matching Helios visualizer.
    Supports both hard rasterization and soft differentiable rasterization for image backpropagation.
    """

    def __init__(self, image_size: int = 256):
        super().__init__()
        self.image_size = image_size
        self.geo_builder = HeliosPlantGeometryBuilder()
        self.COLOR_GROUND = torch.tensor([0.72, 0.62, 0.50], dtype=torch.float32)
        self._glctx = None

    def _get_nvdiffrast_context(self, device):
        if self._glctx is None:
            if _HAS_NVDIFFRAST:
                try:
                    self._glctx = dr.RasterizeCudaContext(device=device)
                except Exception:
                    self._glctx = dr.RasterizeGLContext(device=device)
            else:
                self._glctx = None
        return self._glctx

    def forward(
        self,
        mesh_dict: Dict[str, torch.Tensor],
        azimuth_deg: float = 0.0,
        elevation_deg: float = 90.0,
        camera_height: float = 5.0,
        background: str = "ground",
        light_dir: Optional[torch.Tensor] = None,
        differentiable: bool = False,
        focus_plant: bool = True
    ) -> torch.Tensor:
        verts = mesh_dict['vertices']     # (V, 3)
        faces = mesh_dict['faces']        # (F, 3)
        normals = mesh_dict['normals']    # (V, 3)
        colors = mesh_dict['colors']      # (V, 3)
        organ_types = mesh_dict.get('organ_types', None)

        device = verts.device
        H = W = self.image_size

        if verts.shape[0] == 0 or faces.shape[0] == 0:
            if background == "ground":
                bg = self.COLOR_GROUND.to(device).view(3, 1, 1).repeat(1, H, W)
            elif background == "white":
                bg = torch.ones((3, H, W), device=device)
            else:
                bg = torch.zeros((3, H, W), device=device)
            return bg

        # Camera Matrices matching Helios C++ --focus-plant
        view_mat, proj_mat, _ = compute_focus_plant_camera(
            verts, organ_types, azimuth_deg, elevation_deg, camera_height, aspect_ratio=1.0, focus_plant=focus_plant
        )

        # Transform Vertices to Camera & NDC Space
        v_hom = torch.cat([verts, torch.ones((verts.shape[0], 1), device=device)], dim=-1)
        v_cam = (view_mat @ v_hom.T).T       # (V, 4)
        v_ndc = (proj_mat @ v_cam.T).T       # (V, 4)

        # Perspective divide
        w_div = v_ndc[:, 3:4].clamp(min=1e-5)
        pts_ndc = v_ndc[:, :3] / w_div        # (V, 3)

        # Map NDC [-1, 1] to Screen Pixels [0, W], [0, H]
        screen_x = (pts_ndc[:, 0] * 0.5 + 0.5) * (W - 1)
        screen_y = (-pts_ndc[:, 1] * 0.5 + 0.5) * (H - 1)
        screen_z = pts_ndc[:, 2]

        # Multi-Source Sunlight + Ambient Shading matching Helios C++
        if light_dir is None:
            light_dir = torch.tensor([0.3, -0.4, 0.86], device=device)
        light_dir = light_dir / torch.linalg.norm(light_dir)

        diffuse = torch.abs((normals * light_dir.unsqueeze(0)).sum(dim=-1)).clamp(min=0.0)
        shaded_colors = colors * (0.45 + 0.55 * diffuse.unsqueeze(-1))
        shaded_colors = shaded_colors.clamp(0.0, 1.0)

        # Fast nvdiffrast rasterization path
        glctx = self._get_nvdiffrast_context(device)
        if _HAS_NVDIFFRAST and glctx is not None:
            mvp = proj_mat @ view_mat  # (4, 4)
            v_hom = torch.cat([verts, torch.ones((verts.shape[0], 1), device=device)], dim=-1)
            v_clip = (v_hom @ mvp.T).contiguous()  # (V, 4), row-vector convention matching nvdiffrast

            # nvdiffrast rasterize expects int32 faces
            faces_i32 = faces.to(torch.int32)
            rast_out, _ = dr.rasterize(
                glctx, v_clip.unsqueeze(0), faces_i32, resolution=(H, W), grad_db=False
            )  # (1, H, W, 4)

            # Interpolate shaded colors
            shaded_colors_b = shaded_colors.unsqueeze(0).contiguous()  # (1, V, 3)
            rgb_rast, _ = dr.interpolate(shaded_colors_b, rast_out, faces_i32)
            # (1, H, W, 3)

            # Background composite
            mask = rast_out[..., 3:4] > 0  # (1, H, W, 1)
            if background == "ground":
                bg = self.COLOR_GROUND.to(device).view(1, 1, 1, 3)
            elif background == "white":
                bg = torch.ones((1, 1, 1, 3), device=device)
            else:
                bg = torch.zeros((1, 1, 1, 3), device=device)

            rgb_out = torch.where(mask, rgb_rast, bg)
            return rgb_out.squeeze(0).permute(2, 0, 1).flip(1)  # (3, H, W); match Helios row-0 = bottom

        # Fallback: original slow PyTorch CPU/GPU loop rasterizer
        z_buffer = torch.full((H, W), 1e9, dtype=torch.float32, device=device)
        rgb_buffer = torch.zeros((H, W, 3), dtype=torch.float32, device=device)

        if background == "ground":
            rgb_buffer[:, :] = self.COLOR_GROUND.to(device)
        elif background == "white":
            rgb_buffer[:, :] = 1.0

        f_v0, f_v1, f_v2 = faces[:, 0], faces[:, 1], faces[:, 2]

        p0_x, p0_y, p0_z = screen_x[f_v0], screen_y[f_v0], screen_z[f_v0]
        p1_x, p1_y, p1_z = screen_x[f_v1], screen_y[f_v1], screen_z[f_v1]
        p2_x, p2_y, p2_z = screen_x[f_v2], screen_y[f_v2], screen_z[f_v2]

        c0, c1, c2 = shaded_colors[f_v0], shaded_colors[f_v1], shaded_colors[f_v2]

        num_faces = faces.shape[0]
        chunk_size = 500

        for f_start in range(0, num_faces, chunk_size):
            f_end = min(num_faces, f_start + chunk_size)

            x0, y0, z0 = p0_x[f_start:f_end], p0_y[f_start:f_end], p0_z[f_start:f_end]
            x1, y1, z1 = p1_x[f_start:f_end], p1_y[f_start:f_end], p1_z[f_start:f_end]
            x2, y2, z2 = p2_x[f_start:f_end], p2_y[f_start:f_end], p2_z[f_start:f_end]

            col0, col1, col2 = c0[f_start:f_end], c1[f_start:f_end], c2[f_start:f_end]

            min_x = torch.clamp(torch.min(torch.min(x0, x1), x2).floor().long(), 0, W - 1)
            max_x = torch.clamp(torch.max(torch.max(x0, x1), x2).ceil().long(), 0, W - 1)
            min_y = torch.clamp(torch.min(torch.min(y0, y1), y2).floor().long(), 0, H - 1)
            max_y = torch.clamp(torch.max(torch.max(y0, y1), y2).ceil().long(), 0, H - 1)

            for i in range(f_end - f_start):
                bx0, bx1 = min_x[i].item(), max_x[i].item()
                by0, by1 = min_y[i].item(), max_y[i].item()

                if bx1 < bx0 or by1 < by0:
                    continue

                grid_y, grid_x = torch.meshgrid(
                    torch.arange(by0, by1 + 1, device=device, dtype=torch.float32),
                    torch.arange(bx0, bx1 + 1, device=device, dtype=torch.float32),
                    indexing='ij'
                )

                denom = (y1[i] - y2[i]) * (x0[i] - x2[i]) + (x2[i] - x1[i]) * (y0[i] - y2[i])
                if torch.abs(denom) < 1e-6:
                    continue

                w0 = ((y1[i] - y2[i]) * (grid_x - x2[i]) + (x2[i] - x1[i]) * (grid_y - y2[i])) / denom
                w1 = ((y2[i] - y0[i]) * (grid_x - x2[i]) + (x0[i] - x2[i]) * (grid_y - y2[i])) / denom
                w2 = 1.0 - w0 - w1

                inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)

                if inside.sum() > 0:
                    z_pix = w0 * z0[i] + w1 * z1[i] + w2 * z2[i]
                    col_pix = w0.unsqueeze(-1) * col0[i] + w1.unsqueeze(-1) * col1[i] + w2.unsqueeze(-1) * col2[i]

                    sub_z = z_buffer[by0:by1 + 1, bx0:bx1 + 1]
                    mask_depth = inside & (z_pix < sub_z)

                    if mask_depth.sum() > 0:
                        z_buffer[by0:by1 + 1, bx0:bx1 + 1] = torch.where(mask_depth, z_pix, sub_z)
                        rgb_buffer[by0:by1 + 1, bx0:bx1 + 1] = torch.where(
                            mask_depth.unsqueeze(-1), col_pix, rgb_buffer[by0:by1 + 1, bx0:bx1 + 1]
                        )

        return rgb_buffer.permute(2, 0, 1).flip(1)  # match Helios row-0 = bottom

    def render_organ_type_buffer(
        self,
        mesh_dict: Dict[str, torch.Tensor],
        azimuth_deg: float = 0.0,
        elevation_deg: float = 90.0,
        camera_height: float = 5.0,
        focus_plant: bool = True,
        image_size: Optional[int] = None,
        hfov_override_deg: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Rasterize an organ-type ID buffer (H, W) using the same camera/projection as RGB.
        Background = -1, foreground = organ_type id from mesh_dict['organ_types'].
        """
        verts = mesh_dict['vertices'].detach()
        faces = mesh_dict['faces'].detach()
        organ_types = mesh_dict.get('organ_types', None)

        device = verts.device
        H = W = image_size if image_size is not None else self.image_size

        if verts.shape[0] == 0 or faces.shape[0] == 0 or organ_types is None:
            return torch.full((H, W), -1, dtype=torch.int64, device=device)

        view_mat, proj_mat, _ = compute_focus_plant_camera(
            verts, organ_types, azimuth_deg, elevation_deg, camera_height,
            aspect_ratio=1.0, focus_plant=focus_plant, hfov_override_deg=hfov_override_deg
        )

        v_hom = torch.cat([verts, torch.ones((verts.shape[0], 1), device=device)], dim=-1)
        v_cam = (view_mat @ v_hom.T).T
        v_ndc = (proj_mat @ v_cam.T).T

        w_div = v_ndc[:, 3:4].clamp(min=1e-5)
        pts_ndc = v_ndc[:, :3] / w_div

        screen_x = (pts_ndc[:, 0] * 0.5 + 0.5) * (W - 1)
        screen_y = (-pts_ndc[:, 1] * 0.5 + 0.5) * (H - 1)
        screen_z = pts_ndc[:, 2]

        f_v0, f_v1, f_v2 = faces[:, 0], faces[:, 1], faces[:, 2]
        p0_x, p0_y, p0_z = screen_x[f_v0], screen_y[f_v0], screen_z[f_v0]
        p1_x, p1_y, p1_z = screen_x[f_v1], screen_y[f_v1], screen_z[f_v1]
        p2_x, p2_y, p2_z = screen_x[f_v2], screen_y[f_v2], screen_z[f_v2]
        t0, t1, t2 = organ_types[f_v0], organ_types[f_v1], organ_types[f_v2]

        # Fast nvdiffrast organ-type buffer
        glctx = self._get_nvdiffrast_context(device)
        if _HAS_NVDIFFRAST and glctx is not None:
            mvp = proj_mat @ view_mat
            v_hom = torch.cat([verts, torch.ones((verts.shape[0], 1), device=device)], dim=-1)
            v_clip = (v_hom @ mvp.T).contiguous()
            faces_i32 = faces.to(torch.int32)
            rast_out, _ = dr.rasterize(
                glctx, v_clip.unsqueeze(0), faces_i32, resolution=(H, W), grad_db=False
            )

            # Per-vertex organ type: all three vertices of a face share the same type
            organ_types_b = organ_types.unsqueeze(0).unsqueeze(-1).float().contiguous()  # (1, V, 1)
            type_rast, _ = dr.interpolate(organ_types_b, rast_out, faces_i32)
            type_rast = type_rast.squeeze(0).squeeze(-1)  # (H, W)

            mask = rast_out[0, ..., 3] > 0  # (H, W)
            type_buffer = torch.full((H, W), -1, dtype=torch.int64, device=device)
            type_buffer[mask] = torch.round(type_rast[mask]).to(torch.int64)
            return type_buffer.flip(0)  # match Helios row-0 = bottom

        # Fallback: original slow PyTorch CPU/GPU loop
        type_buffer = torch.full((H, W), -1, dtype=torch.int64, device=device)
        z_buffer = torch.full((H, W), 1e9, dtype=torch.float32, device=device)

        num_faces = faces.shape[0]
        chunk_size = 500

        for f_start in range(0, num_faces, chunk_size):
            f_end = min(num_faces, f_start + chunk_size)

            x0, y0, z0 = p0_x[f_start:f_end], p0_y[f_start:f_end], p0_z[f_start:f_end]
            x1, y1, z1 = p1_x[f_start:f_end], p1_y[f_start:f_end], p1_z[f_start:f_end]
            x2, y2, z2 = p2_x[f_start:f_end], p2_y[f_start:f_end], p2_z[f_start:f_end]
            ty0, ty1, ty2 = t0[f_start:f_end], t1[f_start:f_end], t2[f_start:f_end]

            min_x = torch.clamp(torch.min(torch.min(x0, x1), x2).floor().long(), 0, W - 1)
            max_x = torch.clamp(torch.max(torch.max(x0, x1), x2).ceil().long(), 0, W - 1)
            min_y = torch.clamp(torch.min(torch.min(y0, y1), y2).floor().long(), 0, H - 1)
            max_y = torch.clamp(torch.max(torch.max(y0, y1), y2).ceil().long(), 0, H - 1)

            for i in range(f_end - f_start):
                bx0, bx1 = min_x[i].item(), max_x[i].item()
                by0, by1 = min_y[i].item(), max_y[i].item()

                if bx1 < bx0 or by1 < by0:
                    continue

                grid_y, grid_x = torch.meshgrid(
                    torch.arange(by0, by1 + 1, device=device, dtype=torch.float32),
                    torch.arange(bx0, bx1 + 1, device=device, dtype=torch.float32),
                    indexing='ij'
                )

                denom = (y1[i] - y2[i]) * (x0[i] - x2[i]) + (x2[i] - x1[i]) * (y0[i] - y2[i])
                if torch.abs(denom) < 1e-6:
                    continue

                w0 = ((y1[i] - y2[i]) * (grid_x - x2[i]) + (x2[i] - x1[i]) * (grid_y - y2[i])) / denom
                w1 = ((y2[i] - y0[i]) * (grid_x - x2[i]) + (x0[i] - x2[i]) * (grid_y - y2[i])) / denom
                w2 = 1.0 - w0 - w1

                inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)

                if inside.sum() > 0:
                    z_pix = w0 * z0[i] + w1 * z1[i] + w2 * z2[i]
                    # majority-vote organ type at pixel center
                    type_pix = torch.where(
                        w0 > w1,
                        torch.where(w0 > w2, ty0[i], ty2[i]),
                        torch.where(w1 > w2, ty1[i], ty2[i])
                    )

                    sub_z = z_buffer[by0:by1 + 1, bx0:bx1 + 1]
                    mask_depth = inside & (z_pix < sub_z)

                    if mask_depth.sum() > 0:
                        z_buffer[by0:by1 + 1, bx0:bx1 + 1] = torch.where(mask_depth, z_pix, sub_z)
                        type_buffer[by0:by1 + 1, bx0:bx1 + 1] = torch.where(
                            mask_depth, type_pix, type_buffer[by0:by1 + 1, bx0:bx1 + 1]
                        )

        return type_buffer.flip(0)  # match Helios row-0 = bottom

    def render_mesh(
        self,
        mesh_dict: Dict[str, torch.Tensor],
        azimuth_deg: float = 0.0,
        elevation_deg: float = 90.0,
        camera_height: float = 5.0,
        background: str = "ground",
        differentiable: bool = False,
        focus_plant: bool = True,
        image_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Helper alias for forward."""
        return self.forward(
            mesh_dict,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            camera_height=camera_height,
            background=background,
            differentiable=differentiable,
            focus_plant=focus_plant,
        )

    def render_depth(
        self,
        mesh_dict: Dict[str, torch.Tensor],
        azimuth_deg: float = 0.0,
        elevation_deg: float = 90.0,
        camera_height: float = 5.0,
        focus_plant: bool = True,
        image_size: Optional[int] = None,
    ) -> torch.Tensor:
        verts = mesh_dict['vertices'].detach()
        faces = mesh_dict['faces'].detach()
        device = verts.device
        H = W = image_size if image_size is not None else self.image_size

        if verts.shape[0] == 0 or faces.shape[0] == 0:
            return torch.zeros((H, W), dtype=torch.float32, device=device)

        view_mat, proj_mat, _ = compute_focus_plant_camera(
            verts, mesh_dict.get('organ_types', None), azimuth_deg, elevation_deg, camera_height,
            aspect_ratio=1.0, focus_plant=focus_plant
        )

        glctx = self._get_nvdiffrast_context(device)
        if _HAS_NVDIFFRAST and glctx is not None:
            mvp = proj_mat @ view_mat
            v_hom = torch.cat([verts, torch.ones((verts.shape[0], 1), device=device)], dim=-1)
            v_clip = (v_hom @ mvp.T).contiguous()
            faces_i32 = faces.to(torch.int32)
            rast_out, _ = dr.rasterize(glctx, v_clip.unsqueeze(0), faces_i32, resolution=(H, W), grad_db=False)

            v_cam = (view_mat @ v_hom.T).T
            depth_verts = (-v_cam[:, 2:3]).unsqueeze(0).contiguous()
            depth_rast, _ = dr.interpolate(depth_verts, rast_out, faces_i32)
            depth_map = depth_rast.squeeze(0).squeeze(-1)
            mask = rast_out[0, ..., 3] > 0
            depth_map = torch.where(mask, depth_map, torch.zeros_like(depth_map))
            return depth_map.flip(0)

        return torch.zeros((H, W), dtype=torch.float32, device=device)

    def render_organ_array(
        self,
        organ_array,
        azimuth_deg: float = 0.0,
        elevation_deg: float = 90.0,
        camera_height: float = 5.0,
        background: str = "ground",
        device: torch.device = torch.device('cpu'),
        differentiable: bool = False,
        focus_plant: bool = True,
        existence_threshold: float = 0.5,
    ) -> torch.Tensor:
        mesh_dict = self.geo_builder.build_mesh_from_organ_array(
            organ_array, device=device, existence_threshold=existence_threshold
        )
        return self.forward(
            mesh_dict,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            camera_height=camera_height,
            background=background,
            differentiable=differentiable,
            focus_plant=focus_plant
        )
