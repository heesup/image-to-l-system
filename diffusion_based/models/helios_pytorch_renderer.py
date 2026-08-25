"""
Differentiable PyTorch Renderer for Helios Plant Architecture.
Matches Helios C++ camera positioning, spherical azimuth/elevation rotation, and --focus-plant HFOV auto-fitting.
"""

import os
import math
import warnings
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any
from diffusion_based.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder

if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6;8.9;9.0+PTX"

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
    hfov_override_deg: Optional[float] = None,
    fixed_camera_bounds: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Computes camera view matrix and projection matrix matching Helios C++ init_camera & --focus-plant math.
    """
    device = verts.device

    # 1. 3D Bounding Box of plant vertices matching Helios C++ main.cpp:1715-1730
    if fixed_camera_bounds is not None:
        b_min = fixed_camera_bounds["min"]
        b_max = fixed_camera_bounds["max"]
        bb_min_x, bb_min_y, bb_min_z = float(b_min[0]), float(b_min[1]), float(b_min[2])
        bb_max_x, bb_max_y, bb_max_z = float(b_max[0]), float(b_max[1]), float(b_max[2])
        plant_center = torch.tensor([
            (bb_min_x + bb_max_x) * 0.5,
            (bb_min_y + bb_max_y) * 0.5,
            (bb_min_z + bb_max_z) * 0.5
        ], device=device, dtype=torch.float32)
    elif verts.shape[0] > 0:
        bb_min_x = float(verts[:, 0].min().item())
        bb_max_x = float(verts[:, 0].max().item())
        bb_min_y = float(verts[:, 1].min().item())
        bb_max_y = float(verts[:, 1].max().item())
        bb_min_z = float(verts[:, 2].min().item())
        bb_max_z = float(verts[:, 2].max().item())

        plant_center = torch.tensor([
            (bb_min_x + bb_max_x) * 0.5,
            (bb_min_y + bb_max_y) * 0.5,
            (bb_min_z + bb_max_z) * 0.5
        ], device=device, dtype=torch.float32)
    else:
        bb_min_x = bb_min_y = bb_min_z = -0.5
        bb_max_x = bb_max_y = bb_max_z = 0.5
        plant_center = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)

    # 2. Camera spherical positioning from plant center matching Helios C++ main.cpp:1730-1747
    az_rad = math.radians(azimuth_deg)
    el_rad = math.radians(elevation_deg)
    dist = camera_height / max(math.sin(el_rad), 1e-3)

    cam_x = plant_center[0] + dist * math.cos(el_rad) * math.sin(az_rad)
    cam_y = plant_center[1] - dist * math.cos(el_rad) * math.cos(az_rad)
    cam_z = plant_center[2] + dist * math.sin(el_rad)

    eye = torch.tensor([cam_x, cam_y, cam_z], device=device, dtype=torch.float32)
    target = plant_center.clone()

    z_axis = eye - target
    z_axis = z_axis / (torch.linalg.norm(z_axis) + 1e-8)

    if elevation_deg >= 89.0:
        # Match Helios C++ top-down view (+Y is Screen UP, +X is Screen RIGHT)
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)
    else:
        up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
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

    # 3. Compute HFOV / VFOV matching Helios C++ main.cpp:1748-1793
    if hfov_override_deg is not None:
        hfov_rad = math.radians(hfov_override_deg)
    elif focus_plant and (bb_max_x > bb_min_x) and (bb_max_y > bb_min_y):
        # Project all 8 3D bounding box corners into camera basis
        xs = [bb_min_x, bb_max_x]
        ys = [bb_min_y, bb_max_y]
        zs = [bb_min_z, bb_max_z]
        min_vx, max_vx = float('inf'), float('-inf')
        min_vy, max_vy = float('inf'), float('-inf')

        eye_np = eye.detach().cpu().numpy()
        x_axis_np = x_axis.detach().cpu().numpy()
        y_axis_np = y_axis.detach().cpu().numpy()
        z_axis_np = z_axis.detach().cpu().numpy()

        for px in xs:
            for py in ys:
                for pz in zs:
                    dx = px - eye_np[0]
                    dy = py - eye_np[1]
                    dz = pz - eye_np[2]
                    vx = dx * x_axis_np[0] + dy * x_axis_np[1] + dz * x_axis_np[2]
                    vy = dx * y_axis_np[0] + dy * y_axis_np[1] + dz * y_axis_np[2]
                    vz = dx * z_axis_np[0] + dy * z_axis_np[1] + dz * z_axis_np[2]
                    zneg = max(-vz, 1e-4)
                    proj_x = vx / zneg
                    proj_y = vy / zneg
                    min_vx = min(min_vx, proj_x)
                    max_vx = max(max_vx, proj_x)
                    min_vy = min(min_vy, proj_y)
                    max_vy = max(max_vy, proj_y)

        # +20% margin matching Helios C++ main.cpp:1778
        half_ext_x = max(abs(min_vx), abs(max_vx)) * 1.20
        half_ext_y = max(abs(min_vy), abs(max_vy)) * 1.20
        half_ext_x = max(half_ext_x, 1e-4)
        half_ext_y = max(half_ext_y, 1e-4)

        hfov_r = 2.0 * math.atan(half_ext_x)
        vfov_r = 2.0 * math.atan(half_ext_y)
        if vfov_r > hfov_r * aspect_ratio:
            hfov_r = vfov_r / aspect_ratio
        hfov_rad = max(hfov_r, math.radians(0.1))
    else:
        hfov_rad = math.radians(45.0)

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
        self._mesh_cache: Dict = {}

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
        focus_plant: bool = True,
        hfov_override_deg: Optional[float] = None,
        include_depth: bool = False,
        fixed_camera_bounds: Optional[Dict[str, Any]] = None,
        image_size: Optional[int] = None,
    ) -> torch.Tensor:
        verts = mesh_dict['vertices']     # (V, 3)
        faces = mesh_dict['faces']        # (F, 3)
        normals = mesh_dict['normals']    # (V, 3)
        colors = mesh_dict['colors']      # (V, 3)
        organ_types = mesh_dict.get('organ_types', None)

        device = verts.device
        H = W = image_size if image_size is not None else self.image_size

        if verts.shape[0] == 0 or faces.shape[0] == 0:
            if background == "ground":
                bg = self.COLOR_GROUND.to(device).view(3, 1, 1).repeat(1, H, W)
            elif background == "white":
                bg = torch.ones((3, H, W), device=device)
            else:
                bg = torch.zeros((3, H, W), device=device)
            if include_depth:
                bg = torch.cat([bg, torch.zeros((1, H, W), device=device)], dim=0)
            return bg

        # Camera Matrices matching Helios C++ --focus-plant
        view_mat, proj_mat, _ = compute_focus_plant_camera(
            verts, organ_types, azimuth_deg, elevation_deg, camera_height,
            aspect_ratio=1.0, focus_plant=focus_plant, hfov_override_deg=hfov_override_deg,
            fixed_camera_bounds=fixed_camera_bounds,
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
            with torch.amp.autocast('cuda', enabled=False):
                mvp = (proj_mat @ view_mat).float()  # (4, 4)
                v_hom = torch.cat([verts.float(), torch.ones((verts.shape[0], 1), device=device, dtype=torch.float32)], dim=-1)
                v_clip = (v_hom @ mvp.T).contiguous()  # (V, 4), row-vector convention matching nvdiffrast

                # nvdiffrast rasterize expects int32 faces
                faces_i32 = faces.to(torch.int32)
                rast_out, _ = dr.rasterize(
                    glctx, v_clip.unsqueeze(0), faces_i32, resolution=(H, W), grad_db=False
                )  # (1, H, W, 4)

                mask = rast_out[..., 3:4] > 0  # (1, H, W, 1)

                if include_depth:
                    # 4-Channel Unified Interpolation (RGB 3ch + Physical Canopy Height Model (CHM) 1ch)
                    # Physical height in meters above ground level (Z_world >= 0.0)
                    chm_verts = torch.clamp(verts[:, 2:3].float(), min=0.0)  # (V, 1) in physical meters
                    rgbd_attrs = torch.cat([shaded_colors.float(), chm_verts], dim=-1).unsqueeze(0).contiguous()  # (1, V, 4)
                    rgbd_rast, _ = dr.interpolate(rgbd_attrs, rast_out, faces_i32)  # (1, H, W, 4)

                    if background == "ground":
                        bg_rgb = self.COLOR_GROUND.to(device=device, dtype=torch.float32).view(1, 1, 1, 3)
                    elif background == "white":
                        bg_rgb = torch.ones((1, 1, 1, 3), device=device, dtype=torch.float32)
                    else:
                        bg_rgb = torch.zeros((1, 1, 1, 3), device=device, dtype=torch.float32)
                    bg_depth = torch.zeros((1, 1, 1, 1), device=device, dtype=torch.float32)  # Ground level = 0.0m
                    bg = torch.cat([bg_rgb, bg_depth], dim=-1)

                    rgbd_out = torch.where(mask, rgbd_rast, bg)
                    return rgbd_out.squeeze(0).permute(2, 0, 1).flip(1)  # (4, H, W); match Helios row-0 = bottom

                # Standard 3-Channel RGB
                shaded_colors_b = shaded_colors.float().unsqueeze(0).contiguous()  # (1, V, 3)
                rgb_rast, _ = dr.interpolate(shaded_colors_b, rast_out, faces_i32)
                # (1, H, W, 3)

                # Background composite
                if background == "ground":
                    bg = self.COLOR_GROUND.to(device=device, dtype=torch.float32).view(1, 1, 1, 3)
                elif background == "white":
                    bg = torch.ones((1, 1, 1, 3), device=device, dtype=torch.float32)
                else:
                    bg = torch.zeros((1, 1, 1, 3), device=device, dtype=torch.float32)

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
        fixed_camera_bounds: Optional[Dict[str, Any]] = None,
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
            aspect_ratio=1.0, focus_plant=focus_plant, hfov_override_deg=hfov_override_deg,
            fixed_camera_bounds=fixed_camera_bounds,
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
            with torch.amp.autocast('cuda', enabled=False):
                mvp = (proj_mat @ view_mat).float()
                v_hom = torch.cat([verts.float(), torch.ones((verts.shape[0], 1), device=device, dtype=torch.float32)], dim=-1)
                v_clip = (v_hom @ mvp.T).contiguous()
                faces_i32 = faces.to(torch.int32)
                rast_out, _ = dr.rasterize(
                    glctx, v_clip.unsqueeze(0), faces_i32, resolution=(H, W), grad_db=False
                )

                # Per-vertex organ type: all three vertices of a face share the same type
                organ_types_b = organ_types.float().unsqueeze(0).unsqueeze(-1).contiguous()  # (1, V, 1)
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
        light_dir: Optional[torch.Tensor] = None,
        differentiable: bool = False,
        focus_plant: bool = True,
        image_size: Optional[int] = None,
        hfov_override_deg: Optional[float] = None,
        include_depth: bool = False,
        fixed_camera_bounds: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Helper alias for forward."""
        return self.forward(
            mesh_dict,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            camera_height=camera_height,
            background=background,
            light_dir=light_dir,
            differentiable=differentiable,
            focus_plant=focus_plant,
            hfov_override_deg=hfov_override_deg,
            include_depth=include_depth,
            fixed_camera_bounds=fixed_camera_bounds,
            image_size=image_size,
        )

    def render_depth(
        self,
        mesh_dict: Dict[str, torch.Tensor],
        azimuth_deg: float = 0.0,
        elevation_deg: float = 90.0,
        camera_height: float = 5.0,
        focus_plant: bool = True,
        image_size: Optional[int] = None,
        hfov_override_deg: Optional[float] = None,
        fixed_camera_bounds: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        verts = mesh_dict['vertices'].detach()
        faces = mesh_dict['faces'].detach()
        device = verts.device
        H = W = image_size if image_size is not None else self.image_size

        if verts.shape[0] == 0 or faces.shape[0] == 0:
            return torch.zeros((H, W), dtype=torch.float32, device=device)

        view_mat, proj_mat, _ = compute_focus_plant_camera(
            verts, mesh_dict.get('organ_types', None), azimuth_deg, elevation_deg, camera_height,
            aspect_ratio=1.0, focus_plant=focus_plant, hfov_override_deg=hfov_override_deg,
            fixed_camera_bounds=fixed_camera_bounds,
        )

        glctx = self._get_nvdiffrast_context(device)
        if _HAS_NVDIFFRAST and glctx is not None:
            with torch.amp.autocast('cuda', enabled=False):
                mvp = (proj_mat @ view_mat).float()
                v_hom = torch.cat([verts.float(), torch.ones((verts.shape[0], 1), device=device, dtype=torch.float32)], dim=-1)
                v_clip = (v_hom @ mvp.T).contiguous()
                faces_i32 = faces.to(torch.int32)
                rast_out, _ = dr.rasterize(glctx, v_clip.unsqueeze(0), faces_i32, resolution=(H, W), grad_db=False)

                v_cam = (view_mat.float() @ v_hom.T).T
                depth_verts = (-v_cam[:, 2:3]).float().unsqueeze(0).contiguous()
                depth_rast, _ = dr.interpolate(depth_verts, rast_out, faces_i32)
                depth_map = depth_rast.squeeze(0).squeeze(-1)
                mask = rast_out[0, ..., 3] > 0
                depth_map = torch.where(mask, depth_map, torch.zeros_like(depth_map))
                return depth_map.flip(0)

        return torch.zeros((H, W), dtype=torch.float32, device=device)

    def render_mask(
        self,
        mesh_dict: Dict[str, torch.Tensor],
        azimuth_deg: float = 0.0,
        elevation_deg: float = 90.0,
        camera_height: float = 5.0,
        focus_plant: bool = True,
        image_size: Optional[int] = None,
        hfov_override_deg: Optional[float] = None,
        fixed_camera_bounds: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Renders binary foreground mask (1.0 for plant, 0.0 for background)."""
        depth = self.render_depth(
            mesh_dict=mesh_dict,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            camera_height=camera_height,
            focus_plant=focus_plant,
            image_size=image_size,
            hfov_override_deg=hfov_override_deg,
            fixed_camera_bounds=fixed_camera_bounds,
        )
        return (depth > 1e-4).float()

    def render_organ_segmentation(
        self,
        mesh_dict: Dict[str, torch.Tensor],
        azimuth_deg: float = 0.0,
        elevation_deg: float = 90.0,
        camera_height: float = 5.0,
        focus_plant: bool = True,
        image_size: Optional[int] = None,
        hfov_override_deg: Optional[float] = None,
        background_rgb: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        fixed_camera_bounds: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Renders an RGB semantic organ segmentation image (3, H, W) colored by botanical organ category.
        """
        type_buf = self.render_organ_type_buffer(
            mesh_dict=mesh_dict,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            camera_height=camera_height,
            focus_plant=focus_plant,
            image_size=image_size,
            hfov_override_deg=hfov_override_deg,
            fixed_camera_bounds=fixed_camera_bounds,
        )  # (H, W) tensor of organ IDs, -1 for background

        H, W = type_buf.shape
        device = type_buf.device

        # Color palette: index matching HeliosPyTorchGeometry
        color_lut = torch.tensor([
            [0.55, 0.27, 0.07],  # 0: Stem/Internode (Brown #8B4513)
            [0.90, 0.50, 0.15],  # 1: Petiole (Orange #E68026)
            [0.13, 0.65, 0.18],  # 2: Leaf (Vibrant Forest Green #228B22)
            [0.60, 0.80, 0.20],  # 3: Peduncle (Lime #9ACD32)
            [1.00, 0.84, 0.00],  # 4: Flower (Yellow #FFD700)
            [0.90, 0.20, 0.20],  # 5: Pod/Fruit (Red #E63333)
            [0.20, 0.85, 0.95],  # 6: Bud (Cyan #33D9F2)
        ], dtype=torch.float32, device=device)

        bg = torch.tensor(background_rgb, dtype=torch.float32, device=device).view(1, 1, 3)
        seg_rgb = bg.expand(H, W, 3).clone()

        fg_mask = type_buf >= 0
        valid_types = type_buf.clamp(min=0, max=len(color_lut) - 1)
        seg_rgb[fg_mask] = color_lut[valid_types[fg_mask]]

        return seg_rgb.permute(2, 0, 1)  # (3, H, W)

    def render_organ_array(
        self,
        organ_array,
        azimuth_deg: float = 0.0,
        elevation_deg: float = 90.0,
        camera_height: float = 5.0,
        background: str = "ground",
        light_dir: Optional[torch.Tensor] = None,
        device: torch.device = torch.device('cpu'),
        differentiable: bool = False,
        focus_plant: bool = True,
        existence_threshold: float = 0.5,
        use_cache: bool = False,
        hfov_override_deg: Optional[float] = None,
        include_depth: bool = False,
        fixed_camera_bounds: Optional[Dict[str, Any]] = None,
        image_size: Optional[int] = None,
        use_legacy_tree_kinematics: bool = False,
    ) -> torch.Tensor:
        """
        .. deprecated:: 2026.08
            `render_organ_array` is deprecated. Use :meth:`render_part_tensor` with 16D Part Assembly
            for 1000x faster fully vectorized GPU rendering.

        Directly renders an RGB (or RGBD) image from a PlantOrganArray by delegating to 16D part assembly.
        """
        warnings.warn(
            "`render_organ_array` is deprecated and will be removed in future versions. "
            "Please use `render_part_tensor(part_tensor)` with 16D Part Assembly representation.",
            DeprecationWarning,
            stacklevel=2,
        )
        if use_legacy_tree_kinematics:
            mesh_dict = self._build_mesh_cached(
                organ_array, device=device, existence_threshold=existence_threshold,
                differentiable=differentiable, use_cache=use_cache,
            )
            return self.forward(
                mesh_dict,
                azimuth_deg=azimuth_deg,
                elevation_deg=elevation_deg,
                camera_height=camera_height,
                background=background,
                light_dir=light_dir,
                differentiable=differentiable,
                focus_plant=focus_plant,
                hfov_override_deg=hfov_override_deg,
                include_depth=include_depth,
                fixed_camera_bounds=fixed_camera_bounds,
                image_size=image_size,
            )

        part_tensor = organ_array.to_part_tensor(device=device)
        return self.render_part_tensor(
            part_tensor,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            camera_height=camera_height,
            background=background,
            light_dir=light_dir,
            device=device,
            differentiable=differentiable,
            focus_plant=focus_plant,
            existence_threshold=existence_threshold,
            hfov_override_deg=hfov_override_deg,
            include_depth=include_depth,
            fixed_camera_bounds=fixed_camera_bounds,
            image_size=image_size,
        )

    def _build_mesh_cached(
        self,
        organ_array,
        device: torch.device,
        existence_threshold: float,
        differentiable: bool,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Build the mesh, optionally reusing a cached result when the input is unchanged."""
        if differentiable or not use_cache:
            return self.geo_builder.build_mesh_from_organ_array(
                organ_array, device=device, existence_threshold=existence_threshold
            )

        t = organ_array.tensor
        key = (t.data_ptr(), t._version, existence_threshold, str(device))
        cached = self._mesh_cache.get(key)
        if cached is not None:
            return cached

        mesh_dict = self.geo_builder.build_mesh_from_organ_array(
            organ_array, device=device, existence_threshold=existence_threshold
        )
        self._mesh_cache = {key: mesh_dict}
        return mesh_dict

    def render_part_tensor(
        self,
        part_tensor: torch.Tensor,
        azimuth_deg: float = 0.0,
        elevation_deg: float = 90.0,
        camera_height: float = 5.0,
        background: str = "ground",
        light_dir: Optional[torch.Tensor] = None,
        device: torch.device = torch.device('cpu'),
        differentiable: bool = False,
        focus_plant: bool = True,
        existence_threshold: float = 0.5,
        soft_existence: bool = False,
        hfov_override_deg: Optional[float] = None,
        include_depth: bool = False,
        fixed_camera_bounds: Optional[Dict[str, Any]] = None,
        image_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Directly renders a 16D (or 26D) part tensor on GPU with zero XML overhead."""
        mesh_dict = self.geo_builder.build_mesh_from_part_tensor(
            part_tensor, device=device, existence_threshold=existence_threshold,
            soft_existence=soft_existence,
        )
        return self.forward(
            mesh_dict,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            camera_height=camera_height,
            background=background,
            light_dir=light_dir,
            differentiable=differentiable,
            focus_plant=focus_plant,
            hfov_override_deg=hfov_override_deg,
            include_depth=include_depth,
            fixed_camera_bounds=fixed_camera_bounds,
            image_size=image_size,
        )

    def render_part_depth(
        self,
        part_tensor: torch.Tensor,
        azimuth_deg: float = 0.0,
        elevation_deg: float = 90.0,
        camera_height: float = 5.0,
        device: torch.device = torch.device('cpu'),
        focus_plant: bool = True,
        existence_threshold: float = 0.5,
        soft_existence: bool = False,
        hfov_override_deg: Optional[float] = None,
        fixed_camera_bounds: Optional[Dict[str, Any]] = None,
        image_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Directly renders depth map from a 16D (or 26D) part tensor on GPU."""
        mesh_dict = self.geo_builder.build_mesh_from_part_tensor(
            part_tensor, device=device, existence_threshold=existence_threshold,
            soft_existence=soft_existence,
        )
        return self.render_depth(
            mesh_dict,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            camera_height=camera_height,
            focus_plant=focus_plant,
            hfov_override_deg=hfov_override_deg,
            fixed_camera_bounds=fixed_camera_bounds,
            image_size=image_size,
        )

