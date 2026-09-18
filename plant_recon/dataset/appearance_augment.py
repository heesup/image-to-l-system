"""Appearance augmentation for the cached 16-channel training images: turn a flat-shaded plant on a
uniform tan ground into something whose pixel statistics resemble a real rover photo.

Why (docs/handovers/20260916-sim-to-real-assessment §1.2, §5): the network reads ONLY the RGB planes
(`DINORayEncoder.forward` takes channels 0:3 of every level), every training image is a flat PyTorch
render on one tan colour with no augmentation at all, and swapping those pixels for a Helios raytraced
render of the SAME plant at the SAME framing costs strict P 38.1 -> 28.6 raw / 66.3 -> 59.2 refined,
with the DAP probe off by 25-51 days and the existence gate dropping 40% of the nodes. Pixels, not
structure, are the measured sim-to-real gap.

What this does, per sample, consistently across all four zoom levels (they are one physical scene):

1. **Real soil background.** One 1.2 m soil tile is sampled from the bank of bare-soil patches cut out
   of the actual AgML GEMINI rover frames (`tools/build_soil_bank.py`), mirror-tiled to size so the
   grain keeps its true physical scale, and each zoom level takes its own concentric sub-window of it.
2. **Geometric shading.** The CHM channel is the plant's height field, so its gradient gives a surface
   normal; Lambertian shading under a random sun direction plus a Blinn-Phong highlight replaces the
   renderer's flat per-organ colour with the light/dark/specular variation a raytracer produces.
3. **Cast shadow.** The plant mask is displaced by `height * cot(elevation)` along the sun azimuth --
   the geometrically correct offset for a nadir view -- blurred, and used to darken the soil.
4. **Photometric jitter.** Brightness, contrast, saturation, hue, blur and sensor noise, drawn once per
   sample and applied to every level.

Two lighting regimes, one drawn per sample (`p_rover`). The real captures mix them, so covering only one
would leave half the target domain unseen:

* **sun** -- a directional light at 25-75 degrees elevation from any azimuth, with a cast shadow.
* **rover lamps** -- the lamps ride with the nadir camera, so the light arrives along the view axis. Cast
  shadows hide behind the objects throwing them (`shad_k` ~ 0), the light is NEAR rather than parallel so
  its intensity falls off with distance and the canopy top is lit more strongly than the soil, its
  footprint vignettes toward the frame edges, and its colour temperature differs from daylight.

The plant silhouette is exact (`chm > 0`), so nothing here needs a segmentation step, and the depth
planes are never modified: only the 12 RGB planes change, which is exactly the input the network reads.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

CACHE_WINDOW_M = 1.2      # generate_cache.py: reference_window_size at zoom 1x
ZOOMS = (1.0, 2.0, 4.0, 8.0)


@dataclass
class AugmentConfig:
    p_apply: float = 1.0          # fraction of samples augmented (the rest stay flat)
    p_soil: float = 0.9           # of the augmented ones, fraction that get a real soil background
    shade_strength: float = 0.55  # 0 = keep flat colour, 1 = full Lambertian term
    specular: float = 0.35        # Blinn-Phong highlight weight
    shadow_strength: float = 0.45
    sun_elev_deg: tuple = (25.0, 75.0)
    # The real captures mix TWO lighting regimes (Heesup, 2026-09-17): natural sunlight, and the rover's
    # own artificial lamps inside the tunnel cart. They look nothing alike, so one sample draws one of them.
    p_rover: float = 0.5          # fraction of augmented samples lit by the rover's lamps instead of the sun
    lamp_height_m: float = 1.5    # the rover camera's known height; the lamps ride with it
    lamp_falloff: float = 0.7     # 0 = no inverse-square falloff with distance, 1 = the full physical term
    vignette: float = 0.35        # lateral drop-off away from the lamp's footprint
    rover_tint: float = 0.06      # per-channel gain spread: LED colour temperature vs daylight
    brightness: float = 0.25
    contrast: float = 0.25
    saturation: float = 0.30
    hue: float = 0.03
    blur_p: float = 0.3
    noise_std: float = 0.02


class SoilBank:
    """Bare-soil patches from real frames, held at the bank's own pixels-per-metre."""

    def __init__(self, root: str, device="cpu", max_patches: int = 400, scale_jitter: float = 2.0):
        import json
        from PIL import Image
        import numpy as np
        root = Path(root)
        meta = json.load(open(root / "meta.json"))
        self.px_per_m = float(meta["px_per_m"])
        files = sorted((root / "patches").glob("*.jpg"))[:max_patches]
        if not files:
            raise RuntimeError(f"no soil patches under {root}/patches")
        # uint8, and built once in the parent process: DataLoader workers fork, so the buffers are
        # shared copy-on-write instead of costing ~660 MB of float32 per worker.
        self.patches: List[torch.Tensor] = [
            torch.from_numpy(np.array(Image.open(f).convert("RGB"))).permute(2, 0, 1).contiguous().to(device)
            for f in files]
        self.device = device
        self.scale_jitter = float(scale_jitter)

    def levels(self, window_m_list, out_px: int, gen: torch.Generator):
        """One tile per zoom level, all cut CONCENTRICALLY from the same patch at its native resolution.

        The levels are one physical scene, so they must share a patch and a centre. Cutting each level
        from the patch directly is also what keeps the fine levels sharp: cutting only the widest level
        and then re-cropping THAT for the zoomed ones resamples twice, and a zoom-4x window ends up
        carrying 64 px of real detail stretched over 256 -- the visible haze in the 2026-09-17 panel.
        """
        idx = int(torch.randint(len(self.patches), (1,), generator=gen).item())
        p = self.patches[idx].float() / 255.0
        P = p.shape[-1]
        jit = float(torch.empty(1).uniform_(1.0 / self.scale_jitter, self.scale_jitter, generator=gen).item())
        flip = int(torch.randint(2, (1,), generator=gen).item())
        rot = int(torch.randint(4, (1,), generator=gen).item())
        if flip:
            p = torch.flip(p, [-1])
        if rot:
            p = torch.rot90(p, rot, (-2, -1))
        widest = max(float(w) for w in window_m_list)
        need_w = min(P, max(8, int(round(widest * self.px_per_m / jit))))
        cx = int(torch.randint(need_w // 2, P - need_w // 2 + 1, (1,), generator=gen).item())
        cy = int(torch.randint(need_w // 2, P - need_w // 2 + 1, (1,), generator=gen).item())
        out = []
        for w in window_m_list:
            n = min(P, max(8, int(round(float(w) * self.px_per_m / jit))))
            x0 = min(max(0, cx - n // 2), P - n); y0 = min(max(0, cy - n // 2), P - n)
            crop = p[:, y0:y0 + n, x0:x0 + n]
            out.append(F.interpolate(crop.unsqueeze(0), size=(out_px, out_px), mode="bilinear",
                                     align_corners=False, antialias=(n > out_px)).squeeze(0))
        return out

    def tile(self, window_m: float, out_px: int, gen: torch.Generator) -> torch.Tensor:
        """(3, out_px, out_px) of soil covering `window_m` metres, at true grain scale."""
        p = self.patches[int(torch.randint(len(self.patches), (1,), generator=gen).item())]
        # Grain scale: a patch covers ~0.5 m, a zoom-1x window 1.2 m, so covering the wide levels at true
        # scale would mean tiling -- and a mirrored tile leaves conspicuous symmetry the network could learn.
        # Instead the window is cut at a RANDOM scale around true (scale_jitter): across the dataset soil
        # grain then carries no consistent size cue, so the network cannot read plant scale off the ground,
        # which is the property that matters here (a real photo's grain is correctly scaled; a wrongly but
        # *consistently* scaled one would teach a false cue).
        p = p.float() / 255.0
        jit = float(torch.empty(1).uniform_(1.0 / self.scale_jitter, self.scale_jitter, generator=gen).item())
        need = max(8, int(round(window_m * self.px_per_m / jit)))
        need = min(need, p.shape[-1])
        if need < p.shape[-1]:
            x0 = int(torch.randint(p.shape[-1] - need + 1, (1,), generator=gen).item())
            y0 = int(torch.randint(p.shape[-2] - need + 1, (1,), generator=gen).item())
            p = p[:, y0:y0 + need, x0:x0 + need]
        if int(torch.randint(2, (1,), generator=gen).item()):
            p = torch.flip(p, [-1])
        k = int(torch.randint(4, (1,), generator=gen).item())
        if k:
            p = torch.rot90(p, k, (-2, -1))
        return F.interpolate(p.unsqueeze(0), size=(out_px, out_px), mode="bilinear", align_corners=False).squeeze(0)


def _normals_from_depth(chm: torch.Tensor, px_per_m: float) -> torch.Tensor:
    """(3, H, W) unit normals of the height field; x/y gradients in metres per metre."""
    d = chm.unsqueeze(0).unsqueeze(0)
    kx = torch.tensor([[-0.5, 0.0, 0.5]], dtype=d.dtype, device=d.device).view(1, 1, 1, 3)
    ky = kx.view(1, 1, 3, 1)
    gx = F.conv2d(F.pad(d, (1, 1, 0, 0), mode="replicate"), kx).squeeze(0).squeeze(0) * px_per_m
    gy = F.conv2d(F.pad(d, (0, 0, 1, 1), mode="replicate"), ky).squeeze(0).squeeze(0) * px_per_m
    n = torch.stack([-gx, -gy, torch.ones_like(gx)], 0)
    return n / n.norm(dim=0, keepdim=True).clamp(min=1e-6)


def _shift_zero(t: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Translate by (dy, dx) with ZERO fill, not the circular wrap `torch.roll` gives.

    A cast shadow is displaced by height * cot(elevation), which at the high zoom levels is a large
    fraction of the frame -- 146 px of 256 at zoom 4x with an 8 cm canopy and a 25 degree sun. Under
    `torch.roll` the part that leaves one edge reappears at the opposite one, so a single plant painted
    two or three shadows (spotted in the DAP 25 panel, 2026-09-17). Physically the shadow simply leaves
    the frame, which is what zero fill does.
    """
    out = torch.zeros_like(t)
    H, W = t.shape[-2:]
    y0, y1 = max(0, dy), min(H, H + dy)
    x0, x1 = max(0, dx), min(W, W + dx)
    if y1 > y0 and x1 > x0:
        out[..., y0:y1, x0:x1] = t[..., y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out


def _rgb_to_gray(rgb: torch.Tensor) -> torch.Tensor:
    w = torch.tensor([0.299, 0.587, 0.114], dtype=rgb.dtype, device=rgb.device).view(3, 1, 1)
    return (rgb * w).sum(0, keepdim=True)


def _hue_rotate(rgb: torch.Tensor, deg: float) -> torch.Tensor:
    """Small hue rotation in YIQ space -- enough for a +-few-degree jitter, no HSV round trip."""
    import math
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    m = torch.tensor([
        [0.299 + 0.701 * c + 0.168 * s, 0.587 - 0.587 * c + 0.330 * s, 0.114 - 0.114 * c - 0.497 * s],
        [0.299 - 0.299 * c - 0.328 * s, 0.587 + 0.413 * c + 0.035 * s, 0.114 - 0.114 * c + 0.292 * s],
        [0.299 - 0.300 * c + 1.250 * s, 0.587 - 0.588 * c - 1.050 * s, 0.114 + 0.886 * c - 0.203 * s],
    ], dtype=rgb.dtype, device=rgb.device)
    return torch.einsum("ij,jhw->ihw", m, rgb)


def augment_image16(img16: torch.Tensor, soil: Optional[SoilBank] = None,
                    cfg: AugmentConfig = AugmentConfig(), gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """(16, S, S) cached image -> augmented copy. RGB planes only; the CHM planes pass through."""
    import math
    if gen is None:
        gen = torch.Generator(device="cpu")
    out = img16.clone()
    if float(torch.rand(1, generator=gen).item()) > cfg.p_apply:
        return out
    S = img16.shape[-1]
    u = lambda a, b: float(torch.empty(1).uniform_(a, b, generator=gen).item())  # noqa: E731

    # one scene: one light, one soil tile, one photometric draw for every zoom level
    rover = float(torch.rand(1, generator=gen).item()) < cfg.p_rover
    if rover:
        # The lamps sit with the nadir camera, so the light arrives almost along the view axis. Two
        # consequences separate this regime from sunlight: cast shadows hide behind the objects that
        # throw them (nothing to see from the camera), and the light is NEAR, so its intensity falls off
        # with distance -- the top of the canopy is lit more strongly than the soil, which parallel
        # sunlight never does.
        tilt = math.radians(u(0.0, 18.0)); az0 = u(0.0, 2 * math.pi)
        sun_az, sun_el = az0, math.pi / 2 - tilt
    else:
        sun_az, sun_el = u(0.0, 2 * math.pi), math.radians(u(*cfg.sun_elev_deg))
    L = torch.tensor([math.cos(sun_az) * math.cos(sun_el), math.sin(sun_az) * math.cos(sun_el), math.sin(sun_el)],
                     dtype=torch.float32, device=img16.device)
    lamp_uv = (u(-0.35, 0.35), u(-0.35, 0.35))           # lamp footprint centre, in normalised image coords
    vig_k = u(0.4, 1.0) * cfg.vignette if rover else 0.0
    fall_k = u(0.5, 1.0) * cfg.lamp_falloff if rover else 0.0
    tint = [u(-cfg.rover_tint, cfg.rover_tint) for _ in range(3)] if rover else [0.0, 0.0, 0.0]
    V = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=img16.device)   # nadir view
    Hv = (L + V); Hv = Hv / Hv.norm().clamp(min=1e-6)
    use_soil = soil is not None and float(torch.rand(1, generator=gen).item()) < cfg.p_soil
    soil_levels = ([t.to(img16.device) for t in soil.levels([CACHE_WINDOW_M / z for z in ZOOMS], S, gen)]
                   if use_soil else None)
    b_, c_, s_, h_ = u(-cfg.brightness, cfg.brightness), u(-cfg.contrast, cfg.contrast), \
        u(-cfg.saturation, cfg.saturation), u(-cfg.hue, cfg.hue) * 180.0
    do_blur = float(torch.rand(1, generator=gen).item()) < cfg.blur_p
    shade_k, spec_k = u(0.5, 1.0) * cfg.shade_strength, u(0.5, 1.0) * cfg.specular
    # a shadow thrown along the view axis is hidden by the object throwing it
    shad_k = u(0.0, 0.08) if rover else u(0.5, 1.0) * cfg.shadow_strength

    for li, z in enumerate(ZOOMS):
        rgb = (img16[4 * li:4 * li + 3] * 0.5 + 0.5).clamp(0, 1)
        chm = img16[4 * li + 3]
        fg = (chm > 0).float().unsqueeze(0)
        window_m = CACHE_WINDOW_M / z
        px_per_m = S / window_m

        bg = soil_levels[li].clone() if soil_levels is not None else rgb.clone()

        # cast shadow: displace the silhouette by height * cot(elevation) along the sun azimuth
        shadow = torch.zeros_like(fg)
        if shad_k > 0:
            h_mean = float(chm[chm > 0].mean()) if bool((chm > 0).any()) else 0.0
            off = h_mean / max(math.tan(sun_el), 0.2) * px_per_m
            dx, dy = int(round(-math.cos(sun_az) * off)), int(round(math.sin(sun_az) * off))
            shadow = _shift_zero(fg, dy, dx)
            kk = max(3, int(S * 0.02) | 1)
            shadow = F.avg_pool2d(F.pad(shadow.unsqueeze(0), (kk // 2,) * 4, mode="replicate"), kk, 1).squeeze(0)
            shadow = (shadow * (1.0 - fg)).clamp(0, 1)
        bg = bg * (1.0 - shad_k * shadow)

        # geometric shading of the plant from the height field
        plant = rgb
        if shade_k > 0 or spec_k > 0:
            n = _normals_from_depth(chm, px_per_m)
            lam = (n * L.view(3, 1, 1)).sum(0, keepdim=True).clamp(min=0.0)
            plant = plant * (1.0 - shade_k + shade_k * (0.35 + 0.65 * lam))
            if spec_k > 0:
                spec = (n * Hv.view(3, 1, 1)).sum(0, keepdim=True).clamp(min=0.0) ** 24
                plant = plant + spec_k * spec
        if fall_k > 0:
            # inverse-square falloff from a lamp `lamp_height_m` above the ground: the canopy top is
            # closer to it than the soil is. Normalised so the ground keeps its exposure and only the
            # relative gradient is added.
            dist = (cfg.lamp_height_m - chm).clamp(min=0.2)
            plant = plant * (1.0 + fall_k * ((cfg.lamp_height_m / dist) ** 2 - 1.0))
        comp = (fg * plant + (1.0 - fg) * bg).clamp(0, 1)
        if vig_k > 0:
            yy = torch.linspace(-1, 1, S, device=comp.device).view(S, 1).expand(S, S)
            xx = torch.linspace(-1, 1, S, device=comp.device).view(1, S).expand(S, S)
            r2 = ((xx - lamp_uv[0]) ** 2 + (yy - lamp_uv[1]) ** 2) / 2.0
            comp = comp * (1.0 - vig_k * r2).clamp(min=0.15)
        if any(abs(t) > 1e-6 for t in tint):
            comp = comp * torch.tensor(tint, device=comp.device).view(3, 1, 1).add(1.0)

        # photometric, identical across levels
        comp = (comp + b_).clamp(0, 1)
        g = _rgb_to_gray(comp)
        comp = (g + (comp - g) * (1.0 + s_)).clamp(0, 1)
        m = comp.mean()
        comp = (m + (comp - m) * (1.0 + c_)).clamp(0, 1)
        if abs(h_) > 1e-3:
            comp = _hue_rotate(comp, h_).clamp(0, 1)
        if do_blur:
            k = 3
            comp = F.avg_pool2d(F.pad(comp.unsqueeze(0), (k // 2,) * 4, mode="replicate"), k, 1).squeeze(0)
        if cfg.noise_std > 0:
            comp = (comp + torch.randn(comp.shape, generator=gen).to(comp.device) * cfg.noise_std).clamp(0, 1)

        out[4 * li:4 * li + 3] = (comp - 0.5) / 0.5
    return out


class AppearanceAugmentedDataset(torch.utils.data.Dataset):
    """Wraps the training dataset so only the TRAINING stream is augmented.

    The trainer reads its evaluation and held-out batches straight from the same `PartArrayDataset`
    object, so augmenting inside that class would change the evaluation too and break comparability with
    every earlier reading. Wrapping the training stream alone keeps evaluation on clean pixels.
    """

    def __init__(self, base, soil_bank: Optional[SoilBank] = None, cfg: AugmentConfig = AugmentConfig(), seed: int = 0):
        self.base, self.soil, self.cfg, self.seed = base, soil_bank, cfg, int(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        d = self.base[i]
        img = d.get("image")
        if isinstance(img, torch.Tensor) and img.shape[0] >= 16:
            g = torch.Generator().manual_seed((self.seed * 1000003 + int(i)) % (2 ** 31))
            d["image"] = augment_image16(img.float(), self.soil, self.cfg, g)
        return d

    def __getattr__(self, name):   # samples, image_size, ... stay reachable for the samplers
        # guard against recursion if the attribute is asked for before __init__ has run (unpickling)
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)
