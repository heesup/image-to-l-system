"""How much of the phytomer latent is recoverable from the image at all, and at what resolution?

The 2026-09-17 finding (assessment §6.2) is that Stage 3's latent stays at the dataset mean under two
different objectives and even with correct node positions, which points at what Stage 3 is *given*
rather than at how it is trained. This script measures that directly, with the network taken out of
the loop: a ridge probe from frozen DINOv2 features to the GT PhytomerVAE latent, trained on one set
of plants and scored on held-out plants, under three conditionings:

  token  the current one -- one bilinear sample of the 16x16 patch grid of the whole-plant image, i.e.
         one token per 7.5 cm at zoom 1x, taken at the node's exact projected position
  roi    the proposal -- a `--window_cm` crop around the node cut from a high-resolution render of the
         same plant, resized to the backbone's 224 px (12 cm at 224 px = 0.5 mm/px)
  pos    no image at all: the node's own 3D position, as a floor

Reading it: if `roi` recovers per-node latent variance that `token` does not, the conditioning pipeline
is the limit and the per-node ROI patch is worth building. If `roi` also fails, the limit is the latent
or the single top-down view itself -- part of a 10-slot packet (petiole curvature, three leaflet
orientations, four reproductive organs) is simply not visible from nadir -- and the generative readout
with realistic spread is the right answer rather than better conditioning.

The probe is deliberately linear: it asks whether the information is present and linearly decodable in
the frozen representation, not whether some network could be trained to extract it.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from plant_recon.dataset.part_array_dataset import PartArrayDataset, FM_BASE_START, FM_OT_END  # noqa: E402
from plant_recon.eval.eval_hierarchical_self_consistency import decode_predictions_to_part_tensor  # noqa: E402
from plant_recon.models.helios_pytorch_renderer import HeliosPyTorchRenderer, compute_focus_plant_camera  # noqa: E402
from plant_recon.models.phytomer_vae import PhytomerVAE  # noqa: E402
from plant_recon.eval.ckpt_compat import fix_ckpt_args

CACHE_WINDOW_M = 1.2
CACHE_CAMERA_HEIGHT = 5.0


def project(points_w, view, proj, size):
    """(N,3) world -> (N,2) pixel coordinates in a size x size render, +y up in world = up in image."""
    p = torch.cat([points_w, torch.ones_like(points_w[:, :1])], -1)
    clip = (proj @ (view @ p.T)).T
    ndc = clip[:, :2] / clip[:, 3:4].clamp(min=1e-6)
    x = (ndc[:, 0] * 0.5 + 0.5) * size
    y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * size
    return torch.stack([x, y], -1)


def crops_around(img, uv, win_px, out_px):
    """(3,S,S) image, (N,2) pixel centres -> (N,3,out_px,out_px), zero-padded past the border."""
    S = img.shape[-1]
    half = win_px / 2.0
    out = []
    for i in range(uv.shape[0]):
        cx, cy = float(uv[i, 0]), float(uv[i, 1])
        x0, y0 = int(round(cx - half)), int(round(cy - half))
        x1, y1 = x0 + int(round(win_px)), y0 + int(round(win_px))
        pad_l, pad_t, pad_r, pad_b = max(0, -x0), max(0, -y0), max(0, x1 - S), max(0, y1 - S)
        c = img[:, max(0, y0):min(S, y1), max(0, x0):min(S, x1)]
        if min(c.shape[-2:]) == 0:
            out.append(torch.zeros(3, out_px, out_px, device=img.device)); continue
        if pad_l or pad_t or pad_r or pad_b:
            c = F.pad(c.unsqueeze(0), (pad_l, pad_r, pad_t, pad_b)).squeeze(0)
        out.append(F.interpolate(c.unsqueeze(0), size=(out_px, out_px), mode="bilinear", align_corners=False).squeeze(0))
    return torch.stack(out)


@torch.no_grad()
def dino_features(backbone, imgs, dev, mode="cls_mean", bs=64):
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(imgs), bs):
        b = ((imgs[i:i + bs].to(dev) - mean) / std)
        if b.shape[-1] != 224:
            b = F.interpolate(b, size=(224, 224), mode="bilinear", align_corners=False)
        f = backbone.forward_features(b)
        out.append(torch.cat([f["x_norm_clstoken"], f["x_norm_patchtokens"].mean(1)], -1).float().cpu())
    return torch.cat(out)


def load_backbone(spec, dev):
    """'name@px' -> (model, kind, px). Three families, one interface.

    dinov2 / dinov3 are ViTs whose positional embeddings interpolate, so the input size is free and sets
    the token pitch (224 -> 16x16 -> 75 mm per token at a 1.2 m window; 448 -> 32x32 -> 37.5 mm). The
    224 in the repo's encoder is its own constant, not a limit of the backbone. Swin is hierarchical with
    window attention, so its cost is linear rather than quadratic in image area and a fine stride stays
    affordable -- the reason it is worth testing at all.
    """
    name, _, rest = spec.partition("@")
    px_s, _, stage = rest.partition("#")
    px = int(px_s) if px_s else 224
    stage = int(stage) if stage else 4
    if name.startswith("dinov2"):
        m = torch.hub.load("facebookresearch/dinov2", name, pretrained=True, verbose=False)
        kind = "dino"
    elif name.startswith("dinov3"):
        from plant_recon.models.dinov2_ray_encoder import BACKBONES, _load_backbone
        m = _load_backbone(BACKBONES[name], pretrained=True)
        kind = "dino"
    elif name.startswith("swin"):
        # torchvision Swin's `features` is [patch-embed(stride 4), stage1, merge(8), stage2, merge(16),
        # stage3, merge(32), stage4]; its FULL output is stride 32, which at 512 px is the same 75 mm cell
        # as DINOv2 at 224 and would miss the point. Slicing to an earlier stage is where the hierarchy
        # pays: `#4` (the default) ends after stage2 at stride 8.
        import torchvision
        m = getattr(torchvision.models, name)(weights="DEFAULT")
        m.features = torch.nn.Sequential(*list(m.features)[:stage])
        kind = "swin"
    else:
        raise SystemExit(f"unknown backbone {name}")
    return m.to(dev).eval(), kind, px


@torch.no_grad()
def spatial_map(backbone, kind, img_rgb, px, dev):
    """(1, C, g, g) spatial feature map of the whole plant, for any of the three families."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    b = F.interpolate(((img_rgb.unsqueeze(0).to(dev) - mean) / std), size=(px, px), mode="bilinear", align_corners=False)
    if kind == "dino":
        tok = backbone.forward_features(b)["x_norm_patchtokens"]
        g = int(round(tok.shape[1] ** 0.5))
        return tok.reshape(1, g, g, -1).permute(0, 3, 1, 2).float()
    # torchvision Swin: features() ends at the last stage as (B, H, W, C)
    f = backbone.features(b)
    return f.permute(0, 3, 1, 2).float()


@torch.no_grad()
def token_features(backbone, kind, img_rgb, uv_norm, px, dev):
    """The current conditioning, generalised: bilinear sample of the whole-plant feature map at each node."""
    fmap = spatial_map(backbone, kind, img_rgb, px, dev)
    grid = uv_norm.view(1, -1, 1, 2).to(dev)
    return F.grid_sample(fmap, grid, mode="bilinear", align_corners=False).squeeze(-1).squeeze(0).T.cpu()


def ridge_r2(xtr, ytr, xte, yte, alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)):
    """Best held-out R^2 over a small ridge sweep, plus the alpha that achieved it.

    R^2 is computed against the TRAINING-set mean latent, so 0 means 'no better than predicting the
    mean' -- the same bar the substitution ablation's r2_vs_mean uses.
    """
    xtr = torch.cat([xtr, torch.ones(len(xtr), 1)], 1).double()
    xte = torch.cat([xte, torch.ones(len(xte), 1)], 1).double()
    ytr_, yte_ = ytr.double(), yte.double()
    mu = ytr_.mean(0, keepdim=True)
    denom = ((yte_ - mu) ** 2).sum()
    best = (-float("inf"), None)
    xtx = xtr.T @ xtr
    xty = xtr.T @ ytr_
    for a in alphas:
        reg = a * torch.eye(xtx.shape[0], dtype=torch.float64)
        reg[-1, -1] = 0.0
        try:
            w = torch.linalg.solve(xtx + reg, xty)
        except Exception:
            continue
        r2 = float(1.0 - ((yte_ - xte @ w) ** 2).sum() / denom)
        if r2 > best[0]:
            best = (r2, a)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="any lineage checkpoint: supplies data_dir / cache_dir / VAE path")
    ap.add_argument("--n_plants", type=int, default=120)
    ap.add_argument("--test_frac", type=float, default=0.25)
    ap.add_argument("--render_px", type=int, default=1536, help="high-resolution render the ROI crops are cut from")
    ap.add_argument("--window_cm", type=float, default=12.0, help="ROI window side in centimetres")
    ap.add_argument("--max_nodes_per_plant", type=int, default=120)
    ap.add_argument("--out_dir", default="outputs/logs/20260917/latent_ceiling")
    ap.add_argument("--save_check", action="store_true", help="save one overlay verifying the 3D->pixel projection")
    ap.add_argument("--backbones", default="dinov2_vits14@224,dinov2_vits14@448",
                    help="comma list of 'name@input_px' for the token arm -- the ladder that asks whether a finer "
                         "token grid reduces the mean collapse. The ROI arm always uses dinov2_vits14@224 on its "
                         "crop, so it stays comparable across runs.")
    a = ap.parse_args()

    dev = torch.device("cuda:0")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    args = fix_ckpt_args(ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"]))
    M = args["slots_per_phytomer"]
    ds = PartArrayDataset(data_root=args["data_dir"], max_nodes=args["max_phytomers"] * M, cache_dir=args["cache_dir"],
                          pkt_cache_dir=args.get("pkt_cache_dir") or None, species="cowpea", image_size=256)
    pvae = PhytomerVAE(latent_dim=args["phytomer_latent_dim"], residual_dim=args.get("phytomer_residual_dim", 8),
                       hidden_dim=256).to(dev).eval()
    pvae.load_state_dict(torch.load(args["phytomer_vae_checkpoint"], map_location=dev, weights_only=True))
    renderer = HeliosPyTorchRenderer(image_size=a.render_px).to(dev)
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True, verbose=False).to(dev).eval()
    specs = [x.strip() for x in a.backbones.split(",") if x.strip()]
    bb = {}
    for sp in specs:
        m, kind, px = load_backbone(sp, dev)
        g = int(round(spatial_map(m, kind, torch.zeros(3, 256, 256, device=dev), px, dev).shape[-1]))
        bb[sp] = (m, kind, px, g)
        print(f"  {sp}: {g}x{g} feature grid -> {1200.0 / g:.1f} mm per cell at a 1.2 m window", flush=True)

    g = torch.Generator().manual_seed(0)
    idxs = torch.randperm(len(ds), generator=g)[: a.n_plants].tolist()
    rows = {f"token:{sp}": [] for sp in specs}
    rows.update({"roi": [], "pos": [], "y": [], "plant": []})
    os.makedirs(a.out_dir, exist_ok=True)
    win_px = a.window_cm / 100.0 / CACHE_WINDOW_M * a.render_px

    for n, i in enumerate(idxs):
        it = ds[i]
        pk = it.get("pkt")
        if pk is None:
            continue
        nodes = it["nodes"].to(dev); ex = it["existence_mask"].to(dev)
        parts = decode_predictions_to_part_tensor(nodes[:, FM_BASE_START:], nodes[:, :FM_OT_END].argmax(-1), ex, device=dev)
        if parts.shape[0] == 0:
            continue
        mesh = renderer.geo_builder.build_mesh_from_part_tensor(parts, device=dev)
        v = mesh["vertices"]
        centre = 0.5 * (v.min(0).values + v.max(0).values)
        view, proj, _ = compute_focus_plant_camera(v, None, azimuth_deg=0.0, elevation_deg=90.0,
                                                    camera_height=CACHE_CAMERA_HEIGHT, focus_plant=False,
                                                    reference_window_size=CACHE_WINDOW_M, zoom_factor=1.0,
                                                    center_override=centre)
        with torch.no_grad():
            rgbd = renderer.forward(mesh, azimuth_deg=0.0, elevation_deg=90.0, camera_height=CACHE_CAMERA_HEIGHT,
                                     background="ground", focus_plant=False, include_depth=True,
                                     image_size=a.render_px, zoom_factor=1.0, reference_window_size=CACHE_WINDOW_M,
                                     center_override=centre)
        img = rgbd[:3].clamp(0, 1)
        cen = pk["centers"].to(dev).float()
        lat = pvae.encode(pvae.pack_input(pk["packets"].to(dev).float(), pk["presence"].to(dev)))[0].float().cpu()
        K = min(cen.shape[0], a.max_nodes_per_plant)
        cen, lat = cen[:K], lat[:K]
        uv = project(cen, view, proj, a.render_px)
        inside = (uv[:, 0] > 0) & (uv[:, 0] < a.render_px) & (uv[:, 1] > 0) & (uv[:, 1] < a.render_px)
        if inside.sum() < 3:
            continue
        cen, lat, uv = cen[inside], lat[inside.cpu()], uv[inside]
        if a.save_check and n == 0:
            from PIL import Image
            ov = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
            for p_ in uv.cpu().numpy():
                x_, y_ = int(p_[0]), int(p_[1])
                ov[max(0, y_ - 3):y_ + 3, max(0, x_ - 3):x_ + 3] = [255, 0, 0]
                
            Image.fromarray(ov).save(os.path.join(a.out_dir, "projection_check.png"))
            print("projection overlay ->", os.path.join(a.out_dir, "projection_check.png"), flush=True)
        rois = crops_around(img, uv, win_px, 224)
        rows["roi"].append(dino_features(backbone, rois, dev))
        uv_norm = (uv / a.render_px) * 2.0 - 1.0
        for sp, (m_, kind_, px_, _) in bb.items():
            rows[f"token:{sp}"].append(token_features(m_, kind_, img, uv_norm, px_, dev))
        rows["pos"].append((cen - centre).cpu())
        rows["y"].append(lat)
        rows["plant"].append(torch.full((lat.shape[0],), n))
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(idxs)} plants, {sum(len(x) for x in rows['y'])} nodes", flush=True)

    arms = [k for k in rows if k.startswith("token:")] + ["roi", "pos"]
    X = {k: torch.cat(rows[k]) for k in arms}
    y = torch.cat(rows["y"]); plant = torch.cat(rows["plant"])
    plants = plant.unique()
    n_test = max(1, int(len(plants) * a.test_frac))
    test_plants = set(plants[-n_test:].tolist())
    te = torch.tensor([int(p) in test_plants for p in plant.tolist()])
    print(f"\n{len(y)} nodes from {len(plants)} plants | train {int((~te).sum())} / test {int(te.sum())} nodes "
          f"| ROI window {a.window_cm} cm = {win_px:.0f} px of a {a.render_px} px render "
          f"({a.window_cm * 10 / 224:.2f} mm per backbone pixel)")
    res = {}
    for name in ["pos"] + [k for k in arms if k.startswith("token:")] + ["roi"]:
        r2, alpha = ridge_r2(X[name][~te], y[~te], X[name][te], y[te])
        grid = bb[name.split(":", 1)[1]][3] if name.startswith("token:") else None
        res[name] = {"r2": round(r2, 4), "alpha": alpha, "dim": int(X[name].shape[1]),
                     "grid": grid, "mm_per_cell": round(1200.0 / grid, 1) if grid else None}
        pitch = f"{1200.0 / grid:5.1f} mm/cell" if grid else ("  12 cm crop" if name == "roi" else "   no image")
        print(f"  {name:<26} dim {X[name].shape[1]:>5}  {pitch}  held-out R^2 over the mean latent: {r2:+.4f}  (alpha {alpha})")
    json.dump({"checkpoint": a.checkpoint, "n_nodes": int(len(y)), "n_plants": int(len(plants)),
               "window_cm": a.window_cm, "render_px": a.render_px, "results": res},
              open(os.path.join(a.out_dir, "latent_ceiling.json"), "w"), indent=1)
    print("saved", os.path.join(a.out_dir, "latent_ceiling.json"))


if __name__ == "__main__":
    main()
