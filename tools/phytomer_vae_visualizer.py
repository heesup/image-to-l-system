"""PhytomerVAE-64 latent visualizer (Gradio web app).

PCA latent cloud (2D clickable + 3D rotatable) + 64D latent sliders -> decode
-> re-anchor (reference frame) -> HeliosPyTorchRenderer RGB + depth.

Data: precomputed by tools/precompute_phytomer_latent_pca.py into
dataset/cache/phytomer_gui_cache/ (z, pca, proj2d/3d, presence, refs,
centers, packets, dap, meta.json).

Usage (workspace root):
    .../bin/python tools/phytomer_vae_visualizer.py \
        --cache dataset/cache/phytomer_gui_cache \
        --ckpt diffusion_based/checkpoints/phytomer_vae_v2/phytomer_vae_64d_best.pt \
        --server-name 0.0.0.0 --server-port 7860
"""

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from diffusion_based.models.phytomer_vae import PhytomerVAE
from diffusion_based.dataset.phytomer_packets import (
    apply_reference_rotation,
    assemble_packets,
)
from diffusion_based.dataset.part_array_dataset import (
    decode_fm,
    FM_OT_END,
    FM_BASE_START,
    FM_ROT_START,
    FM_SCALE_START,
    FM_CURV,
    BASE_SCALE,
)
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer

# Leaf mesh quality options (HeliosPlantGeometryBuilder.leaf_mode):
#   "high"   = highres cowpea OBJ leaf assets (~1.4k verts/leaf)
#   "low"    = lightweight alpha-cutout leaf (~80 verts/leaf)
LEAF_MODES = {"high": "obj", "low": "lowpoly"}

ORGAN_NAMES = {
    0: "NONE", 1: "root_meta", 2: "shoot_meta", 3: "internode", 4: "petiole",
    5: "leaf", 6: "peduncle", 7: "bud_dormant", 8: "bud_active",
    9: "flower_closed", 10: "flower_open", 11: "fruit", 12: "bud_aborted",
}

SLOT_ROLES = ["stem", "petiole", "leaflet1", "leaflet2", "leaflet3",
              "peduncle", "repro1", "repro2", "repro3", "repro4"]

PCA2D_SIZE = 700  # px, matplotlib image side


class PhytomerVisualizer:
    def __init__(self, cache_dir: str, ckpt: str, device: torch.device):
        self.device = device
        self.meta = json.load(open(os.path.join(cache_dir, "meta.json")))
        self.z = torch.load(os.path.join(cache_dir, "z.pt"), map_location="cpu")
        self.proj2d = torch.load(os.path.join(cache_dir, "proj2d.pt"), map_location="cpu")
        self.proj3d = torch.load(os.path.join(cache_dir, "proj3d.pt"), map_location="cpu")
        self.presence = torch.load(os.path.join(cache_dir, "presence.pt"), map_location="cpu")
        self.refs = torch.load(os.path.join(cache_dir, "refs.pt"), map_location="cpu")
        self.packets = torch.load(os.path.join(cache_dir, "packets.pt"), map_location="cpu")
        centers_path = os.path.join(cache_dir, "centers.pt")
        self.centers = (torch.load(centers_path, map_location="cpu")
                        if os.path.exists(centers_path) else None)
        dap_path = os.path.join(cache_dir, "dap.pt")
        self.dap = torch.load(dap_path, map_location="cpu") if os.path.exists(dap_path) else None
        self.n = self.z.shape[0]
        self.latent_dim = self.meta["latent_dim"]

        self.model = PhytomerVAE(latent_dim=self.latent_dim, hidden_dim=256).to(device)
        self.model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        self.model.eval()
        self.renderer = HeliosPyTorchRenderer(image_size=256).to(device)

        # Slider bounds: full data range with a small margin (percentile-based
        # bounds from precompute are too tight for real latents).
        zmin = self.z.min(dim=0).values.numpy()
        zmax = self.z.max(dim=0).values.numpy()
        pad = 0.1 * (zmax - zmin) + 0.1
        self.slider_lo = zmin - pad
        self.slider_hi = zmax + pad

        # Latent dims are NOT equal: std spans ~0.05 (dead) .. ~1.15 (live).
        # Raw sliders are shown variance-sorted (live first) with σ in labels.
        self.z_std = self.z.std(dim=0).numpy()
        self.dim_order = np.argsort(-self.z_std)

        # Coarse PCA control: pca16 (16 comps, 98.2% var) → top-10 ≈ 81%.
        # Each PC move shifts many correlated dims → always visibly changes.
        with open(os.path.join(cache_dir, "pca16.pt"), "rb") as f:
            pca16 = pickle.load(f)
        self.n_pc = 10
        self.pc_comps = np.ascontiguousarray(pca16.components_[: self.n_pc]).astype(np.float32)
        self.pc_mean = pca16.mean_.astype(np.float32)
        self.pc_sigma = np.sqrt(pca16.explained_variance_[: self.n_pc]).astype(np.float32)
        # PC slider bounds: DATA range (+pad), not ±3σ — outlier packets sit
        # beyond 3σ and gradio raises ValueError setting a slider out of range.
        scores = (self.z.numpy() - self.pc_mean) @ self.pc_comps.T
        smin, smax = scores.min(0), scores.max(0)
        spad = 0.05 * (smax - smin) + 1e-3
        self.pc_lo = (smin - spad).astype(np.float32)
        self.pc_hi = (smax + spad).astype(np.float32)

        # Organ-type combination per packet (unique ID per present-type set).
        # Groups the 13 organ types into 7 functional categories; the combo is
        # the sorted tuple of categories present in the packet.
        self.combo_ids, self.combo_names = self._compute_combos()

        # 2D scatter image (matplotlib) + pixel->point mapping.
        self._pca2d_img, self._pca2d_ax = self._render_pca2d("combo", "viridis")

    # ------------------------------------------------------------- combos
    def _compute_combos(self):
        """Maps each packet to an organ-type-combination ID.

        Categories: internode(3), petiole(4), leaf(5), peduncle(6),
        flower(9,10), fruit(11); bud(7,8,12) and meta(1,2) are dropped so the
        combo reflects the 6 main functional categories. Returns
        (ids (P,) int64, names list[str] indexable by id).
        """
        labels = self.packets[:, :, :FM_OT_END].argmax(-1)  # (P, 8)
        cat_of = {3: "internode", 4: "petiole", 5: "leaf", 6: "peduncle",
                  9: "flower", 10: "flower", 11: "fruit"}
        combos = []
        for i in range(self.n):
            pres = self.presence[i]
            types = sorted({cat_of[int(t)] for t in labels[i][pres].tolist()
                            if int(t) in cat_of})
            combos.append(tuple(types))
        names = sorted(set(combos))
        name_to_id = {n: k for k, n in enumerate(names)}
        ids = np.array([name_to_id[c] for c in combos], dtype=np.int64)
        return ids, names

    # ------------------------------------------------------------- PCA 2D
    def _color_values(self, color_by: str) -> np.ndarray:
        if color_by == "dap" and self.dap is not None:
            return self.dap.numpy()
        if color_by == "slots":
            return self.presence.float().sum(dim=1).numpy()
        if color_by == "combo":
            return self.combo_ids
        return np.zeros(self.n)

    def _render_pca2d(self, color_by: str, cmap: str):
        proj = self.proj2d.numpy()
        cvals = self._color_values(color_by)
        evr = self.meta["pca_evr"]
        fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
        fig.patch.set_facecolor("#10131a")
        ax.set_facecolor("#161a22")
        if color_by == "dap":
            sc = ax.scatter(proj[:, 0], proj[:, 1], c=cvals, s=3, cmap=cmap,
                            vmin=0, vmax=100, alpha=0.7)
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("DAP", color="#ced4da")
            cb.ax.yaxis.set_tick_params(color="#ced4da")
            plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#ced4da")
        elif color_by == "slots":
            sc = ax.scatter(proj[:, 0], proj[:, 1], c=cvals, s=3, cmap=cmap,
                            vmin=0, vmax=10, alpha=0.7)
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("slots", color="#ced4da")
            cb.ax.yaxis.set_tick_params(color="#ced4da")
            plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#ced4da")
        elif color_by == "combo":
            # Categorical: one color per organ-type combination + legend.
            n_combos = len(self.combo_names)
            cmap_obj = plt.get_cmap(cmap)
            colors = [cmap_obj(k / max(n_combos - 1, 1)) for k in range(n_combos)]
            for k in range(n_combos):
                mask = cvals == k
                if mask.sum() == 0:
                    continue
                ax.scatter(proj[mask, 0], proj[mask, 1], s=3, c=[colors[k]],
                           alpha=0.7, label="+".join(self.combo_names[k]))
            ax.legend(markerscale=4, fontsize=6, loc="upper left",
                      facecolor="#161a22", edgecolor="#3a4150",
                      labelcolor="#ced4da", ncol=2)
        else:
            ax.scatter(proj[:, 0], proj[:, 1], s=3, c="#4dabf7", alpha=0.7)
        ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)", color="#ced4da")
        ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)", color="#ced4da")
        ax.tick_params(colors="#ced4da")
        for spine in ax.spines.values():
            spine.set_color("#3a4150")
        ax.set_title(f"PCA 2D latent cloud ({self.n:,} packets) — click a point",
                     color="#ced4da", fontsize=11)
        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return img, ax

    def _pca2d_click_to_idx(self, x_px: int, y_px: int) -> int:
        """Maps matplotlib image pixel coords -> nearest packet index.

        The image is the full figure canvas (origin top-left). The axes occupy
        a fractional sub-rectangle of the figure (get_position), with the axes
        origin at bottom-left.
        """
        h, w = self._pca2d_img.shape[:2]
        ax = self._pca2d_ax
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        pos = ax.get_position()
        fx = (x_px / w - pos.x0) / pos.width
        fy = (1.0 - y_px / h - pos.y0) / pos.height
        x = xlim[0] + fx * (xlim[1] - xlim[0])
        y = ylim[0] + fy * (ylim[1] - ylim[0])
        proj = self.proj2d.numpy()
        d = (proj[:, 0] - x) ** 2 + (proj[:, 1] - y) ** 2
        return int(d.argmin())

    # ------------------------------------------------------------- PCA 3D
    def _scatter3d(self, color_by: str, colorscale: str) -> go.Figure:
        proj = self.proj3d.numpy()
        cvals = self._color_values(color_by)
        evr = self.meta["pca_evr"]
        fig = go.Figure()
        if color_by == "combo":
            # Categorical: one trace per combination (legend = combo names).
            n_combos = len(self.combo_names)
            cmap_obj = plt.get_cmap(colorscale.lower())
            for k in range(n_combos):
                mask = cvals == k
                if mask.sum() == 0:
                    continue
                rgb = [int(255 * v) for v in cmap_obj(k / max(n_combos - 1, 1))[:3]]
                fig.add_trace(go.Scatter3d(
                    x=proj[mask, 0], y=proj[mask, 1], z=proj[mask, 2],
                    mode="markers", name="+".join(self.combo_names[k]),
                    marker=dict(size=2.5, color=f"rgb({rgb[0]},{rgb[1]},{rgb[2]})",
                                opacity=0.7),
                    customdata=np.nonzero(mask)[0],
                    hovertemplate="idx %{customdata}<extra></extra>",
                ))
        else:
            if color_by == "dap":
                cmin, cmax, ctitle = 0.0, 100.0, "DAP"
            elif color_by == "slots":
                cmin, cmax, ctitle = 0.0, 10.0, "slots"
            else:
                cmin, cmax, ctitle = 0.0, 1.0, ""
            fig.add_trace(go.Scatter3d(
                x=proj[:, 0], y=proj[:, 1], z=proj[:, 2],
                mode="markers",
                marker=dict(
                    size=2.5, color=cvals, colorscale=colorscale,
                    cmin=cmin, cmax=cmax, showscale=True,
                    colorbar=dict(title=ctitle, thickness=10),
                    opacity=0.7,
                ),
                customdata=np.arange(self.n),
                hovertemplate="idx %{customdata}<extra></extra>",
            ))
        fig.update_layout(
            title=f"PCA 3D (PC1 {evr[0]*100:.1f}%, PC2 {evr[1]*100:.1f}%, PC3 {evr[2]*100:.1f}%)",
            height=520, margin=dict(l=0, r=0, t=50, b=0),
            paper_bgcolor="#10131a", font=dict(color="#ced4da"),
            scene=dict(bgcolor="#161a22"),
        )
        return fig

    # ------------------------------------------------------------- decoding
    def _anchor_for_packet(self, idx: int):
        """Returns (center_m, ref_rot6d) for a real packet.

        Uses the stored cluster center (precompute saves centers.pt); falls
        back to recovering it from the slot-0 relative base (slot 0 is the
        internode whose base IS the node position).
        """
        ref = self.refs[idx].numpy()
        if self.centers is not None:
            center = self.centers[idx].numpy()
        else:
            p = self.packets[idx]
            pr = self.presence[idx]
            if bool(pr[0]):
                center = -(p[0, FM_BASE_START:FM_BASE_START + 3].numpy()) / BASE_SCALE
            else:
                center = np.zeros(3)
        return center, ref

    # ------------------------------------------------- PC / slider mapping
    def _z_from_pc(self, alphas: np.ndarray) -> np.ndarray:
        """PC slider values (n_pc,) -> full z (64,)."""
        return self.pc_mean + np.asarray(alphas, dtype=np.float32) @ self.pc_comps

    def _pc_from_z(self, z: np.ndarray) -> np.ndarray:
        """Full z (64,) -> PC scores (n_pc,) by projection."""
        return (np.asarray(z, dtype=np.float32) - self.pc_mean) @ self.pc_comps.T

    def _sliders_out(self, z: np.ndarray):
        """z (64,) -> [64 variance-ordered raw values] + [n_pc scores]."""
        z = np.asarray(z, dtype=np.float32)
        pc = np.clip(self._pc_from_z(z), self.pc_lo, self.pc_hi)
        return [float(z[i]) for i in self.dim_order] + [float(a) for a in pc]

    def _z_from_ordered(self, values) -> np.ndarray:
        """Variance-ordered raw slider values -> z (64,)."""
        z = np.zeros(self.latent_dim, dtype=np.float32)
        z[self.dim_order] = np.asarray(values, dtype=np.float32)
        return z

    def _decode(self, z: np.ndarray, ref_mode: str, ref_idx: int):
        """z (64,) -> (relative packet, presence, center, ref_rot6d).

        The structural VAE decodes ZEROED base columns; slot bases are then
        deterministically reconstructed from the petiole geometry via
        assemble_packets() (needs the reference-frame rotation in the WORLD
        frame; identity mode passes identity so R_pet = R_rel).
        """
        zt = torch.from_numpy(z.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model.decode(zt, use_rot_branch=True)
        rel = out["recon_packets"][0].cpu()
        presence = out["cls_logits"][0].argmax(-1).cpu() > 0
        if ref_mode == "real" and 0 <= ref_idx < self.n:
            center, ref = self._anchor_for_packet(ref_idx)
        else:
            center, ref = np.zeros(3), np.array([1.0, 0, 0, 0, 1.0, 0])
        ref_t = torch.from_numpy(ref.astype(np.float32)).unsqueeze(0)
        rel = assemble_packets(rel.unsqueeze(0), ref_t)[0]
        return rel, presence, center, ref

    def _build_mesh(self, rel: torch.Tensor, presence: torch.Tensor,
                    center: np.ndarray, ref: np.ndarray, leaf_quality: str):
        """Re-anchors a relative packet and builds the 3D mesh dict.

        leaf_quality: "high" (highres OBJ leaves) or "low" (lightweight).
        """
        rel = rel.clone()
        rel[presence, FM_BASE_START:FM_BASE_START + 3] += (
            torch.from_numpy(center.astype(np.float32)) * BASE_SCALE
        )
        if ref.any():
            rel[presence, FM_ROT_START:FM_ROT_START + 6] = apply_reference_rotation(
                rel[presence, FM_ROT_START:FM_ROT_START + 6],
                torch.from_numpy(ref.astype(np.float32)).unsqueeze(0),
            )
        part14 = decode_fm(rel[presence])
        leaf_mode = LEAF_MODES.get(leaf_quality, "obj")
        return self.renderer.geo_builder.build_mesh_from_part_tensor(
            part14.to(self.device), device=self.device, leaf_mode=leaf_mode)

    def _export_glb(self, rel: torch.Tensor, presence: torch.Tensor,
                    center: np.ndarray, ref: np.ndarray, leaf_quality: str,
                    tag: str) -> str:
        """Builds the mesh and exports it as a GLB file for gr.Model3D.

        Each export gets a UNIQUE path: concurrent Gradio events previously
        raced on a fixed per-tag path (one event reading a half-written file
        from another → "not a glb" assert). Stale files are pruned.
        """
        import uuid
        mesh = self._build_mesh(rel, presence, center, ref, leaf_quality)
        import trimesh
        # NOTE: keep Helios Z-up as-is — RGB renderer and web GLB must show
        # the IDENTICAL view (a Y-up bake here once made them disagree).
        t = trimesh.Trimesh(
            vertices=mesh["vertices"].cpu().numpy(),
            faces=mesh["faces"].cpu().numpy(),
            vertex_colors=(mesh["colors"].cpu().numpy() * 255).astype(np.uint8),
            process=False,
        )
        if len(t.faces) == 0:
            raise ValueError("empty mesh: no faces to export as GLB")
        data = t.export(file_type="glb")
        data = self._patch_glb_double_sided(bytes(data))
        glb_dir = "/tmp/opencode/phytomer_glb"
        os.makedirs(glb_dir, exist_ok=True)
        path = os.path.join(glb_dir, f"phytomer_{tag}_{leaf_quality}_{uuid.uuid4().hex[:8]}.glb")
        with open(path, "wb") as f:
            f.write(data)
        self._prune_glb_dir(glb_dir, keep=20)
        return path

    @staticmethod
    def _prune_glb_dir(glb_dir: str, keep: int = 20):
        """Deletes oldest GLBs beyond `keep` (unique-path exports accumulate)."""
        try:
            files = sorted(
                (os.path.join(glb_dir, f) for f in os.listdir(glb_dir) if f.endswith(".glb")),
                key=os.path.getmtime,
            )
            for f in files[:-keep]:
                os.remove(f)
        except OSError:
            pass

    @staticmethod
    def _patch_glb_double_sided(data: bytes) -> bytes:
        """Returns GLB bytes with a doubleSided material bound to all primitives.

        (Trimesh GLB export omits materials entirely by default; without this
        the default glTF material is single-sided, so leaf backfaces vanish.)
        Pure bytes-in/bytes-out — no file read-back, so concurrent exports
        cannot observe half-written files.
        """
        import struct
        import json as _json
        if data[:4] != b"glTF":
            raise ValueError(f"trimesh did not return GLB bytes (magic={data[:4]!r})")
        json_len = struct.unpack("<I", data[12:16])[0]
        gltf = _json.loads(data[20:20 + json_len])

        mats = gltf.setdefault("materials", [])
        mats.append({"name": "double_sided", "doubleSided": True,
                     "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                                              "metallicFactor": 0.0,
                                              "roughnessFactor": 1.0}})
        mat_idx = len(mats) - 1
        for m in gltf.get("meshes", []):
            for p in m.get("primitives", []):
                p["material"] = mat_idx

        new_json = bytearray(_json.dumps(gltf, separators=(",", ":")).encode())
        # GLB chunks must be 4-byte aligned; pad JSON with spaces.
        while len(new_json) % 4 != 0:
            new_json += b" "
        bin_chunk = data[20 + json_len:]
        total_len = 20 + len(new_json) + len(bin_chunk)
        out = bytearray()
        out += data[:8]  # magic + version
        out += struct.pack("<I", total_len)
        out += struct.pack("<I", len(new_json)) + b"JSON"
        out += new_json
        out += bin_chunk
        return bytes(out)

    def _render(self, rel: torch.Tensor, presence: torch.Tensor,
                center: np.ndarray, ref: np.ndarray, leaf_quality: str = "high"):
        """Re-anchors + renders. Returns (rgb HxWx3, depth HxW, info dict)."""
        mesh = self._build_mesh(rel, presence, center, ref, leaf_quality)
        out = self.renderer.forward(
            mesh, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
            background="ground", focus_plant=True, include_depth=True,
            zoom_factor=1.0,
        )
        rgb = out[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        depth = out[3].clamp(min=0.0).cpu().numpy()
        info = self._packet_info(rel, presence)
        return rgb, depth, info

    def _packet_info(self, rel: torch.Tensor, presence: torch.Tensor) -> str:
        lines = []
        n_slots = presence.shape[0]
        for s in range(n_slots):
            if not bool(presence[s]):
                lines.append(f"slot {s} ({SLOT_ROLES[s]}): —")
                continue
            row = rel[s]
            ot = int(row[:FM_OT_END].argmax())
            base = row[FM_BASE_START:FM_BASE_START + 3].numpy() / BASE_SCALE
            scale = row[FM_SCALE_START:FM_SCALE_START + 3].numpy() / 50.0
            curv = float(row[FM_CURV]) * 60.0
            lines.append(
                f"slot {s} ({SLOT_ROLES[s]}): {ORGAN_NAMES.get(ot, ot)} | "
                f"base ({base[0]*100:.1f}, {base[1]*100:.1f}, {base[2]*100:.1f}) cm | "
                f"scale ({scale[0]*100:.1f}, {scale[1]*100:.1f}, {scale[2]*100:.1f}) cm | "
                f"curv {curv:.0f} deg/m"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------ callbacks
    def on_pca2d_click(self, evt: gr.SelectData):
        x, y = int(evt.index[0]), int(evt.index[1])
        idx = self._pca2d_click_to_idx(x, y)
        z = self.z[idx].numpy()
        msg = f"Loaded packet {idx}"
        if self.dap is not None:
            msg += f" (DAP {self.dap[idx].item():.0f})"
        # Also sets ref_idx=idx: its .change cascade renders with the new
        # sliders (programmatic slider sets don't fire .release themselves).
        return self._sliders_out(z) + [idx, idx, msg]

    def on_load_idx(self, idx: int):
        idx = int(idx) % self.n
        z = self.z[idx].numpy()
        msg = f"Loaded packet {idx}"
        if self.dap is not None:
            msg += f" (DAP {self.dap[idx].item():.0f})"
        msg += f" | combo: {'+'.join(self.combo_names[self.combo_ids[idx]])}"
        return self._sliders_out(z) + [idx, idx, msg]

    def on_sliders(self, leaf_quality: str, *values):
        z = self._z_from_ordered(values)
        rel, presence, center, ref = self._decode(z, "identity", 0)
        rgb, depth, info = self._render(rel, presence, center, ref, leaf_quality)
        glb = self._export_glb(rel, presence, center, ref, leaf_quality, "sliders")
        return rgb, depth, info, glb

    def on_pc_sliders(self, leaf_quality: str, *values):
        """Coarse PC sliders -> set raw sliders + render (always visible change)."""
        z = self._z_from_pc(np.array(values, dtype=np.float32))
        rel, presence, center, ref = self._decode(z, "identity", 0)
        rgb, depth, info = self._render(rel, presence, center, ref, leaf_quality)
        glb = self._export_glb(rel, presence, center, ref, leaf_quality, "pc")
        return [float(z[i]) for i in self.dim_order] + [rgb, depth, info, glb]

    def on_random(self):
        z = np.random.randn(self.latent_dim).astype(np.float32)
        return self._sliders_out(z)

    def on_reset(self):
        return self._sliders_out(np.zeros(self.latent_dim, dtype=np.float32))

    def on_ref_change(self, ref_mode: str, ref_idx: int, leaf_quality: str, *values):
        z = self._z_from_ordered(values)
        rel, presence, center, ref = self._decode(z, ref_mode, int(ref_idx))
        rgb, depth, info = self._render(rel, presence, center, ref, leaf_quality)
        glb = self._export_glb(rel, presence, center, ref, leaf_quality, "ref")
        return rgb, depth, info, glb


def build_app(viz: PhytomerVisualizer):
    meta = viz.meta
    slider_lo = viz.slider_lo.tolist()
    slider_hi = viz.slider_hi.tolist()
    z_mean = meta["z_mean"]

    with gr.Blocks(theme=gr.themes.Base(primary_hue="green"), title="PhytomerVAE-64 Visualizer") as demo:
        gr.Markdown(
            f"# PhytomerVAE-64 Latent Visualizer\n"
            f"**{viz.n:,} packets** · PCA evr {meta['pca_evr'][0]*100:.1f}/{meta['pca_evr'][1]*100:.1f}/{meta['pca_evr'][2]*100:.1f}% · "
            f"decode `use_rot_branch=True`"
        )
        with gr.Row():
            # ---------------- left: PCA cloud ----------------
            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab("PCA 2D (click)"):
                        pca2d_img = gr.Image(value=viz._pca2d_img, height=520,
                                             label="Click a point to load its latent")
                    with gr.Tab("PCA 3D (rotate)"):
                        pca3d = gr.Plot(value=viz._scatter3d("combo", "Viridis"))
                with gr.Row():
                    color_by = gr.Radio(
                        ["combo", "slots", "dap", "none"], value="combo", label="Color by")
                    colorscale = gr.Dropdown(
                        ["Viridis", "Plasma", "Turbo", "Cividis", "Jet"],
                        value="Viridis", label="Colorscale")
                with gr.Row():
                    load_idx = gr.Number(value=0, precision=0, label="Load packet idx")
                    btn_load = gr.Button("Load", variant="secondary")
                status = gr.Markdown("Click a point in the PCA 2D cloud (or enter an index).")

            # ---------------- center: 64D sliders ----------------
            with gr.Column(scale=4):
                with gr.Row():
                    btn_random = gr.Button("Random z ~ N(0,I)", variant="primary")
                    btn_reset = gr.Button("Reset to 0")
                with gr.Row():
                    ref_mode = gr.Radio(
                        ["identity", "real"], value="identity",
                        label="Reference frame")
                    ref_idx = gr.Number(value=0, precision=0, label="Real packet idx")
                leaf_quality = gr.Radio(
                    ["high", "low"], value="high",
                    label="Leaf quality (high = highres OBJ, low = lightweight)")
                # Coarse control: top-10 PCs (~81% of latent variance).
                # Each PC move shifts many correlated dims → always visible.
                gr.Markdown("**Coarse (PC sliders — big moves)**")
                pc_sliders = [
                    gr.Slider(
                        minimum=float(viz.pc_lo[k]),
                        maximum=float(viz.pc_hi[k]),
                        value=0.0, step=float((viz.pc_hi[k] - viz.pc_lo[k]) / 500),
                        label=f"PC{k} ({viz.meta['pca_evr'][k]*100:.1f}%)" if k < 3
                              else f"PC{k}",
                    )
                    for k in range(viz.n_pc)
                ]
                with gr.Accordion("Raw 64D (variance-sorted, σ in label)", open=False):
                    sliders = [
                        gr.Slider(
                            minimum=slider_lo[i], maximum=slider_hi[i],
                            value=0.0, step=0.01,
                            label=f"z[{i}] σ={viz.z_std[i]:.2f}",
                        )
                        for i in viz.dim_order
                    ]

            # ---------------- right: render ----------------
            with gr.Column(scale=5):
                with gr.Tabs(selected="web3d"):
                    with gr.Tab("RGB"):
                        rgb_out = gr.Image(label="RGB render", height=420)
                    with gr.Tab("Depth"):
                        depth_out = gr.Image(label="Depth render", height=420)
                    with gr.Tab("3D (web)", id="web3d"):
                        model3d = gr.Model3D(label="3D mesh", height=420,
                                             clear_color=(0.06, 0.08, 0.1, 1.0))
                info_out = gr.Textbox(label="Packet (relative frame)", lines=10, max_lines=14)

        # ---- events ----
        slider_outs = sliders + pc_sliders
        click_outs = slider_outs + [load_idx, ref_idx, status]
        pca2d_img.select(viz.on_pca2d_click, None, click_outs)
        btn_load.click(viz.on_load_idx, load_idx, click_outs)
        color_by.change(
            lambda cb, cs: viz._render_pca2d(cb, cs.lower())[0], [color_by, colorscale], pca2d_img)
        colorscale.change(
            lambda cb, cs: viz._render_pca2d(cb, cs.lower())[0], [color_by, colorscale], pca2d_img)
        color_by.change(lambda cb, cs: viz._scatter3d(cb, cs), [color_by, colorscale], pca3d)
        colorscale.change(lambda cb, cs: viz._scatter3d(cb, cs), [color_by, colorscale], pca3d)

        btn_random.click(viz.on_random, None, slider_outs)
        btn_reset.click(viz.on_reset, None, slider_outs)

        slider_inputs = sliders
        # NOTE: .release (not .change) — dragging fires dozens of change
        # events that flood the queue (stuck "QUEUED"); render on release.
        for s in sliders:
            s.release(viz.on_sliders, [leaf_quality] + slider_inputs,
                      [rgb_out, depth_out, info_out, model3d])
        for s in pc_sliders:
            s.release(viz.on_pc_sliders, [leaf_quality] + pc_sliders,
                      sliders + [rgb_out, depth_out, info_out, model3d])
        ref_mode.change(viz.on_ref_change, [ref_mode, ref_idx, leaf_quality] + slider_inputs,
                        [rgb_out, depth_out, info_out, model3d])
        ref_idx.change(viz.on_ref_change, [ref_mode, ref_idx, leaf_quality] + slider_inputs,
                       [rgb_out, depth_out, info_out, model3d])
        leaf_quality.change(viz.on_sliders, [leaf_quality] + slider_inputs,
                            [rgb_out, depth_out, info_out, model3d])

        # Initial render via a load event: gradio only serves files from its
        # own cache dir, so the GLB must pass through the queue (not be set
        # directly as a raw /tmp path).
        def initial_render():
            return viz.on_sliders("high", *([0.0] * viz.latent_dim))
        demo.load(initial_render, None, [rgb_out, depth_out, info_out, model3d])

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, default="dataset/cache/phytomer_gui_cache")
    parser.add_argument("--ckpt", type=str,
                        default="diffusion_based/checkpoints/phytomer_vae_v2/phytomer_vae_64d_best.pt")
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    viz = PhytomerVisualizer(args.cache, args.ckpt, device)
    demo = build_app(viz)
    demo.launch(
        server_name=args.server_name, server_port=args.server_port,
        share=args.share, show_error=True,
    )


if __name__ == "__main__":
    main()
