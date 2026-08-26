"""
18D XML Roundtrip Test:
  XML → parse → (N,18) nodes → write_organ_nodes_to_xml → re-parse → (N,18) nodes'
  Checks:
    1. Node count preserved
    2. 18D vector L2 error per-channel
    3. Rendered pixel similarity (18D path)

Run from repo root:
  python diffusion_based/models/test_18d_xml_roundtrip.py
"""
import os, sys, json, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo_root = "/home/lion397/codes/image-to-l-system"
sys.path.insert(0, repo_root)

from diffusion_based.models.helios_xml_parser import HeliosXMLParser, OrganNode3D
from diffusion_based.models.legacy.helios_xml_writer_track_a import write_organ_nodes_to_xml
from diffusion_based.models.legacy.helios_rasterizer_3d_track_a import HeliosGeometryRasterizer
from diffusion_based.models.legacy.differentiable_pipeline_track_a import DifferentiableHeliosRenderer

# ── Config ────────────────────────────────────────────────────────
xml_path = os.path.join(repo_root, "notebooks", "output_dap30_verification",
                        "dap30_gt_seed42_0000_plant_0000.xml")
out_dir  = os.path.join(repo_root, "diffusion_based", "docs",
                        "report1_backprop_vs_difffusion", "images")
os.makedirs(out_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)
IMAGE_SIZE = 256

# ── Step 1: Parse original XML → 18D nodes ───────────────────────
print("\n[1] Parsing original XML …", flush=True)
parser_a = HeliosXMLParser(xml_path); parser_a.parse()
nodes_a   = parser_a.get_all_organ_nodes()
nodes_np_a = np.stack([n.to_vec() for n in nodes_a])  # (N, 18)
parents_a  = np.array([n.parent_idx for n in nodes_a])
print(f"    {len(nodes_a)} organ nodes, 18D array shape: {nodes_np_a.shape}", flush=True)

# ── Step 2: Write 18D → XML roundtrip file ───────────────────────
print("\n[2] Writing roundtrip XML …", flush=True)
roundtrip_xml = os.path.join(out_dir, "_test_roundtrip.xml")
write_organ_nodes_to_xml(nodes_a, roundtrip_xml, plant_age=30)
print(f"    Wrote: {roundtrip_xml}", flush=True)

# ── Step 3: Re-parse roundtrip XML → 18D nodes' ──────────────────
print("\n[3] Re-parsing roundtrip XML …", flush=True)
parser_b = HeliosXMLParser(roundtrip_xml); parser_b.parse()
nodes_b   = parser_b.get_all_organ_nodes()
nodes_np_b = np.stack([n.to_vec() for n in nodes_b])  # (N', 18)
parents_b  = np.array([n.parent_idx for n in nodes_b])
print(f"    {len(nodes_b)} organ nodes, 18D array shape: {nodes_np_b.shape}", flush=True)

# ── Step 4: Per-channel 18D error analysis ───────────────────────
print("\n[4] 18D vector roundtrip error …", flush=True)
N = min(len(nodes_a), len(nodes_b))
a, b = nodes_np_a[:N], nodes_np_b[:N]
per_channel_mae  = np.mean(np.abs(a - b), axis=0)   # (18,)
per_channel_rmse = np.sqrt(np.mean((a - b)**2, axis=0))

CHANNEL_NAMES = [
    "x", "y", "z",                                       # 0-2
    "length", "radius",                                   # 3-4
    "dir_x", "dir_y", "dir_z",                           # 5-7
    "oh_INTERNODE", "oh_PETIOLE", "oh_LEAF",              # 8-10
    "oh_FLORAL_BUD", "oh_FLOWER", "oh_POD",               # 11-13
    "shoot_id", "phytomer_idx",                           # 14-15
    "existence", "head_radius",                           # 16-17
    "parent_idx",                                         # 18
]

print(f"    {'Channel':<18} {'MAE':>10} {'RMSE':>10}")
print(f"    {'-'*40}")
for ch, (name, mae_v, rmse_v) in enumerate(zip(CHANNEL_NAMES, per_channel_mae, per_channel_rmse)):
    flag = "  ⚠" if mae_v > 0.01 else ""
    print(f"    [{ch:2d}] {name:<14} {mae_v:>10.6f} {rmse_v:>10.6f}{flag}", flush=True)

overall_mae  = float(np.mean(np.abs(a - b)))
overall_rmse = float(np.sqrt(np.mean((a - b)**2)))
print(f"\n    Overall MAE={overall_mae:.6f}  RMSE={overall_rmse:.6f}", flush=True)

# ── Step 5: Rendered image comparison (19D path) ─────────────────
print("\n[5] Rendering both 19D tensors …", flush=True)
rasterizer    = HeliosGeometryRasterizer(image_size=IMAGE_SIZE).to(device)
diff_renderer = DifferentiableHeliosRenderer(rasterizer).to(device)

def _render(nodes_np):
    t = torch.tensor(nodes_np, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        rgba = diff_renderer(t, focus_plant=True, background="black")
    return rgba[0, :3].permute(1, 2, 0).cpu().numpy().clip(0, 1)

img_a = _render(nodes_np_a)
img_b = _render(nodes_np_b)

render_mae  = float(np.mean(np.abs(img_a - img_b)))
render_psnr = float(20 * np.log10(1.0 / (np.sqrt(np.mean((img_a - img_b)**2)) + 1e-8)))
print(f"    Pixel MAE={render_mae:.6f}  PSNR={render_psnr:.1f}dB", flush=True)

# ── Step 6: Save report JSON ──────────────────────────────────────
report = {
    "xml_original":      xml_path,
    "xml_roundtrip":     roundtrip_xml,
    "n_nodes_original":  len(nodes_a),
    "n_nodes_roundtrip": len(nodes_b),
    "overall_mae_18d":   overall_mae,
    "overall_rmse_18d":  overall_rmse,
    "per_channel_mae":   {name: float(v) for name, v in zip(CHANNEL_NAMES, per_channel_mae)},
    "render_pixel_mae":  render_mae,
    "render_pixel_psnr": render_psnr,
}
json_out = os.path.join(out_dir, "18d_xml_roundtrip_report.json")
with open(json_out, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nSaved report → {json_out}", flush=True)

# ── Step 7: Figure ────────────────────────────────────────────────
BG = "#0d0f1a"; WHITE = "#e8eaf6"; GREEN = "#38ef7d"; CYAN = "#00d2ff"

fig = plt.figure(figsize=(18, 10), dpi=150, facecolor=BG)
gs  = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3)

def _iax(pos, img, title, color):
    ax = fig.add_subplot(pos)
    ax.imshow(img); ax.axis("off"); ax.set_facecolor(BG)
    ax.set_title(title, color=color, fontsize=10, fontweight="bold")

_iax(gs[0, 0], img_a, "Original 18D → Render\n(source XML)", GREEN)
_iax(gs[0, 1], img_b, f"Roundtrip 18D → Render\n(via write/re-parse)", CYAN)

diff_img = np.abs(img_a - img_b).mean(axis=-1)
ax_diff = fig.add_subplot(gs[0, 2])
im = ax_diff.imshow(diff_img, cmap="inferno", vmin=0, vmax=0.15)
ax_diff.set_title(f"Pixel Diff Map\nMAE={render_mae:.5f}  PSNR={render_psnr:.1f}dB",
                  color="#ffdd59", fontsize=10, fontweight="bold")
ax_diff.axis("off")
plt.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04).ax.tick_params(colors="white")

# Per-channel MAE bar chart
ax_bar = fig.add_subplot(gs[0, 3])
ax_bar.set_facecolor("#1a1d29")
colors_bar = ["#ff4b5c" if v > 0.01 else "#38ef7d" for v in per_channel_mae]
ax_bar.barh(range(19), per_channel_mae, color=colors_bar, edgecolor="#333344")
ax_bar.set_yticks(range(19))
ax_bar.set_yticklabels(CHANNEL_NAMES, fontsize=7, color=WHITE)
ax_bar.set_title("Per-Channel MAE\n(red = >0.01)", color=WHITE, fontsize=10, fontweight="bold")
ax_bar.set_xlabel("MAE", color=WHITE, fontsize=8)
ax_bar.tick_params(colors=WHITE)
ax_bar.invert_yaxis()
ax_bar.axvline(0.01, color="yellow", ls=":", lw=1, alpha=0.7)

# Summary text
ax_txt = fig.add_subplot(gs[1, :])
ax_txt.set_facecolor("#0d0f1a")
ax_txt.axis("off")
summary = (
    f"18D XML Roundtrip Test — DAP 30 Plant\n"
    f"  Nodes: {len(nodes_a)} (original) → {len(nodes_b)} (roundtrip)\n"
    f"  18D Overall MAE: {overall_mae:.6f}   RMSE: {overall_rmse:.6f}\n"
    f"  Pixel MAE: {render_mae:.6f}   PSNR: {render_psnr:.1f} dB\n"
    f"  Pipeline: XML → parser.get_all_organ_nodes() → to_vec() → write_organ_nodes_to_xml() → re-parse → to_vec()"
)
ax_txt.text(0.02, 0.5, summary, color=WHITE, fontsize=11, va="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#1a1d29", edgecolor="#38ef7d", lw=2))

fig.suptitle("18D OrganNode3D ↔ XML Roundtrip Fidelity Test",
             color=WHITE, fontsize=14, fontweight="bold")
plt.tight_layout()
fig_out = os.path.join(out_dir, "18d_xml_roundtrip_test.png")
plt.savefig(fig_out, facecolor=BG, bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved figure → {fig_out}", flush=True)
