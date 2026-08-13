"""
3-column pipeline comparison:
  Col 1: C++ Helios binary → JPEG (ground truth)
  Col 2: XML → 18D nodes → DifferentiableHeliosRenderer (Python, fixed bud bug)
  Col 3: 18D nodes → write_organ_nodes_to_xml → new XML → C++ re-render → JPEG

Output: diffusion_based/docs/report1_backprop_vs_difffusion/images/18d_pipeline_comparison.png
"""
import os, sys, subprocess, json, time, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

repo_root = "/home/lion397/codes/image-to-l-system"
sys.path.insert(0, repo_root)

from diffusion_based.models.helios_xml_parser import HeliosXMLParser
from diffusion_based.models.legacy.helios_xml_writer_track_a import write_organ_nodes_to_xml
from diffusion_based.models.legacy.helios_rasterizer_3d_track_a import HeliosGeometryRasterizer
from diffusion_based.models.legacy.differentiable_pipeline_track_a import DifferentiableHeliosRenderer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# ── Paths ─────────────────────────────────────────────────────────
output_dir   = os.path.join(repo_root, "notebooks", "output_dap30_verification")
xml_path     = os.path.join(output_dir, "dap30_gt_seed42_0000_plant_0000.xml")
cpp_binary   = os.path.join(repo_root, "Digital-Crops", "projects",
                             "syntheticdata_generation", "build", "main")
params_file  = os.path.join(repo_root, "Digital-Crops", "projects",
                             "syntheticdata_generation", "params.json")
img_out_dir  = os.path.join(repo_root, "diffusion_based", "docs",
                             "report1_backprop_vs_difffusion", "images")
os.makedirs(img_out_dir, exist_ok=True)

IMAGE_SIZE = 256

# ═══════════════════════════════════════════════════════════════════
# Helper: run C++ binary on an XML and return rendered JPEG as numpy
# ═══════════════════════════════════════════════════════════════════
def render_with_cpp(xml_source_path: str, name: str, work_dir: str) -> np.ndarray:
    """Call C++ binary to render xml_source_path. Returns (H,W,3) float32 [0,1]."""
    env = os.environ.copy()
    if "DISPLAY" not in env:
        env["DISPLAY"] = ":1.0"

    # Write a minimal params.json that points to the XML and sets camera
    with open(params_file) as f:
        params = json.load(f)
    params.setdefault("camera", {}).setdefault("positioning", {})
    params["camera"]["positioning"]["azimuth_angle"] = 0.0
    params["camera"]["positioning"]["camera_height"] = 1.0
    params["camera"]["positioning"]["focusing_plants"] = True
    params.setdefault("metadata", {})["dap"] = 30
    params["metadata"].pop("DAP", None)
    # Point to our XML directly via xml_input_path if supported,
    # otherwise rely on the binary's default output naming
    tmp_params = os.path.join(work_dir, f"{name}_params.json")
    with open(tmp_params, "w") as f:
        json.dump(params, f, indent=2)

    cmd = [
        cpp_binary,
        "--renderer", "vis",
        "--focus-plant",
        "--xml-input", xml_source_path,  # render existing XML (no plant growth)
        "-n", name,
        "--output", work_dir,
        "-f", tmp_params,
    ]
    build_dir = os.path.dirname(cpp_binary)
    print(f"  Running C++: {' '.join(cmd[-6:])}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=build_dir, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    print(f"  C++ finished in {elapsed:.1f}s (rc={res.returncode})", flush=True)

    # Try common output patterns
    for pattern in [
        os.path.join(work_dir, f"{name}_0000_vis.jpeg"),
        os.path.join(work_dir, f"{name}_0000_vis.jpg"),
        os.path.join(work_dir, f"{name}_0000.jpeg"),
        os.path.join(work_dir, f"{name}_0000.jpg"),
    ]:
        if os.path.exists(pattern):
            img = np.array(Image.open(pattern).convert("RGB")).astype(np.float32) / 255.0
            print(f"  Found C++ image: {pattern}  shape={img.shape}", flush=True)
            return img

    # Fallback: print stderr and raise
    print(f"  STDOUT: {res.stdout[-500:]}", flush=True)
    print(f"  STDERR: {res.stderr[-500:]}", flush=True)
    raise FileNotFoundError(f"C++ output image not found in {work_dir} for name={name}")


# ═══════════════════════════════════════════════════════════════════
# Col 1: C++ GT render (existing JPEG from previous run)
# ═══════════════════════════════════════════════════════════════════
print("\n── Col 1: C++ GT image ──", flush=True)
cpp_gt_jpeg = os.path.join(output_dir, "dap30_gt_seed42_0000_vis.jpeg")
if os.path.exists(cpp_gt_jpeg):
    img_col1 = np.array(Image.open(cpp_gt_jpeg).convert("RGB")).astype(np.float32) / 255.0
    print(f"  Loaded existing C++ GT: {cpp_gt_jpeg}  shape={img_col1.shape}", flush=True)
else:
    print("  Existing JPEG not found – re-running C++ binary …", flush=True)
    img_col1 = render_with_cpp(xml_path, "dap30_gt_col1", output_dir)

# ═══════════════════════════════════════════════════════════════════
# Parse 19D nodes from XML (shared between Col 2 and Col 3)
# ═══════════════════════════════════════════════════════════════════
print("\n── Parsing 19D organ nodes ──", flush=True)
parser = HeliosXMLParser(xml_path)
parser.parse()
organ_nodes = parser.get_all_organ_nodes()
print(f"  Parsed {len(organ_nodes)} organ nodes", flush=True)

nodes_np = np.stack([n.to_vec() for n in organ_nodes], axis=0)   # (N, 19)
nodes_t  = torch.tensor(nodes_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1,N,19)

def _resize(arr, h, w):
    from PIL import Image as PILImage
    img_pil = PILImage.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
    return np.array(img_pil.resize((w, h), PILImage.LANCZOS)).astype(np.float32) / 255.0

# ═══════════════════════════════════════════════════════════════════
# Col 2: 19D nodes → DifferentiableHeliosRenderer
# ═══════════════════════════════════════════════════════════════════
print("\n── Col 2: 19D → DifferentiableHeliosRenderer ──", flush=True)
rasterizer    = HeliosGeometryRasterizer(image_size=IMAGE_SIZE).to(device)
diff_renderer = DifferentiableHeliosRenderer(rasterizer).to(device)

bg_col1 = _resize(img_col1, IMAGE_SIZE, IMAGE_SIZE)
with torch.no_grad():
    rgba_18d = diff_renderer(nodes_t, focus_plant=True, background=bg_col1)
img_col2 = rgba_18d[0, :3].permute(1, 2, 0).cpu().numpy().clip(0, 1)
print(f"  19D render shape: {img_col2.shape}", flush=True)

# ═══════════════════════════════════════════════════════════════════
# Col 3: 19D nodes → XML → C++ re-render
# ═══════════════════════════════════════════════════════════════════
print("\n── Col 3: 19D → write_organ_nodes_to_xml → C++ re-render ──", flush=True)
roundtrip_xml = os.path.join(img_out_dir, "_roundtrip_dap30.xml")
write_organ_nodes_to_xml(organ_nodes, roundtrip_xml, plant_age=30)
print(f"  Wrote roundtrip XML: {roundtrip_xml}", flush=True)

img_col3 = render_with_cpp(roundtrip_xml, "dap30_roundtrip_col3", img_out_dir)

# ═══════════════════════════════════════════════════════════════════
# Metrics (against C++ GT resized to match Python render)
# ═══════════════════════════════════════════════════════════════════

h2, w2 = img_col2.shape[:2]
ref      = _resize(img_col1, h2, w2)
img_col3 = _resize(img_col3, h2, w2)  # normalise C++ output to same size as Python renderer

def mae(a, b):  return float(np.mean(np.abs(a - b)))
def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return float(20 * np.log10(1.0 / (np.sqrt(max(mse, 1e-12)))))

mae2, psnr2 = mae(ref, img_col2), psnr(ref, img_col2)
mae3, psnr3 = mae(ref, img_col3), psnr(ref, img_col3)
print(f"\n  Col 2 (19D Python) vs GT C++: MAE={mae2:.5f}  PSNR={psnr2:.1f}dB", flush=True)
print(f"  Col 3 (Roundtrip C++) vs GT C++: MAE={mae3:.5f}  PSNR={psnr3:.1f}dB", flush=True)

# ═══════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════
BG    = "#0d0f1a"
WHITE = "#e8eaf6"
C1    = "#ffffff"
C2    = "#38ef7d"
C3    = "#00d2ff"

fig, axes = plt.subplots(1, 3, figsize=(17, 6.5), dpi=160)
fig.patch.set_facecolor(BG)

panels = [
    (img_col1, "① C++ Helios GT\n(Ground Truth Renderer)", C1, None),
    (img_col2, f"② XML → 19D → DifferentiableRenderer\nMAE={mae2:.5f}  PSNR={psnr2:.1f}dB",  C2, mae2),
    (img_col3, f"③ 19D → XML Roundtrip → C++ Re-render\nMAE={mae3:.5f}  PSNR={psnr3:.1f}dB", C3, mae3),
]

for ax, (img, title, color, _) in zip(axes, panels):
    ax.imshow(img)
    ax.set_title(title, color=color, fontsize=11, fontweight="bold", pad=10)
    ax.axis("off")
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2.5)
        spine.set_visible(True)

fig.suptitle(
    "Path B: XML → 19D OrganNode3D → DifferentiableRenderer / XML Roundtrip",
    color=WHITE, fontsize=13, fontweight="bold", y=1.01,
)

# Subtitle pipeline labels
fig.text(0.17, -0.01,
         "C++ Helios binary → JPEG",
         ha="center", color="#888899", fontsize=8.5,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1d2e", edgecolor="#444466"))
fig.text(0.50, -0.01,
         "parser.get_all_organ_nodes() → to_vec() → (N,19) → DifferentiableHeliosRenderer",
         ha="center", color="#888899", fontsize=8.5,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1d2e", edgecolor="#444466"))
fig.text(0.83, -0.01,
         "(N,19) → write_organ_nodes_to_xml() → XML → C++ binary → JPEG",
         ha="center", color="#888899", fontsize=8.5,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1d2e", edgecolor="#444466"))

plt.tight_layout(pad=1.2)
out_path = os.path.join(img_out_dir, "18d_pipeline_comparison.png")
plt.savefig(out_path, facecolor=BG, bbox_inches="tight", dpi=160)
plt.close()
print(f"\nSaved → {out_path}", flush=True)
