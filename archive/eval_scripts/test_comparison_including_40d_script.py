import os, json, math, torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer

DEVICE = torch.device('cuda:0')
IMG_SIZE = 512
renderer = HeliosPyTorchRenderer(image_size=IMG_SIZE).to(DEVICE)

daps = [10, 35, 70]
n_rows = len(daps)
n_cols = 5

fig, axes = plt.subplots(n_rows, n_cols, figsize=(26, 5.5 * n_rows))
fig.patch.set_facecolor('#0B0D17')

col_titles = [
    'Helios C++\nInput RGB',
    'Helios C++\nGT Mask (_masks.json)',
    '40D Typed Tensor Render\n(Direct render_organ_array)',
    '94D Legacy FK Render\n(to_legacy_tensor_diff)',
    'Mask Comparison Overlay\n(Green: Match, Red: FP, Blue: FN)'
]
col_colors = ['#E2E8F0', '#38BDF8', '#F59E0B', '#4ADE80', '#F472B6']

for col, (title, color) in enumerate(zip(col_titles, col_colors)):
    axes[0, col].set_title(title, fontsize=13, fontweight='bold', color=color, pad=10)

for row_idx, d in enumerate(daps):
    print(f"Processing DAP {d}...")
    prefix = f'cowpea_dap{d:03d}_seed00_caz000_h1.0_se045_saz180_0000'
    xml_path = f'dataset/helios_data/cowpea/{prefix}_plant_0000.xml'
    img_path = f'dataset/helios_data/cowpea/{prefix}_rad.jpeg'
    cam_path = f'dataset/helios_data/cowpea/{prefix}_camera.json'
    mask_path = f'dataset/helios_data/cowpea/{prefix}_masks.json'

    # 1. Helios GT RGB
    pil_gt = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    gt_rgb = np.array(pil_gt) / 255.0

    # 2. Camera
    with open(cam_path) as f:
        cam_data = json.load(f)
    f_len = cam_data.get('camera_properties', {}).get('focal_length', 50.0)
    s_w = cam_data.get('camera_properties', {}).get('sensor_width', 35.0)
    cam_h = float(cam_data.get('acquisition_properties', {}).get('camera_height_m', 5.0))
    cam_el = float(cam_data.get('acquisition_properties', {}).get('camera_angle_deg', 90.0))
    cam_hfov = 2.0 * math.degrees(math.atan((s_w * 0.5) / max(f_len, 1e-3)))

    # 3. GT Mask
    with open(mask_path) as f:
        mask_data = json.load(f)
    src_w = mask_data.get('images', [{}])[0].get('width', 720)
    src_h = mask_data.get('images', [{}])[0].get('height', 720)
    sx = IMG_SIZE / src_w
    sy = IMG_SIZE / src_h

    canvas = Image.new('L', (IMG_SIZE, IMG_SIZE), 0)
    draw = ImageDraw.Draw(canvas)
    for ann in mask_data.get('annotations', []):
        for poly in ann.get('segmentation', []):
            pts = [(poly[i] * sx, poly[i+1] * sy) for i in range(0, len(poly), 2)]
            if len(pts) >= 3:
                draw.polygon(pts, fill=1)
    gt_mask = np.array(canvas, dtype=np.uint8)

    # 4. 40D Typed Tensor Render
    arr_40d = PlantOrganArray.from_xml_file(xml_path) # Typed (N, 40)
    render_40d_t = renderer.render_organ_array(
        arr_40d, azimuth_deg=0.0, elevation_deg=cam_el, camera_height=cam_h,
        background='white', focus_plant=False, hfov_override_deg=cam_hfov, device=DEVICE
    )
    rgb_40d = render_40d_t.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()

    # 5. 94D Legacy FK Render
    leg_tensor = arr_40d.to_legacy_tensor_diff()
    arr_94d = PlantOrganArray(leg_tensor, raw_metadata=[])
    mesh_94d = renderer.geo_builder.build_mesh_from_organ_array(arr_94d, device=DEVICE)
    render_94d_t = renderer.render_mesh(
        mesh_94d, azimuth_deg=0.0, elevation_deg=cam_el, camera_height=cam_h,
        background='white', focus_plant=False, hfov_override_deg=cam_hfov, image_size=IMG_SIZE
    )
    rgb_94d = render_94d_t.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()

    depth_t = renderer.render_depth(
        mesh_94d, azimuth_deg=0.0, elevation_deg=cam_el, camera_height=cam_h,
        focus_plant=False, hfov_override_deg=cam_hfov, image_size=IMG_SIZE
    )
    pytorch_mask = (depth_t.detach().cpu().numpy() > 0).astype(np.uint8)

    # 6. Metrics
    intersection = np.logical_and(gt_mask > 0, pytorch_mask > 0)
    union = np.logical_or(gt_mask > 0, pytorch_mask > 0)
    iou = float(np.sum(intersection)) / max(float(np.sum(union)), 1.0)
    dice = 2.0 * float(np.sum(intersection)) / max(float(np.sum(gt_mask > 0) + np.sum(pytorch_mask > 0)), 1.0)
    precision = float(np.sum(intersection)) / max(float(np.sum(pytorch_mask > 0)), 1.0)
    recall = float(np.sum(intersection)) / max(float(np.sum(gt_mask > 0)), 1.0)
    print(f'DAP {d:02d}: IoU={iou*100:.2f}%, Dice={dice*100:.2f}%, Prec={precision*100:.2f}%, Rec={recall*100:.2f}%')

    # 7. Error Overlay Map
    overlay_rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    overlay_rgb[np.logical_and(gt_mask > 0, pytorch_mask == 0)] = [0.15, 0.45, 1.0]   # Blue: Helios only
    overlay_rgb[np.logical_and(gt_mask == 0, pytorch_mask > 0)] = [1.0, 0.25, 0.25]  # Red: PyTorch only
    overlay_rgb[intersection] = [0.2, 0.9, 0.35]                                     # Green: Match

    # Plots
    axes[row_idx, 0].imshow(gt_rgb)
    axes[row_idx, 0].set_ylabel(f'DAP {d:02d}\n({arr_40d.tensor.shape[0]} Organs)', fontsize=12, fontweight='bold', color='white', rotation=0, labelpad=55, va='center')
    axes[row_idx, 0].axis('off')

    gt_mask_disp = np.stack([gt_mask * 0.2, gt_mask * 0.75, gt_mask * 0.95], axis=-1)
    axes[row_idx, 1].imshow(gt_mask_disp)
    axes[row_idx, 1].axis('off')
    axes[row_idx, 1].text(0.03, 0.03, f'GT Pixels: {np.sum(gt_mask):,}', transform=axes[row_idx, 1].transAxes,
                         fontsize=9, color='#38BDF8', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

    axes[row_idx, 2].imshow(rgb_40d)
    axes[row_idx, 2].axis('off')
    axes[row_idx, 2].text(0.03, 0.03, '40D Typed Render', transform=axes[row_idx, 2].transAxes,
                         fontsize=9, color='#F59E0B', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

    axes[row_idx, 3].imshow(rgb_94d)
    axes[row_idx, 3].axis('off')
    axes[row_idx, 3].text(0.03, 0.03, f'94D FK Pixels: {np.sum(pytorch_mask):,}', transform=axes[row_idx, 3].transAxes,
                         fontsize=9, color='#4ADE80', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

    axes[row_idx, 4].imshow(overlay_rgb)
    axes[row_idx, 4].axis('off')
    axes[row_idx, 4].text(0.03, 0.03, f'IoU: {iou*100:.1f}% | Dice: {dice*100:.1f}%\nPrec: {precision*100:.1f}% | Rec: {recall*100:.1f}%',
                         transform=axes[row_idx, 4].transAxes,
                         fontsize=10, fontweight='bold', color='white',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='#111827', edgecolor='#4B5563', alpha=0.85))

plt.subplots_adjust(wspace=0.06, hspace=0.12)
out_path = 'diffusion_based/eval/test_comparison_including_40d.png'
plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='#0B0D17')
print('Successfully saved:', out_path)
