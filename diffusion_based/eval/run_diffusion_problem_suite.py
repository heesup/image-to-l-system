"""
Diffusion-based inverse rendering problem suite.

Runs the trained image-to-graph diffusion model on three difficulty levels:
  1. Easy:   input is the GT image, model generates the full plant array.
  2. Medium: input is a partial/cropped or lower-resolution image.
  3. Hard:   input is a heavily corrupted/occluded version of the image.

Output layout matches run_backprop_problem_suite.py:
    diffusion_based/eval/output/<xml_name>_diffusion/<xml_name>_diffusion_problem_{easy,medium,hard}.png
    diffusion_based/eval/output/<xml_name>_diffusion/<xml_name>_diffusion_problem_suite_metrics.json
"""

import os
import sys
import json
import re
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.organ_array_diffuser import PlantOrganArrayDiffuser
from diffusion_based.dataset.organ_array_dataset import OrganArrayDataset
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.models.plant_organ_array import PlantOrganArray


class DDPMScheduler:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_ssim_numpy(img1, img2):
    try:
        from skimage.metrics import structural_similarity as ssim
        min_dim = min(img1.shape[0], img1.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        return float(ssim(img1, img2, channel_axis=2, data_range=1.0, win_size=win_size))
    except Exception:
        mse = float(np.mean((img1 - img2) ** 2))
        return float(max(0.0, 1.0 - 5.0 * mse))


def denormalize_image(image_tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(3, 1, 1)
    return (image_tensor * std + mean).clamp(0.0, 1.0)


def normalize_image_tensor(rgb: torch.Tensor, image_size: int) -> torch.Tensor:
    """Normalize a rendered (3, H, W) tensor with ImageNet stats and resize."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return transform(rgb)


def _extract_dap_and_name(xml_path: str) -> Tuple[str, int]:
    """Infer a clean base name and DAP from an XML path like dap30_gt_0000_plant_0000.xml."""
    base = os.path.basename(xml_path)
    name = base.replace(".xml", "")
    m = re.search(r"dap(\d+)", name, re.IGNORECASE)
    dap = int(m.group(1)) if m else 10
    return name, dap


def postprocess_prediction(pred_x0: torch.Tensor, dataset: OrganArrayDataset, top_k_active: int = 24) -> PlantOrganArray:
    """Convert a single denoised prediction (N, 94) into a valid PlantOrganArray."""
    N = pred_x0.shape[0]
    denorm = dataset.denormalize(pred_x0)
    denorm[:, :93] = torch.clamp(denorm[:, :93], min=0.0)

    # Clamp discrete / count-like channels to valid integer ranges
    denorm[:, 16] = torch.clamp(torch.round(denorm[:, 16]), 1, 20)
    denorm[:, 31] = torch.clamp(torch.round(denorm[:, 31]), 0, 3)
    denorm[:, 44] = torch.clamp(torch.round(denorm[:, 44]), 0, 1)
    denorm[:, 51] = torch.clamp(torch.round(denorm[:, 51]), 1, 20)
    denorm[:, 55] = torch.clamp(torch.round(denorm[:, 55]), 0, 3)
    denorm[:, 60] = torch.clamp(torch.round(denorm[:, 60]), 0, 1)
    denorm[:, 71] = torch.clamp(torch.round(denorm[:, 71]), 0, 4)

    exist = torch.clamp(denorm[:, -1], 0.0, 1.0)
    active = exist > 0.5
    num_active = int(active.sum().item())
    if num_active == 0:
        _, topk_idx = torch.topk(exist, k=min(top_k_active, N))
        denorm[topk_idx, -1] = 1.0
        num_active = min(top_k_active, N)
    else:
        denorm[:, -1] = active.float()
    return PlantOrganArray(tensor=denorm.cpu())


def sample_organ_array_with_snapshots(
    model: PlantOrganArrayDiffuser,
    image: torch.Tensor,
    scheduler: DDPMScheduler,
    dataset: OrganArrayDataset,
    steps: int = 50,
    snapshot_steps: List[int] = None,
    top_k_active: int = 24,
) -> Tuple[PlantOrganArray, Dict[int, PlantOrganArray]]:
    """
    Deterministic DDIM reverse sampling with intermediate snapshots.

    Returns:
        final PlantOrganArray and dict {step_index: PlantOrganArray}.
    """
    device = image.device
    model.eval()

    B = 1
    N = model.max_nodes

    if snapshot_steps is None:
        snapshot_steps = [0, max(1, steps // 4), steps // 2, 3 * steps // 4, steps - 1]
    snapshot_steps = sorted(set(int(s) for s in snapshot_steps if 0 <= s < steps))

    x_t = torch.randn(B, N, 94, device=device)
    step_indices = torch.linspace(scheduler.timesteps - 1, 0, steps, device=device).long()

    snapshots: Dict[int, PlantOrganArray] = {}

    with torch.no_grad():
        for idx, t in enumerate(step_indices):
            t_batch = torch.tensor([t], device=device).long()
            pred_x0 = model(x_t, t_batch, image.unsqueeze(0))["pred_x0"]

            if idx in snapshot_steps:
                snapshots[idx] = postprocess_prediction(pred_x0[0], dataset, top_k_active)
                print(f"  Snapshot step {idx:02d} (t={t.item():.0f}): {int((snapshots[idx].tensor[:, -1] > 0.5).sum().item())} active nodes")

            alpha_t = scheduler.alphas_cumprod[t].clamp(min=1e-6)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt((1.0 - alpha_t).clamp(min=1e-6))

            # Convert x0-prediction to noise-prediction, then DDIM deterministic step.
            pred_noise = (x_t - sqrt_alpha_t * pred_x0) / sqrt_one_minus_alpha_t
            pred_noise = torch.nan_to_num(pred_noise, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

            if idx < len(step_indices) - 1:
                t_prev = step_indices[idx + 1]
                alpha_prev = scheduler.alphas_cumprod[t_prev].clamp(min=1e-6)
                sqrt_alpha_prev = torch.sqrt(alpha_prev)
                sqrt_one_minus_alpha_prev = torch.sqrt(1.0 - alpha_prev)
                x_t = sqrt_alpha_prev * pred_x0 + sqrt_one_minus_alpha_prev * pred_noise
                x_t = torch.nan_to_num(x_t, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
            else:
                x_t = pred_x0

    final = postprocess_prediction(x_t[0], dataset, top_k_active)
    print(f"  Final step {steps - 1}: {int((final.tensor[:, -1] > 0.5).sum().item())} active nodes")
    return final, snapshots


def render_organ_array(organ_array: PlantOrganArray, renderer: HeliosPyTorchRenderer, device: torch.device) -> torch.Tensor:
    try:
        rgb = renderer.render_organ_array(
            organ_array,
            azimuth_deg=0.0,
            elevation_deg=90.0,
            camera_height=1.0,
            background="black",
            device=device,
            differentiable=False,
            focus_plant=True,
            existence_threshold=0.5,
        )
    except Exception as e:
        print(f"Rendering failed: {e}")
        rgb = torch.zeros(3, renderer.image_size, renderer.image_size, device=device)
    return rgb


def make_problem_inputs(image: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Create easy/medium/hard corrupted versions of the input image."""
    C, H, W = image.shape

    # Easy: original image
    easy = image.clone()

    # Medium: random 50x50 occlusion patch + slight Gaussian noise
    medium = image.clone()
    occlusion_mask = torch.zeros_like(medium)
    ox = np.random.randint(0, H - 50)
    oy = np.random.randint(0, W - 50)
    occlusion_mask[:, ox:ox+50, oy:oy+50] = 1.0
    medium = torch.where(occlusion_mask > 0, torch.zeros_like(medium), medium)
    noise = torch.randn_like(medium) * 0.05
    medium = (medium + noise).clamp(image.min(), image.max())

    # Hard: downsample to 64x64 then resize back + heavy noise + larger occlusion
    hard = image.clone()
    small = F.interpolate(hard.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False).squeeze(0)
    hard = F.interpolate(small.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)

    occ2 = torch.zeros_like(hard)
    ox2 = np.random.randint(0, H - 100)
    oy2 = np.random.randint(0, W - 100)
    occ2[:, ox2:ox2+100, oy2:oy2+100] = 1.0
    hard = torch.where(occ2 > 0, torch.zeros_like(hard), hard)
    hard = (hard + torch.randn_like(hard) * 0.15).clamp(image.min(), image.max())

    return {
        "easy": easy,
        "medium": medium,
        "hard": hard,
    }


def plot_problem(
    target_rgb_np: np.ndarray,
    history: Dict[str, Any],
    problem: str,
    caption: str,
    output_path: str,
    dap: int = 10,
):
    """Plot target + intermediate diffusion snapshots + loss/SSIM curves + final diff map."""
    snapshot_steps = history["snapshot_steps"]
    images = history["images"]
    loss_curve = history["loss"]
    ssim_curve = history["ssim"]
    final_img = images[-1][1]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor="black")
    for row in axes:
        for ax in row:
            ax.set_facecolor("black")

    fig.suptitle(caption, color="white", fontsize=14, fontweight="bold", y=0.98)

    axes[0, 0].imshow(target_rgb_np)
    axes[0, 0].set_title(f"Target Helios GT\n(DAP {dap})", color="white", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    for idx, (step_num, img, loss_v, ssim_v) in enumerate(images):
        if idx < 3:
            ax = axes[0, idx + 1]
        else:
            ax = axes[1, 0]
        ax.imshow(img)
        ax.set_title(
            f"Step {step_num:02d}\nLoss={loss_v:.4f} | SSIM={ssim_v:.4f}",
            color="cyan", fontsize=12, fontweight="bold"
        )
        ax.axis("off")

    axes[1, 1].plot(loss_curve, color="crimson", linewidth=2.5)
    axes[1, 1].set_title("Loss Curve", color="white", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Diffusion Step", color="white")
    axes[1, 1].set_ylabel("MSE vs Target", color="crimson")
    axes[1, 1].tick_params(colors="white")
    axes[1, 1].grid(True, linestyle="--", alpha=0.3)

    axes[1, 2].plot(ssim_curve, color="springgreen", linewidth=2.5)
    axes[1, 2].set_title("SSIM Curve", color="white", fontsize=12, fontweight="bold")
    axes[1, 2].set_xlabel("Diffusion Step", color="white")
    axes[1, 2].set_ylabel("SSIM", color="springgreen")
    axes[1, 2].tick_params(colors="white")
    axes[1, 2].grid(True, linestyle="--", alpha=0.3)

    final_diff = np.abs(final_img - target_rgb_np)
    im = axes[1, 3].imshow(final_diff.mean(axis=-1), cmap="inferno", vmin=0.0, vmax=0.2)
    axes[1, 3].set_title(f"Final Diff Map\nMAE={np.mean(final_diff):.5f}", color="gold", fontsize=12, fontweight="bold")
    axes[1, 3].axis("off")
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"Saved {problem} figure to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="diffusion_based/checkpoints/organ_array_diffuser_norm.pt")
    parser.add_argument("--single_xml", type=str, default="dataset/helios_data/cowpea_dap001_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml")
    parser.add_argument("--data_root", type=str, default="dataset/helios_data")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save outputs. Default is diffusion_based/eval/output/<xml_name>_diffusion")
    parser.add_argument("--max_nodes", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    xml_name, dap = _extract_dap_and_name(args.single_xml)
    if args.output_dir is None:
        args.output_dir = os.path.join("diffusion_based", "eval", "output", f"{xml_name}_diffusion")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"Running diffusion problem suite on device: {device}")
    print(f"Source XML: {args.single_xml} (DAP {dap})")
    print(f"Output dir: {args.output_dir}")

    dataset = OrganArrayDataset(
        data_root=args.data_root,
        max_nodes=args.max_nodes,
        image_size=args.image_size,
        single_xml_path=args.single_xml,
        device=device,
    )

    model = PlantOrganArrayDiffuser(
        max_nodes=args.max_nodes,
        node_dim=94,
        embed_dim=256,
        num_layers=4,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {args.checkpoint}")

    scheduler = DDPMScheduler(timesteps=1000)

    # Use the Helios-rendered GT image as the easy input.
    # If a matching *_vis.jpeg exists, dataset[0]["image"] is used; otherwise render directly.
    gt_organ_array = PlantOrganArray.from_xml_file(args.single_xml)
    renderer = HeliosPyTorchRenderer(image_size=args.image_size).to(device)
    target_rgb = render_organ_array(gt_organ_array, renderer, device)
    target_rgb_np = target_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)

    try:
        gt_image = dataset[0]["image"].to(device)
    except FileNotFoundError:
        gt_image = normalize_image_tensor(target_rgb, args.image_size).to(device)

    problem_inputs = make_problem_inputs(gt_image)

    captions = {
        "easy": f"DIFFUSION DAP{dap} - EASY: GT image as condition. Full 94D PlantOrganArray generation.",
        "medium": f"DIFFUSION DAP{dap} - MEDIUM: Occluded + noisy image as condition.",
        "hard": f"DIFFUSION DAP{dap} - HARD: Heavily corrupted low-res image as condition.",
    }

    snapshot_steps = [0, max(1, args.steps // 4), args.steps // 2, 3 * args.steps // 4, args.steps - 1]

    all_metrics = {}

    for problem in ["easy", "medium", "hard"]:
        print(f"\n=== PROBLEM {problem.upper()} ===")
        input_img = problem_inputs[problem]

        final_array, snapshots = sample_organ_array_with_snapshots(
            model, input_img, scheduler, dataset, steps=args.steps, snapshot_steps=snapshot_steps
        )

        # Render snapshots and final, recording MAE/SSIM
        history = {"snapshot_steps": snapshot_steps, "images": [], "loss": [], "ssim": []}
        for step_num in snapshot_steps:
            arr = snapshots[step_num]
            rgb = render_organ_array(arr, renderer, device)
            rgb_np = rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            mae = float(np.mean(np.abs(rgb_np - target_rgb_np)))
            ssim = compute_ssim_numpy(rgb_np, target_rgb_np)
            history["images"].append((step_num, rgb_np, mae, ssim))
            history["loss"].append(mae)
            history["ssim"].append(ssim)

        # Final
        final_rgb = render_organ_array(final_array, renderer, device)
        final_np = final_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)
        final_mae = float(np.mean(np.abs(final_np - target_rgb_np)))
        final_ssim = compute_ssim_numpy(final_np, target_rgb_np)

        metrics = {"mae": final_mae, "ssim": final_ssim}
        all_metrics[problem] = metrics
        print(f"  Final MAE={final_mae:.6f} SSIM={final_ssim:.4f}")

        output_path = os.path.join(args.output_dir, f"{xml_name}_diffusion_problem_{problem}.png")
        plot_problem(target_rgb_np, history, problem, captions[problem], output_path, dap=dap)

        xml_path = os.path.join(args.output_dir, f"{xml_name}_diffusion_problem_{problem}_pred.xml")
        final_array.write_xml(xml_path)
        print(f"  Saved predicted XML to {xml_path}")

    metrics_path = os.path.join(args.output_dir, f"{xml_name}_diffusion_problem_suite_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics saved to {metrics_path}")
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
