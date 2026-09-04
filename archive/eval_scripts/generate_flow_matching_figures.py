"""
Flow-Matching Generation + Render-Loss-Guided Sampling figures.

Trains/loads the part flow-matching model, samples part tensors (from a Gaussian
prior or an empty-plant prior), optionally steers sampling with a render-loss
guidance term, renders the results, and compares against ground truth.

Outputs:
  docs/results/assets/fig5_vit_diffusion_generative.png
  docs/results/assets/fig6_loss_convergence_trajectories.png
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.dataset.part_array_dataset import (
    PartArrayDataset, FM_OT_END, EMPTY_IDX,
)
from diffusion_based.models.part_flow_matching import PartFlowMatchingModel
from diffusion_based.training.flow_matching import FlowMatchingScheduler
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.models.plant_organ_array import (
    PlantOrganArray, P_COL_EXISTENCE, P_COL_ORGAN_TYPE,
)


def _to_np(t):
    return t.permute(1, 2, 0).cpu().clamp(0, 1).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="diffusion_based/checkpoints/fm/part_flow_matching.pt")
    parser.add_argument("--assets_dir", default="docs/results/assets")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--guidance_weight", type=float, default=0.0)
    parser.add_argument("--empty_prior", action="store_true")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--max_nodes", type=int, default=2048)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.assets_dir, exist_ok=True)

    # Dataset (for normalization + a few GT samples)
    ds = PartArrayDataset(
        data_root="dataset/helios_data", max_nodes=args.max_nodes,
        image_size=args.image_size, device=device, use_gt_renderer_image=True,
        cache_dir="dataset/helios_data_14d_cache",
    )

    # Model
    model = PartFlowMatchingModel(
        max_nodes=args.max_nodes, node_dim=ds.node_dim,
        image_size=args.image_size, patch_size=8, embed_dim=256,
        encoder_layers=6, decoder_layers=4, num_heads=8,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded flow-matching checkpoint (epoch {ckpt.get('epoch','?')})")

    scheduler = FlowMatchingScheduler()
    renderer = HeliosPyTorchRenderer(image_size=args.image_size).to(device)

    # Pick a few GT samples for comparison
    sample_idxs = [0, 1, 2]
    gt_imgs, gt_img_tensors, gt_nodes = [], [], []
    for i in sample_idxs:
        s = ds[i]
        gt_imgs.append(_to_np(s["image"]))
        gt_img_tensors.append(s["image"])
        gt_nodes.append(s["nodes"])

    # Guidance function: render x_t (denormalized) and compare to a target GT image.
    def make_guidance(target_img):
        def _g(x_denorm):
            # x_denorm: (B, N, 16) world part tensor
            p = x_denorm[0]
            out = renderer.render_part_tensor_multimodal(
                p, camera_height=5.0, elevation_deg=89.88, device=device,
                return_mask=False, return_organ_masks=False, soft_existence=True,
            )
            return F.l1_loss(out["rgb"], target_img)
        return _g

    # Sample from the model (optionally with guidance toward the first GT image).
    target_img = gt_img_tensors[0].to(device) if args.guidance_weight > 0 else None
    guidance_fn = make_guidance(target_img) if target_img is not None else None

    x0 = None
    if args.empty_prior:
        # True Zero-Plant Prior: all coordinates zero, all slots EMPTY
        x0 = torch.zeros((1, args.max_nodes, ds.node_dim), device=device)
        x0[:, :, EMPTY_IDX] = 1.0

    with torch.no_grad():
        cond_img = gt_img_tensors[0].unsqueeze(0).to(device)
        x1 = scheduler.sample(
            model, cond_img, num_steps=args.num_steps, node_dim=ds.node_dim,
            max_nodes=args.max_nodes, device=device, x0=x0,
            guidance_fn=guidance_fn, guidance_weight=args.guidance_weight,
            denormalize_fn=ds.decode_fm,
        )

    # Render the generated part tensor.
    gen_p = ds.decode_fm(x1[0])
    with torch.no_grad():
        gen_rgb = renderer.render_part_tensor_multimodal(
            gen_p, camera_height=5.0, elevation_deg=89.88, device=device,
            return_mask=False, return_organ_masks=False,
        )["rgb"]

    # Figure: GT vs generated (vs guided).
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#1a1a2e")
    axes[0].imshow(gt_imgs[0]); axes[0].set_title("Ground Truth", color="white")
    axes[1].imshow(_to_np(gen_rgb)); axes[1].set_title("Flow-Matching Generated", color="white")
    axes[2].imshow(np.abs(_to_np(gen_rgb) - gt_imgs[0]), cmap="hot")
    axes[2].set_title("Abs Diff", color="white")
    for ax in axes:
        ax.axis("off"); ax.set_facecolor("#0d0d1a")
    out = os.path.join(args.assets_dir, "fig5_vit_diffusion_generative.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
