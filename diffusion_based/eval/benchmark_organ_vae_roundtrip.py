"""
Benchmark script: Trains OrganLatentVAE on organ cache and evaluates
full 3D rendering round-trip fidelity using HeliosPyTorchRenderer.

Produces:
- Numerical accuracy table (Silhouette IoU, Depth MAE, Cls Acc, Scale MAE)
- docs/results/assets/fig_organ_vae_roundtrip_comparison.png
"""

import os
import sys
import time
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.organ_latent_vae import OrganLatentVAE
from diffusion_based.dataset.part_array_dataset import (
    PartArrayDataset,
    NUM_ORGAN_TYPES,
    FM_OT_END,
    FM_BASE_START,
    FM_BASE_END,
    FM_ROT_START,
    FM_ROT_END,
    FM_SCALE_START,
    FM_SCALE_END,
    FM_CURV,
    FM_NODE_DIM,
    BASE_SCALE,
    SCALE_SCALE,
    CURV_SCALE,
    decode_fm,
)
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder


def collect_organ_dataset(dataset: PartArrayDataset, num_plants: int = 500) -> torch.Tensor:
    """Collects active physical organ vectors (classes 3..12) from dataset."""
    print(f"Collecting organ vectors from {num_plants} cached plants...")
    all_organs = []
    
    indices = np.linspace(0, len(dataset) - 1, num_plants, dtype=int)
    for idx in indices:
        item = dataset[idx]
        nodes = item["nodes"]  # (N_max, 26)
        exist = item["existence_mask"]  # (N_max,)
        ot = nodes[:, :FM_OT_END].argmax(dim=-1)
        
        # Keep active physical organs (classes >= 3: internode, petiole, leaf, etc.)
        valid = (exist > 0.5) & (ot >= 3)
        if valid.any():
            all_organs.append(nodes[valid])
            
    all_organs_t = torch.cat(all_organs, dim=0)
    print(f"Collected {all_organs_t.shape[0]:,} valid 3D physical organs.")
    return all_organs_t


def train_organ_vae(
    organs: torch.Tensor,
    latent_dim: int = 16,
    hidden_dim: int = 128,
    epochs: int = 25,
    batch_size: int = 2048,
    lr: float = 1e-3,
    beta_kl: float = 5e-4,
    device: torch.device = torch.device("cuda:0"),
) -> OrganLatentVAE:
    """Trains the OrganLatentVAE on GPU."""
    model = OrganLatentVAE(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    dataset = TensorDataset(organs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    print(f"\n--- Training OrganLatentVAE ({latent_dim}D Latent) on {device} ---")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_acc = 0.0
        n_batches = 0
        
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            losses = model.compute_loss(out, batch_x, beta_kl=beta_kl)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += losses["loss"].item()
            total_recon += losses["recon_loss"].item()
            total_kl += losses["loss_kl"].item()
            total_acc += losses["cls_acc"].item()
            n_batches += 1
            
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == epochs:
            avg_loss = total_loss / n_batches
            avg_recon = total_recon / n_batches
            avg_kl = total_kl / n_batches
            avg_acc = (total_acc / n_batches) * 100.0
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}) | "
                f"Cls Acc: {avg_acc:.1f}% | Time: {time.time() - start_time:.1f}s"
            )
            
    return model


def run_roundtrip_rendering_benchmark(
    model: OrganLatentVAE,
    dataset: PartArrayDataset,
    renderer: HeliosPyTorchRenderer,
    test_dap_indices: list,
    output_image_path: str = "docs/results/assets/fig_organ_vae_roundtrip_comparison.png",
    device: torch.device = torch.device("cuda:0"),
):
    """Encodes plant organ arrays to latent z, decodes back, and compares 3D renderings."""
    model.eval()
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    
    num_samples = len(test_dap_indices)
    fig, axes = plt.subplots(num_samples, 5, figsize=(22, 4.4 * num_samples))
    fig.patch.set_facecolor("#111216")
    
    metrics = {
        "iou": [],
        "depth_mae": [],
        "cls_acc": [],
        "base_mae_cm": [],
        "scale_mae_cm": [],
    }
    
    with torch.no_grad():
        for row_idx, sample_idx in enumerate(test_dap_indices):
            sample = dataset[sample_idx]
            dap = int(sample["dap"].item())
            nodes = sample["nodes"].to(device)
            exist = sample["existence_mask"].to(device)
            
            # Active organs
            ot = nodes[:, :FM_OT_END].argmax(dim=-1)
            valid_mask = (exist > 0.5) & (ot >= 3)
            num_organs = int(valid_mask.sum().item())
            
            if num_organs == 0:
                continue
                
            gt_valid_nodes = nodes[valid_mask]  # (M, 26)
            
            # 1. Full VAE Round Trip: x_26 -> z_16 -> recon_26
            out = model(gt_valid_nodes)
            recon_nodes = out["recon_26d"]
            
            # Quantitative Metric Computations
            gt_cls = ot[valid_mask]
            pred_cls = out["cls_logits"].argmax(dim=-1)
            cls_acc = (pred_cls == gt_cls).float().mean().item() * 100.0
            
            base_err_cm = (out["base"] - gt_valid_nodes[:, FM_BASE_START:FM_BASE_END]).abs().mean().item() / BASE_SCALE * 100.0
            scale_err_cm = (out["scale"] - gt_valid_nodes[:, FM_SCALE_START:FM_SCALE_END]).abs().mean().item() / SCALE_SCALE * 100.0
            
            # 2. Decode to Canonical 14D Part Tensors for 3D Rendering
            gt_14d = decode_fm(gt_valid_nodes)
            recon_14d = decode_fm(recon_nodes)
            
            # 3. Differentiably render Ground Truth 3D Mesh
            gt_mesh = renderer.geo_builder.build_mesh_from_part_tensor(gt_14d, device=device)
            gt_render = renderer.forward(
                gt_mesh,
                azimuth_deg=0.0,
                elevation_deg=90.0,
                camera_height=5.0,
                background="ground",
                focus_plant=True,
                include_depth=True,
                zoom_factor=1.0,
                reference_window_size=1.2,
            )
            gt_rgb = gt_render[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            gt_depth = gt_render[3].clamp(min=0.0).cpu().numpy()
            
            # 4. Differentiably render VAE Reconstructed 3D Mesh
            recon_mesh = renderer.geo_builder.build_mesh_from_part_tensor(recon_14d, device=device)
            recon_render = renderer.forward(
                recon_mesh,
                azimuth_deg=0.0,
                elevation_deg=90.0,
                camera_height=5.0,
                background="ground",
                focus_plant=True,
                include_depth=True,
                zoom_factor=1.0,
                reference_window_size=1.2,
            )
            recon_rgb = recon_render[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            recon_depth = recon_render[3].clamp(min=0.0).cpu().numpy()
            
            # 5. Compute Visual Quality Metrics
            gt_mask = (gt_depth > 0.005)
            recon_mask = (recon_depth > 0.005)
            inter = np.logical_and(gt_mask, recon_mask).sum()
            union = np.logical_or(gt_mask, recon_mask).sum()
            iou = float(inter / max(union, 1)) * 100.0
            
            canopy_mask = np.logical_or(gt_mask, recon_mask)
            depth_mae_cm = float(np.abs(recon_depth[canopy_mask] - gt_depth[canopy_mask]).mean()) * 100.0 if canopy_mask.any() else 0.0
            
            metrics["iou"].append(iou)
            metrics["depth_mae"].append(depth_mae_cm)
            metrics["cls_acc"].append(cls_acc)
            metrics["base_mae_cm"].append(base_err_cm)
            metrics["scale_mae_cm"].append(scale_err_cm)
            
            # Depth Error Heatmap
            err_map = np.abs(recon_depth - gt_depth)
            err_map[~canopy_mask] = 0.0
            
            # Plotting
            ax_row = axes[row_idx] if num_samples > 1 else axes
            
            # Col 0: GT RGB
            ax_row[0].imshow(gt_rgb)
            ax_row[0].set_title(f"1. Ground Truth 3D Mesh\n(DAP {dap}, {num_organs} organs)", color="#FFD166", fontsize=11, fontweight="bold")
            ax_row[0].axis("off")
            
            # Col 1: GT Depth
            ax_row[1].imshow(gt_depth, cmap="viridis", vmin=0.0, vmax=0.6)
            ax_row[1].set_title("2. GT 3D Depth (CHM)", color="#06D6A0", fontsize=11, fontweight="bold")
            ax_row[1].axis("off")
            
            # Col 2: VAE Reconstructed RGB
            ax_row[2].imshow(recon_rgb)
            ax_row[2].set_title(f"3. VAE Round-Trip (z in R^16)\nIoU: {iou:.1f}% | ClsAcc: {cls_acc:.1f}%", color="#118AB2", fontsize=11, fontweight="bold")
            ax_row[2].axis("off")
            
            # Col 3: VAE Depth
            ax_row[3].imshow(recon_depth, cmap="viridis", vmin=0.0, vmax=0.6)
            ax_row[3].set_title("4. VAE Reconstructed Depth", color="#06D6A0", fontsize=11, fontweight="bold")
            ax_row[3].axis("off")
            
            # Col 4: Depth Error Heatmap
            im_err = ax_row[4].imshow(err_map * 100.0, cmap="inferno", vmin=0.0, vmax=10.0)
            ax_row[4].set_title(f"5. Depth Error (cm)\nMAE: {depth_mae_cm:.2f} cm", color="#EF476F", fontsize=11, fontweight="bold")
            ax_row[4].axis("off")
            
    fig.suptitle(
        f"Option B: Organ Latent VAE (26D -> z in R^16 -> 26D) Round-Trip Benchmark\n"
        f"Mean Silhouette IoU: {np.mean(metrics['iou']):.1f}% | "
        f"Mean 3D Depth MAE: {np.mean(metrics['depth_mae']):.2f} cm | "
        f"Cls Acc: {np.mean(metrics['cls_acc']):.1f}% | "
        f"Scale MAE: {np.mean(metrics['scale_mae_cm']):.2f} cm",
        fontsize=16,
        fontweight="bold",
        color="white",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_image_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"\nSaved round-trip benchmark figure to {output_image_path}")
    
    print("\n" + "=" * 60)
    print("ORGAN LATENT VAE (OPTION B) BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Organ Classification Accuracy : {np.mean(metrics['cls_acc']):.2f} %")
    print(f"  Organ Base Position MAE       : {np.mean(metrics['base_mae_cm']):.2f} cm")
    print(f"  Organ Scale MAE (Length/Width): {np.mean(metrics['scale_mae_cm']):.2f} cm")
    print(f"  Rendered Silhouette IoU       : {np.mean(metrics['iou']):.2f} %")
    print(f"  Rendered 3D Canopy Depth MAE  : {np.mean(metrics['depth_mae']):.2f} cm")
    print("=" * 60)
    return metrics


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = PartArrayDataset(
        data_root="dataset/helios_data/cowpea",
        max_nodes=512 * 8,
        cache_dir="dataset/cache/cowpea_curv26",
        species="cowpea",
        image_size=128,
    )
    
    renderer = HeliosPyTorchRenderer(image_size=256).to(device)
    
    # 1. Collect organ training data
    organs = collect_organ_dataset(dataset, num_plants=600)
    
    # 2. Train OrganLatentVAE
    vae_model = train_organ_vae(
        organs=organs,
        latent_dim=16,
        hidden_dim=256,
        epochs=35,
        batch_size=2048,
        lr=1e-3,
        beta_kl=3e-4,
        device=device,
    )
    
    # Save VAE checkpoint
    os.makedirs("diffusion_based/checkpoints/organ_vae", exist_ok=True)
    vae_ckpt_path = "diffusion_based/checkpoints/organ_vae/organ_latent_vae_best.pt"
    torch.save(vae_model.state_dict(), vae_ckpt_path)
    print(f"Saved trained Organ VAE weights to {vae_ckpt_path}")
    
    # 3. Select diverse mature plant DAPs (e.g., DAP 15, DAP 30, DAP 50, DAP 80)
    # The dataset is sorted by DAP, so we pick indices corresponding to different DAPs
    test_indices = [1500, 3000, 5000, 8000]
    
    run_roundtrip_rendering_benchmark(
        model=vae_model,
        dataset=dataset,
        renderer=renderer,
        test_dap_indices=test_indices,
        output_image_path="docs/results/assets/fig_organ_vae_roundtrip_comparison.png",
        device=device,
    )


if __name__ == "__main__":
    main()
