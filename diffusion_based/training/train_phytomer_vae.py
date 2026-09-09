"""
Train PhytomerVAE on canonical 8-slot phytomer packets from the cache dataset.

Packet extraction mirrors HierarchicalBotanicalMatcher clustering
(petiole bases + standalone internodes), so training targets are exactly the
supervision units the anchor-level flow matcher will consume.

Usage (workspace root):
    /home/lion397/.conda/envs/digital-crops/bin/python \\
        diffusion_based/training/train_phytomer_vae.py \\
        --latent-dim 64 --epochs 60 --max-files 4000
"""

import argparse
import glob
import os
import random
import sys
import time
from typing import Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from diffusion_based.models.phytomer_vae import PhytomerVAE
from diffusion_based.dataset.phytomer_packets import build_phytomer_packets, strip_base


def collect_packets(
    cache_dir: str,
    max_files: int,
    seed: int = 0,
    packet_cache: str = "",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extracts (P, 8, 26) packets + (P, 8) presence + (P, 6) reference_rots
    from cache files.

    Returns stacked (packets, presence) float tensors. Results are cached to
    `packet_cache` (.pt) so repeat runs skip extraction.
    """
    if packet_cache and os.path.exists(packet_cache):
        print(f"Loading cached packets from {packet_cache}")
        blob = torch.load(packet_cache, map_location="cpu", weights_only=False)
        print(f"  {blob['packets'].shape[0]:,} packets "
              f"(drop rate {blob['drop_stats'].get('dropped_organs', 0) / max(blob['drop_stats'].get('total_organs', 1), 1) * 100:.2f}%)")
        ref = blob.get("reference_rots")
        # Structural assembly: strip base columns (deterministic from petiole
        # geometry) so the VAE never sees base.
        return strip_base(blob["packets"]), blob["presence"], ref

    files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    rng = random.Random(seed)
    rng.shuffle(files)
    files = files[:max_files]

    all_packets, all_presence, all_refs = [], [], []
    stats: dict = {}
    t0 = time.time()
    for i, f in enumerate(files):
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if not (isinstance(d, dict) and "nodes" in d):
            continue
        packets, presence, _, refs = build_phytomer_packets(
            d["nodes"], d.get("existence_mask"), drop_stats=stats,
            phytomer_ids=d.get("phytomer_ids"),
        )
        if packets.shape[0] > 0:
            all_packets.append(packets)
            all_presence.append(presence)
            all_refs.append(refs)
        if (i + 1) % 500 == 0:
            print(f"  [{i + 1}/{len(files)}] packets={sum(p.shape[0] for p in all_packets):,} "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)

    packets_t = torch.cat(all_packets, dim=0)
    presence_t = torch.cat(all_presence, dim=0)
    refs_t = torch.cat(all_refs, dim=0)
    # Structural assembly: base columns are deterministic from petiole geometry,
    # so strip them before training (VAE never sees base; 216D -> 192D input).
    packets_t = strip_base(packets_t)
    print(f"Extracted {packets_t.shape[0]:,} packets from {len(files)} files "
          f"in {time.time() - t0:.0f}s "
          f"(drop rate {stats.get('dropped_organs', 0) / max(stats.get('total_organs', 1), 1) * 100:.2f}%)")
    if packet_cache:
        os.makedirs(os.path.dirname(os.path.abspath(packet_cache)), exist_ok=True)
        torch.save({"packets": packets_t, "presence": presence_t,
                    "reference_rots": refs_t, "drop_stats": stats}, packet_cache)
        print(f"Saved packet cache to {packet_cache}")
    return packets_t, presence_t, refs_t


def main():
    parser = argparse.ArgumentParser(description="Train PhytomerVAE on cache phytomer packets")
    parser.add_argument("--cache-dir", type=str, default="dataset/cache/cowpea_curv26")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta-kl", type=float, default=1e-3)
    parser.add_argument("--rot-weight", type=float, default=2.0,
                        help="Weight of the rotation term in recon loss (2.0 = OrganLatentVAE parity).")
    parser.add_argument("--symmetry-aware-rot", action="store_true",
                        help="Spin-invariant axis-direction loss on cylindrical slots "
                             "(stem/petiole/peduncle); full Frobenius elsewhere.")
    parser.add_argument("--ortho-reg-weight", type=float, default=0.0,
                        help="Orthonormality penalty on raw 6D outputs.")
    parser.add_argument("--rot-branch", action="store_true",
                        help="Route rotation through the dedicated branch "
                             "(separate pathway from z, not the shared backbone).")
    parser.add_argument("--hungarian-roles", action="store_true",
                        help="Resolve multi-GT role ambiguity by optimal bipartite matching "
                             "instead of fixed canonical order. REQUIRES warm-start "
                             "(--init-checkpoint) from a converged canonical model.")
    parser.add_argument("--init-checkpoint", type=str, default="",
                        help="Warm-start weights (.pt state_dict). Required with --hungarian-roles.")
    parser.add_argument("--max-files", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--packet-cache", type=str,
                        default="/tmp/opencode/phytomer_packets_4k.pt")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="diffusion_based/checkpoints/phytomer_vae")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    packets, presence, _refs = collect_packets(args.cache_dir, args.max_files, args.seed, args.packet_cache)

    n_total = packets.shape[0]
    n_val = min(max(256, int(0.05 * n_total)), n_total // 2)
    n_train = n_total - n_val
    train_pack, val_pack = packets[:n_train], packets[n_train:]
    train_pres, val_pres = presence[:n_train], presence[n_train:]
    # Shuffle train split deterministically
    perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(args.seed))
    train_pack, train_pres = train_pack[perm], train_pres[perm]
    train_loader = DataLoader(TensorDataset(train_pack, train_pres),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"Train packets: {n_train:,}, val packets: {val_pack.shape[0]:,}")

    model = PhytomerVAE(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PhytomerVAE-{args.latent_dim}D: {n_params:,} params")
    if args.init_checkpoint:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device, weights_only=True))
        print(f"Warm-started from {args.init_checkpoint}")
        if args.hungarian_roles:
            print("  hungarian-roles ON: assignment resolved optimally from sane predictions.")
    elif args.hungarian_roles:
        print("WARNING: --hungarian-roles without --init-checkpoint: assignment is random "
              "until predictions stabilize; expect slow/noisy early epochs.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_val = float("inf")
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = {"loss": 0.0, "recon_loss": 0.0, "loss_kl": 0.0, "cls_acc": 0.0,
               "loss_cls": 0.0, "loss_base": 0.0, "loss_rot": 0.0,
               "loss_scale": 0.0, "loss_curv": 0.0}
        nb = 0
        for bp, br in train_loader:
            bp, br = bp.to(device), br.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(bp, br, use_rot_branch=args.rot_branch)
            losses = model.compute_loss(
                out, bp, br, beta_kl=args.beta_kl, rot_weight=args.rot_weight,
                symmetry_aware_rot=args.symmetry_aware_rot,
                ortho_reg_weight=args.ortho_reg_weight,
                hungarian_roles=args.hungarian_roles,
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for k in tot:
                tot[k] += float(losses[k].item()) if torch.is_tensor(losses[k]) else float(losses[k])
            nb += 1
        scheduler.step()

        # Validation (no grad, mu encoding)
        model.eval()
        with torch.no_grad():
            v_batches, v_recon, v_acc = 0, 0.0, 0.0
            for i in range(0, val_pack.shape[0], args.batch_size):
                vp = val_pack[i:i + args.batch_size].to(device)
                vr = val_pres[i:i + args.batch_size].to(device)
                out = model(vp, vr, use_rot_branch=args.rot_branch)
                vl = model.compute_loss(out, vp, vr, beta_kl=args.beta_kl)
                v_recon += float(vl["recon_loss"])
                v_acc += float(vl["cls_acc"])
                v_batches += 1
            v_recon /= max(v_batches, 1)
            v_acc /= max(v_batches, 1)

        print(f"Epoch {epoch:03d}/{args.epochs} | loss {tot['loss']/nb:.4f} "
              f"(recon {tot['recon_loss']/nb:.4f}, kl {tot['loss_kl']/nb:.4f}) | "
              f"[cls {tot['loss_cls']/nb:.4f} base {tot['loss_base']/nb:.4f} "
              f"rot {tot['loss_rot']/nb:.4f} scale {tot['loss_scale']/nb:.4f} "
              f"curv {tot['loss_curv']/nb:.4f}] | "
              f"train_acc {tot['cls_acc']/nb*100:.1f}% | val_recon {v_recon:.4f} "
              f"val_acc {v_acc*100:.1f}% | {time.time()-t_start:.0f}s", flush=True)
        if v_recon < best_val:
            best_val = v_recon
            torch.save(model.state_dict(),
                       os.path.join(args.checkpoint_dir, f"phytomer_vae_{args.latent_dim}d_best.pt"))
            print(f"  -> new best (val recon {v_recon:.4f}), checkpoint saved.")

    torch.save(model.state_dict(),
               os.path.join(args.checkpoint_dir, f"phytomer_vae_{args.latent_dim}d_last.pt"))
    print(f"Done. Best val recon: {best_val:.4f}. Checkpoints in {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
