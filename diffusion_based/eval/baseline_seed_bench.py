"""Quick single-seed leaf-mask IoU benchmark (no Helios timing)."""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from diffusion_based.eval.dap30_multi_seed_panel import process_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="Digital-Crops/projects/syntheticdata_generation/build/output_rad_dap30")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--generic-leaves", action="store_true", default=False)
    args = parser.parse_args()

    for seed in args.seeds:
        res = process_seed(args.base_dir, seed, "diffusion_based/eval/output", use_generic_leaves=args.generic_leaves, time_helios=False)
        print(f"\nSeed {seed}: GT={res['helios_mask'].sum()}, PyTorch={res['pytorch_mask'].sum()}, IoU={res['iou']:.4f}, Dice={res['dice']:.4f}")


if __name__ == "__main__":
    main()