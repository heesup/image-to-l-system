#!/bin/zsh
# Background training script for the 3D plant diffusion model.
# Use with: nohup /Users/lion397/codes/l-systems-gnn/scripts/train_3d_background.sh > /Users/lion397/codes/l-systems-gnn/train_3d_background.log 2>&1 &

export PYTHONUNBUFFERED=1
source /Users/lion397/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate l-system

python -m diffusion_based.training.train_diffusion_3d \
  --epochs 200 \
  --batch_size 2 \
  --max_nodes 2048 \
  --lr 3e-4 \
  --helios_data_root /Users/lion397/codes/l-systems-gnn/Digital-Crops/projects/syntheticdata_generation/build/output \
  --num_samples 100 \
  --pretrain_existence_epochs 10 \
  --save_path diffusion_based/checkpoints/diffusion_model_3d_200ep.pt \
  --best_save_path diffusion_based/checkpoints/best_3d_model_200ep.pt
