# Next Steps: Continue 3D Plant Diffusion on Ubuntu + NVIDIA GPU

> This plan assumes the repository has been cloned on an Ubuntu machine with an NVIDIA GPU and a working CUDA/PyTorch environment.

## 1. Environment Setup

1. **Clone the repo** (if not already present):
   ```bash
   git clone https://github.com/heesup/image-to-l-system.git
   cd image-to-l-system
   ```

2. **Create a Conda environment** (recommended) or use an existing CUDA-enabled PyTorch env:
   ```bash
   conda create -n l-system python=3.10 -y
   conda activate l-system
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers peft accelerate pillow matplotlib opencv-python-headless pyyaml tqdm
   ```

3. **Verify GPU access**:
   ```bash
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
   ```

## 2. Prepare the Helios Dataset

The repo no longer includes the `Digital-Crops` submodule. On the Ubuntu machine you must regenerate or copy the Helios dataset:

1. **Option A — Regenerate locally**:
   - Build/obtain the Helios generation binary at:
     `Digital-Crops/projects/syntheticdata_generation/build/main`
   - Run the generator (adjust paths for Linux):
     ```bash
     python -m dataset.generate_helios_dataset \
       --helios_binary Digital-Crops/projects/syntheticdata_generation/build/main \
       --output Digital-Crops/projects/syntheticdata_generation/build/output \
       --dap_values 5 10 15 20 25 30 35 40 45 50 55 60 \
       --seeds 0 1 2 3 4 \
       --renderer vis \
       --workers 8
     ```

2. **Option B — Copy from macOS**:
   - Copy `Digital-Crops/projects/syntheticdata_generation/build/output/` from the macOS machine.
   - This directory is git-ignored; place it in the same relative path.

3. **Verify the dataset**:
   ```bash
   python -c "
   from dataset.helios_dataset import HeliosPlantDataset
   ds = HeliosPlantDataset('Digital-Crops/projects/syntheticdata_generation/build/output', max_nodes=2048)
   print('samples:', len(ds))
   print('node counts:', [(s['dap'].item()*90, s['num_nodes'].item()) for s in ds][:5])
   "
   ```

## 3. Run Long Training on GPU

The macOS run only reached a few epochs; the full model needs 100–500 epochs.

1. **Start the background training script** (uses CUDA by default; the code auto-detects via `get_device()`):
   ```bash
   nohup ./scripts/train_3d_background.sh > train_3d_background.log 2>&1 &
   ```
   This trains 200 epochs with `max_nodes=2048`, batch_size=2, and 10 epochs of existence-only pretraining.

2. **If the GPU has ≥16 GB VRAM**, increase batch size for faster training:
   ```bash
   python -m diffusion_based.training.train_diffusion_3d \
     --epochs 200 --batch_size 4 --max_nodes 2048 --lr 3e-4 \
     --helios_data_root Digital-Crops/projects/syntheticdata_generation/build/output \
     --num_samples 100 --pretrain_existence_epochs 10 \
     --save_path diffusion_based/checkpoints/diffusion_model_3d_200ep.pt \
     --best_save_path diffusion_based/checkpoints/best_3d_model_200ep.pt
   ```

3. **Monitor loss**:
   ```bash
   tail -f train_3d_background.log
   ```
   Expect val_loss to drop below 0.3–0.5 by epoch 50–100.

## 4. Validate Existence & Budget Heads

After ~20–50 epochs, check whether the model predicts reasonable node counts:

```bash
python -c "
from diffusion_based.models.graph_diffuser_3d import PlantGraphDiffuser3D
from dataset.helios_dataset import HeliosPlantDataset
import torch

device = 'cuda'
model = PlantGraphDiffuser3D(max_nodes=2048, node_dim=15).to(device)
model.load_state_dict(torch.load('diffusion_based/checkpoints/best_3d_model_200ep.pt', map_location=device)['model'])
model.eval()

ds = HeliosPlantDataset('Digital-Crops/projects/syntheticdata_generation/build/output', max_nodes=2048)
s = ds[0]
img = s['image'].unsqueeze(0).to(device)
dap = s['dap'].unsqueeze(0).to(device)
pose = s['camera_pose'].unsqueeze(0).to(device)
noisy = torch.randn(1, 2048, 15).to(device)
exist = torch.ones(1, 2048, 1).to(device)
t = torch.tensor([500]).to(device)

with torch.no_grad():
    out = model(noisy, exist, t, img, camera_poses=pose, dap=dap)
    pred_budget = out['pred_node_budget'][0].item() * 2048
    true_budget = (s['existence_mask'] > 0.5).sum().item()
    print(f'pred_budget={pred_budget:.0f}, true_budget={true_budget}')
"
```

## 5. Run Inference & Visualize

After training, generate reconstructions on real Helios images:

```bash
python -c "
from diffusion_based.eval.visualize_diffusion_3d import run_inference_on_real_image
run_inference_on_real_image(
    jpeg_path='Digital-Crops/projects/syntheticdata_generation/build/output/cowpea_0000_vis.jpeg',
    xml_path='Digital-Crops/projects/syntheticdata_generation/build/output/cowpea_0000_plant_0000.xml',
    dap=5,
    steps=50,
    existence_threshold=0.3,
    max_nodes=2048,
    save_path='diffusion_based/plots/real_dap5_200ep.png'
)
"
```

Open `diffusion_based/plots/real_dap5_200ep.png` and compare to the ground truth.

## 6. Likely Issues to Debug on GPU

| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| OOM at N=2048 | Cross-attention still too heavy | Drop `max_nodes` to 1024 or use mixed precision (`torch.cuda.amp`) |
| Existence all 1s | Insufficient pretraining or class imbalance | Increase `--pretrain_existence_epochs` to 20–50 |
| Reconstructed points scattered | Coord loss dominates, topology weak | Increase `loss_parent` weight and `loss_snap3d` weight |
| Budget head way off | No signal from DAP | Verify `dap` tensor is correctly normalized and passed |
| Slow training | MPS→CUDA still has overhead | Use `pin_memory=True`, `num_workers>0`, and `torch.compile(model)` if PyTorch ≥2.0 |

## 7. Suggested Hyperparameter Grid

Run short 10-epoch sweeps to find the best mix:

| Config | `--pretrain_existence_epochs` | `loss_existence` weight | `loss_budget` weight | Notes |
|--------|-------------------------------|------------------------|----------------------|-------|
| Baseline | 10 | 2.0 | 1.0 | Current |
| Strong existence | 30 | 5.0 | 2.0 | If pruning still poor |
| Topology focus | 10 | 2.0 | 1.0 | Also raise `loss_parent` to 2.0 and `loss_snap3d` to 1.5 |

## 8. Definition of Done

- [ ] 200-epoch checkpoint trained on GPU
- [ ] `best_3d_model_200ep.pt` exists and loads cleanly
- [ ] Validation loss < 0.3
- [ ] `pred_node_budget` is within ±20% of true node count on validation samples
- [ ] Inference reconstruction visually matches ground truth structure (right number of nodes, correct plant region)

## 9. Optional Follow-Ups

1. **L-System grammar extraction**: Convert the predicted 3D graph back into L-System production rules (the original project goal).
2. **Stochastic L-Systems**: Train on grammars with probabilistic rules.
3. **Real-world images**: Test on actual photographs (requires camera calibration and background removal).

---

*Last updated after macOS Phase 2 development. Continue by running Step 2 then Step 3.*
