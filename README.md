# Image-to-L-System

An end-to-end deep learning project written in **PyTorch** (optimized for **Apple Silicon MPS**) to estimate L-System grammars (axioms and production rules) and rendering parameters (turn angle, iterations, step size) directly from 2D rendered plant images.

Reference Blog Post: [Understanding L-Systems](https://heesup.github.io/posts/2026/06/20/lsystems.html)

---

## Key Features

- **Synthetic Dataset Engine (`dataset/`)**:
  - Deterministic 2D L-System rewriting parser and Turtle graphics renderer with automatic canvas auto-centering.
  - Multi-tier plant templates (Binary plants, Monopodial bushes, Ternary trees, Ferns, Vines).
- **Two-Stage Training Pipeline (`training/`)**:
  - **Stage 1 (Supervised Fine-Tuning - SFT)**: Trains Vision Encoder + Transformer Decoder / VLM on synthetic image-JSON data.
  - **Stage 2 (Render-in-the-Loop RL)**: Reinforcement learning fine-tuning using visual render Mask Intersection-over-Union (IoU) rewards.
- **Apple Silicon Acceleration**:
  - Native support for PyTorch `mps` backend (`torch.device("mps")`).
- **Evaluation & Visualization (`eval/`)**:
  - Quantitative metrics (Syntax validity rate, Mean Mask IoU).
  - Side-by-side plot visualizer comparing Ground Truth Plant Images with Model Predicted Renders.

---

## Installation & Environment Setup

Using `mamba` or `conda`:

```bash
# 1. Create environment from environment.yml
mamba env create -f environment.yml

# 2. Activate environment
conda activate l-system
```

---

## Quickstart & Usage

### 1. Synthetic Dataset Generation
Generate 100 synthetic plant images and JSON annotations:
```bash
python -m dataset.generator
```

### 2. Supervised Fine-Tuning (SFT)
Train the base vision-language model on Apple Silicon MPS:
```bash
python -m training.train_sft --data_dir data/synthetic --epochs 5 --device auto
```

### 3. Render-in-the-Loop RL Training
Fine-tune the model using visual render IoU reward feedback:
```bash
python -m training.train_rl --data_dir data/synthetic --epochs 3 --device auto
```

### 4. Single-Image Inference
Run estimation on any arbitrary plant image file:
```bash
python infer.py --image path/to/plant.png --checkpoint checkpoints/rl_model.pt --output_plot plots/my_prediction.png
```

### 5. Out-of-Distribution (OOD) Testing
Test model performance on unseen plant grammars and out-of-distribution parameter ranges:
```bash
python -m eval.test_ood
```

### 6. Quantitative Evaluation
Evaluate model metrics on the synthetic test set:
```bash
python -m eval.evaluate --checkpoint checkpoints/rl_model.pt
```

### 7. Visualization
Generate side-by-side comparison plots:
```bash
python -m eval.visualize --num_samples 5 --output_dir plots
```

### 8. Run Unit Tests
```bash
python -m unittest discover -s tests
```


---

## Project Structure

```
image-to-l-system/
├── environment.yml          # Mamba / Conda environment spec
├── README.md                # Project documentation
├── PLAN.md                  # Detailed architectural plan
├── dataset/
│   ├── lsystem.py           # L-System parser & bracket validator
│   ├── renderer.py          # Auto-centered 2D Turtle renderer
│   ├── generator.py         # Synthetic dataset generator
│   └── dataloader.py        # PyTorch Dataset & DataLoader
├── models/
│   └── vlm_wrapper.py       # VLM / PyTorch model wrapper with MPS support
├── training/
│   ├── rewards.py           # Render IoU & syntax reward metrics
│   ├── train_sft.py         # Supervised fine-tuning script
│   └── train_rl.py          # Render-in-the-loop RL script
├── eval/
│   ├── evaluate.py          # Evaluation script
│   └── visualize.py         # Side-by-side visualization plot generator
└── tests/
    └── test_lsystem.py      # Pytest unit tests
```
