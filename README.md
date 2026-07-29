# Plant Architecture Reconstruction: Language Modeling & Diffusion Approaches

This repository contains end-to-end deep learning methods written in **PyTorch** (optimized for **Apple Silicon MPS**) for 2D and 3D plant architecture reconstruction from plant images.

Reference Blog Post: [Understanding L-Systems](https://heesup.github.io/posts/2026/06/20/lsystems.html)

---

## Repository Structure

```
l-systems-gnn/
├── dataset/                     # [Shared Core] L-System parser, 2D Turtle renderer, Graph Extractor
│   ├── lsystem.py               # L-System grammar parser & bracket validator
│   ├── renderer.py              # Auto-centered 2D Turtle graphics renderer
│   ├── generator.py             # Synthetic plant dataset generator
│   └── graph_extractor.py       # Extractor for 2D node attributes & adjacency matrix (V, A)
├── lm_based/                    # [Language Modeling Approach]
│   ├── README.md                # LM approach documentation
│   ├── infer.py                 # Single-image VLM inference
│   ├── models/                  # Pure L-System VLM / Autoregressive Transformer
│   ├── training/                # Stage 1 SFT & Stage 2 Render-in-the-Loop RL
│   └── eval/                    # OOD testing, visualization, and evaluation scripts
├── diffusion_based/             # [Diffusion-Based Approach]
│   ├── README.md                # Diffusion design roadmap
│   ├── dataset/                 # Plant graph & image paired PyTorch dataset
│   ├── models/                  # Graph Cross-Attention Transformer Diffuser
│   ├── training/                # DDPM noise schedule & graph denoising training loop
│   └── eval/                    # Denoising process visualizer
└── tests/                       # Integrated Pytest & Unittest suite
```

---

## Installation & Setup

```bash
# 1. Create environment
mamba env create -f environment.yml

# 2. Activate environment
conda activate l-system

# 3. Run all unit tests
python -m unittest discover -s tests
```

---

## Quickstart

### Approach 1: Language Modeling (VLM / L-System Token Generation)
```bash
# Supervised Fine-Tuning
python -m lm_based.training.train_sft --data_dir data/synthetic --epochs 5

# Single Image Inference
python -m lm_based.infer --image path/to/plant.png --checkpoint checkpoints/rl_model.pt
```

### Approach 2: Graph Diffusion (Point-Cloud & Adjacency Denoising)
```bash
# Train Plant Graph Diffusion Model
python -m diffusion_based.training.train_diffusion

# Visualize Denoised Plant Graph
python -m diffusion_based.eval.visualize_diffusion
```
