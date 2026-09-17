# Language Modeling (VLM) Based Plant Architecture Reconstruction

This module implements plant grammar and rendering parameter estimation from 2D rendered plant images using a Vision-Language Model (VLM / Autoregressive Transformer).

---

## Key Features

- **End-to-End Token Generation**: Predicts axiom, production rules, turn angle, iterations, step size, and line width directly as JSON tokens.
- **Two-Stage Training Pipeline**:
  - **Stage 1 (Supervised Fine-Tuning - SFT)**: Trains Vision Encoder + Transformer Decoder on synthetic image-JSON targets (`lm_based/training/train_sft.py`).
  - **Stage 2 (Render-in-the-Loop RL)**: Fine-tunes model outputs using visual render Mask Intersection-over-Union (IoU) rewards (`lm_based/training/train_rl.py`).
- **Out-of-Distribution (OOD) Testing**: Evaluates model generalization on unseen plant grammars (`lm_based/eval/test_ood.py`).

---

## Usage

### 1. Supervised Fine-Tuning (SFT)
```bash
python -m lm_based.training.train_sft --data_dir data/synthetic --epochs 5
```

### 2. Render-in-the-Loop RL Training
```bash
python -m lm_based.training.train_rl --data_dir data/synthetic --epochs 3
```

### 3. Single-Image Inference
```bash
python -m lm_based.infer --image path/to/plant.png --checkpoint checkpoints/rl_model.pt --output_plot plots/my_prediction.png
```

### 4. Quantitative Evaluation
```bash
python -m lm_based.eval.evaluate --checkpoint checkpoints/rl_model.pt
```
