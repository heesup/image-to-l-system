# Image-to-L-System Plan: Estimating L-System Grammar & Parameters from Rendered Plant Images

This project implements an end-to-end pipeline in **PyTorch (MPS / Apple Silicon)** using a fine-tuned mini VLM (`SmolVLM` / `PaliGemma` with LoRA) to estimate L-System grammars (axiom & production rules) and geometric parameters (turn angle, iterations, step size) from 2D rendered plant images.

Reference: [Understanding L-Systems Blog Post](https://heesup.github.io/posts/2026/06/20/lsystems.html)

---

## 1. Task Definition & Output Format

Given an input plant image $I \in \mathbb{R}^{3 \times H \times W}$:
1. **Estimate L-System Grammar**:
   - Axiom (e.g. `X`)
   - Production rules (e.g. `X -> F[+X][-X]FX`, `F -> FF`)
2. **Estimate Geometric & Rendering Parameters**:
   - `angle` ($\theta \in [10.0^\circ, 45.0^\circ]$)
   - `iterations` ($N \in \{2, 3, 4, 5\}$)
   - `step_size` ($d$)
   - `line_width` ($w$)

Output Target (JSON instruction string):
```json
{
  "axiom": "X",
  "rules": {"X": "F[+X][-X]FX", "F": "FF"},
  "angle": 25.0,
  "iterations": 4,
  "step_size": 1.0,
  "line_width": 2.0
}
```

---

## 2. Dataset Synthesis Engine (Deterministic vs. Stochastic Rules)

### What are Stochastic L-System Rules?
- **Deterministic Rules**: Every symbol rewrites into a fixed string ($X \to F[+X][-X]FX$). Every render of the grammar produces the exact same image.
- **Stochastic Rules**: Symbols rewrite into strings chosen probabilistically (e.g., $X \xrightarrow{50\%} F[+X][-X]$, $X \xrightarrow{50\%} F[+X]$). Random choices during drawing create natural variations across individual plants.

### Dataset Strategy
- **Tiers 1 & 2 (Initial Target)**: 2D **Deterministic** Branching L-Systems (exact 1-to-1 visual-to-grammar mapping).
- **Tier 3 (Stretch Goal)**: Stochastic L-Systems with probability distributions.

---

## 3. Architecture & Training Pipeline

### Model Choice
- Mini Vision-Language Model: `HuggingFaceTB/SmolVLM-256M-Instruct` or `google/paligemma-3b-pt-224`.
- Fine-Tuning Technique: **LoRA (PEFT)** on attention projection weights (`q_proj`, `v_proj`, `k_proj`, `o_proj`).
- Compute Target: PyTorch on Apple Silicon MPS (`torch.device("mps")`).

### Two-Stage Training
1. **Stage 1: Supervised Fine-Tuning (SFT)**
   - Train mini VLM on synthetic image-text dataset with Cross-Entropy loss over the target JSON tokens.
   - Learns valid JSON structure, bracket matching balance (`[` / `]`), and initial parameter estimates.
2. **Stage 2: Render-Reward Reinforcement Learning (RL / GRPO)**
   - Policy samples predicted L-System parameters for image $I$.
   - Render engine produces candidate plant render $\hat{I}$.
   - **Render-in-the-Loop Reward**:
     $$R = \text{IoU}(\hat{I}, I) - \lambda \cdot \text{SyntaxPenalty}$$

---

## 4. Software Stack & macOS Environment

- **Framework**: PyTorch (`torch`, `torchvision`) with `mps` backend support.
- **Environment Management**: Conda / Mamba via `environment.yml`.
- **Dependencies**: `transformers`, `peft`, `accelerate`, `bitsandbytes`, `pillow`, `matplotlib`, `cairosvg`, `opencv-python`, `pytest`, `tensorboard`, `pyyaml`, `tqdm`.

---

## 5. Directory Structure

```
image-to-l-system/
├── environment.yml
├── README.md
├── PLAN.md
├── dataset/
│   ├── lsystem.py        # L-system parser & expansion logic
│   ├── renderer.py       # 2D Turtle graphics renderer
│   ├── generator.py      # Dataset generator
│   └── dataloader.py     # PyTorch / HF DataLoader
├── models/
│   └── vlm_wrapper.py    # SmolVLM / PaliGemma + LoRA wrapper
├── training/
│   ├── rewards.py        # Render IoU & SSIM reward metrics
│   ├── train_sft.py      # Supervised Fine-Tuning script
│   └── train_rl.py       # Render-in-the-Loop RL script
├── eval/
│   ├── evaluate.py       # Metrics evaluation script
│   └── visualize.py      # Side-by-side comparison renderer
└── tests/
    └── test_lsystem.py   # Unit test suite
```

---

## 6. Remote Repository Setup
- GitHub Remote: `https://github.com/heesup/image-to-l-system.git`
