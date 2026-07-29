# Diffusion-Based Plant Architecture Reconstruction

This module implements a **Set-Based Graph & Adjacency Diffusion Model** to reconstruct 2D/3D plant branch topology from input plant images.

---

## Technical Concept

1. **Scattered Material & Denoising**:
   - Plant nodes are initialized from isotropic Gaussian noise: $X_T \sim \mathcal{N}(0, I)$, with edge adjacency probabilities $A_T \sim \text{Bernoulli}(0.5)$.
   - During the reverse diffusion process $t \to t-1$, a Graph Cross-Attention UNet conditions on image features from the target plant $I_{\text{target}}$ to steer scattered points into structured plant branch geometry.
2. **Dynamic Node Count**:
   - Each node contains a continuous existence/confidence score $e_i \in [0, 1]$.
   - Denoising naturally prunes unused nodes ($e_i \to 0$) and keeps active plant junctions/tips ($e_i \to 1$).
3. **Graph Assembly**:
   - Node compatibility and directional alignment determine edge connection probabilities $A_{ij}$.
   - Minimum Spanning Tree / Directed Acyclic Graph (DAG) extraction constructs the final tree graph.

---

## Directory Overview

```
diffusion_based/
├── README.md                     # Documentation & technical design
├── dataset/
│   └── graph_dataset.py          # PyTorch dataset for paired plant images & graph structures
├── models/
│   └── graph_diffuser.py         # Graph Cross-Attention Transformer Denoising Model
├── training/
│   └── train_diffusion.py        # DDPM/DDIM diffusion training script
└── eval/
    └── visualize_diffusion.py    # Denoising trajectory visualizer
```
