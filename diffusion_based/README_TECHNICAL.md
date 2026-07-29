# Plant Architecture Reconstruction via 5D Organ Primitive Diffusion & Tree Topology Estimation

## Abstract
This document outlines the technical design, mathematical formulation, and architectural implementation for reconstructing 2D/3D plant geometries from images. Rather than relying on rigid keypoints (like human skeleton models) or unconstrained edge probabilities, our approach parameterizes plants as a set of **5D Organ Primitives (Stem Segments)** and learns reverse-diffusion score functions coupled with a **Categorical Parent-Softmax Tree Constraint** and **Joint Snap Physics Losses**.

---

## 1. Mathematical Problem Formulation

### 1.1 5D Organ Primitive Representation
Each plant is represented as a set of $N_{\max}$ candidate organ primitives $\mathcal{P} = \{p_i\}_{i=1}^{N_{\max}}$ alongside an existence confidence vector $\mathbf{e} \in [0, 1]^{N_{\max}}$ and a parent index vector $\mathbf{p} \in \{0 \dots N_{\max}-1\}^{N_{\max}}$.

Each primitive $p_i$ is a normalized 5-dimensional tuple:
$$p_i = (x_i, y_i, \bar{\theta}_i, l_i, w_i) \in [-1, 1]^5$$

- $(x_i, y_i) \in [0, 1]^2$: Normalized 2D coordinates of the stem segment base.
- $\bar{\theta}_i = \theta_i / \pi \in [-1, 1]$: Normalized growth direction angle in radians.
- $l_i \in [0, 1]$: Normalized stem segment length.
- $w_i \in [0, 1]$: Normalized stem line width.

The tip coordinate $\vec{x}_{i, \text{tip}}$ of primitive $i$ is calculated geometrically:
$$\vec{x}_{i, \text{tip}} = \left( x_i + l_i \cdot \cos(\pi \bar{\theta}_i), \; y_i - l_i \cdot \sin(\pi \bar{\theta}_i) \right)$$

---

## 2. Forward & Reverse Diffusion Process

### 2.1 Forward Process (Noise Addition)
For timesteps $t \sim \text{Uniform}(1, T)$ with a linear variance schedule $\beta_1, \dots, \beta_T$:
$$q(X_t | X_0) = \mathcal{N}\left(X_t; \sqrt{\bar{\alpha}_t} X_0, (1 - \bar{\alpha}_t) \mathbf{I}\right)$$
$$X_t = \sqrt{\bar{\alpha}_t} X_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$
where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.

### 2.2 Reverse Process with Direct $x_0$-Parameterization
Instead of predicting abstract noise $\epsilon$, our network directly predicts the target organ attributes $\hat{X}_0 = \text{Model}(X_t, t, I_{\text{target}})$. The reverse step blends $X_t$ toward $\hat{X}_0$:
$$X_{t-1} = (1 - \gamma_t) X_t + \gamma_t \hat{X}_0$$
where $\gamma_t = 0.2 + 0.8 \cdot \left(\frac{T - t}{T}\right)$.

---

## 3. Combined Loss Formulation

The network is trained end-to-end using a multi-objective loss:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{x_0} + 0.5 \cdot \mathcal{L}_{\text{existence}} + 0.5 \cdot \mathcal{L}_{\text{parent}} + 0.2 \cdot \mathcal{L}_{\text{snap}}$$

### 3.1 Direct Parameter MSE Loss ($\mathcal{L}_{x_0}$)
$$\mathcal{L}_{x_0} = \frac{1}{N} \sum_{i=1}^N \| \hat{p}_{0, i} - p_{0, i} \|^2$$

### 3.2 Categorical Parent Cross-Entropy Loss ($\mathcal{L}_{\text{parent}}$)
To guarantee a valid Directed Tree topology without disconnected branches, the network predicts categorical logits $\hat{\mathbf{S}} \in \mathbb{R}^{N \times N}$ over all candidate parent nodes:
$$\mathcal{L}_{\text{parent}} = - \frac{1}{N} \sum_{v=1}^N \log \frac{\exp(\hat{S}_{v, p_v})}{\sum_{u=1}^N \exp(\hat{S}_{v, u})}$$

### 3.3 Organ Joint Snap Loss ($\mathcal{L}_{\text{snap}}$)
Forces the base of child primitive $v$ to physically snap to the tip of parent primitive $u = p_v$:
$$\mathcal{L}_{\text{snap}} = \frac{1}{\sum A_{uv}} \sum_{u, v} A_{uv} \cdot \left( \| \hat{x}_{u, \text{tip}} - \hat{x}_{v, \text{base}} \|^2 + \| \hat{y}_{u, \text{tip}} - \hat{y}_{v, \text{base}} \|^2 \right)$$

---

## 4. Architectural Implementation & Modules

### 4.1 `MultiScaleSpatialEncoder` (`diffusion_based/models/graph_diffuser.py`)
Replaces classification-biased ResNet-18 backbones with a multi-resolution feature pyramid that pools $128 \times 128$ (low-level edges), $64 \times 64$ (branch junctions), and $32 \times 32$ (global semantics) into 1,024 spatial key-value feature tokens:
```python
class MultiScaleSpatialEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat0 = self.stem(x)       # 128x128
        feat1 = self.layer1(feat0) # 64x64
        feat2 = self.layer2(feat1) # 32x32
        feat3 = self.layer3(feat2) # 16x16
        # Combine into (B, embed_dim, 32, 32) -> 1,024 Tokens
        return torch.cat([self.proj1(feat1), self.proj2(feat2), self.proj3(feat3)], dim=1)
```

### 4.2 `PlantGraphExtractor` (`dataset/graph_extractor.py`)
Parses execution traces of 2D L-System Turtle graphics into 5D Organ Primitives:
```python
nodes.append((nx, ny, math.radians(curr_h) / math.pi, norm_step_len, w_val / 20.0))
```

### 4.3 Reverse Sampler & Visualizer (`diffusion_based/eval/visualize_diffusion.py`)
Generates 4-panel progression figures (Input Image, Step 999 Noise, Step 489 Mid Assembly, Step 0 Reconstructed Tree Graph) rendering tapered stem primitives and parent-child snap joints.
