import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from typing import Dict, Tuple

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal Timestep Embeddings for Diffusion."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MultiScaleSpatialEncoder(nn.Module):
    """Multi-Scale Spatial Feature Encoder (U-Net / DenseNet Style).
    Extracts fine-grained edge details (128x128) + mid-level junction features (64x64) + high-level semantics (32x32).
    Output is pooled to 16x16=256 tokens to keep cross-attention O(N*M) manageable for large N.
    """

    def __init__(self, out_dim: int = 256, output_tokens: int = 16):
        super().__init__()
        self.output_tokens = output_tokens
        weights = ResNet18_Weights.DEFAULT
        resnet = resnet18(weights=weights)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # 128x128, 64-ch
        self.layer1 = resnet.layer1  # 64x64, 64-ch
        self.layer2 = resnet.layer2  # 32x32, 128-ch
        self.layer3 = resnet.layer3  # 16x16, 256-ch

        self.proj1 = nn.Sequential(nn.AdaptiveAvgPool2d((output_tokens, output_tokens)), nn.Conv2d(64, out_dim // 4, 1))
        self.proj2 = nn.Sequential(nn.AdaptiveAvgPool2d((output_tokens, output_tokens)), nn.Conv2d(128, out_dim // 2, 1))
        self.proj3 = nn.Sequential(nn.AdaptiveAvgPool2d((output_tokens, output_tokens)), nn.Conv2d(256, out_dim // 4, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat0 = self.stem(x)
        feat1 = self.layer1(feat0)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)

        p1 = self.proj1(feat1)
        p2 = self.proj2(feat2)
        p3 = self.proj3(feat3)

        # Concatenate multi-scale spatial features -> (B, out_dim, output_tokens, output_tokens)
        multi_scale = torch.cat([p1, p2, p3], dim=1)
        return multi_scale

class PlantGraphDiffuser(nn.Module):
    """Multi-Scale Spatial Encoder & Direct x0-Prediction Graph Diffuser.
    Denoises 5D plant organ primitives (norm_x, norm_y, norm_theta, norm_length, norm_width)
    and predicts categorical Parent Index distributions for Tree Topology.
    """

    def __init__(self, max_nodes: int = 64, node_dim: int = 5, embed_dim: int = 256, num_layers: int = 4):
        super().__init__()
        self.max_nodes = max_nodes
        self.embed_dim = embed_dim

        # Multi-Scale Dense Spatial Vision Encoder (32x32 = 1024 spatial tokens)
        self.vision_encoder = MultiScaleSpatialEncoder(out_dim=embed_dim)

        # Timestep Embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Node Input Projection & Learned Node Identity Position Embeddings
        self.node_proj = nn.Linear(node_dim + 1, embed_dim)
        self.node_pos_emb = nn.Embedding(max_nodes, embed_dim)

        # Transformer Cross-Attention Layers (1024 Spatial Tokens)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=512,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        # Prediction Heads (Direct x0 prediction for Organ Attributes)
        self.node_pred_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, node_dim)
        )
        self.existence_pred_head = nn.Linear(embed_dim, 1)

        # Pairwise Parent-Child Softmax Logits Head (B, N, N)
        self.parent_pred_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, noisy_nodes: torch.Tensor, noisy_existence: torch.Tensor, timesteps: torch.Tensor, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            noisy_nodes: (B, N, 5)
            noisy_existence: (B, N, 1)
            timesteps: (B,)
            images: (B, 3, H, W)
        """
        B, N, _ = noisy_nodes.shape
        device = noisy_nodes.device

        # 1. Extract Multi-Scale High-Resolution Spatial Key/Value Features (32x32 = 1024 spatial tokens)
        img_feats = self.vision_encoder(images)           # (B, embed_dim, 32, 32)
        img_feats = img_feats.flatten(2).permute(0, 2, 1) # (B, 1024, embed_dim)

        # 2. Compute Timestep Embeddings
        t_emb = self.time_emb(timesteps).unsqueeze(1)      # (B, 1, embed_dim)

        # 3. Project Noisy Node Inputs with Learned Node Position Embeddings
        node_in = torch.cat([noisy_nodes, noisy_existence], dim=-1) # (B, N, 6)
        node_indices = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        h_nodes = self.node_proj(node_in) + t_emb + self.node_pos_emb(node_indices) # (B, N, embed_dim)

        # 4. Pass through High-Res Cross-Attention Transformer
        for layer in self.transformer_layers:
            h_nodes = layer(tgt=h_nodes, memory=img_feats)        # (B, N, embed_dim)

        # 5. Predict Direct Organ x0 & Existence
        pred_x0 = torch.clamp(self.node_pred_head(h_nodes), 0.0, 1.0)        # Un-saturated coordinates in [0, 1]
        pred_existence_logits = self.existence_pred_head(h_nodes).squeeze(-1) # (B, N)

        # 6. Predict Parent Categorical Logits with k-NN Spatial Candidate Pruning (k = 8)
        coords = pred_x0[:, :, :2] # (B, N, 2) 2D predicted base positions
        dist_matrix = torch.cdist(coords, coords) # (B, N, N) pairwise 2D distance matrix

        k_val = min(8, N)
        _, knn_indices = torch.topk(dist_matrix, k=k_val, largest=False, dim=-1) # (B, N, k)

        knn_mask = torch.zeros(B, N, N, dtype=torch.bool, device=device)
        knn_mask.scatter_(2, knn_indices, True)

        h_i = h_nodes.unsqueeze(2).repeat(1, 1, N, 1)
        h_j = h_nodes.unsqueeze(1).repeat(1, N, 1, 1)
        pair_feat = torch.cat([h_i, h_j], dim=-1)                  # (B, N, N, embed_dim*2)
        pred_parent_logits = self.parent_pred_head(pair_feat).squeeze(-1) # (B, N, N)

        # Apply k-NN Spatial Pruning: mask out distant non-knn candidate pairs
        pred_parent_logits = pred_parent_logits.masked_fill(~knn_mask, -10000.0)

        # Infer noise from x0: eps = (x_t - sqrt(alpha_bar) * x0) / sqrt(1 - alpha_bar)
        pred_node_noise = noisy_nodes - pred_x0

        return {
            "pred_x0": pred_x0,
            "pred_node_noise": pred_node_noise,
            "pred_existence_logits": pred_existence_logits,
            "pred_adj_logits": pred_parent_logits,
            "pred_parent_logits": pred_parent_logits
        }
