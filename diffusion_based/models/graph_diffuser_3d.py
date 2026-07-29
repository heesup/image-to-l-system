import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from typing import Dict, Tuple
from diffusion_based.models.graph_diffuser import SinusoidalPosEmb, MultiScaleSpatialEncoder

class PlantGraphDiffuser3D(nn.Module):
    """3D Vision-Conditioned Graph Diffuser.
    Denoises 7D 3D Botanical Organ Primitives (x, y, z, theta, phi, length, is_leaf) from a 2D Projection Image.
    Uses 3D Euclidean k-NN Spatial Candidate Pruning (k = 8).
    """

    def __init__(self, max_nodes: int = 64, node_dim: int = 7, embed_dim: int = 256, num_layers: int = 4):
        super().__init__()
        self.max_nodes = max_nodes
        self.embed_dim = embed_dim

        # 2D Multi-Scale Vision Encoder (32x32 = 1024 spatial tokens)
        self.vision_encoder = MultiScaleSpatialEncoder(out_dim=embed_dim)

        # Timestep Embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 7D Node Projection & Learned Position Embeddings
        self.node_proj = nn.Linear(node_dim + 1, embed_dim)
        self.node_pos_emb = nn.Embedding(max_nodes, embed_dim)

        # Transformer Cross-Attention Layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=512,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        # Prediction Heads (7D 3D Organ Attributes)
        self.node_pred_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, node_dim)
        )
        self.existence_pred_head = nn.Linear(embed_dim, 1)

        # Pairwise 3D Parent Logits Head
        self.parent_pred_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, noisy_nodes: torch.Tensor, noisy_existence: torch.Tensor, timesteps: torch.Tensor, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            noisy_nodes: (B, N, 7) 7D 3D Organ Primitives (x, y, z, theta, phi, length, is_leaf)
            noisy_existence: (B, N, 1)
            timesteps: (B,)
            images: (B, 3, H, W) 2D Projection Input Image
        """
        B, N, _ = noisy_nodes.shape
        device = noisy_nodes.device

        # 1. Extract 2D Spatial Vision Key/Value Features (B, 1024, embed_dim)
        img_feats = self.vision_encoder(images)
        img_feats = img_feats.flatten(2).permute(0, 2, 1)

        # 2. Compute Timestep Embeddings
        t_emb = self.time_emb(timesteps).unsqueeze(1)

        # 3. Project 7D Node Inputs with Learned Position Embeddings
        node_in = torch.cat([noisy_nodes, noisy_existence], dim=-1) # (B, N, 8)
        node_indices = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        h_nodes = self.node_proj(node_in) + t_emb + self.node_pos_emb(node_indices) # (B, N, embed_dim)

        # 4. Pass through Cross-Attention Transformer
        for layer in self.transformer_layers:
            h_nodes = layer(tgt=h_nodes, memory=img_feats)

        # 5. Predict Direct 7D 3D Organ Attributes & Existence
        pred_x0 = torch.clamp(self.node_pred_head(h_nodes), 0.0, 1.0)
        pred_existence_logits = self.existence_pred_head(h_nodes).squeeze(-1)

        # 6. Predict 3D Parent Topology Logits with 3D Spatial k-NN Pruning (k = 8)
        coords_3d = pred_x0[:, :, :3] # (B, N, 3) 3D Position (x, y, z)
        dist_matrix_3d = torch.cdist(coords_3d, coords_3d) # (B, N, N) 3D Euclidean distance

        k_val = min(8, N)
        _, knn_indices = torch.topk(dist_matrix_3d, k=k_val, largest=False, dim=-1)

        knn_mask = torch.zeros(B, N, N, dtype=torch.bool, device=device)
        knn_mask.scatter_(2, knn_indices, True)

        h_i = h_nodes.unsqueeze(2).repeat(1, 1, N, 1)
        h_j = h_nodes.unsqueeze(1).repeat(1, N, 1, 1)
        pair_feat = torch.cat([h_i, h_j], dim=-1)
        pred_parent_logits = self.parent_pred_head(pair_feat).squeeze(-1)

        pred_node_noise = noisy_nodes - pred_x0

        return {
            "pred_x0": pred_x0,
            "pred_node_noise": pred_node_noise,
            "pred_existence_logits": pred_existence_logits,
            "pred_parent_logits": pred_parent_logits
        }
