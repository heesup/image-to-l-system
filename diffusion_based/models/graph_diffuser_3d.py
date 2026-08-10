import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from torchvision.models import resnet18, ResNet18_Weights

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
    """Multi-Scale Spatial Feature Encoder for 3D Graph Diffuser."""
    def __init__(self, out_dim: int = 256, output_tokens: int = 16):
        super().__init__()
        self.output_tokens = output_tokens
        weights = ResNet18_Weights.DEFAULT
        resnet = resnet18(weights=weights)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.proj1 = nn.Sequential(nn.AdaptiveAvgPool2d((output_tokens, output_tokens)), nn.Conv2d(64, out_dim // 4, 1))
        self.proj2 = nn.Sequential(nn.AdaptiveAvgPool2d((output_tokens, output_tokens)), nn.Conv2d(128, out_dim // 4, 1))
        self.proj3 = nn.Sequential(nn.AdaptiveAvgPool2d((output_tokens, output_tokens)), nn.Conv2d(256, out_dim // 2, 1))
        self.final_proj = nn.Linear(out_dim, out_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)

        x0 = self.stem(img)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        p1 = self.proj1(x1)
        p2 = self.proj2(x2)
        p3 = self.proj3(x3)

        feat_map = torch.cat([p1, p2, p3], dim=1)
        B, C, H, W = feat_map.shape
        tokens = feat_map.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return self.final_proj(tokens)

from diffusion_based.models.knn_attention import KNNTransformerDecoderLayer


class PlantGraphDiffuser3D(nn.Module):
    """3D Vision-Conditioned Graph Diffuser.

    Denoises 15D 3D botanical organ primitives from a 2D projection image.
    Node feature layout:
        0-2: x, y, z
        3:   length / scale
        4:   radius / thickness
        5-7: pitch, yaw, roll
        8-11: organ_type one-hot (internode, petiole, leaf, floral_bud)
        12:  shoot_id
        13:  phytomer_idx
        14:  existence

    Uses sparse 3D Euclidean k-NN parent prediction (k = 16) to keep
    topology prediction at O(N*k) memory/compute instead of O(N^2).
    """

    def __init__(self, max_nodes: int = 2048, node_dim: int = 15,
                 embed_dim: int = 256, num_layers: int = 4,
                 k_nearest: int = 16):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_dim = node_dim
        self.embed_dim = embed_dim
        self.k_nearest = k_nearest

        # 2D Multi-Scale Vision Encoder (32x32 = 1024 spatial tokens)
        self.vision_encoder = MultiScaleSpatialEncoder(out_dim=embed_dim)

        # Timestep Embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Optional Camera Pose Angle Encoder (Azimuth, Elevation)
        self.pose_encoder = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # DAP (Days After Planting) Condition Encoder
        self.dap_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 15D Node Projection & Learned Position Embeddings
        self.node_proj = nn.Linear(node_dim + 1, embed_dim)
        self.node_pos_emb = nn.Embedding(max_nodes, embed_dim)

        # k-NN Transformer Decoder Layers (O(N*k) self-attention).
        # Cross-attention is expensive at N=2048, so apply it only to every other
        # layer (first and last) while keeping k-NN self-attention in every layer.
        self.transformer_layers = nn.ModuleList([
            KNNTransformerDecoderLayer(
                d_model=embed_dim,
                nhead=8,
                k=k_nearest,
                dim_feedforward=512,
                dropout=0.1,
                use_cross_attention=(i == 0 or i == num_layers - 1)
            ) for i in range(num_layers)
        ])

        # Prediction Heads (15D 3D Organ Attributes)
        self.node_pred_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, node_dim)
        )
        self.existence_pred_head = nn.Linear(embed_dim, 1)

        # Organ Type Classification Head (4-class)
        self.organ_type_head = nn.Linear(embed_dim, 4)

        # DAP-based adaptive node-budget head: predicts expected active node count.
        # Trained as a regression over the true active node count / max_nodes.
        self.node_budget_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

        # Sparse k-NN Parent Prediction Head
        # Only predicts logits for the k nearest neighbor candidates.
        self.parent_pred_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)
        )

    def _compute_knn_indices(self, coords: torch.Tensor) -> torch.Tensor:
        """Return (B, N, k) indices of k-nearest neighbors by Euclidean distance."""
        B, N, _ = coords.shape
        k_val = min(self.k_nearest, N)
        dist_matrix = torch.cdist(coords, coords)  # (B, N, N)
        _, knn_indices = torch.topk(dist_matrix, k=k_val, largest=False, dim=-1)
        return knn_indices

    def _sparse_parent_logits(self, h_nodes: torch.Tensor,
                              knn_indices: torch.Tensor) -> torch.Tensor:
        """Compute (B, N, k) parent logits for k-NN candidates only."""
        B, N, _ = h_nodes.shape
        k_val = knn_indices.shape[-1]

        # h_i: (B, N, 1, embed_dim) -> expand to (B, N, k, embed_dim)
        h_i = h_nodes.unsqueeze(2).expand(-1, -1, k_val, -1)

        # Gather neighbor features: h_j[candidate] for each query node
        # knn_indices: (B, N, k) -> (B, N, k, 1) for gather on dim 1
        h_j = torch.gather(
            h_nodes.unsqueeze(2).expand(-1, -1, N, -1),  # (B, N, N, embed_dim)
            dim=2,
            index=knn_indices.unsqueeze(-1).expand(-1, -1, -1, self.embed_dim)
        )  # (B, N, k, embed_dim)

        pair_feat = torch.cat([h_i, h_j], dim=-1)  # (B, N, k, 2*embed_dim)
        logits = self.parent_pred_head(pair_feat).squeeze(-1)  # (B, N, k)
        return logits

    def forward(self, noisy_nodes: torch.Tensor, noisy_existence: torch.Tensor,
                timesteps: torch.Tensor, images: torch.Tensor,
                camera_poses: torch.Tensor = None,
                dap: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            noisy_nodes: (B, N, 15) 15D organ primitives
            noisy_existence: (B, N, 1)
            timesteps: (B,)
            images: (B, 3, H, W) 2D projection input image
            camera_poses: (B, 2) optional camera angles (azimuth_norm, elevation_norm)
            dap: (B, 1) optional normalized days-after-planting

        Returns:
            pred_x0: (B, N, 15)
            pred_node_noise: (B, N, 15)
            pred_existence_logits: (B, N)
            pred_parent_logits: (B, N, k)
            pred_parent_candidates: (B, N, k)  # k-NN candidate indices
            pred_organ_type_logits: (B, N, 4)
            pred_node_budget: (B,)
        """
        B, N, _ = noisy_nodes.shape
        device = noisy_nodes.device

        # 1. Extract 2D spatial vision key/value features (B, 1024, embed_dim)
        img_feats = self.vision_encoder(images)
        img_feats = img_feats.flatten(2).permute(0, 2, 1)

        # Inject camera pose angle condition if provided
        if camera_poses is not None:
            pose_emb = self.pose_encoder(camera_poses).unsqueeze(1)
            img_feats = img_feats + pose_emb

        # 2. Compute timestep embeddings
        t_emb = self.time_emb(timesteps).unsqueeze(1)

        # 3. Project 15D node inputs with learned position embeddings
        node_in = torch.cat([noisy_nodes, noisy_existence], dim=-1)  # (B, N, 16)
        node_indices = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        h_nodes = self.node_proj(node_in) + t_emb + self.node_pos_emb(node_indices)

        # 4. Inject DAP condition if provided
        if dap is not None:
            dap_emb = self.dap_encoder(dap.view(B, 1)).unsqueeze(1)  # (B, 1, embed_dim)
            h_nodes = h_nodes + dap_emb

        # 5. Compute k-NN graph from initial 3D coordinates and reuse through layers
        knn_indices = self._compute_knn_indices(noisy_nodes[:, :, :3])  # (B, N, k)

        # Pass through k-NN transformer decoder layers
        for layer in self.transformer_layers:
            h_nodes = layer(tgt=h_nodes, memory=img_feats, knn_indices=knn_indices)

        # 6. Predict direct 15D organ attributes & existence
        pred_x0 = torch.clamp(self.node_pred_head(h_nodes), 0.0, 1.0)
        pred_existence_logits = self.existence_pred_head(h_nodes).squeeze(-1)

        # 7. Predict 4-class organ type
        pred_organ_type_logits = self.organ_type_head(h_nodes)  # (B, N, 4)

        # 8. Predict DAP-based adaptive node budget (normalized [0,1])
        # Use the mean pooled node feature + global DAP/image context.
        pooled_h = h_nodes.mean(dim=1)  # (B, embed_dim)
        pred_node_budget = self.node_budget_head(pooled_h).squeeze(-1)  # (B,)

        # 9. Predict sparse 3D parent topology logits for k-NN candidates
        coords_3d = pred_x0[:, :, :3]  # (B, N, 3)
        parent_candidates = self._compute_knn_indices(coords_3d)  # (B, N, k)
        pred_parent_logits = self._sparse_parent_logits(h_nodes, parent_candidates)

        # Infer noise from x0
        pred_node_noise = noisy_nodes - pred_x0

        return {
            "pred_x0": pred_x0,
            "pred_node_noise": pred_node_noise,
            "pred_existence_logits": pred_existence_logits,
            "pred_parent_logits": pred_parent_logits,
            "pred_parent_candidates": parent_candidates,
            "pred_organ_type_logits": pred_organ_type_logits,
            "pred_node_budget": pred_node_budget,
        }
