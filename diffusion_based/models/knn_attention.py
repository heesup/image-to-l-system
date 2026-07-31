"""Sparse k-NN self-attention for scalable graph transformers.

Replaces dense O(N^2) self-attention with O(N*k) attention over a dynamic
k-nearest-neighbor graph in 3D coordinate space.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class KNNSelfAttention(nn.Module):
    """Multi-head self-attention restricted to k-nearest spatial neighbors.

    Args:
        d_model: model dimension
        nhead: number of attention heads
        k: number of neighbors each node attends to
        dropout: dropout probability
    """

    def __init__(self, d_model: int, nhead: int, k: int = 16, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.k = k

        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, knn_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model) node features
            knn_indices: (B, N, k) neighbor indices for each node
        Returns:
            (B, N, d_model) attended features
        """
        B, N, _ = x.shape
        k_val = min(self.k, N)
        knn_indices = knn_indices[:, :, :k_val]

        # Query from every node
        q = self.q_proj(x)  # (B, N, d)
        q = q.view(B, N, self.nhead, self.head_dim).transpose(1, 2)  # (B, h, N, d_h)

        # Gather k neighbor features (B, N, k, d)
        neighbors = torch.gather(
            x.unsqueeze(2).expand(-1, -1, N, -1),
            dim=2,
            index=knn_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_model)
        )

        # Project to keys and values
        kv = self.kv_proj(neighbors)  # (B, N, k, 2*d)
        k_feats, v_feats = kv.chunk(2, dim=-1)

        # Reshape to multi-head: (B, h, N, k, d_h)
        k_feats = k_feats.view(B, N, k_val, self.nhead, self.head_dim).permute(0, 3, 1, 2, 4)
        v_feats = v_feats.view(B, N, k_val, self.nhead, self.head_dim).permute(0, 3, 1, 2, 4)

        # Scaled dot-product attention: q (B,h,N,1,d_h) @ k^T (B,h,N,d_h,k) -> (B,h,N,1,k)
        scores = torch.matmul(q.unsqueeze(3), k_feats.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v_feats).squeeze(3)  # (B, h, N, d_h)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.out_proj(out)


class KNNCrossAttention(nn.Module):
    """Standard multi-head cross-attention to image/memory tokens.

    Memory size M is fixed (e.g. 1024 image tokens), so complexity is O(N*M),
    which is acceptable even for large N.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return self.attn(query, memory, memory, need_weights=False)[0]


class KNNTransformerDecoderLayer(nn.Module):
    """Decoder layer with k-NN self-attention + cross-attention + FFN.

    Replaces PyTorch's nn.TransformerDecoderLayer for O(N*k) node self-attention.
    """

    def __init__(self, d_model: int, nhead: int, k: int = 16,
                 dim_feedforward: int = 512, dropout: float = 0.1,
                 use_cross_attention: bool = True):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.self_attn = KNNSelfAttention(d_model, nhead, k=k, dropout=dropout)
        if use_cross_attention:
            self.cross_attn = KNNCrossAttention(d_model, nhead, dropout=dropout)
            self.norm2 = nn.LayerNorm(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                knn_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tgt: (B, N, d_model) node features
            memory: (B, M, d_model) image/memory features
            knn_indices: (B, N, k) spatial neighbor indices
        Returns:
            (B, N, d_model) updated node features
        """
        # k-NN self-attention with residual + pre-norm
        h = self.norm1(tgt)
        h = tgt + self.self_attn(h, knn_indices)

        # Cross-attention to image features (disabled in some middle layers)
        if self.use_cross_attention:
            h = h + self.cross_attn(self.norm2(h), memory)

        # Feed-forward
        out = h + self.ffn(self.norm3(h))
        return out
