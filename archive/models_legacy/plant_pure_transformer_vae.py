"""
Pure Transformer Global Plant VAE (No separate ResNet blocks).

A unified, pure Vision/Graph Transformer architecture where all token-mixing (Self/Cross-Attention)
and channel-mixing (4x FFN with GeLU and LayerNorm) are performed entirely within standard
Transformer Encoder and Decoder stacks.
"""

import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_based.models.plant_organ_array import (
    NUM_FEATURES_TYPED,
    T_COL_ORGAN_TYPE,
    T_COL_SHOOT_TYPE,
    T_COL_BUD_STATE,
    T_COL_BUD_IS_TERMINAL,
    T_COL_EXISTENCE,
    T_COL_SHOOT_ID,
    T_COL_PARENT_SHOOT_ID,
    T_COL_PARENT_NODE_IDX,
    T_COL_PHYTOMER_IDX,
    T_COL_PITCH,
    T_COL_YAW,
    T_COL_ROLL,
    T_COL_CURVATURE,
    T_COL_PHYLLOTACTIC_ANGLE,
    T_COL_CURV_PERT_0,
    T_COL_CURV_PERT_1,
    T_COL_YAW_PERT_0,
    T_COL_YAW_PERT_1,
    T_COL_FLOWER_AZIMUTH,
    T_COL_LENGTH,
    T_COL_RADIUS,
    T_COL_SCALE,
    T_COL_CURRENT_LEAF_SCALE_FACTOR,
    T_COL_LEAFLET_SCALE,
    ORGAN_LEAF,
)
from diffusion_based.models.plant_vae import (
    OrganFeatureNormalizer,
    ANGULAR_COLS,
    NUM_ANGULAR_COLS,
    NUM_ORGAN_TYPES,
)


class PlantPureTransformerVAE(nn.Module):
    """
    Pure Transformer Architecture for Global 512D Plant VAE.
    """

    def __init__(
        self,
        in_features: int = NUM_FEATURES_TYPED,
        latent_dim: int = 512,
        hidden_dim: int = 512,
        ffn_dim: int = 2048,
        max_organs: int = 2048,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_organs = max_organs
        self.normalizer = OrganFeatureNormalizer()

        # 1. Direct Linear Projection of 118D Features
        self.organ_type_embed = nn.Embedding(NUM_ORGAN_TYPES, 32)
        self.shoot_type_embed = nn.Embedding(2, 16)
        self.bud_state_embed = nn.Embedding(6, 16)
        self.terminal_embed = nn.Embedding(2, 8)

        in_enc_dim = (in_features - 4 - NUM_ANGULAR_COLS) + (2 * NUM_ANGULAR_COLS) + 32 + 16 + 16 + 8
        self.in_proj = nn.Linear(in_enc_dim, hidden_dim)
        self.in_norm = nn.LayerNorm(hidden_dim)

        # 2. Pure Transformer Encoder Stack (6 layers with 2048D FFN)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers)

        # Learnable global pooling query
        self.global_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.global_pool_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 3. Pure Transformer Decoder Stack (6 layers with 2048D FFN)
        self.tree_pos_embed = nn.Embedding(max_organs, 128)
        self.tree_phytomer_embed = nn.Embedding(128, 128)
        self.tree_shoot_embed = nn.Embedding(32, 128)
        self.tree_organ_embed = nn.Embedding(NUM_ORGAN_TYPES, 128)
        self.tree_query_proj = nn.Linear(512, hidden_dim)

        # FiLM / AdaLN modulation on input queries
        self.film_gen = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_layers)
        self.out_norm = nn.LayerNorm(hidden_dim)

        # Direct Output Prediction Heads from Transformer final LayerNorm
        self.head_organ_type = nn.Linear(hidden_dim, NUM_ORGAN_TYPES)
        self.head_shoot_type = nn.Linear(hidden_dim, 2)
        self.head_bud_state = nn.Linear(hidden_dim, 6)
        self.head_is_terminal = nn.Linear(hidden_dim, 2)
        self.head_existence = nn.Linear(hidden_dim, 1)
        self.head_continuous = nn.Linear(hidden_dim, in_features)
        self.head_angles = nn.Linear(hidden_dim, 2 * NUM_ANGULAR_COLS)

    def _extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.normalizer.normalize(x)
        ot = x[..., T_COL_ORGAN_TYPE].long().clamp(0, NUM_ORGAN_TYPES - 1)
        st = x[..., T_COL_SHOOT_TYPE].long().clamp(0, 1)
        bs = x[..., T_COL_BUD_STATE].long().clamp(0, 5)
        term = x[..., T_COL_BUD_IS_TERMINAL].long().clamp(0, 1)

        ot_emb = self.organ_type_embed(ot)
        st_emb = self.shoot_type_embed(st)
        bs_emb = self.bud_state_embed(bs)
        term_emb = self.terminal_embed(term)

        rad = x[..., ANGULAR_COLS] * (math.pi / 180.0)
        circ_ang = torch.cat([torch.cos(rad), torch.sin(rad)], dim=-1)

        mask_cont = torch.ones(self.in_features, dtype=torch.bool, device=x.device)
        for col in [T_COL_ORGAN_TYPE, T_COL_SHOOT_TYPE, T_COL_BUD_STATE, T_COL_BUD_IS_TERMINAL] + ANGULAR_COLS:
            mask_cont[col] = False

        cont_x = norm_x[..., mask_cont]
        feat = torch.cat([cont_x, circ_ang, ot_emb, st_emb, bs_emb, term_emb], dim=-1)
        h = self.in_proj(feat)
        return self.in_norm(h)

    def encode(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        h = self._extract_tokens(x)

        key_padding_mask = ~mask if mask is not None else None
        h = self.transformer_encoder(h, src_key_padding_mask=key_padding_mask)

        q = self.global_query.expand(B, -1, -1)
        global_h, _ = self.global_pool_attn(
            query=q, key=h, value=h, key_padding_mask=key_padding_mask
        )
        global_h = global_h.squeeze(1)

        mu = self.fc_mu(global_h)
        logvar = self.fc_logvar(global_h).clamp(-10.0, 10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _build_tree_queries(self, target_x: Optional[torch.Tensor], target_len: int, device: torch.device) -> torch.Tensor:
        if target_x is not None and target_x.shape[1] >= target_len:
            pos_ids = torch.arange(target_len, device=device).unsqueeze(0).expand(target_x.shape[0], -1)
            phyt_ids = target_x[:, :target_len, T_COL_PHYTOMER_IDX].long().clamp(0, 127)
            shoot_ids = target_x[:, :target_len, T_COL_SHOOT_ID].long().clamp(0, 31)
            ot_ids = target_x[:, :target_len, T_COL_ORGAN_TYPE].long().clamp(0, NUM_ORGAN_TYPES - 1)
        else:
            pos_ids = torch.arange(target_len, device=device).unsqueeze(0)
            phyt_ids = (pos_ids // 8).clamp(0, 127)
            shoot_ids = (pos_ids // 64).clamp(0, 31)
            ot_ids = (pos_ids % 6).clamp(0, NUM_ORGAN_TYPES - 1)

        q_pos = self.tree_pos_embed(pos_ids)
        q_phyt = self.tree_phytomer_embed(phyt_ids)
        q_shoot = self.tree_shoot_embed(shoot_ids)
        q_ot = self.tree_organ_embed(ot_ids)

        q_cat = torch.cat([q_pos, q_phyt, q_shoot, q_ot], dim=-1)
        return self.tree_query_proj(q_cat)

    def decode_raw(
        self, z_global: torch.Tensor, target_len: int = 2048, tree_x: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B = z_global.shape[0]
        device = z_global.device

        tree_q = self._build_tree_queries(tree_x, target_len, device)
        if tree_q.shape[0] != B:
            tree_q = tree_q.expand(B, -1, -1)

        film = self.film_gen(z_global)
        gamma, beta = film.chunk(2, dim=-1)
        modulated_q = tree_q * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        z_memory = z_global.unsqueeze(1)
        h = self.transformer_decoder(tgt=modulated_q, memory=z_memory)
        h = self.out_norm(h)

        return {
            "organ_type_logits": self.head_organ_type(h),
            "shoot_type_logits": self.head_shoot_type(h),
            "bud_state_logits": self.head_bud_state(h),
            "is_terminal_logits": self.head_is_terminal(h),
            "existence_logits": self.head_existence(h),
            "cont_norm": self.head_continuous(h),
            "angle_vecs": self.head_angles(h),
        }

    def decode(
        self, z_global: torch.Tensor, target_len: int = 2048, tree_x: Optional[torch.Tensor] = None, hard_categoricals: bool = True
    ) -> torch.Tensor:
        if z_global.dim() == 1:
            z_global = z_global.unsqueeze(0)
            single = True
        else:
            single = False

        raw = self.decode_raw(z_global, target_len=target_len, tree_x=tree_x)
        cont = self.normalizer.denormalize(raw["cont_norm"])

        exist_val = torch.sigmoid(raw["existence_logits"]).squeeze(-1)
        if hard_categoricals:
            ot_val = raw["organ_type_logits"].argmax(dim=-1).float()
            st_val = raw["shoot_type_logits"].argmax(dim=-1).float()
            bs_val = raw["bud_state_logits"].argmax(dim=-1).float()
            term_val = raw["is_terminal_logits"].argmax(dim=-1).float()
        else:
            ot_val = cont[..., T_COL_ORGAN_TYPE]
            st_val = cont[..., T_COL_SHOOT_TYPE]
            bs_val = cont[..., T_COL_BUD_STATE]
            term_val = cont[..., T_COL_BUD_IS_TERMINAL]

        angle_vecs = raw["angle_vecs"]
        cos_pred = angle_vecs[..., :NUM_ANGULAR_COLS]
        sin_pred = angle_vecs[..., NUM_ANGULAR_COLS:]
        angle_deg_pred = torch.atan2(sin_pred, cos_pred) * (180.0 / math.pi)

        recovered_angles = {}
        for i, col_idx in enumerate(ANGULAR_COLS):
            deg = angle_deg_pred[..., i:i+1]
            if col_idx in (T_COL_PHYLLOTACTIC_ANGLE, T_COL_FLOWER_AZIMUTH):
                deg = torch.where(deg < 0, deg + 360.0, deg)
            recovered_angles[col_idx] = deg

        cols = []
        for col_i in range(self.in_features):
            if col_i == T_COL_ORGAN_TYPE:
                cols.append(ot_val.unsqueeze(-1))
            elif col_i == T_COL_SHOOT_TYPE:
                cols.append(st_val.unsqueeze(-1))
            elif col_i == T_COL_BUD_STATE:
                cols.append(bs_val.unsqueeze(-1))
            elif col_i == T_COL_BUD_IS_TERMINAL:
                cols.append(term_val.unsqueeze(-1))
            elif col_i == T_COL_EXISTENCE:
                cols.append(exist_val.unsqueeze(-1))
            elif col_i in recovered_angles:
                cols.append(recovered_angles[col_i])
            else:
                cols.append(cont[..., col_i:col_i+1])

        out = torch.cat(cols, dim=-1)
        return out.squeeze(0) if single else out

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        B, N, _ = x.shape
        mu, logvar = self.encode(x, mask=mask)
        z_global = self.reparameterize(mu, logvar)
        raw = self.decode_raw(z_global, target_len=N, tree_x=x)
        recon_x = self.decode(z_global, target_len=N, tree_x=x, hard_categoricals=False)
        return recon_x, mu, logvar, raw


def compute_pure_transformer_vae_loss(
    model: PlantPureTransformerVAE,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    beta: float = 1e-4,
) -> Dict[str, torch.Tensor]:
    recon_x, mu, logvar, raw = model(x, mask=mask)

    target_exist = (x[..., T_COL_EXISTENCE] > 0.5).float()
    if mask is not None:
        exist_mask = mask & (target_exist > 0.5)
    else:
        exist_mask = (target_exist > 0.5)

    if exist_mask.sum() == 0:
        exist_mask = torch.ones_like(target_exist, dtype=torch.bool)

    loss_exist = F.binary_cross_entropy_with_logits(raw["existence_logits"].squeeze(-1), target_exist)

    target_ot = x[..., T_COL_ORGAN_TYPE].long().clamp(0, NUM_ORGAN_TYPES - 1)
    target_st = x[..., T_COL_SHOOT_TYPE].long().clamp(0, 1)
    target_bs = x[..., T_COL_BUD_STATE].long().clamp(0, 5)
    target_term = x[..., T_COL_BUD_IS_TERMINAL].long().clamp(0, 1)

    loss_ot = F.cross_entropy(raw["organ_type_logits"][exist_mask], target_ot[exist_mask])
    loss_st = F.cross_entropy(raw["shoot_type_logits"][exist_mask], target_st[exist_mask])
    loss_bs = F.cross_entropy(raw["bud_state_logits"][exist_mask], target_bs[exist_mask])
    loss_term = F.cross_entropy(raw["is_terminal_logits"][exist_mask], target_term[exist_mask])
    loss_cls = loss_ot + 0.5 * loss_st + 0.5 * loss_bs + 0.5 * loss_term

    target_norm = model.normalizer.normalize(x)
    loss_geom = F.smooth_l1_loss(raw["cont_norm"][exist_mask], target_norm[exist_mask], beta=0.001)

    dim_cols = [T_COL_LENGTH, T_COL_RADIUS, T_COL_SCALE, T_COL_CURRENT_LEAF_SCALE_FACTOR, T_COL_LEAFLET_SCALE]
    loss_dim = F.l1_loss(raw["cont_norm"][exist_mask][:, dim_cols], target_norm[exist_mask][:, dim_cols])

    is_leaf = exist_mask & (target_ot == ORGAN_LEAF)
    if is_leaf.sum() > 0:
        loss_leaf_scale = F.l1_loss(raw["cont_norm"][is_leaf][:, T_COL_SCALE], target_norm[is_leaf][:, T_COL_SCALE])
    else:
        loss_leaf_scale = torch.tensor(0.0, device=x.device)

    target_ang_rad = x[..., ANGULAR_COLS] * (math.pi / 180.0)
    target_vec = torch.cat([torch.cos(target_ang_rad), torch.sin(target_ang_rad)], dim=-1)
    loss_angle_vec = F.smooth_l1_loss(raw["angle_vecs"][exist_mask], target_vec[exist_mask], beta=0.001)

    diff_ang_rad = (x[..., ANGULAR_COLS] - recon_x[..., ANGULAR_COLS]) * (math.pi / 180.0)
    loss_angle_geo = (1.0 - torch.cos(diff_ang_rad[exist_mask])).mean()
    loss_angle = 10.0 * loss_angle_vec + 5.0 * loss_angle_geo

    loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

    total_loss = (
        50.0 * loss_geom +
        50.0 * loss_dim +
        50.0 * loss_leaf_scale +
        10.0 * loss_cls +
        5.0 * loss_exist +
        10.0 * loss_angle +
        beta * loss_kl
    )

    return {
        "loss": total_loss,
        "loss_geom": loss_geom.detach(),
        "loss_dim": loss_dim.detach(),
        "loss_cls": loss_cls.detach(),
        "loss_exist": loss_exist.detach(),
        "loss_angle": loss_angle.detach(),
        "loss_kl": loss_kl.detach(),
    }
