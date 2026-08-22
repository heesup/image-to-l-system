"""
Plant Variational Autoencoder (Plant VAE) for 40D Typed Plant Organ Vectors.

Provides:
1. PlantOrganVAE: Per-organ feature manifold compression (40D <-> z_organ in R^16).
2. PlantSequenceVAE / PlantSetVAE: Full plant canopy sequence compression ((N, 40) <-> z_plant in R^256).
3. End-to-end differentiable reconstruction with full gradient connectivity to HeliosPyTorchRenderer.
4. Lossless serialization & round-trip to/from Helios XML.
"""

import math
from typing import Dict, Tuple, Optional, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    NUM_FEATURES_TYPED,
    NUM_ORGAN_TYPES,
    T_COL_PLANT_ID,
    T_COL_PLANT_AGE,
    T_COL_BASE_X,
    T_COL_BASE_Y,
    T_COL_BASE_Z,
    T_COL_SHOOT_ID,
    T_COL_PARENT_SHOOT_ID,
    T_COL_PARENT_NODE_IDX,
    T_COL_PARENT_PETIOLE_IDX,
    T_COL_PHYTOMER_IDX,
    T_COL_CHILD_INDEX,
    T_COL_ORGAN_TYPE,
    T_COL_SHOOT_TYPE,
    T_COL_LENGTH,
    T_COL_RADIUS,
    T_COL_SCALE,
    T_COL_PITCH,
    T_COL_YAW,
    T_COL_ROLL,
    T_COL_CURVATURE,
    T_COL_PHYLLOTACTIC_ANGLE,
    T_COL_LENGTH_MAX,
    T_COL_LENGTH_SEGMENTS,
    T_COL_CURV_PERT_0,
    T_COL_CURV_PERT_1,
    T_COL_YAW_PERT_0,
    T_COL_YAW_PERT_1,
    T_COL_CURRENT_LEAF_SCALE_FACTOR,
    T_COL_TAPER,
    T_COL_RADIAL_SUBDIVISIONS,
    T_COL_LEAFLET_SCALE,
    T_COL_LEAFLET_OFFSET,
    T_COL_BUD_STATE,
    T_COL_BUD_PARENT_INDEX,
    T_COL_BUD_IS_TERMINAL,
    T_COL_FRUIT_SCALE,
    T_COL_FLOWER_AZIMUTH,
    T_COL_FLOWER_OFFSET,
    T_COL_RESERVED,
    T_COL_EXISTENCE,
    ORGAN_ROOT_META,
    ORGAN_SHOOT_META,
    ORGAN_INTERNODE,
    ORGAN_PETIOLE,
    ORGAN_LEAF,
    ORGAN_BUD,
    ORGAN_PEDUNCLE,
    ORGAN_FLOWER,
)


class OrganFeatureNormalizer(nn.Module):
    """Normalizes the continuous columns of 40D typed organ tensors using scale priors."""
    def __init__(self):
        super().__init__()
        # Predefined scales for crop architecture (m, degrees, counts)
        scales = torch.ones(NUM_FEATURES_TYPED, dtype=torch.float32)
        scales[T_COL_PLANT_AGE] = 100.0
        scales[T_COL_BASE_X:T_COL_BASE_Z+1] = 2.0
        scales[T_COL_SHOOT_ID] = 50.0
        scales[T_COL_PARENT_SHOOT_ID] = 50.0
        scales[T_COL_PARENT_NODE_IDX] = 50.0
        scales[T_COL_PARENT_PETIOLE_IDX] = 5.0
        scales[T_COL_PHYTOMER_IDX] = 30.0
        scales[T_COL_CHILD_INDEX] = 5.0
        scales[T_COL_LENGTH] = 0.05
        scales[T_COL_RADIUS] = 0.005
        scales[T_COL_SCALE] = 0.05
        scales[T_COL_PITCH] = 180.0
        scales[T_COL_YAW] = 180.0
        scales[T_COL_ROLL] = 180.0
        scales[T_COL_CURVATURE] = 90.0
        scales[T_COL_PHYLLOTACTIC_ANGLE] = 360.0
        scales[T_COL_LENGTH_MAX] = 0.05
        scales[T_COL_LENGTH_SEGMENTS] = 10.0
        scales[T_COL_CURV_PERT_0:T_COL_YAW_PERT_1+1] = 45.0
        scales[T_COL_CURRENT_LEAF_SCALE_FACTOR] = 1.0
        scales[T_COL_TAPER] = 1.0
        scales[T_COL_RADIAL_SUBDIVISIONS] = 10.0
        scales[T_COL_LEAFLET_SCALE] = 1.0
        scales[T_COL_LEAFLET_OFFSET] = 1.0
        scales[T_COL_BUD_PARENT_INDEX] = 5.0
        scales[T_COL_FRUIT_SCALE] = 0.05
        scales[T_COL_FLOWER_AZIMUTH] = 360.0
        scales[T_COL_FLOWER_OFFSET] = 1.0
        self.register_buffer("scales", scales)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / (self.scales.to(x.device) + 1e-7)

    def denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * self.scales.to(x_norm.device)


# Angular column indices for circular representation (degrees)
ANGULAR_COLS = [
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
]
NUM_ANGULAR_COLS = len(ANGULAR_COLS)


class ResBlock(nn.Module):
    """Residual MLP Block with LayerNorm, GELU, and Dropout."""
    def __init__(self, dim: int, dropout: float = 0.05):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class PlantOrganVAE(nn.Module):
    """
    Per-Organ Variational Autoencoder with Circular Angle Manifold.
    Compresses individual 40D organ vectors into a continuous latent space z in R^latent_dim.
    """
    def __init__(
        self,
        in_features: int = NUM_FEATURES_TYPED,
        latent_dim: int = 512,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.normalizer = OrganFeatureNormalizer()

        # Organ Type & Categorical Embeddings
        self.organ_type_embed = nn.Embedding(NUM_ORGAN_TYPES, 32)
        self.shoot_type_embed = nn.Embedding(2, 16)
        self.bud_state_embed = nn.Embedding(6, 16)
        self.terminal_embed = nn.Embedding(2, 8)

        # Input features: (40 - 4 categoricals - 10 angle cols + 2*10 sin/cos + 32+16+16+8 = 118)
        in_enc_dim = (in_features - 4 - NUM_ANGULAR_COLS) + (2 * NUM_ANGULAR_COLS) + 32 + 16 + 16 + 8
        self.in_proj = nn.Linear(in_enc_dim, hidden_dim)

        self.encoder_res = nn.Sequential(
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: Latent -> Project -> ResBlocks -> Heads
        self.dec_proj = nn.Linear(latent_dim, hidden_dim)
        self.decoder_res = nn.Sequential(
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
        )

        # Decoder Heads
        self.head_organ_type = nn.Linear(hidden_dim, NUM_ORGAN_TYPES)
        self.head_shoot_type = nn.Linear(hidden_dim, 2)
        self.head_bud_state = nn.Linear(hidden_dim, 6)
        self.head_is_terminal = nn.Linear(hidden_dim, 2)
        self.head_existence = nn.Linear(hidden_dim, 1)

        # Continuous attributes regression (normalized)
        self.head_continuous = nn.Linear(hidden_dim, in_features)
        # Circular angle unit vectors (2D per angle: cos, sin)
        self.head_angles = nn.Linear(hidden_dim, 2 * NUM_ANGULAR_COLS)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (..., 40) typed organ tensor
        Returns:
            mu: (..., latent_dim)
            logvar: (..., latent_dim)
        """
        device = x.device
        norm_x = self.normalizer.normalize(x)

        # Extract & clamp categoricals
        ot = x[..., T_COL_ORGAN_TYPE].long().clamp(0, NUM_ORGAN_TYPES - 1)
        st = x[..., T_COL_SHOOT_TYPE].long().clamp(0, 1)
        bs = x[..., T_COL_BUD_STATE].long().clamp(0, 5)
        term = x[..., T_COL_BUD_IS_TERMINAL].long().clamp(0, 1)

        ot_emb = self.organ_type_embed(ot)
        st_emb = self.shoot_type_embed(st)
        bs_emb = self.bud_state_embed(bs)
        term_emb = self.terminal_embed(term)

        # Circular angle encodings (cos, sin in radians)
        ang_rad = x[..., ANGULAR_COLS] * (math.pi / 180.0)
        cos_ang = torch.cos(ang_rad)
        sin_ang = torch.sin(ang_rad)
        circ_ang = torch.cat([cos_ang, sin_ang], dim=-1)

        # Mask out raw categorical and raw angular columns
        mask_cont = torch.ones(self.in_features, dtype=torch.bool, device=device)
        mask_cont[T_COL_ORGAN_TYPE] = False
        mask_cont[T_COL_SHOOT_TYPE] = False
        mask_cont[T_COL_BUD_STATE] = False
        mask_cont[T_COL_BUD_IS_TERMINAL] = False
        for c in ANGULAR_COLS:
            mask_cont[c] = False

        cont_x = norm_x[..., mask_cont]
        feat = torch.cat([cont_x, circ_ang, ot_emb, st_emb, bs_emb, term_emb], dim=-1)

        h = self.in_proj(feat)
        h = self.encoder_res(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-10.0, 10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_raw(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decodes latent vector into categorical logits, normalized continuous parameters, and circular angle vectors.
        """
        h = self.dec_proj(z)
        h = self.decoder_res(h)

        organ_type_logits = self.head_organ_type(h)
        shoot_type_logits = self.head_shoot_type(h)
        bud_state_logits = self.head_bud_state(h)
        is_terminal_logits = self.head_is_terminal(h)
        existence_logits = self.head_existence(h)
        cont_norm = self.head_continuous(h)
        angle_vecs = self.head_angles(h)

        return {
            "organ_type_logits": organ_type_logits,
            "shoot_type_logits": shoot_type_logits,
            "bud_state_logits": bud_state_logits,
            "is_terminal_logits": is_terminal_logits,
            "existence_logits": existence_logits,
            "cont_norm": cont_norm,
            "angle_vecs": angle_vecs,
        }

    def decode(self, z: torch.Tensor, hard_categoricals: bool = True) -> torch.Tensor:
        """
        Decodes latent vector z into full 40D typed organ tensor with continuous gradient connectivity.
        """
        raw = self.decode_raw(z)
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

        # Decode circular angles via atan2
        angle_vecs = raw["angle_vecs"]  # (..., 2 * NUM_ANGULAR_COLS)
        cos_pred = angle_vecs[..., :NUM_ANGULAR_COLS]
        sin_pred = angle_vecs[..., NUM_ANGULAR_COLS:]
        angle_rad_pred = torch.atan2(sin_pred, cos_pred)
        angle_deg_pred = angle_rad_pred * (180.0 / math.pi)

        # Map angles to respective domain
        recovered_angles = {}
        for i, col_idx in enumerate(ANGULAR_COLS):
            deg = angle_deg_pred[..., i:i+1]
            if col_idx in (T_COL_PHYLLOTACTIC_ANGLE, T_COL_FLOWER_AZIMUTH):
                # [0, 360]
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
            elif col_i in (T_COL_PLANT_ID, T_COL_SHOOT_ID, T_COL_PARENT_SHOOT_ID,
                           T_COL_PARENT_NODE_IDX, T_COL_PARENT_PETIOLE_IDX,
                           T_COL_PHYTOMER_IDX, T_COL_CHILD_INDEX, T_COL_BUD_PARENT_INDEX):
                if hard_categoricals:
                    cols.append(torch.round(cont[..., col_i:col_i+1]))
                else:
                    cols.append(cont[..., col_i:col_i+1])
            elif col_i in (T_COL_LENGTH, T_COL_RADIUS, T_COL_SCALE, T_COL_LENGTH_MAX,
                           T_COL_CURRENT_LEAF_SCALE_FACTOR, T_COL_LEAFLET_SCALE, T_COL_FRUIT_SCALE):
                cols.append(cont[..., col_i:col_i+1].clamp(min=0.0))
            else:
                cols.append(cont[..., col_i:col_i+1])

        return torch.cat(cols, dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        raw = self.decode_raw(z)
        x_recon = self.decode(z, hard_categoricals=False)
        return x_recon, mu, logvar, raw


class PlantTransformerVAE(nn.Module):
    """
    Global Plant Canopy Sequence Variational Autoencoder.
    Compresses an entire plant (N, 40) organ sequence into a single global latent vector z_plant in R^latent_dim (e.g. 256).
    """
    def __init__(
        self,
        in_features: int = NUM_FEATURES_TYPED,
        latent_dim: int = 256,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        max_organs: int = 1600,
    ):
        super().__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.max_organs = max_organs
        self.normalizer = OrganFeatureNormalizer()

        # Tokenizer
        self.organ_vae = PlantOrganVAE(in_features=in_features, latent_dim=32, hidden_dim=64)
        self.organ_proj = nn.Linear(32, d_model)
        self.pos_emb = nn.Embedding(max_organs + 1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.05, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

        # Latent to Decoder Query Projector
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.query_pos_emb = nn.Embedding(max_organs, d_model)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.05, activation="gelu", batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output Projector back to Organ Latent (latent_dim of organ_vae)
        self.out_proj = nn.Linear(d_model, self.organ_vae.latent_dim)

    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, 40)
            mask: (B, N) boolean tensor where True = valid, False = padding
        Returns:
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        B, N, _ = x.shape
        device = x.device

        # 1. Per-organ tokenization via organ_vae encoder
        org_mu, _ = self.organ_vae.encode(x)  # (B, N, organ_latent_dim)
        tokens = self.organ_proj(org_mu)     # (B, N, d_model)

        # Add positional embedding
        positions = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        tokens = tokens + self.pos_emb(positions)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, N+1, d_model)

        # Key padding mask: True means IGNORED in PyTorch transformer
        if mask is not None:
            # CLS token is never masked
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
            full_mask = torch.cat([cls_mask, mask], dim=1)
            src_key_padding_mask = ~full_mask
        else:
            src_key_padding_mask = None

        h = self.transformer_encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        cls_out = h[:, 0, :]  # (B, d_model)

        mu = self.fc_mu(cls_out)
        logvar = self.fc_logvar(cls_out).clamp(-10.0, 10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z_plant: torch.Tensor, target_len: int = 100, hard_categoricals: bool = True) -> torch.Tensor:
        """
        Decodes plant latent vector z_plant into full (B, target_len, 40) organ sequence.
        """
        B = z_plant.shape[0]
        device = z_plant.device
        memory = self.latent_proj(z_plant).unsqueeze(1)  # (B, 1, d_model)

        # Target positional queries
        query_pos = torch.arange(target_len, device=device).unsqueeze(0).expand(B, target_len)
        tgt = self.query_pos_emb(query_pos)  # (B, target_len, d_model)

        h_dec = self.transformer_decoder(tgt, memory)  # (B, target_len, d_model)
        z_organs = self.out_proj(h_dec)  # (B, target_len, organ_latent_dim)

        # Decode each organ token through organ_vae decoder
        recon_organs = self.organ_vae.decode(z_organs, hard_categoricals=hard_categoricals)
        return recon_organs

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        mu, logvar = self.encode(x, mask=mask)
        z_plant = self.reparameterize(mu, logvar)
        recon_x = self.decode(z_plant, target_len=N, hard_categoricals=False)
        return recon_x, mu, logvar


def compute_organ_vae_loss(
    model: PlantOrganVAE,
    x: torch.Tensor,
    beta: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    """
    Computes multi-task reconstruction loss and KL divergence for PlantOrganVAE.
    """
    device = x.device
    x_recon, mu, logvar, raw = model(x)

    # 1. Existence Mask Loss
    target_exist = (x[..., T_COL_EXISTENCE] > 0.5).float()
    loss_exist = F.binary_cross_entropy_with_logits(
        raw["existence_logits"].squeeze(-1), target_exist
    )

    # Mask for existing organs only
    exist_mask = (target_exist > 0.5)
    if exist_mask.sum() == 0:
        exist_mask = torch.ones_like(target_exist, dtype=torch.bool)

    # 2. Categorical Classification Losses
    target_ot = x[..., T_COL_ORGAN_TYPE].long().clamp(0, NUM_ORGAN_TYPES - 1)
    target_st = x[..., T_COL_SHOOT_TYPE].long().clamp(0, 1)
    target_bs = x[..., T_COL_BUD_STATE].long().clamp(0, 5)
    target_term = x[..., T_COL_BUD_IS_TERMINAL].long().clamp(0, 1)

    loss_ot = F.cross_entropy(raw["organ_type_logits"][exist_mask], target_ot[exist_mask])
    loss_st = F.cross_entropy(raw["shoot_type_logits"][exist_mask], target_st[exist_mask])
    loss_bs = F.cross_entropy(raw["bud_state_logits"][exist_mask], target_bs[exist_mask])
    loss_term = F.cross_entropy(raw["is_terminal_logits"][exist_mask], target_term[exist_mask])

    loss_cls = loss_ot + 0.5 * loss_st + 0.5 * loss_bs + 0.5 * loss_term

    # 3. Continuous Parameter Loss (Normalized Smooth L1 + Explicit Physical Dimensions)
    target_norm = model.normalizer.normalize(x)
    loss_geom = F.smooth_l1_loss(
        raw["cont_norm"][exist_mask], target_norm[exist_mask], beta=0.001
    )

    dim_cols = [T_COL_LENGTH, T_COL_RADIUS, T_COL_SCALE, T_COL_CURRENT_LEAF_SCALE_FACTOR, T_COL_LEAFLET_SCALE]
    loss_dim = F.l1_loss(
        raw["cont_norm"][exist_mask][:, dim_cols],
        target_norm[exist_mask][:, dim_cols]
    )

    # Explicit leaf scale priority loss
    is_leaf = exist_mask & (target_ot == 4)  # ORGAN_LEAF
    if is_leaf.sum() > 0:
        loss_leaf_scale = F.l1_loss(raw["cont_norm"][is_leaf][:, T_COL_SCALE], target_norm[is_leaf][:, T_COL_SCALE])
    else:
        loss_leaf_scale = torch.tensor(0.0, device=device)

    # 4. Circular Angular Unit Vector Loss + Geodesic Periodic Loss
    target_ang_rad = x[..., ANGULAR_COLS] * (math.pi / 180.0)
    target_cos = torch.cos(target_ang_rad)
    target_sin = torch.sin(target_ang_rad)
    target_vec = torch.cat([target_cos, target_sin], dim=-1)

    pred_vec = raw["angle_vecs"]
    loss_angle_vec = F.smooth_l1_loss(pred_vec[exist_mask], target_vec[exist_mask], beta=0.001)

    diff_ang_rad = (x[..., ANGULAR_COLS] - x_recon[..., ANGULAR_COLS]) * (math.pi / 180.0)
    loss_angle_geo = (1.0 - torch.cos(diff_ang_rad[exist_mask])).mean()
    loss_angle = 30.0 * loss_angle_vec + 20.0 * loss_angle_geo

    # 3D Vector Alignment Loss
    p_pred = x_recon[..., T_COL_PITCH] * (math.pi / 180.0)
    y_pred = x_recon[..., T_COL_YAW] * (math.pi / 180.0)
    l_pred = x_recon[..., T_COL_LENGTH].clamp(min=0.0)
    p_gt = x[..., T_COL_PITCH] * (math.pi / 180.0)
    y_gt = x[..., T_COL_YAW] * (math.pi / 180.0)
    l_gt = x[..., T_COL_LENGTH].clamp(min=0.0)

    dir_pred = torch.stack([torch.sin(y_pred) * torch.cos(p_pred), torch.sin(p_pred), torch.cos(y_pred) * torch.cos(p_pred)], dim=-1)
    dir_gt = torch.stack([torch.sin(y_gt) * torch.cos(p_gt), torch.sin(p_gt), torch.cos(y_gt) * torch.cos(p_gt)], dim=-1)
    loss_dir_3d = (1.0 - (dir_pred * dir_gt).sum(dim=-1))[exist_mask].mean()
    loss_disp_3d = F.smooth_l1_loss((dir_pred * l_pred.unsqueeze(-1))[exist_mask], (dir_gt * l_gt.unsqueeze(-1))[exist_mask], beta=0.001)
    loss_fk_3d = 50.0 * loss_dir_3d + 100.0 * loss_disp_3d

    # 5. KL Divergence
    loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

    total_loss = (
        100.0 * loss_geom +
        100.0 * loss_dim +
        100.0 * loss_leaf_scale +
        10.0 * loss_cls +
        5.0 * loss_exist +
        loss_angle +
        loss_fk_3d +
        beta * loss_kl
    )

    return {
        "loss": total_loss,
        "loss_geom": loss_geom.detach(),
        "loss_cls": loss_cls.detach(),
        "loss_exist": loss_exist.detach(),
        "loss_angle": loss_angle.detach(),
        "loss_fk_3d": loss_fk_3d.detach(),
        "loss_kl": loss_kl.detach(),
    }
