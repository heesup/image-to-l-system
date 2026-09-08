"""
Organ-level Variational Autoencoder (OrganLatentVAE).

Compresses heterogeneous 26D/14D plant organ parameters into a compact,
spherical standard Gaussian latent space z in R^16 ~ N(0, I).

Solves:
1. Dimensional heterogeneity (discrete class vs continuous positions vs rotations vs scales).
2. Gradient imbalance between large scale dimensions (length) and small scale dimensions (width).
3. Impossible geometric configurations (guarantees positive leaf scales and valid manifold).
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_based.dataset.part_array_dataset import (
    NUM_ORGAN_TYPES,
    FM_OT_END,
    FM_BASE_START,
    FM_BASE_END,
    FM_ROT_START,
    FM_ROT_END,
    FM_SCALE_START,
    FM_SCALE_END,
    FM_CURV,
    FM_NODE_DIM,
    BASE_SCALE,
    SCALE_SCALE,
    CURV_SCALE,
    decode_fm,
)


class OrganLatentVAE(nn.Module):
    """Compact 1D Organ-Level Variational Autoencoder."""

    def __init__(
        self,
        in_dim: int = 26,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        num_classes: int = NUM_ORGAN_TYPES,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Encoder: 26D -> hidden -> (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: latent_dim -> hidden -> specialized output heads
        self.decoder_backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Specialized prediction heads
        self.head_cls = nn.Linear(hidden_dim, num_classes)
        self.head_base = nn.Linear(hidden_dim, 3)
        self.head_rot = nn.Linear(hidden_dim, 6)
        self.head_scale = nn.Linear(hidden_dim, 3)
        self.head_curv = nn.Linear(hidden_dim, 1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes (N, 26) organ vectors to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-10.0, 10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decodes latent vector z in R^latent_dim into reconstructed organ parameters."""
        h = self.decoder_backbone(z)
        pred_cls_logits = self.head_cls(h)
        pred_base = self.head_base(h)
        pred_rot = self.head_rot(h)
        # Positive scale constraint using softplus
        pred_scale = F.softplus(self.head_scale(h)) + 1e-4
        pred_curv = self.head_curv(h)

        # Assemble back into 26D layout
        pred_probs = F.softmax(pred_cls_logits, dim=-1)
        pred_26d = torch.cat([pred_probs, pred_base, pred_rot, pred_scale, pred_curv], dim=-1)

        return {
            "cls_logits": pred_cls_logits,
            "base": pred_base,
            "rot": pred_rot,
            "scale": pred_scale,
            "curv": pred_curv,
            "recon_26d": pred_26d,
        }

    def decode_to_part_tensor(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decodes latent tensor (*, 16) into canonical 14D part tensor and class probabilities.

        Returns:
            part_14d: (*, 14) tensor [cls_float, base/BASE_SCALE, rot6d, scale/SCALE_SCALE, curv/CURV_SCALE]
            probs: (*, num_classes) softmax class probabilities
        """
        decoded = self.decode(z)
        probs = F.softmax(decoded["cls_logits"], dim=-1)
        pred_cls = decoded["cls_logits"].argmax(dim=-1, keepdim=True).float()

        norm_base = decoded["base"] / BASE_SCALE
        norm_rot = decoded["rot"]
        norm_scale = decoded["scale"] / SCALE_SCALE
        norm_curv = decoded["curv"] / CURV_SCALE

        part_14d = torch.cat([pred_cls, norm_base, norm_rot, norm_scale, norm_curv], dim=-1)
        return part_14d, probs

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        out["mu"] = mu
        out["logvar"] = logvar
        out["z"] = z
        return out

    def compute_loss(
        self,
        pred: Dict[str, torch.Tensor],
        target: torch.Tensor,
        beta_kl: float = 1e-3,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes composite VAE loss:
        Reconstruction Loss = Cls CE + Base SmoothL1 + Rot SmoothL1 + Scale SmoothL1 + Curv SmoothL1
        Regularization = beta * KL(N(mu, sigma^2) || N(0, I))
        """
        target_cls = target[:, :FM_OT_END].argmax(dim=-1)
        target_base = target[:, FM_BASE_START:FM_BASE_END]
        target_rot = target[:, FM_ROT_START:FM_ROT_END]
        target_scale = target[:, FM_SCALE_START:FM_SCALE_END]
        target_curv = target[:, FM_CURV:FM_NODE_DIM]

        loss_cls = F.cross_entropy(pred["cls_logits"], target_cls)
        # Scale relative log-space loss: gives equal gradient weight to 2mm radius and 35cm length
        log_pred_s = torch.log(pred["scale"] + 1e-3)
        log_target_s = torch.log(target_scale + 1e-3)
        loss_scale = F.smooth_l1_loss(log_pred_s, log_target_s) + 0.1 * F.smooth_l1_loss(pred["scale"], target_scale)

        # SO(3) Matrix Frobenius Rotation Loss for sub-degree orientation accuracy
        x_p = F.normalize(pred["rot"][:, 0:3], dim=-1)
        z_p = F.normalize(torch.cross(x_p, pred["rot"][:, 3:6], dim=-1), dim=-1)
        y_p = torch.cross(z_p, x_p, dim=-1)
        R_pred = torch.stack([x_p, y_p, z_p], dim=-1)

        x_t = F.normalize(target_rot[:, 0:3], dim=-1)
        z_t = F.normalize(torch.cross(x_t, target_rot[:, 3:6], dim=-1), dim=-1)
        y_t = torch.cross(z_t, x_t, dim=-1)
        R_target = torch.stack([x_t, y_t, z_t], dim=-1)

        loss_rot = F.mse_loss(R_pred, R_target) + 0.5 * F.smooth_l1_loss(pred["rot"], target_rot)
        loss_base = F.smooth_l1_loss(pred["base"], target_base)
        loss_curv = F.smooth_l1_loss(pred["curv"], target_curv)

        # KL Divergence
        mu, logvar = pred["mu"], pred["logvar"]
        loss_kl = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1))

        recon_loss = loss_cls + 3.0 * loss_base + 2.0 * loss_rot + 3.0 * loss_scale + 0.5 * loss_curv
        total_loss = recon_loss + beta_kl * loss_kl

        # Metrics
        pred_cls = pred["cls_logits"].argmax(dim=-1)
        acc = (pred_cls == target_cls).float().mean()

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "loss_cls": loss_cls,
            "loss_base": loss_base,
            "loss_rot": loss_rot,
            "loss_scale": loss_scale,
            "loss_curv": loss_curv,
            "loss_kl": loss_kl,
            "cls_acc": acc,
        }
