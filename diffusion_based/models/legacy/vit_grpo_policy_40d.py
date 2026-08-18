"""
GRPO (Group Relative Policy Optimization) Policy Wrapper for ViTImageToOrganArray.

Wraps the ViT Image -> PlantOrganArray model into a stochastic policy that can:
1. Sample G candidate organ arrays for a given input image with exploration noise.
2. Compute log probabilities log \pi(A | I) for continuous params, organ types, and existence.
3. Compute KL divergence against a frozen reference policy \pi_ref.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional, Any

from diffusion_based.models.legacy.vit_image_to_organ_array_40d import ViTImageToOrganArray
from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.dataset.legacy.organ_array_dataset_40d import OrganArrayDataset


class ViTGRPOPolicy(nn.Module):
    """
    Stochastic Actor Policy wrapping ViTImageToOrganArray for GRPO reinforcement learning.
    """

    def __init__(
        self,
        base_model: ViTImageToOrganArray,
        init_log_std: float = -2.0,
        continuous_std_fixed: bool = False,
    ):
        super().__init__()
        self.model = base_model
        self.node_dim = base_model.node_dim
        self.max_nodes = base_model.max_nodes
        self.num_organ_types = base_model.num_organ_types

        # Continuous channels (38 channels, excluding categorical col 11 and existence col 39)
        self.categorical_col = 11
        self.existence_col = 39
        self.continuous_cols = [c for c in range(self.node_dim) if c not in (self.categorical_col, self.existence_col)]
        self.n_continuous = len(self.continuous_cols)

        if continuous_std_fixed:
            self.log_std = nn.Parameter(torch.full((self.n_continuous,), init_log_std), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.full((self.n_continuous,), init_log_std), requires_grad=True)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(images)

    def sample_group(
        self,
        images: torch.Tensor,
        group_size: int = 4,
        dataset: Optional[OrganArrayDataset] = None,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Sample G candidate plant organ arrays per input image.

        Args:
            images: (B, 3, H, W) normalized image batch.
            group_size: number of candidate rollouts G per prompt.
            dataset: OrganArrayDataset for denormalization.
            temperature: sampling temperature for categorical organ types and existence.

        Returns:
            Dict containing:
                - 'candidates': list of B lists of G PlantOrganArray objects
                - 'log_probs': (B, G) tensor of trajectory log-probabilities
                - 'sampled_tensors': (B, G, N, node_dim) normalized node tensors
        """
        B = images.shape[0]
        device = images.device

        # Expand images across group dimension: (B*G, 3, H, W)
        expanded_images = images.repeat_interleave(group_size, dim=0)

        outputs = self.model(expanded_images)
        pred_x0 = outputs["pred_x0"]               # (B*G, N, node_dim)
        organ_type_logits = outputs["organ_type_logits"]  # (B*G, N, num_organ_types)
        existence_logits = outputs["existence_logits"]    # (B*G, N)

        BG, N, _ = pred_x0.shape

        # 1. Sample Continuous Parameters: N(mu, sigma^2)
        mu_cont = pred_x0[:, :, self.continuous_cols]     # (B*G, N, n_cont)
        std_cont = torch.exp(self.log_std).clamp(1e-4, 1.0).view(1, 1, -1)
        eps = torch.randn_like(mu_cont)
        sampled_cont = mu_cont + eps * std_cont           # (B*G, N, n_cont)

        # Log prob of continuous Gaussian
        log_prob_cont = -0.5 * (((sampled_cont - mu_cont) / std_cont) ** 2 + 2 * self.log_std.view(1, 1, -1) + math.log(2 * math.pi))
        log_prob_cont = log_prob_cont.sum(dim=-1)         # (B*G, N)

        # 2. Sample Categorical Organ Types
        type_dist = torch.distributions.Categorical(logits=organ_type_logits / max(temperature, 1e-4))
        sampled_types = type_dist.sample()                # (B*G, N)
        log_prob_types = type_dist.log_prob(sampled_types)  # (B*G, N)

        # 3. Sample Existence via Bernoulli
        exist_probs = torch.sigmoid(existence_logits / max(temperature, 1e-4)).clamp(1e-6, 1.0 - 1e-6)
        exist_dist = torch.distributions.Bernoulli(probs=exist_probs)
        sampled_exist = exist_dist.sample()               # (B*G, N)
        log_prob_exist = exist_dist.log_prob(sampled_exist)  # (B*G, N)

        # Total per-node log prob (only active nodes contribute to continuous and type log-prob)
        per_node_log_prob = log_prob_exist + sampled_exist * (log_prob_cont + log_prob_types)
        trajectory_log_prob = per_node_log_prob.sum(dim=-1).view(B, group_size)  # (B, G)

        # Assemble full (B*G, N, node_dim) normalized tensor
        sampled_tensor = torch.zeros((BG, N, self.node_dim), device=device)
        sampled_tensor[:, :, self.continuous_cols] = sampled_cont
        sampled_tensor[:, :, self.categorical_col] = sampled_types.float()
        sampled_tensor[:, :, self.existence_col] = sampled_exist

        sampled_tensors_reshaped = sampled_tensor.view(B, group_size, N, self.node_dim)

        # Convert to PlantOrganArray instances
        candidates = []
        for b in range(B):
            group_candidates = []
            for g in range(group_size):
                raw = sampled_tensors_reshaped[b, g].clone()
                if dataset is not None:
                    denorm = dataset.denormalize(raw)
                    denorm[:, self.continuous_cols] = torch.clamp(denorm[:, self.continuous_cols], min=0.0)
                    denorm[:, self.categorical_col] = torch.round(denorm[:, self.categorical_col]).clamp(0, 7)
                    denorm[:, self.existence_col] = torch.clamp(denorm[:, self.existence_col], 0.0, 1.0)
                else:
                    denorm = raw
                group_candidates.append(PlantOrganArray(tensor=denorm.cpu()))
            candidates.append(group_candidates)

        return {
            "candidates": candidates,
            "log_probs": trajectory_log_prob,
            "sampled_tensors": sampled_tensors_reshaped,
            "mu_cont": mu_cont.view(B, group_size, N, self.n_continuous),
            "organ_type_logits": organ_type_logits.view(B, group_size, N, self.num_organ_types),
            "existence_logits": existence_logits.view(B, group_size, N),
        }

    def evaluate_log_probs(
        self,
        images: torch.Tensor,
        sampled_tensors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log-probabilities and entropy of previously sampled tensors under current policy.

        Args:
            images: (B, 3, H, W)
            sampled_tensors: (B, G, N, node_dim)

        Returns:
            log_probs: (B, G)
            entropy: (B, G)
        """
        B, G, N, D = sampled_tensors.shape
        expanded_images = images.repeat_interleave(G, dim=0)

        outputs = self.model(expanded_images)
        pred_x0 = outputs["pred_x0"]
        organ_type_logits = outputs["organ_type_logits"]
        existence_logits = outputs["existence_logits"]

        flat_sampled = sampled_tensors.view(B * G, N, D)
        sampled_cont = flat_sampled[:, :, self.continuous_cols]
        sampled_types = flat_sampled[:, :, self.categorical_col].long().clamp(0, self.num_organ_types - 1)
        sampled_exist = flat_sampled[:, :, self.existence_col]

        # Continuous log prob
        mu_cont = pred_x0[:, :, self.continuous_cols]
        std_cont = torch.exp(self.log_std).clamp(1e-4, 1.0).view(1, 1, -1)
        log_prob_cont = -0.5 * (((sampled_cont - mu_cont) / std_cont) ** 2 + 2 * self.log_std.view(1, 1, -1) + math.log(2 * math.pi))
        log_prob_cont = log_prob_cont.sum(dim=-1)

        # Categorical log prob
        type_dist = torch.distributions.Categorical(logits=organ_type_logits)
        log_prob_types = type_dist.log_prob(sampled_types)

        # Existence log prob
        exist_probs = torch.sigmoid(existence_logits).clamp(1e-6, 1.0 - 1e-6)
        exist_dist = torch.distributions.Bernoulli(probs=exist_probs)
        log_prob_exist = exist_dist.log_prob(sampled_exist)

        per_node_log_prob = log_prob_exist + sampled_exist * (log_prob_cont + log_prob_types)
        total_log_prob = per_node_log_prob.sum(dim=-1).view(B, G)

        entropy = (type_dist.entropy() + exist_dist.entropy()).sum(dim=-1).view(B, G)
        return total_log_prob, entropy
