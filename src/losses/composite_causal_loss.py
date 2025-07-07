"""
Composite Causal Loss Implementation for CausalHES.

This module implements the complete causal independence loss as specified in the paper:
L_causal = α_MI * L_MI + α_adv * L_adv + α_dcor * L_dcor

Following paper Section 3.3: Causal Independence Constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata


class MINENetwork(nn.Module):
    """
    Mutual Information Neural Estimator (MINE) network.

    Reference: Belghazi et al. "Mutual Information Neural Estimation" (2018)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.network(x)


class DiscriminatorNetwork(nn.Module):
    """
    Discriminator network for adversarial independence training.

    Predicts whether embeddings come from joint or marginal distributions.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


def distance_correlation(X, Y):
    """
    Compute distance correlation between two matrices.

    Reference: Székely et al. "Measuring and testing dependence by correlation of distances" (2007)

    Args:
        X: First set of embeddings [batch_size, dim1]
        Y: Second set of embeddings [batch_size, dim2]

    Returns:
        Distance correlation value
    """
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()

    n = X.shape[0]
    if n < 2:
        return torch.tensor(0.0)

    # Compute pairwise distances
    a = squareform(pdist(X))
    b = squareform(pdist(Y))

    # Double centering
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    # Compute distance covariance and variances
    dcov_AB = np.sqrt(np.mean(A * B))
    dvar_A = np.sqrt(np.mean(A * A))
    dvar_B = np.sqrt(np.mean(B * B))

    # Distance correlation
    if dvar_A > 0 and dvar_B > 0:
        dcor = dcov_AB / np.sqrt(dvar_A * dvar_B)
    else:
        dcor = 0.0

    return torch.tensor(dcor, dtype=torch.float32)


class CompositeCausalLoss(nn.Module):
    """
    Composite causal independence loss combining MINE, adversarial, and distance correlation.

    Implements Equation 6 from the paper:
    L_causal = α_MI * L_MI + α_adv * L_adv + α_dcor * L_dcor
    """

    def __init__(
        self,
        base_dim: int,
        weather_dim: int,
        alpha_mi: float = 1.0,
        alpha_adv: float = 0.5,
        alpha_dcor: float = 0.3,
        mine_hidden_dim: int = 64,
        disc_hidden_dim: int = 64,
    ):
        """
        Initialize composite causal loss.

        Args:
            base_dim: Dimension of base load embeddings
            weather_dim: Dimension of weather embeddings
            alpha_mi: Weight for mutual information loss
            alpha_adv: Weight for adversarial loss
            alpha_dcor: Weight for distance correlation loss
            mine_hidden_dim: Hidden dimension for MINE network
            disc_hidden_dim: Hidden dimension for discriminator
        """
        super().__init__()

        self.alpha_mi = alpha_mi
        self.alpha_adv = alpha_adv
        self.alpha_dcor = alpha_dcor

        # MINE network for mutual information estimation
        self.mine_network = MINENetwork(base_dim + weather_dim, mine_hidden_dim)

        # Discriminator for adversarial training
        self.discriminator = DiscriminatorNetwork(
            base_dim + weather_dim, disc_hidden_dim
        )

        # Loss functions
        self.bce_loss = nn.BCELoss()

    def mine_loss(
        self, base_embedding: torch.Tensor, weather_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MINE-based mutual information loss.

        Following Equation 7 in the paper.
        """
        batch_size = base_embedding.size(0)

        # Joint distribution
        joint = torch.cat([base_embedding, weather_embedding], dim=1)

        # Marginal distribution (shuffle weather embeddings)
        shuffled_indices = torch.randperm(batch_size, device=base_embedding.device)
        weather_shuffled = weather_embedding[shuffled_indices]
        marginal = torch.cat([base_embedding, weather_shuffled], dim=1)

        # MINE estimation: E[T(x,y)] - log(E[exp(T(x,y'))])
        joint_scores = self.mine_network(joint)
        marginal_scores = self.mine_network(marginal)

        mi_estimate = torch.mean(joint_scores) - torch.log(
            torch.mean(torch.exp(marginal_scores)) + 1e-8
        )

        return mi_estimate  # We want to minimize MI, so return positive value

    def adversarial_loss(
        self, base_embedding: torch.Tensor, weather_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adversarial independence loss.

        Following Equations 8-9 in the paper.
        """
        batch_size = base_embedding.size(0)

        # Joint distribution
        joint = torch.cat([base_embedding, weather_embedding], dim=1)

        # Marginal distribution (shuffle weather embeddings)
        shuffled_indices = torch.randperm(batch_size, device=base_embedding.device)
        weather_shuffled = weather_embedding[shuffled_indices]
        marginal = torch.cat([base_embedding, weather_shuffled], dim=1)

        # Discriminator predictions
        joint_pred = self.discriminator(joint)
        marginal_pred = self.discriminator(marginal)

        # Labels: 1 for joint, 0 for marginal
        joint_labels = torch.ones_like(joint_pred)
        marginal_labels = torch.zeros_like(marginal_pred)

        # Generator loss (we want to fool the discriminator)
        # Make joint distribution look like marginal (minimize discriminator confidence)
        generator_loss = self.bce_loss(joint_pred, marginal_labels) + self.bce_loss(
            marginal_pred, joint_labels
        )

        return generator_loss

    def distance_correlation_loss(
        self, base_embedding: torch.Tensor, weather_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distance correlation loss.

        Following Equation 10 in the paper.
        """
        dcor = distance_correlation(base_embedding, weather_embedding)
        return dcor.to(base_embedding.device)

    def forward(
        self, base_embedding: torch.Tensor, weather_embedding: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute composite causal independence loss.

        Args:
            base_embedding: Base load embeddings [batch_size, base_dim]
            weather_embedding: Weather embeddings [batch_size, weather_dim]

        Returns:
            Dictionary containing individual and total losses
        """
        # Individual loss components
        mi_loss = self.mine_loss(base_embedding, weather_embedding)
        adv_loss = self.adversarial_loss(base_embedding, weather_embedding)
        dcor_loss = self.distance_correlation_loss(base_embedding, weather_embedding)

        # Composite loss (Equation 6)
        total_loss = (
            self.alpha_mi * mi_loss
            + self.alpha_adv * adv_loss
            + self.alpha_dcor * dcor_loss
        )

        return {
            "total_causal_loss": total_loss,
            "mi_loss": mi_loss,
            "adversarial_loss": adv_loss,
            "dcor_loss": dcor_loss,
        }

    def update_discriminator(
        self, base_embedding: torch.Tensor, weather_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Update discriminator for adversarial training.

        This should be called separately to train the discriminator.
        """
        batch_size = base_embedding.size(0)

        # Joint distribution
        joint = torch.cat([base_embedding.detach(), weather_embedding.detach()], dim=1)

        # Marginal distribution
        shuffled_indices = torch.randperm(batch_size, device=base_embedding.device)
        weather_shuffled = weather_embedding[shuffled_indices].detach()
        marginal = torch.cat([base_embedding.detach(), weather_shuffled], dim=1)

        # Discriminator predictions
        joint_pred = self.discriminator(joint)
        marginal_pred = self.discriminator(marginal)

        # Labels: 1 for joint, 0 for marginal
        joint_labels = torch.ones_like(joint_pred)
        marginal_labels = torch.zeros_like(marginal_pred)

        # Discriminator loss (standard binary classification)
        disc_loss = self.bce_loss(joint_pred, joint_labels) + self.bce_loss(
            marginal_pred, marginal_labels
        )

        return disc_loss
