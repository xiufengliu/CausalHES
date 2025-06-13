"""
Custom loss functions for CausalHES framework.

This module implements various loss functions for enforcing causal independence
and source separation quality in the CSSAE model.

Key Loss Functions:
1. Reconstruction Loss: Standard MSE for autoencoder training
2. Mutual Information Loss: Minimizes I(Z_base; Z_weather) for independence
3. Adversarial Loss: Uses discriminator to enforce independence
4. Distance Correlation Loss: Minimizes distance correlation between embeddings
5. Composite Causal Loss: Combines multiple independence constraints
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Callable
import logging

from .utils.logging import get_logger


class ReconstructionLoss(nn.Module):
    """
    Standard reconstruction loss for autoencoder training.

    Computes MSE between original and reconstructed load profiles.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_true, y_pred):
        """
        Compute reconstruction loss.

        Args:
            y_true: Original load profiles
            y_pred: Reconstructed load profiles

        Returns:
            Reconstruction loss
        """
        return self.mse(y_true, y_pred)


class MutualInformationLoss(nn.Module):
    """
    Mutual Information loss for enforcing statistical independence.

    Uses MINE (Mutual Information Neural Estimation) to estimate and minimize
    the mutual information between base load and weather embeddings.

    Reference: Belghazi et al. "Mutual Information Neural Estimation" (2018)
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.logger = get_logger(self.__class__.__name__)

        # Build MINE network for MI estimation
        self._build_mine_network()

    def _build_mine_network(self):
        """Build the MINE network for MI estimation."""
        # MINE network takes concatenated embeddings and outputs a scalar
        self.mine_network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, base_embedding, weather_embedding):
        """
        Compute mutual information loss.

        Args:
            base_embedding: Base load embeddings [batch_size, base_dim]
            weather_embedding: Weather embeddings [batch_size, weather_dim]

        Returns:
            Negative mutual information (to minimize MI)
        """
        batch_size = base_embedding.shape[0]

        # Joint distribution: concatenate embeddings
        joint = torch.cat([base_embedding, weather_embedding], dim=-1)

        # Marginal distribution: shuffle weather embeddings
        shuffled_indices = torch.randperm(batch_size)
        weather_shuffled = weather_embedding[shuffled_indices]
        marginal = torch.cat([base_embedding, weather_shuffled], dim=-1)

        # MINE estimation: E[T(x,y)] - log(E[exp(T(x,y'))])
        joint_scores = self.mine_network(joint)
        marginal_scores = self.mine_network(marginal)

        # Compute MI estimate
        mi_estimate = torch.mean(joint_scores) - torch.log(
            torch.mean(torch.exp(marginal_scores)) + 1e-8
        )

        # Return negative MI to minimize it
        return -mi_estimate


class AdversarialIndependenceLoss(nn.Module):
    """
    Adversarial loss for enforcing independence between embeddings.

    Uses a discriminator network that tries to predict weather embeddings
    from base embeddings. The encoder is trained to fool this discriminator,
    similar to Gradient Reversal Layer (GRL) approach.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.logger = get_logger(self.__class__.__name__)

        # Build discriminator network
        self._build_discriminator()

        # Binary crossentropy for adversarial training
        self.bce = nn.BCELoss()

    def _build_discriminator(self):
        """Build discriminator network."""
        self.discriminator = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, base_embedding, weather_embedding):
        """
        Compute adversarial independence loss.

        The discriminator tries to distinguish between:
        - Real pairs: (base_embedding, weather_embedding) → label 1
        - Fake pairs: (base_embedding, shuffled_weather) → label 0

        The encoder is trained to make base_embedding uninformative about weather.

        Args:
            base_embedding: Base load embeddings
            weather_embedding: Weather embeddings

        Returns:
            Adversarial loss for the encoder (to fool discriminator)
        """
        batch_size = base_embedding.shape[0]

        # Create real and fake pairs
        real_pairs = torch.cat([base_embedding, weather_embedding], dim=-1)

        # Shuffle weather embeddings for fake pairs
        shuffled_indices = torch.randperm(batch_size)
        weather_shuffled = weather_embedding[shuffled_indices]
        fake_pairs = torch.cat([base_embedding, weather_shuffled], dim=-1)

        # Discriminator predictions
        real_pred = self.discriminator(real_pairs)
        fake_pred = self.discriminator(fake_pairs)

        # Labels
        real_labels = torch.ones_like(real_pred)
        fake_labels = torch.zeros_like(fake_pred)

        # Discriminator loss (not used for encoder training)
        disc_loss_real = self.bce(real_pred, real_labels)
        disc_loss_fake = self.bce(fake_pred, fake_labels)
        disc_loss = disc_loss_real + disc_loss_fake

        # Encoder loss: fool the discriminator
        # We want the discriminator to predict 0 (fake) for real pairs
        # This means base_embedding doesn't contain weather information
        encoder_loss = self.bce(real_pred, fake_labels)

        return encoder_loss


class DistanceCorrelationLoss(nn.Module):
    """
    Distance correlation loss for measuring dependence between embeddings.

    Distance correlation is a measure of dependence between random vectors
    that is zero if and only if the vectors are independent.

    Reference: Székely et al. "Measuring and testing dependence by correlation of distances" (2007)
    """

    def __init__(self):
        super().__init__()

    def _distance_matrix(self, x):
        """Compute pairwise distance matrix."""
        # x shape: [batch_size, dim]
        x_expanded = x.unsqueeze(1)  # [batch_size, 1, dim]
        y_expanded = x.unsqueeze(0)  # [1, batch_size, dim]

        # Compute pairwise distances
        distances = torch.norm(x_expanded - y_expanded, dim=-1)  # [batch_size, batch_size]
        return distances

    def _centered_distance_matrix(self, distances):
        """Compute centered distance matrix."""
        n = distances.shape[0]

        # Row means
        row_means = torch.mean(distances, dim=1, keepdim=True)

        # Column means
        col_means = torch.mean(distances, dim=0, keepdim=True)

        # Grand mean
        grand_mean = torch.mean(distances)

        # Centered matrix
        centered = distances - row_means - col_means + grand_mean

        return centered

    def forward(self, base_embedding, weather_embedding):
        """
        Compute distance correlation between embeddings.

        Args:
            base_embedding: Base load embeddings
            weather_embedding: Weather embeddings

        Returns:
            Distance correlation (to be minimized)
        """
        # Compute distance matrices
        dist_base = self._distance_matrix(base_embedding)
        dist_weather = self._distance_matrix(weather_embedding)

        # Center the distance matrices
        centered_base = self._centered_distance_matrix(dist_base)
        centered_weather = self._centered_distance_matrix(dist_weather)

        # Compute distance covariance
        dcov_xy = torch.sqrt(torch.mean(centered_base * centered_weather))
        dcov_xx = torch.sqrt(torch.mean(centered_base * centered_base))
        dcov_yy = torch.sqrt(torch.mean(centered_weather * centered_weather))

        # Distance correlation
        dcor = dcov_xy / (torch.sqrt(dcov_xx * dcov_yy) + 1e-8)

        return dcor


class CompositeCausalLoss(nn.Module):
    """
    Composite loss that combines multiple causal independence constraints.

    This loss combines:
    1. Mutual Information Loss (MINE-based)
    2. Adversarial Independence Loss
    3. Distance Correlation Loss

    Each component enforces independence from a different perspective.
    """

    def __init__(self,
                 base_dim: int,
                 weather_dim: int,
                 mi_weight: float = 1.0,
                 adversarial_weight: float = 0.5,
                 dcor_weight: float = 0.3):
        super().__init__()

        self.mi_weight = mi_weight
        self.adversarial_weight = adversarial_weight
        self.dcor_weight = dcor_weight

        # Initialize component losses
        self.mi_loss = MutualInformationLoss(input_dim=base_dim + weather_dim)
        self.adversarial_loss = AdversarialIndependenceLoss(input_dim=base_dim + weather_dim)
        self.dcor_loss = DistanceCorrelationLoss()

        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(f"Composite causal loss initialized with weights: "
                        f"MI={mi_weight}, Adversarial={adversarial_weight}, dCor={dcor_weight}")

    def forward(self, base_embedding, weather_embedding):
        """
        Compute composite causal loss.

        Args:
            base_embedding: Base load embeddings
            weather_embedding: Weather embeddings

        Returns:
            Weighted combination of independence losses
        """
        # Compute individual losses
        mi_loss = self.mi_loss(base_embedding, weather_embedding)
        adversarial_loss = self.adversarial_loss(base_embedding, weather_embedding)
        dcor_loss = self.dcor_loss(base_embedding, weather_embedding)

        # Weighted combination
        total_loss = (
            self.mi_weight * mi_loss +
            self.adversarial_weight * adversarial_loss +
            self.dcor_weight * dcor_loss
        )

        return total_loss


class CSSAELoss(nn.Module):
    """
    Complete loss function for CSSAE training.
    """

    def __init__(self,
                 base_dim: int,
                 weather_dim: int,
                 reconstruction_weight: float = 1.0,
                 causal_weight: float = 0.1,
                 mi_weight: float = 1.0,
                 adversarial_weight: float = 0.5,
                 dcor_weight: float = 0.3):
        super().__init__()

        self.reconstruction_weight = reconstruction_weight
        self.causal_weight = causal_weight

        self.reconstruction_loss = ReconstructionLoss()
        self.causal_loss = CompositeCausalLoss(
            base_dim=base_dim,
            weather_dim=weather_dim,
            mi_weight=mi_weight,
            adversarial_weight=adversarial_weight,
            dcor_weight=dcor_weight
        )

    def forward(self, y_true, outputs):
        """
        Compute complete CSSAE loss.

        Args:
            y_true: True load profiles
            outputs: Dictionary with model outputs

        Returns:
            Total loss
        """
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(
            y_true,
            outputs['total_reconstruction']
        )

        # Causal independence loss
        causal_loss_value = self.causal_loss(
            outputs['base_embedding'],
            outputs['weather_embedding']
        )

        # Total loss
        total_loss = (
            self.reconstruction_weight * recon_loss +
            self.causal_weight * causal_loss_value
        )

        return total_loss
