"""
Complete CSSAE Loss Implementation.

This module implements the complete loss function for the Causal Source Separation Autoencoder
as specified in the paper, combining reconstruction and causal independence losses.

Following paper Equations 11-12:
- Stage 1: L_pretrain = L_rec + λ_causal * L_causal  
- Stage 2: L_total = L_rec + λ_causal * L_causal + λ_cluster * L_cluster
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from .composite_causal_loss import CompositeCausalLoss


class CSSAELoss(nn.Module):
    """
    Complete CSSAE loss function implementing the paper's methodology.

    Combines:
    1. Reconstruction loss (MSE)
    2. Composite causal independence loss (MINE + Adversarial + Distance Correlation)
    3. Clustering loss (KL divergence for DEC)
    """

    def __init__(
        self,
        base_dim: int,
        weather_dim: int,
        lambda_causal: float = 0.1,
        lambda_cluster: float = 0.5,
        alpha_mi: float = 1.0,
        alpha_adv: float = 0.5,
        alpha_dcor: float = 0.3,
    ):
        """
        Initialize CSSAE loss.

        Args:
            base_dim: Dimension of base load embeddings
            weather_dim: Dimension of weather embeddings
            lambda_causal: Weight for causal independence loss
            lambda_cluster: Weight for clustering loss
            alpha_mi: Weight for mutual information in causal loss
            alpha_adv: Weight for adversarial loss in causal loss
            alpha_dcor: Weight for distance correlation in causal loss
        """
        super().__init__()

        self.lambda_causal = lambda_causal
        self.lambda_cluster = lambda_cluster

        # Reconstruction loss
        self.reconstruction_loss = nn.MSELoss()

        # Composite causal independence loss
        self.causal_loss = CompositeCausalLoss(
            base_dim=base_dim,
            weather_dim=weather_dim,
            alpha_mi=alpha_mi,
            alpha_adv=alpha_adv,
            alpha_dcor=alpha_dcor,
        )

    def reconstruction_forward(
        self, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.

        Following Equation 3 in the paper:
        L_rec = E[||x^(l) - (ŝ_base + ŝ_weather)||²]
        """
        return self.reconstruction_loss(y_true, y_pred)

    def clustering_loss(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Compute clustering loss using KL divergence.

        Following Equation 16 in the paper:
        L_cluster = KL(P || Q) = Σᵢ Σₖ p_ik * log(p_ik / q_ik)

        Args:
            q: Soft assignment probabilities [batch_size, n_clusters]
            p: Target distribution [batch_size, n_clusters]
        """
        # Add small epsilon to prevent log(0)
        epsilon = 1e-8
        q = torch.clamp(q, epsilon, 1.0)
        p = torch.clamp(p, epsilon, 1.0)

        kl_div = torch.sum(p * torch.log(p / q), dim=1)
        return torch.mean(kl_div)

    def forward(
        self,
        y_true: torch.Tensor,
        model_outputs: Dict[str, torch.Tensor],
        stage: str = "pretrain",
        target_distribution: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute complete CSSAE loss.

        Args:
            y_true: Original load profiles
            model_outputs: Dictionary from CausalHESModel.forward()
            stage: Training stage ("pretrain" or "joint")
            target_distribution: Target distribution for clustering (required for joint stage)

        Returns:
            Dictionary containing all loss components
        """
        # Extract model outputs
        y_pred = model_outputs["load_reconstruction"]
        base_embedding = model_outputs["base_embedding"]
        weather_embedding = model_outputs.get("weather_embedding")
        cluster_probs = model_outputs.get("cluster_probs")

        # 1. Reconstruction loss (always computed)
        rec_loss = self.reconstruction_forward(y_true, y_pred)

        # 2. Causal independence loss (always computed)
        causal_losses = {}
        if weather_embedding is not None:
            causal_losses = self.causal_loss(base_embedding, weather_embedding)
            total_causal_loss = causal_losses["total_causal_loss"]
        else:
            # If no weather data, skip causal loss
            total_causal_loss = torch.tensor(0.0, device=y_true.device)
            causal_losses = {
                "total_causal_loss": total_causal_loss,
                "mi_loss": total_causal_loss,
                "adversarial_loss": total_causal_loss,
                "dcor_loss": total_causal_loss,
            }

        # 3. Clustering loss (only for joint training stage)
        cluster_loss = torch.tensor(0.0, device=y_true.device)
        if (
            stage == "joint"
            and cluster_probs is not None
            and target_distribution is not None
        ):
            cluster_loss = self.clustering_loss(cluster_probs, target_distribution)

        # 4. Total loss based on training stage
        if stage == "pretrain":
            # Stage 1: L_pretrain = L_rec + λ_causal * L_causal (Equation 11)
            total_loss = rec_loss + self.lambda_causal * total_causal_loss
        else:  # stage == "joint"
            # Stage 2: L_total = L_rec + λ_causal * L_causal + λ_cluster * L_cluster (Equation 12)
            total_loss = (
                rec_loss
                + self.lambda_causal * total_causal_loss
                + self.lambda_cluster * cluster_loss
            )

        # Return comprehensive loss dictionary
        loss_dict = {
            "total_loss": total_loss,
            "reconstruction_loss": rec_loss,
            "causal_loss": total_causal_loss,
            "clustering_loss": cluster_loss,
            "stage": stage,
        }

        # Add detailed causal loss components
        loss_dict.update({f"causal_{k}": v for k, v in causal_losses.items()})

        return loss_dict

    def update_discriminator(
        self, model_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Update discriminator for adversarial training.

        This should be called separately during training.
        """
        base_embedding = model_outputs["base_embedding"]
        weather_embedding = model_outputs.get("weather_embedding")

        if weather_embedding is not None:
            return self.causal_loss.update_discriminator(
                base_embedding, weather_embedding
            )
        else:
            return torch.tensor(0.0, device=base_embedding.device)
