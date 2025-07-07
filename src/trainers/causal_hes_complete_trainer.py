"""
Complete CausalHES Training Implementation.

This module implements the two-stage training strategy as specified in the paper:
1. Stage 1: CSSAE Pre-training (Algorithm 1)
2. Stage 2: Joint Training with Clustering (Algorithm 2)

Following paper Section 3.5: Optimization Strategy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path
import json

from ..models.causal_hes_model import CausalHESModel
from ..losses.cssae_loss import CSSAELoss
from ..utils.logging import get_logger
from sklearn.cluster import KMeans


class CausalHESTrainer:
    """
    Complete trainer for CausalHES framework implementing the paper's methodology.

    Implements Algorithms 1 and 2 from the paper:
    - Algorithm 1: CSSAE Pre-training for Causal Source Separation
    - Algorithm 2: Joint Training for Causal Clustering
    """

    def __init__(
        self,
        model: CausalHESModel,
        loss_fn: CSSAELoss,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        """
        Initialize CausalHES trainer.

        Args:
            model: CausalHES model instance
            loss_fn: CSSAE loss function
            device: Training device
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.logger = get_logger(self.__class__.__name__)

        # Optimizers
        self.main_optimizer = optim.Adam(
            list(model.parameters())
            + list(loss_fn.causal_loss.mine_network.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Separate optimizer for discriminator
        self.discriminator_optimizer = optim.Adam(
            loss_fn.causal_loss.discriminator.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Training history
        self.history = {
            "pretrain": {"losses": [], "epochs": []},
            "joint": {"losses": [], "epochs": []},
        }

    def stage1_pretrain(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        patience: int = 10,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Stage 1: CSSAE Pre-training for Causal Source Separation.

        Implements Algorithm 1 from the paper.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            patience: Early stopping patience
            save_path: Path to save best model

        Returns:
            Training results dictionary
        """
        self.logger.info("Starting Stage 1: CSSAE Pre-training")
        self.logger.info(f"Training for {epochs} epochs with patience {patience}")

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            # Training phase
            train_losses = self._train_epoch_stage1(train_loader)

            # Validation phase
            val_losses = {}
            if val_loader is not None:
                val_losses = self._validate_epoch_stage1(val_loader)

            # Logging
            self._log_epoch_results(epoch, train_losses, val_losses, stage="pretrain")

            # Early stopping
            current_val_loss = val_losses.get("total_loss", train_losses["total_loss"])
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()

                if save_path:
                    self._save_checkpoint(
                        save_path, epoch, train_losses, val_losses, stage="pretrain"
                    )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.logger.info("Stage 1 pre-training completed")
        return {
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch + 1,
            "history": self.history["pretrain"],
        }

    def stage2_joint_training(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 30,
        patience: int = 10,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Stage 2: Joint Training for Causal Clustering.

        Implements Algorithm 2 from the paper.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            patience: Early stopping patience
            save_path: Path to save best model

        Returns:
            Training results and cluster assignments
        """
        self.logger.info("Starting Stage 2: Joint Training with Clustering")

        # Initialize cluster centroids using K-means on base embeddings
        self._initialize_cluster_centroids(train_loader)

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            # Training phase
            train_losses = self._train_epoch_stage2(train_loader)

            # Validation phase
            val_losses = {}
            if val_loader is not None:
                val_losses = self._validate_epoch_stage2(val_loader)

            # Logging
            self._log_epoch_results(epoch, train_losses, val_losses, stage="joint")

            # Early stopping
            current_val_loss = val_losses.get("total_loss", train_losses["total_loss"])
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()

                if save_path:
                    self._save_checkpoint(
                        save_path, epoch, train_losses, val_losses, stage="joint"
                    )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # Get final cluster assignments
        final_assignments = self._get_cluster_assignments(train_loader)

        self.logger.info("Stage 2 joint training completed")
        return {
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch + 1,
            "cluster_assignments": final_assignments,
            "history": self.history["joint"],
        }

    def _train_epoch_stage1(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train one epoch for Stage 1 (pre-training)."""
        self.model.train()
        epoch_losses = {}

        for batch_idx, (load_data, weather_data) in enumerate(train_loader):
            load_data = load_data.to(self.device)
            weather_data = weather_data.to(self.device)

            # Forward pass
            model_outputs = self.model(load_data, weather_data)

            # Compute losses
            losses = self.loss_fn(load_data, model_outputs, stage="pretrain")

            # Update main model
            self.main_optimizer.zero_grad()
            losses["total_loss"].backward(retain_graph=True)
            self.main_optimizer.step()

            # Update discriminator
            disc_loss = self.loss_fn.update_discriminator(model_outputs)
            self.discriminator_optimizer.zero_grad()
            disc_loss.backward()
            self.discriminator_optimizer.step()

            # Accumulate losses
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value.item())

        # Average losses
        return {key: np.mean(values) for key, values in epoch_losses.items()}

    def _train_epoch_stage2(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train one epoch for Stage 2 (joint training)."""
        self.model.train()
        epoch_losses = {}

        for batch_idx, (load_data, weather_data) in enumerate(train_loader):
            load_data = load_data.to(self.device)
            weather_data = weather_data.to(self.device)

            # Forward pass
            model_outputs = self.model(load_data, weather_data)

            # Compute target distribution for clustering
            cluster_probs = model_outputs["cluster_probs"]
            target_dist = self._compute_target_distribution(cluster_probs)

            # Compute losses
            losses = self.loss_fn(
                load_data, model_outputs, stage="joint", target_distribution=target_dist
            )

            # Update main model
            self.main_optimizer.zero_grad()
            losses["total_loss"].backward(retain_graph=True)
            self.main_optimizer.step()

            # Update discriminator
            disc_loss = self.loss_fn.update_discriminator(model_outputs)
            self.discriminator_optimizer.zero_grad()
            disc_loss.backward()
            self.discriminator_optimizer.step()

            # Accumulate losses
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value.item())

        return {key: np.mean(values) for key, values in epoch_losses.items()}

    def _validate_epoch_stage1(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate one epoch for Stage 1."""
        self.model.eval()
        epoch_losses = {}

        with torch.no_grad():
            for load_data, weather_data in val_loader:
                load_data = load_data.to(self.device)
                weather_data = weather_data.to(self.device)

                model_outputs = self.model(load_data, weather_data)
                losses = self.loss_fn(load_data, model_outputs, stage="pretrain")

                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = []
                    epoch_losses[key].append(value.item())

        return {key: np.mean(values) for key, values in epoch_losses.items()}

    def _validate_epoch_stage2(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate one epoch for Stage 2."""
        self.model.eval()
        epoch_losses = {}

        with torch.no_grad():
            for load_data, weather_data in val_loader:
                load_data = load_data.to(self.device)
                weather_data = weather_data.to(self.device)

                model_outputs = self.model(load_data, weather_data)
                cluster_probs = model_outputs["cluster_probs"]
                target_dist = self._compute_target_distribution(cluster_probs)

                losses = self.loss_fn(
                    load_data,
                    model_outputs,
                    stage="joint",
                    target_distribution=target_dist,
                )

                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = []
                    epoch_losses[key].append(value.item())

        return {key: np.mean(values) for key, values in epoch_losses.items()}

    def _initialize_cluster_centroids(self, train_loader: DataLoader):
        """Initialize cluster centroids using K-means on base embeddings."""
        self.logger.info("Initializing cluster centroids with K-means")

        self.model.eval()
        base_embeddings = []

        with torch.no_grad():
            for load_data, weather_data in train_loader:
                load_data = load_data.to(self.device)
                weather_data = weather_data.to(self.device)

                model_outputs = self.model(load_data, weather_data)
                base_embeddings.append(model_outputs["base_embedding"].cpu().numpy())

        base_embeddings = np.vstack(base_embeddings)

        # Run K-means
        kmeans = KMeans(n_clusters=self.model.n_clusters, random_state=42)
        kmeans.fit(base_embeddings)

        # Set cluster centroids in the model
        centroids = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32, device=self.device
        )
        self.model.clustering_layer.clusters.data = centroids

        self.logger.info(f"Initialized {self.model.n_clusters} cluster centroids")

    def _compute_target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute target distribution for clustering.

        Following Equation 15 in the paper:
        p_ik = (q_ik² / f_k) / Σ_k' (q_ik'² / f_k')
        """
        # Compute frequency of each cluster
        f_k = torch.sum(q, dim=0)  # [n_clusters]

        # Compute target distribution
        numerator = q**2 / (f_k + 1e-8)  # Add epsilon to prevent division by zero
        denominator = torch.sum(numerator, dim=1, keepdim=True)

        p = numerator / (denominator + 1e-8)
        return p

    def _get_cluster_assignments(self, data_loader: DataLoader) -> np.ndarray:
        """Get final cluster assignments for all data."""
        self.model.eval()
        assignments = []

        with torch.no_grad():
            for load_data, weather_data in data_loader:
                load_data = load_data.to(self.device)
                weather_data = weather_data.to(self.device)

                model_outputs = self.model(load_data, weather_data)
                cluster_probs = model_outputs["cluster_probs"]

                # Hard assignments
                hard_assignments = torch.argmax(cluster_probs, dim=1)
                assignments.append(hard_assignments.cpu().numpy())

        return np.concatenate(assignments)

    def _log_epoch_results(
        self,
        epoch: int,
        train_losses: Dict[str, float],
        val_losses: Dict[str, float],
        stage: str,
    ):
        """Log epoch results."""
        self.logger.info(f"[{stage.upper()}] Epoch {epoch}")
        self.logger.info(f"  Train Loss: {train_losses['total_loss']:.6f}")

        if val_losses:
            self.logger.info(f"  Val Loss: {val_losses['total_loss']:.6f}")

        # Store in history
        self.history[stage]["epochs"].append(epoch)
        self.history[stage]["losses"].append({"train": train_losses, "val": val_losses})

    def _save_checkpoint(
        self,
        save_path: str,
        epoch: int,
        train_losses: Dict[str, float],
        val_losses: Dict[str, float],
        stage: str,
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "stage": stage,
            "model_state_dict": self.model.state_dict(),
            "loss_fn_state_dict": self.loss_fn.state_dict(),
            "main_optimizer_state_dict": self.main_optimizer.state_dict(),
            "discriminator_optimizer_state_dict": self.discriminator_optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "history": self.history,
        }

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint saved: {save_path}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.loss_fn.load_state_dict(checkpoint["loss_fn_state_dict"])
        self.main_optimizer.load_state_dict(checkpoint["main_optimizer_state_dict"])
        self.discriminator_optimizer.load_state_dict(
            checkpoint["discriminator_optimizer_state_dict"]
        )
        self.history = checkpoint["history"]

        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
