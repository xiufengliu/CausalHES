"""
Deep clustering methods for household energy segmentation.

This module implements various deep learning-based clustering approaches
including Deep Embedded Clustering (DEC) and weather-fused variants.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from typing import Optional, Tuple, Union, Dict

from .base import BaseDeepClusteringMethod, BaseMultiModalClusteringMethod
from ..models.clustering_layers import ClusteringLayer
from ..utils.logging import LoggerMixin


class DeepEmbeddedClustering(BaseDeepClusteringMethod, LoggerMixin):
    """
    Deep Embedded Clustering (DEC) implementation.

    This class implements the DEC algorithm which jointly learns
    feature representations and cluster assignments using PyTorch.

    Reference: Xie et al. "Unsupervised Deep Embedding for Clustering Analysis" (2016)
    """

    def __init__(
        self,
        n_clusters: int,
        embedding_dim: int = 10,
        alpha: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        random_state: Optional[int] = None,
    ):
        """
        Initialize DEC model.

        Args:
            n_clusters: Number of clusters
            embedding_dim: Dimension of learned embeddings
            alpha: Degrees of freedom for Student's t-distribution
            device: Device to run the model on
            random_state: Random state for reproducibility
        """
        super().__init__(n_clusters, embedding_dim, random_state)
        self.alpha = alpha
        self.device = device
        self.encoder_ = None
        self.clustering_layer = None
        self.optimizer = None
        self.history_ = []

    def initialize_with_autoencoder(self, autoencoder):
        """
        Initialize DEC with a pre-trained autoencoder.

        Args:
            autoencoder: Pre-trained PyTorch autoencoder model
        """
        self.encoder_ = autoencoder.encoder.to(self.device)

        # Initialize clustering layer
        self.clustering_layer = ClusteringLayer(
            n_clusters=self.n_clusters,
            embedding_dim=self.embedding_dim,
            alpha=self.alpha,
        ).to(self.device)

        self.logger.info("DEC initialized with pre-trained autoencoder")

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        epochs: int = 30,
        batch_size: int = 32,
        tolerance: float = 0.001,
        update_interval: int = 1,
        verbose: int = 1,
    ) -> "DeepEmbeddedClustering":
        """
        Fit the DEC model.

        Args:
            X: Input data
            y: Ignored, present for API consistency
            epochs: Number of training epochs
            batch_size: Batch size for training
            tolerance: Convergence tolerance
            update_interval: Interval for updating target distribution
            verbose: Verbosity level

        Returns:
            self: Fitted DEC object
        """
        if self.encoder_ is None:
            raise ValueError("DEC must be initialized with an autoencoder first")

        self.logger.info(f"Training DEC for {epochs} epochs...")

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Generate initial embeddings
        self.encoder_.eval()
        with torch.no_grad():
            embeddings = self.encoder_(X_tensor).cpu().numpy()

        # Initialize cluster centroids with K-means
        kmeans = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init="auto"
        )
        kmeans.fit(embeddings)
        initial_centroids = torch.FloatTensor(kmeans.cluster_centers_).to(self.device)

        # Set initial cluster centroids
        self.clustering_layer.clusters.data = initial_centroids

        # Setup optimizer
        self.optimizer = optim.Adam(
            list(self.encoder_.parameters()) + list(self.clustering_layer.parameters()),
            lr=0.001,
        )

        # Training loop
        y_pred_last = np.zeros(len(X), dtype=int)

        # Create data loader
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            # Update target distribution P
            if epoch % update_interval == 0:
                self.encoder_.eval()
                self.clustering_layer.eval()

                with torch.no_grad():
                    embeddings = self.encoder_(X_tensor)
                    q = self.clustering_layer(embeddings)
                    q_np = q.cpu().numpy()
                    p_np = self._target_distribution(q_np)
                    p_tensor = torch.FloatTensor(p_np).to(self.device)

                # Check for convergence
                y_pred = np.argmax(q_np, axis=1)
                delta_label = np.sum(y_pred != y_pred_last).astype(float) / len(y_pred)
                y_pred_last = y_pred

                if verbose:
                    self.logger.info(
                        f"Epoch {epoch+1}/{epochs} - Labels changed: {delta_label:.4f}"
                    )

                if epoch > 0 and delta_label < tolerance:
                    self.logger.info(f"Converged at epoch {epoch+1}")
                    break

            # Training phase
            self.encoder_.train()
            self.clustering_layer.train()
            epoch_loss = []

            for batch_idx, (batch_X,) in enumerate(dataloader):
                batch_X = batch_X.to(self.device)

                # Get corresponding target distribution for this batch
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(p_tensor))
                batch_p = p_tensor[batch_start:batch_end]

                self.optimizer.zero_grad()

                # Forward pass
                embeddings = self.encoder_(batch_X)
                q = self.clustering_layer(embeddings)

                # KL divergence loss
                loss = F.kl_div(torch.log(q + 1e-8), batch_p, reduction="batchmean")

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss.append(loss.item())

            if verbose:
                avg_loss = np.mean(epoch_loss)
                self.logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

            self.history_.append(avg_loss)

        # Final predictions
        self.labels_ = self.predict(X)
        self.is_fitted_ = True

        self.logger.info("DEC training completed")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Args:
            X: Input data

        Returns:
            Cluster labels
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        X_tensor = torch.FloatTensor(X).to(self.device)

        self.encoder_.eval()
        self.clustering_layer.eval()

        with torch.no_grad():
            embeddings = self.encoder_(X_tensor)
            q = self.clustering_layer(embeddings)

        return torch.argmax(q, dim=1).cpu().numpy()

    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """
        Get learned embeddings for input data.

        Args:
            X: Input data

        Returns:
            Learned embeddings
        """
        if self.encoder_ is None:
            raise ValueError("Encoder not available")

        X_tensor = torch.FloatTensor(X).to(self.device)

        self.encoder_.eval()
        with torch.no_grad():
            embeddings = self.encoder_(X_tensor)

        return embeddings.cpu().numpy()

    def _target_distribution(self, q: np.ndarray) -> np.ndarray:
        """
        Compute target distribution P from current soft assignments Q.

        Args:
            q: Current soft assignments

        Returns:
            Target distribution P
        """
        weight = q**2 / np.sum(q, axis=0)
        return (weight.T / np.sum(weight, axis=1)).T


class WeatherFusedDEC(BaseMultiModalClusteringMethod, LoggerMixin):
    """
    Weather-fused Deep Embedded Clustering.

    This class extends DEC to incorporate weather information using
    multi-modal fusion and attention mechanisms with PyTorch.
    """

    def __init__(
        self,
        n_clusters: int,
        embedding_dim: int = 10,
        alpha: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        random_state: Optional[int] = None,
    ):
        """
        Initialize Weather-fused DEC model.

        Args:
            n_clusters: Number of clusters
            embedding_dim: Dimension of learned embeddings
            alpha: Degrees of freedom for Student's t-distribution
            device: Device to run the model on
            random_state: Random state for reproducibility
        """
        super().__init__(n_clusters, embedding_dim, random_state)
        self.alpha = alpha
        self.device = device
        self.encoder_ = None
        self.clustering_layer = None
        self.optimizer = None
        self.history_ = []

    def initialize_with_autoencoder(self, weather_autoencoder):
        """
        Initialize with a pre-trained weather-fused autoencoder.

        Args:
            weather_autoencoder: Pre-trained weather-fused PyTorch autoencoder
        """
        self.encoder_ = weather_autoencoder.encoder.to(self.device)

        # Initialize clustering layer
        self.clustering_layer = ClusteringLayer(
            n_clusters=self.n_clusters,
            embedding_dim=self.embedding_dim,
            alpha=self.alpha,
        ).to(self.device)

        self.logger.info("Weather-fused DEC initialized with pre-trained autoencoder")

    def fit_predict(
        self,
        X_primary: np.ndarray,
        X_secondary: np.ndarray,
        epochs: int = 30,
        batch_size: int = 32,
        tolerance: float = 0.001,
        update_interval: int = 1,
        verbose: int = 1,
    ) -> np.ndarray:
        """
        Fit the Weather-fused DEC model and return cluster labels.

        Args:
            X_primary: Primary input data (load shapes)
            X_secondary: Secondary input data (weather data)
            epochs: Number of training epochs
            batch_size: Batch size for training
            tolerance: Convergence tolerance
            update_interval: Interval for updating target distribution
            verbose: Verbosity level

        Returns:
            Cluster labels
        """
        if self.encoder_ is None:
            raise ValueError(
                "Weather-fused DEC must be initialized with an autoencoder first"
            )

        self.logger.info(f"Training Weather-fused DEC for {epochs} epochs...")

        # Convert to PyTorch tensors
        X_primary_tensor = torch.FloatTensor(X_primary).to(self.device)
        X_secondary_tensor = torch.FloatTensor(X_secondary).to(self.device)

        # Generate initial embeddings
        self.encoder_.eval()
        with torch.no_grad():
            embeddings = (
                self.encoder_([X_primary_tensor, X_secondary_tensor]).cpu().numpy()
            )

        # Initialize cluster centroids with K-means
        kmeans = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init="auto"
        )
        kmeans.fit(embeddings)
        initial_centroids = torch.FloatTensor(kmeans.cluster_centers_).to(self.device)

        # Set initial cluster centroids
        self.clustering_layer.clusters.data = initial_centroids

        # Setup optimizer
        self.optimizer = optim.Adam(
            list(self.encoder_.parameters()) + list(self.clustering_layer.parameters()),
            lr=0.001,
        )

        # Training loop (simplified for multi-modal)
        for epoch in range(epochs):
            self.encoder_.train()
            self.clustering_layer.train()

            # Forward pass
            embeddings = self.encoder_([X_primary_tensor, X_secondary_tensor])
            q = self.clustering_layer(embeddings)

            # Compute target distribution
            with torch.no_grad():
                q_np = q.cpu().numpy()
                p_np = self._target_distribution(q_np)
                p_tensor = torch.FloatTensor(p_np).to(self.device)

            # KL divergence loss
            loss = F.kl_div(torch.log(q + 1e-8), p_tensor, reduction="batchmean")

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if verbose and epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")

            self.history_.append(loss.item())

        # Final predictions
        self.encoder_.eval()
        self.clustering_layer.eval()

        with torch.no_grad():
            embeddings = self.encoder_([X_primary_tensor, X_secondary_tensor])
            q = self.clustering_layer(embeddings)
            labels = torch.argmax(q, dim=1).cpu().numpy()

        self.labels_ = labels
        self.is_fitted_ = True

        self.logger.info("Weather-fused DEC training completed")
        return labels

    def _target_distribution(self, q: np.ndarray) -> np.ndarray:
        """
        Compute target distribution P from current soft assignments Q.

        Args:
            q: Current soft assignments

        Returns:
            Target distribution P
        """
        weight = q**2 / np.sum(q, axis=0)
        return (weight.T / np.sum(weight, axis=1)).T
