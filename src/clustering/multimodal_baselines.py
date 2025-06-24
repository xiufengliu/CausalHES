"""
Recent multi-modal clustering methods for household energy segmentation.

This module implements state-of-the-art multi-modal clustering approaches
including attention-based fusion, contrastive learning, and graph-based methods.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph
from typing import Optional, Tuple, Dict, Union, List
import math

from .base import BaseMultiModalClusteringMethod
from ..models.clustering_layers import ClusteringLayer
from ..utils.logging import LoggerMixin


class AttentionFusedDEC(BaseMultiModalClusteringMethod, LoggerMixin):
    """
    Attention-based Multi-modal Deep Embedded Clustering.
    
    This method uses attention mechanisms to learn adaptive fusion weights
    between load and weather modalities for clustering.
    
    Reference: Recent advances in attention-based multi-modal learning (2020+)
    """
    
    def __init__(self,
                 n_clusters: int,
                 embedding_dim: int = 10,
                 alpha: float = 1.0,
                 attention_type: str = 'cross',  # 'self', 'cross', 'co'
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 random_state: Optional[int] = None):
        """
        Initialize attention-fused DEC.
        
        Args:
            n_clusters: Number of clusters
            embedding_dim: Dimension of learned embeddings
            alpha: Degrees of freedom for Student's t-distribution
            attention_type: Type of attention mechanism
            device: Device to run the model on
            random_state: Random state for reproducibility
        """
        super().__init__(n_clusters, embedding_dim, random_state)
        self.alpha = alpha
        self.attention_type = attention_type
        self.device = device
        self.encoder_ = None
        self.clustering_layer = None
        self.optimizer = None
        self.history_ = []
        
    def fit(self, X_primary: np.ndarray, X_secondary: Optional[np.ndarray] = None,
            y: Optional[np.ndarray] = None,
            epochs: int = 100, batch_size: int = 32,
            tolerance: float = 0.001, update_interval: int = 1,
            learning_rate: float = 1e-3) -> 'AttentionFusedDEC':
        """
        Fit attention-fused DEC model.
        
        Args:
            X_primary: Primary input data (load shapes)
            X_secondary: Secondary input data (weather data)
            y: Ignored, present for API consistency
            epochs: Number of training epochs
            batch_size: Batch size for training
            tolerance: Convergence tolerance
            update_interval: Interval for updating target distribution
            learning_rate: Learning rate for optimization
            
        Returns:
            self: Fitted attention-fused DEC object
        """
        if X_secondary is None:
            raise ValueError("Attention-fused DEC requires secondary data (weather)")
            
        self.log_info(f"Training attention-fused DEC for {epochs} epochs...")
        
        # Reshape data if needed
        if X_primary.ndim == 3:
            X_primary = X_primary.squeeze(-1)
        if X_secondary.ndim == 3:
            X_secondary = X_secondary.squeeze(-1)
            
        load_dim = X_primary.shape[1]
        weather_dim = X_secondary.shape[1]
        
        # Create attention-based encoder
        self.encoder_ = AttentionFusionEncoder(
            load_dim=load_dim,
            weather_dim=weather_dim,
            embedding_dim=self.embedding_dim,
            attention_type=self.attention_type
        ).to(self.device)
        
        # Initialize clustering layer
        self.clustering_layer = ClusteringLayer(
            n_clusters=self.n_clusters,
            embedding_dim=self.embedding_dim,
            alpha=self.alpha
        ).to(self.device)
        
        # Convert to tensors
        X_load_tensor = torch.FloatTensor(X_primary).to(self.device)
        X_weather_tensor = torch.FloatTensor(X_secondary).to(self.device)
        
        # Generate initial embeddings
        self.encoder_.eval()
        with torch.no_grad():
            embeddings = self.encoder_(X_load_tensor, X_weather_tensor).cpu().numpy()
            
        # Initialize cluster centroids with K-means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        kmeans.fit(embeddings)
        initial_centroids = torch.FloatTensor(kmeans.cluster_centers_).to(self.device)
        
        # Set initial cluster centroids
        self.clustering_layer.clusters.data = initial_centroids
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            list(self.encoder_.parameters()) + list(self.clustering_layer.parameters()),
            lr=learning_rate
        )
        
        # Training loop
        y_pred_last = np.zeros(len(X_primary), dtype=int)
        
        # Create data loader
        dataset = TensorDataset(X_load_tensor, X_weather_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            # Update target distribution P
            if epoch % update_interval == 0:
                self.encoder_.eval()
                self.clustering_layer.eval()
                
                with torch.no_grad():
                    embeddings = self.encoder_(X_load_tensor, X_weather_tensor)
                    q = self.clustering_layer(embeddings)
                    q_np = q.cpu().numpy()
                    p_np = self._target_distribution(q_np)
                    p_tensor = torch.FloatTensor(p_np).to(self.device)
                    
                # Check for convergence
                y_pred = np.argmax(q_np, axis=1)
                delta_label = np.sum(y_pred != y_pred_last).astype(float) / len(y_pred)
                y_pred_last = y_pred
                
                if epoch > 0 and delta_label < tolerance:
                    self.log_info(f"Converged at epoch {epoch+1}")
                    break
                    
            # Training phase
            self.encoder_.train()
            self.clustering_layer.train()
            epoch_loss = []
            
            for batch_idx, (batch_load, batch_weather) in enumerate(dataloader):
                batch_load = batch_load.to(self.device)
                batch_weather = batch_weather.to(self.device)
                
                # Get corresponding target distribution for this batch
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(p_tensor))
                batch_p = p_tensor[batch_start:batch_end]
                
                self.optimizer.zero_grad()
                
                # Forward pass
                embeddings = self.encoder_(batch_load, batch_weather)
                q = self.clustering_layer(embeddings)
                
                # KL divergence loss
                loss = F.kl_div(torch.log(q + 1e-8), batch_p, reduction='batchmean')
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss.append(loss.item())
                
            if epoch % 20 == 0:
                avg_loss = np.mean(epoch_loss)
                self.log_info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
                
            self.history_.append(avg_loss)
            
        # Final predictions
        self.labels_ = self.predict(X_primary, X_secondary)
        self.is_fitted_ = True
        
        self.log_info("Attention-fused DEC training completed")
        return self
        
    def predict(self, X_primary: np.ndarray, X_secondary: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """Predict cluster labels for new data."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        if X_secondary is None:
            raise ValueError("Weather data required for prediction")
            
        if X_primary.ndim == 3:
            X_primary = X_primary.squeeze(-1)
        if X_secondary.ndim == 3:
            X_secondary = X_secondary.squeeze(-1)
            
        X_load_tensor = torch.FloatTensor(X_primary).to(self.device)
        X_weather_tensor = torch.FloatTensor(X_secondary).to(self.device)
        
        self.encoder_.eval()
        self.clustering_layer.eval()
        
        with torch.no_grad():
            embeddings = self.encoder_(X_load_tensor, X_weather_tensor)
            q = self.clustering_layer(embeddings)
            
        return torch.argmax(q, dim=1).cpu().numpy()
        
    def get_embeddings(self, X_primary: np.ndarray, X_secondary: np.ndarray) -> np.ndarray:
        """Get learned embeddings for input data."""
        if self.encoder_ is None:
            raise ValueError("Encoder not available")
            
        if X_primary.ndim == 3:
            X_primary = X_primary.squeeze(-1)
        if X_secondary.ndim == 3:
            X_secondary = X_secondary.squeeze(-1)
            
        X_load_tensor = torch.FloatTensor(X_primary).to(self.device)
        X_weather_tensor = torch.FloatTensor(X_secondary).to(self.device)
        
        self.encoder_.eval()
        with torch.no_grad():
            embeddings = self.encoder_(X_load_tensor, X_weather_tensor)
            
        return embeddings.cpu().numpy()
        
    def _target_distribution(self, q: np.ndarray) -> np.ndarray:
        """Compute target distribution P from current soft assignments Q."""
        weight = q ** 2 / np.sum(q, axis=0)
        return (weight.T / np.sum(weight, axis=1)).T


class ContrastiveMVClustering(BaseMultiModalClusteringMethod, LoggerMixin):
    """
    Contrastive Multi-view Clustering.
    
    This method uses contrastive learning to learn consistent representations
    across different modalities (views) and then performs clustering.
    
    Reference: Contrastive multi-view representation learning (2020+)
    """
    
    def __init__(self,
                 n_clusters: int,
                 embedding_dim: int = 10,
                 temperature: float = 0.07,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 random_state: Optional[int] = None):
        """
        Initialize contrastive multi-view clustering.
        
        Args:
            n_clusters: Number of clusters
            embedding_dim: Dimension of learned embeddings
            temperature: Temperature parameter for contrastive loss
            device: Device to run the model on
            random_state: Random state for reproducibility
        """
        super().__init__(n_clusters, embedding_dim, random_state)
        self.temperature = temperature
        self.device = device
        self.load_encoder_ = None
        self.weather_encoder_ = None
        self.kmeans_ = None
        self.history_ = []
        
    def fit(self, X_primary: np.ndarray, X_secondary: Optional[np.ndarray] = None,
            y: Optional[np.ndarray] = None,
            epochs: int = 100, batch_size: int = 32,
            learning_rate: float = 1e-3) -> 'ContrastiveMVClustering':
        """
        Fit contrastive multi-view clustering model.
        
        Args:
            X_primary: Primary input data (load shapes)
            X_secondary: Secondary input data (weather data)
            y: Ignored, present for API consistency
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            
        Returns:
            self: Fitted contrastive multi-view clustering object
        """
        if X_secondary is None:
            raise ValueError("Contrastive MV clustering requires secondary data (weather)")
            
        self.log_info(f"Training contrastive multi-view clustering for {epochs} epochs...")
        
        # Reshape data if needed
        if X_primary.ndim == 3:
            X_primary = X_primary.squeeze(-1)
        if X_secondary.ndim == 3:
            X_secondary = X_secondary.squeeze(-1)
            
        load_dim = X_primary.shape[1]
        weather_dim = X_secondary.shape[1]
        
        # Create view-specific encoders
        self.load_encoder_ = ViewEncoder(load_dim, self.embedding_dim).to(self.device)
        self.weather_encoder_ = ViewEncoder(weather_dim, self.embedding_dim).to(self.device)
        
        optimizer = optim.Adam(
            list(self.load_encoder_.parameters()) + list(self.weather_encoder_.parameters()),
            lr=learning_rate
        )
        
        # Convert to tensors
        X_load_tensor = torch.FloatTensor(X_primary).to(self.device)
        X_weather_tensor = torch.FloatTensor(X_secondary).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_load_tensor, X_weather_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            
            for batch_idx, (batch_load, batch_weather) in enumerate(dataloader):
                batch_load = batch_load.to(self.device)
                batch_weather = batch_weather.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass through encoders
                z_load = self.load_encoder_(batch_load)
                z_weather = self.weather_encoder_(batch_weather)
                
                # Contrastive loss
                loss = self._contrastive_loss(z_load, z_weather)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            if epoch % 20 == 0:
                avg_loss = epoch_loss / len(dataloader)
                self.log_info(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f}")
                
            self.history_.append(epoch_loss / len(dataloader))
            
        # Extract embeddings and perform clustering
        self.log_info("Extracting embeddings and performing clustering...")
        
        # Use concatenated view embeddings for clustering
        self.load_encoder_.eval()
        self.weather_encoder_.eval()
        
        with torch.no_grad():
            z_load = self.load_encoder_(X_load_tensor)
            z_weather = self.weather_encoder_(X_weather_tensor)
            # Concatenate embeddings from both views
            embeddings = torch.cat([z_load, z_weather], dim=1).cpu().numpy()
            
        # Apply K-means clustering
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        self.labels_ = self.kmeans_.fit_predict(embeddings)
        self.cluster_centers_ = self.kmeans_.cluster_centers_
        self.embeddings_ = embeddings
        self.is_fitted_ = True
        
        self.log_info("Contrastive multi-view clustering completed")
        return self
        
    def predict(self, X_primary: np.ndarray, X_secondary: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """Predict cluster labels for new data."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        if X_secondary is None:
            raise ValueError("Weather data required for prediction")
            
        embeddings = self.get_embeddings(X_primary, X_secondary)
        return self.kmeans_.predict(embeddings)
        
    def get_embeddings(self, X_primary: np.ndarray, X_secondary: np.ndarray) -> np.ndarray:
        """Get learned embeddings for input data."""
        if self.load_encoder_ is None or self.weather_encoder_ is None:
            raise ValueError("Encoders not available")
            
        if X_primary.ndim == 3:
            X_primary = X_primary.squeeze(-1)
        if X_secondary.ndim == 3:
            X_secondary = X_secondary.squeeze(-1)
            
        X_load_tensor = torch.FloatTensor(X_primary).to(self.device)
        X_weather_tensor = torch.FloatTensor(X_secondary).to(self.device)
        
        self.load_encoder_.eval()
        self.weather_encoder_.eval()
        
        with torch.no_grad():
            z_load = self.load_encoder_(X_load_tensor)
            z_weather = self.weather_encoder_(X_weather_tensor)
            embeddings = torch.cat([z_load, z_weather], dim=1)
            
        return embeddings.cpu().numpy()
        
    def _contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss between two views.
        
        Args:
            z1: Embeddings from view 1
            z2: Embeddings from view 2
            
        Returns:
            Contrastive loss
        """
        batch_size = z1.size(0)
        
        # Normalize embeddings
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z1_norm, z2_norm.T) / self.temperature
        
        # Create labels (positive pairs are on the diagonal)
        labels = torch.arange(batch_size, device=self.device)
        
        # Compute cross-entropy loss for both directions
        loss_12 = F.cross_entropy(sim_matrix, labels)
        loss_21 = F.cross_entropy(sim_matrix.T, labels)
        
        return (loss_12 + loss_21) / 2


class GraphMultiModalClustering(BaseMultiModalClusteringMethod, LoggerMixin):
    """
    Graph-based Multi-modal Clustering.
    
    This method constructs graphs from different modalities and uses
    graph-based clustering techniques.
    """
    
    def __init__(self,
                 n_clusters: int,
                 embedding_dim: int = 10,
                 k_neighbors: int = 10,
                 fusion_type: str = 'concatenate',  # 'concatenate', 'average', 'weighted'
                 random_state: Optional[int] = None):
        """
        Initialize graph-based multi-modal clustering.
        
        Args:
            n_clusters: Number of clusters
            embedding_dim: Dimension of learned embeddings
            k_neighbors: Number of neighbors for k-NN graph construction
            fusion_type: How to fuse graphs from different modalities
            random_state: Random state for reproducibility
        """
        super().__init__(n_clusters, embedding_dim, random_state)
        self.k_neighbors = k_neighbors
        self.fusion_type = fusion_type
        self.fused_graph_ = None
        self.embeddings_ = None
        
    def fit(self, X_primary: np.ndarray, X_secondary: Optional[np.ndarray] = None,
            y: Optional[np.ndarray] = None) -> 'GraphMultiModalClustering':
        """
        Fit graph-based multi-modal clustering.
        
        Args:
            X_primary: Primary input data (load shapes)
            X_secondary: Secondary input data (weather data)
            y: Ignored, present for API consistency
            
        Returns:
            self: Fitted graph multi-modal clustering object
        """
        if X_secondary is None:
            raise ValueError("Graph multi-modal clustering requires secondary data (weather)")
            
        self.log_info("Fitting graph-based multi-modal clustering...")
        
        # Reshape data if needed
        if X_primary.ndim == 3:
            X_primary = X_primary.squeeze(-1)
        if X_secondary.ndim == 3:
            X_secondary = X_secondary.squeeze(-1)
            
        # Construct k-NN graphs for each modality
        self.log_info("Constructing k-NN graphs...")
        
        # Load graph
        load_graph = kneighbors_graph(
            X_primary, 
            n_neighbors=self.k_neighbors, 
            mode='connectivity',
            include_self=False
        )
        
        # Weather graph
        weather_graph = kneighbors_graph(
            X_secondary,
            n_neighbors=self.k_neighbors,
            mode='connectivity', 
            include_self=False
        )
        
        # Fuse graphs
        if self.fusion_type == 'concatenate':
            # Simple addition of adjacency matrices
            self.fused_graph_ = load_graph + weather_graph
        elif self.fusion_type == 'average':
            # Average of adjacency matrices
            self.fused_graph_ = (load_graph + weather_graph) / 2
        elif self.fusion_type == 'weighted':
            # Weighted combination (can be learned or fixed)
            weight_load = 0.6  # Can be made learnable
            weight_weather = 0.4
            self.fused_graph_ = weight_load * load_graph + weight_weather * weather_graph
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
            
        # Perform spectral clustering on fused graph
        self.log_info("Performing spectral clustering...")
        
        # Compute normalized Laplacian
        laplacian = csgraph.laplacian(self.fused_graph_, normed=True)
        
        # Compute eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(laplacian.toarray())
        
        # Use smallest k eigenvectors as embeddings
        self.embeddings_ = eigenvecs[:, :self.embedding_dim]
        
        # Apply K-means to spectral embeddings
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        self.labels_ = kmeans.fit_predict(self.embeddings_)
        self.cluster_centers_ = kmeans.cluster_centers_
        self.is_fitted_ = True
        
        self.log_info("Graph-based multi-modal clustering completed")
        return self
        
    def predict(self, X_primary: np.ndarray, X_secondary: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Note: This is a simplified prediction that assigns new points to
        nearest cluster centers. For true out-of-sample prediction,
        a more sophisticated approach would be needed.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        # For simplicity, use the same approach as during training
        # In practice, you might want to use a different strategy for new data
        embeddings = self.get_embeddings(X_primary, X_secondary)
        
        # Find nearest cluster centers
        distances = np.linalg.norm(
            embeddings[:, np.newaxis] - self.cluster_centers_[np.newaxis, :], 
            axis=2
        )
        return np.argmin(distances, axis=1)
        
    def get_embeddings(self, X_primary: np.ndarray, X_secondary: np.ndarray) -> np.ndarray:
        """
        Get embeddings for new data.
        
        Note: This is a simplified approach. For true out-of-sample embeddings,
        you would need to extend the graph and recompute spectral embeddings.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting embeddings")
            
        # For now, return the training embeddings
        # This is just a placeholder - proper out-of-sample extension would be more complex
        return self.embeddings_


# Neural Network Components

class AttentionFusionEncoder(nn.Module):
    """Attention-based multi-modal encoder."""
    
    def __init__(self, load_dim: int, weather_dim: int, embedding_dim: int, 
                 attention_type: str = 'cross'):
        super().__init__()
        self.attention_type = attention_type
        
        # Individual encoders
        self.load_encoder = nn.Sequential(
            nn.Linear(load_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.weather_encoder = nn.Sequential(
            nn.Linear(weather_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Attention mechanism
        if attention_type == 'cross':
            self.cross_attention = CrossModalAttention(64, 64)
        elif attention_type == 'self':
            self.self_attention = SelfAttention(128)  # For concatenated features
        elif attention_type == 'co':
            self.co_attention = CoAttention(64, 64)
        
        # Final projection
        if attention_type in ['cross', 'co']:
            self.projector = nn.Linear(128, embedding_dim)  # 64 + 64
        else:  # self
            self.projector = nn.Linear(128, embedding_dim)
            
    def forward(self, load_data, weather_data):
        """Forward pass through attention fusion encoder."""
        # Individual encoding
        load_features = self.load_encoder(load_data)  # [batch, 64]
        weather_features = self.weather_encoder(weather_data)  # [batch, 64]
        
        # Attention-based fusion
        if self.attention_type == 'cross':
            attended_load = self.cross_attention(load_features, weather_features)
            attended_weather = self.cross_attention(weather_features, load_features)
            fused_features = torch.cat([attended_load, attended_weather], dim=1)
        elif self.attention_type == 'self':
            concat_features = torch.cat([load_features, weather_features], dim=1)
            fused_features = self.self_attention(concat_features)
        elif self.attention_type == 'co':
            attended_load, attended_weather = self.co_attention(load_features, weather_features)
            fused_features = torch.cat([attended_load, attended_weather], dim=1)
            
        # Final projection
        embeddings = self.projector(fused_features)
        return embeddings


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism."""
    
    def __init__(self, query_dim: int, key_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, key_dim)
        self.key_proj = nn.Linear(key_dim, key_dim)
        self.value_proj = nn.Linear(key_dim, key_dim)
        self.scale = math.sqrt(key_dim)
        
    def forward(self, query, key_value):
        """
        Args:
            query: Query features [batch, query_dim]
            key_value: Key and value features [batch, key_dim]
        """
        Q = self.query_proj(query)  # [batch, key_dim]
        K = self.key_proj(key_value)  # [batch, key_dim]
        V = self.value_proj(key_value)  # [batch, key_dim]
        
        # Compute attention weights
        attention_weights = torch.matmul(Q.unsqueeze(1), K.unsqueeze(2)) / self.scale
        attention_weights = F.softmax(attention_weights.squeeze(), dim=-1)
        
        # Apply attention
        attended = attention_weights.unsqueeze(1) * V
        return attended.squeeze(1)


class SelfAttention(nn.Module):
    """Self-attention mechanism."""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.scale = math.sqrt(feature_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input features [batch, feature_dim]
        """
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        
        # Compute self-attention
        attention_weights = torch.matmul(Q, K.T) / self.scale
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        attended = torch.matmul(attention_weights, V)
        return attended


class CoAttention(nn.Module):
    """Co-attention mechanism for two modalities."""
    
    def __init__(self, load_dim: int, weather_dim: int):
        super().__init__()
        self.load_attention = nn.Linear(load_dim + weather_dim, load_dim)
        self.weather_attention = nn.Linear(load_dim + weather_dim, weather_dim)
        
    def forward(self, load_features, weather_features):
        """
        Args:
            load_features: Load modality features [batch, load_dim]
            weather_features: Weather modality features [batch, weather_dim]
        """
        # Concatenate features
        concat_features = torch.cat([load_features, weather_features], dim=1)
        
        # Compute attention weights for each modality
        load_weights = torch.sigmoid(self.load_attention(concat_features))
        weather_weights = torch.sigmoid(self.weather_attention(concat_features))
        
        # Apply attention
        attended_load = load_weights * load_features
        attended_weather = weather_weights * weather_features
        
        return attended_load, attended_weather


class ViewEncoder(nn.Module):
    """View-specific encoder for contrastive learning."""
    
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, embedding_dim)
        )
        
    def forward(self, x):
        """Forward pass through view encoder."""
        return self.encoder(x)