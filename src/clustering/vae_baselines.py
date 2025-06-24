"""
VAE-based disentanglement clustering methods for household energy segmentation.

This module implements various VAE-based approaches including β-VAE, FactorVAE,
and β-TCVAE for learning disentangled representations followed by clustering.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from typing import Optional, Tuple, Dict, Union
import math

from .base import BaseDeepClusteringMethod, BaseMultiModalClusteringMethod
from ..utils.logging import LoggerMixin


class BetaVAEClustering(BaseDeepClusteringMethod, LoggerMixin):
    """
    β-VAE based clustering for disentangled representation learning.
    
    This method uses β-VAE to learn disentangled representations and then
    applies K-means clustering to the learned latent space.
    
    Reference: Higgins et al. "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" (2017)
    """
    
    def __init__(self,
                 n_clusters: int,
                 embedding_dim: int = 10,
                 beta: float = 4.0,
                 hidden_dims: Tuple[int, ...] = (128, 64, 32),
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 random_state: Optional[int] = None):
        """
        Initialize β-VAE clustering.
        
        Args:
            n_clusters: Number of clusters
            embedding_dim: Dimension of learned embeddings (latent dimension)
            beta: β parameter controlling disentanglement vs reconstruction trade-off
            hidden_dims: Hidden layer dimensions for encoder/decoder
            device: Device to run the model on
            random_state: Random state for reproducibility
        """
        super().__init__(n_clusters, embedding_dim, random_state)
        self.beta = beta
        self.hidden_dims = hidden_dims
        self.device = device
        self.vae_model_ = None
        self.kmeans_ = None
        self.history_ = []
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            epochs: int = 100, batch_size: int = 32,
            learning_rate: float = 1e-3) -> 'BetaVAEClustering':
        """
        Fit β-VAE and perform clustering.
        
        Args:
            X: Input data of shape (n_samples, n_timesteps) or (n_samples, n_timesteps, 1)
            y: Ignored, present for API consistency
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            
        Returns:
            self: Fitted β-VAE clustering object
        """
        self.log_info(f"Training β-VAE clustering for {epochs} epochs...")
        
        # Reshape data if needed
        if X.ndim == 3:
            X = X.squeeze(-1)
        
        input_dim = X.shape[1]
        
        # Create β-VAE model
        self.vae_model_ = BetaVAE(
            input_dim=input_dim,
            latent_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            beta=self.beta
        ).to(self.device)
        
        optimizer = optim.Adam(self.vae_model_.parameters(), lr=learning_rate)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.vae_model_.train()
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_recon_loss = 0
            epoch_kld_loss = 0
            
            for batch_idx, (batch_x,) in enumerate(dataloader):
                batch_x = batch_x.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                recon_x, mu, logvar = self.vae_model_(batch_x)
                
                # Compute losses
                recon_loss = F.mse_loss(recon_x, batch_x, reduction='sum')
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                
                # β-VAE loss
                total_loss = recon_loss + self.beta * kld_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kld_loss += kld_loss.item()
                
            # Log progress
            if epoch % 20 == 0:
                avg_loss = epoch_loss / len(X)
                avg_recon = epoch_recon_loss / len(X)
                avg_kld = epoch_kld_loss / len(X)
                self.log_info(f"Epoch {epoch}/{epochs} - Total: {avg_loss:.4f}, "
                            f"Recon: {avg_recon:.4f}, KLD: {avg_kld:.4f}")
                
            self.history_.append({
                'total_loss': epoch_loss / len(X),
                'recon_loss': epoch_recon_loss / len(X),
                'kld_loss': epoch_kld_loss / len(X)
            })
        
        # Extract embeddings and perform clustering
        self.log_info("Extracting embeddings and performing clustering...")
        embeddings = self.get_embeddings(X)
        
        # Apply K-means clustering
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        self.labels_ = self.kmeans_.fit_predict(embeddings)
        self.cluster_centers_ = self.kmeans_.cluster_centers_
        self.embeddings_ = embeddings
        self.is_fitted_ = True
        
        self.log_info("β-VAE clustering completed")
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
            
        embeddings = self.get_embeddings(X)
        return self.kmeans_.predict(embeddings)
        
    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """
        Get learned embeddings for input data.
        
        Args:
            X: Input data
            
        Returns:
            Learned embeddings (mean of latent distribution)
        """
        if self.vae_model_ is None:
            raise ValueError("VAE model not available")
            
        if X.ndim == 3:
            X = X.squeeze(-1)
            
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.vae_model_.eval()
        with torch.no_grad():
            mu, _ = self.vae_model_.encode(X_tensor)
            
        return mu.cpu().numpy()


class FactorVAEClustering(BaseDeepClusteringMethod, LoggerMixin):
    """
    FactorVAE based clustering for disentangled representation learning.
    
    This method uses FactorVAE to learn disentangled representations by
    penalizing the total correlation between latent factors.
    
    Reference: Kim & Mnih "Disentangling by Factorising" (2018)
    """
    
    def __init__(self,
                 n_clusters: int,
                 embedding_dim: int = 10,
                 gamma: float = 6.4,
                 hidden_dims: Tuple[int, ...] = (128, 64, 32),
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 random_state: Optional[int] = None):
        """
        Initialize FactorVAE clustering.
        
        Args:
            n_clusters: Number of clusters
            embedding_dim: Dimension of learned embeddings (latent dimension)
            gamma: γ parameter controlling total correlation penalty
            hidden_dims: Hidden layer dimensions for encoder/decoder
            device: Device to run the model on
            random_state: Random state for reproducibility
        """
        super().__init__(n_clusters, embedding_dim, random_state)
        self.gamma = gamma
        self.hidden_dims = hidden_dims
        self.device = device
        self.vae_model_ = None
        self.discriminator_ = None
        self.kmeans_ = None
        self.history_ = []
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            epochs: int = 100, batch_size: int = 32,
            learning_rate: float = 1e-3) -> 'FactorVAEClustering':
        """
        Fit FactorVAE and perform clustering.
        
        Args:
            X: Input data
            y: Ignored, present for API consistency
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            
        Returns:
            self: Fitted FactorVAE clustering object
        """
        self.log_info(f"Training FactorVAE clustering for {epochs} epochs...")
        
        # Reshape data if needed
        if X.ndim == 3:
            X = X.squeeze(-1)
            
        input_dim = X.shape[1]
        
        # Create FactorVAE model and discriminator
        self.vae_model_ = BetaVAE(  # Use same architecture as β-VAE
            input_dim=input_dim,
            latent_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            beta=1.0  # β=1 for FactorVAE, total correlation handled separately
        ).to(self.device)
        
        self.discriminator_ = Discriminator(self.embedding_dim).to(self.device)
        
        # Optimizers
        vae_optimizer = optim.Adam(self.vae_model_.parameters(), lr=learning_rate)
        disc_optimizer = optim.Adam(self.discriminator_.parameters(), lr=learning_rate)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(epochs):
            epoch_vae_loss = 0
            epoch_disc_loss = 0
            
            for batch_idx, (batch_x,) in enumerate(dataloader):
                batch_x = batch_x.to(self.device)
                batch_size_actual = batch_x.size(0)
                
                # Train VAE
                vae_optimizer.zero_grad()
                
                recon_x, mu, logvar = self.vae_model_(batch_x)
                z = self.vae_model_.reparameterize(mu, logvar)
                
                # VAE losses
                recon_loss = F.mse_loss(recon_x, batch_x, reduction='sum')
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                
                # Total correlation penalty using discriminator
                z_perm = self._permute_dims(z)  # Permute dimensions to break correlations
                d_z = self.discriminator_(z)
                d_z_perm = self.discriminator_(z_perm)
                
                tc_loss = torch.mean(d_z) - torch.mean(d_z_perm)
                
                vae_loss = recon_loss + kld_loss + self.gamma * tc_loss
                vae_loss.backward(retain_graph=True)
                vae_optimizer.step()
                
                # Train Discriminator
                disc_optimizer.zero_grad()
                
                # Discriminator loss (distinguish between correlated and uncorrelated latents)
                d_z = self.discriminator_(z.detach())
                d_z_perm = self.discriminator_(z_perm.detach())
                
                disc_loss = -(torch.mean(d_z) - torch.mean(d_z_perm))
                disc_loss.backward()
                disc_optimizer.step()
                
                epoch_vae_loss += vae_loss.item()
                epoch_disc_loss += disc_loss.item()
                
            # Log progress
            if epoch % 20 == 0:
                avg_vae_loss = epoch_vae_loss / len(X)
                avg_disc_loss = epoch_disc_loss / len(dataloader)
                self.log_info(f"Epoch {epoch}/{epochs} - VAE Loss: {avg_vae_loss:.4f}, "
                            f"Disc Loss: {avg_disc_loss:.4f}")
                
            self.history_.append({
                'vae_loss': epoch_vae_loss / len(X),
                'disc_loss': epoch_disc_loss / len(dataloader)
            })
        
        # Extract embeddings and perform clustering
        self.log_info("Extracting embeddings and performing clustering...")
        embeddings = self.get_embeddings(X)
        
        # Apply K-means clustering
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        self.labels_ = self.kmeans_.fit_predict(embeddings)
        self.cluster_centers_ = self.kmeans_.cluster_centers_
        self.embeddings_ = embeddings
        self.is_fitted_ = True
        
        self.log_info("FactorVAE clustering completed")
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        embeddings = self.get_embeddings(X)
        return self.kmeans_.predict(embeddings)
        
    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Get learned embeddings for input data."""
        if self.vae_model_ is None:
            raise ValueError("VAE model not available")
            
        if X.ndim == 3:
            X = X.squeeze(-1)
            
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.vae_model_.eval()
        with torch.no_grad():
            mu, _ = self.vae_model_.encode(X_tensor)
            
        return mu.cpu().numpy()
        
    def _permute_dims(self, z: torch.Tensor) -> torch.Tensor:
        """Permute dimensions of latent codes to break correlations."""
        batch_size, latent_dim = z.size()
        
        # Create permutation indices for each dimension
        permuted_z = torch.zeros_like(z)
        for dim in range(latent_dim):
            perm_indices = torch.randperm(batch_size, device=z.device)
            permuted_z[:, dim] = z[perm_indices, dim]
            
        return permuted_z


class MultiModalVAEClustering(BaseMultiModalClusteringMethod, LoggerMixin):
    """
    Multi-modal VAE clustering for load and weather data.
    
    This method learns disentangled representations from both load and weather
    modalities using a multi-modal VAE architecture.
    """
    
    def __init__(self,
                 n_clusters: int,
                 embedding_dim: int = 10,
                 beta: float = 4.0,
                 hidden_dims: Tuple[int, ...] = (128, 64, 32),
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 random_state: Optional[int] = None):
        """
        Initialize multi-modal VAE clustering.
        
        Args:
            n_clusters: Number of clusters
            embedding_dim: Dimension of learned embeddings
            beta: β parameter for disentanglement
            hidden_dims: Hidden layer dimensions
            device: Device to run the model on
            random_state: Random state for reproducibility
        """
        super().__init__(n_clusters, embedding_dim, random_state)
        self.beta = beta
        self.hidden_dims = hidden_dims
        self.device = device
        self.mvae_model_ = None
        self.kmeans_ = None
        self.history_ = []
        
    def fit(self, X_primary: np.ndarray, X_secondary: Optional[np.ndarray] = None,
            y: Optional[np.ndarray] = None,
            epochs: int = 100, batch_size: int = 32,
            learning_rate: float = 1e-3) -> 'MultiModalVAEClustering':
        """
        Fit multi-modal VAE and perform clustering.
        
        Args:
            X_primary: Primary input data (load)
            X_secondary: Secondary input data (weather)
            y: Ignored, present for API consistency
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            
        Returns:
            self: Fitted multi-modal VAE clustering object
        """
        if X_secondary is None:
            raise ValueError("Multi-modal VAE requires secondary data (weather)")
            
        self.log_info(f"Training multi-modal VAE clustering for {epochs} epochs...")
        
        # Reshape data if needed
        if X_primary.ndim == 3:
            X_primary = X_primary.squeeze(-1)
        if X_secondary.ndim == 3:
            X_secondary = X_secondary.squeeze(-1)
            
        load_dim = X_primary.shape[1]
        weather_dim = X_secondary.shape[1]
        
        # Create multi-modal VAE model
        self.mvae_model_ = MultiModalVAE(
            load_dim=load_dim,
            weather_dim=weather_dim,
            latent_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            beta=self.beta
        ).to(self.device)
        
        optimizer = optim.Adam(self.mvae_model_.parameters(), lr=learning_rate)
        
        # Convert to tensors
        X_load_tensor = torch.FloatTensor(X_primary).to(self.device)
        X_weather_tensor = torch.FloatTensor(X_secondary).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_load_tensor, X_weather_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.mvae_model_.train()
        for epoch in range(epochs):
            epoch_loss = 0
            
            for batch_idx, (batch_load, batch_weather) in enumerate(dataloader):
                batch_load = batch_load.to(self.device)
                batch_weather = batch_weather.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                recon_load, recon_weather, mu, logvar = self.mvae_model_(batch_load, batch_weather)
                
                # Compute losses
                recon_loss_load = F.mse_loss(recon_load, batch_load, reduction='sum')
                recon_loss_weather = F.mse_loss(recon_weather, batch_weather, reduction='sum')
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                
                total_loss = recon_loss_load + recon_loss_weather + self.beta * kld_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                
            # Log progress
            if epoch % 20 == 0:
                avg_loss = epoch_loss / len(X_primary)
                self.log_info(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")
                
            self.history_.append({'total_loss': epoch_loss / len(X_primary)})
        
        # Extract embeddings and perform clustering
        self.log_info("Extracting embeddings and performing clustering...")
        embeddings = self.get_embeddings(X_primary, X_secondary)
        
        # Apply K-means clustering
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        self.labels_ = self.kmeans_.fit_predict(embeddings)
        self.cluster_centers_ = self.kmeans_.cluster_centers_
        self.embeddings_ = embeddings
        self.is_fitted_ = True
        
        self.log_info("Multi-modal VAE clustering completed")
        return self
        
    def predict(self, X_primary: np.ndarray, X_secondary: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """Predict cluster labels for new data."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        embeddings = self.get_embeddings(X_primary, X_secondary)
        return self.kmeans_.predict(embeddings)
        
    def get_embeddings(self, X_primary: np.ndarray, X_secondary: np.ndarray) -> np.ndarray:
        """Get learned embeddings for input data."""
        if self.mvae_model_ is None:
            raise ValueError("Multi-modal VAE model not available")
            
        if X_primary.ndim == 3:
            X_primary = X_primary.squeeze(-1)
        if X_secondary.ndim == 3:
            X_secondary = X_secondary.squeeze(-1)
            
        X_load_tensor = torch.FloatTensor(X_primary).to(self.device)
        X_weather_tensor = torch.FloatTensor(X_secondary).to(self.device)
        
        self.mvae_model_.eval()
        with torch.no_grad():
            mu, _ = self.mvae_model_.encode(X_load_tensor, X_weather_tensor)
            
        return mu.cpu().numpy()


# Neural Network Components

class BetaVAE(nn.Module):
    """β-VAE model for learning disentangled representations."""
    
    def __init__(self, input_dim: int, latent_dim: int, 
                 hidden_dims: Tuple[int, ...], beta: float = 4.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        """Encode input to latent parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        """Decode latent codes to reconstruction."""
        return self.decoder(z)
        
    def forward(self, x):
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


class Discriminator(nn.Module):
    """Discriminator for FactorVAE total correlation penalty."""
    
    def __init__(self, latent_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )
        
    def forward(self, z):
        """Forward pass through discriminator."""
        return self.network(z)


class MultiModalVAE(nn.Module):
    """Multi-modal VAE for load and weather data."""
    
    def __init__(self, load_dim: int, weather_dim: int, latent_dim: int,
                 hidden_dims: Tuple[int, ...], beta: float = 4.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Load encoder
        load_encoder_layers = []
        prev_dim = load_dim
        for hidden_dim in hidden_dims:
            load_encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        self.load_encoder = nn.Sequential(*load_encoder_layers)
        
        # Weather encoder
        weather_encoder_layers = []
        prev_dim = weather_dim
        for hidden_dim in hidden_dims:
            weather_encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        self.weather_encoder = nn.Sequential(*weather_encoder_layers)
        
        # Fusion and latent layers
        fused_dim = hidden_dims[-1] * 2  # Concatenated features
        self.fusion = nn.Linear(fused_dim, hidden_dims[-1])
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoders
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        self.shared_decoder = nn.Sequential(*decoder_layers)
        self.load_decoder = nn.Linear(prev_dim, load_dim)
        self.weather_decoder = nn.Linear(prev_dim, weather_dim)
        
    def encode(self, load, weather):
        """Encode multi-modal input to latent parameters."""
        load_features = self.load_encoder(load)
        weather_features = self.weather_encoder(weather)
        
        # Fuse features
        fused_features = torch.cat([load_features, weather_features], dim=1)
        h = F.relu(self.fusion(fused_features))
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        """Decode latent codes to multi-modal reconstruction."""
        h = self.shared_decoder(z)
        recon_load = self.load_decoder(h)
        recon_weather = self.weather_decoder(h)
        return recon_load, recon_weather
        
    def forward(self, load, weather):
        """Forward pass through multi-modal VAE."""
        mu, logvar = self.encode(load, weather)
        z = self.reparameterize(mu, logvar)
        recon_load, recon_weather = self.decode(z)
        return recon_load, recon_weather, mu, logvar