"""
Causal inference-based clustering methods for household energy segmentation.

This module implements various causal inference approaches for addressing
confounding in clustering, including do-calculus, instrumental variables,
and domain adaptation methods.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Dict, Union, List
import warnings

from .base import BaseMultiModalClusteringMethod
from ..utils.logging import LoggerMixin


class DoublyRobustClustering(BaseMultiModalClusteringMethod, LoggerMixin):
    """
    Doubly Robust Clustering for causal inference in clustering.
    
    This method uses doubly robust estimation to remove confounding effects
    before clustering. It estimates both the treatment assignment (weather)
    and outcome (load) models.
    """
    
    def __init__(self,
                 n_clusters: int,
                 base_clusterer,
                 propensity_model: str = 'linear',  # 'linear', 'neural'
                 outcome_model: str = 'linear',     # 'linear', 'neural'
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 random_state: Optional[int] = None):
        """
        Initialize doubly robust clustering.
        
        Args:
            n_clusters: Number of clusters
            base_clusterer: Base clustering method to apply after adjustment
            propensity_model: Type of propensity score model
            outcome_model: Type of outcome regression model
            device: Device to run neural models on
            random_state: Random state for reproducibility
        """
        super().__init__(n_clusters, random_state=random_state)
        self.base_clusterer = base_clusterer
        self.propensity_model = propensity_model
        self.outcome_model = outcome_model
        self.device = device
        self.propensity_estimator_ = None
        self.outcome_estimator_ = None
        self.adjusted_data_ = None
        
    def fit(self, X_primary: np.ndarray, X_secondary: Optional[np.ndarray] = None,
            y: Optional[np.ndarray] = None) -> 'DoublyRobustClustering':
        """
        Fit doubly robust clustering.
        
        Args:
            X_primary: Primary input data (load shapes)
            X_secondary: Secondary input data (weather data)
            y: Ignored, present for API consistency
            
        Returns:
            self: Fitted doubly robust clustering object
        """
        if X_secondary is None:
            raise ValueError("Doubly robust clustering requires secondary data (weather)")
            
        self.log_info("Fitting doubly robust clustering...")
        
        # Reshape data if needed
        if X_primary.ndim == 3:
            X_primary = X_primary.squeeze(-1)
        if X_secondary.ndim == 3:
            X_secondary = X_secondary.squeeze(-1)
            
        # Discretize weather data for treatment assignment
        # Use median split as a simple treatment assignment
        weather_treatment = (X_secondary > np.median(X_secondary, axis=0)).astype(float)
        
        # Fit propensity score model (P(T|X))
        self.log_info("Fitting propensity score model...")
        if self.propensity_model == 'linear':
            self.propensity_estimator_ = self._fit_linear_propensity(X_primary, weather_treatment)
        else:
            self.propensity_estimator_ = self._fit_neural_propensity(X_primary, weather_treatment)
            
        # Fit outcome regression model (E[Y|T,X])
        self.log_info("Fitting outcome regression model...")
        if self.outcome_model == 'linear':
            self.outcome_estimator_ = self._fit_linear_outcome(X_primary, weather_treatment)
        else:
            self.outcome_estimator_ = self._fit_neural_outcome(X_primary, weather_treatment)
            
        # Compute doubly robust adjustment
        self.log_info("Computing doubly robust adjustment...")
        self.adjusted_data_ = self._compute_dr_adjustment(X_primary, weather_treatment)
        
        # Apply base clustering to adjusted data
        self.log_info("Applying base clustering to adjusted data...")
        self.base_clusterer.fit(self.adjusted_data_)
        self.labels_ = self.base_clusterer.labels_
        self.cluster_centers_ = getattr(self.base_clusterer, 'cluster_centers_', None)
        self.is_fitted_ = True
        
        self.log_info("Doubly robust clustering completed")
        return self
        
    def predict(self, X_primary: np.ndarray, X_secondary: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """Predict cluster labels for new data."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        if X_secondary is None:
            raise ValueError("Weather data required for prediction")
            
        # Apply same adjustment to new data
        adjusted_data = self._adjust_new_data(X_primary, X_secondary)
        return self.base_clusterer.predict(adjusted_data)
        
    def _fit_linear_propensity(self, X: np.ndarray, T: np.ndarray) -> Dict:
        """Fit linear propensity score models."""
        estimators = {}
        n_timesteps = T.shape[1]
        
        for t in range(n_timesteps):
            # For each timestep, predict treatment from load features
            estimator = LinearRegression()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(X, T[:, t])
            estimators[t] = estimator
            
        return estimators
        
    def _fit_neural_propensity(self, X: np.ndarray, T: np.ndarray) -> nn.Module:
        """Fit neural propensity score model."""
        # Simplified neural propensity model
        model = PropensityNetwork(X.shape[1], T.shape[1]).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        T_tensor = torch.FloatTensor(T).to(self.device)
        
        model.train()
        for epoch in range(50):  # Quick training
            optimizer.zero_grad()
            pred_T = model(X_tensor)
            loss = F.binary_cross_entropy_with_logits(pred_T, T_tensor)
            loss.backward()
            optimizer.step()
            
        return model
        
    def _fit_linear_outcome(self, X: np.ndarray, T: np.ndarray) -> Dict:
        """Fit linear outcome regression models."""
        estimators = {}
        n_timesteps = X.shape[1]
        
        for t in range(n_timesteps):
            # For each timestep, predict load from weather treatment
            estimator = LinearRegression()
            # Use treatment at all timesteps as features
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(T, X[:, t])
            estimators[t] = estimator
            
        return estimators
        
    def _fit_neural_outcome(self, X: np.ndarray, T: np.ndarray) -> nn.Module:
        """Fit neural outcome regression model."""
        model = OutcomeNetwork(T.shape[1], X.shape[1]).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        T_tensor = torch.FloatTensor(T).to(self.device)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        model.train()
        for epoch in range(50):  # Quick training
            optimizer.zero_grad()
            pred_X = model(T_tensor)
            loss = F.mse_loss(pred_X, X_tensor)
            loss.backward()
            optimizer.step()
            
        return model
        
    def _compute_dr_adjustment(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute doubly robust adjustment."""
        n_samples, n_timesteps = X.shape
        adjusted_X = np.zeros_like(X)
        
        for t in range(n_timesteps):
            # Get propensity scores
            if self.propensity_model == 'linear':
                prop_scores = self.propensity_estimator_[t].predict(X)
                prop_scores = np.clip(prop_scores, 0.1, 0.9)  # Avoid extreme values
            else:
                X_tensor = torch.FloatTensor(X).to(self.device)
                with torch.no_grad():
                    prop_scores = torch.sigmoid(self.propensity_estimator_(X_tensor)[:, t]).cpu().numpy()
                prop_scores = np.clip(prop_scores, 0.1, 0.9)
                
            # Get outcome predictions
            if self.outcome_model == 'linear':
                outcome_pred = self.outcome_estimator_[t].predict(T)
            else:
                T_tensor = torch.FloatTensor(T).to(self.device)
                with torch.no_grad():
                    outcome_pred = self.outcome_estimator_(T_tensor)[:, t].cpu().numpy()
                    
            # Doubly robust adjustment
            # Adjusted = Y - E[Y|T,X] + (T-P(T|X))/P(T|X) * (Y - E[Y|T,X])
            residual = X[:, t] - outcome_pred
            weight = (T[:, t] - prop_scores) / prop_scores
            adjusted_X[:, t] = X[:, t] - outcome_pred + weight * residual
            
        return adjusted_X
        
    def _adjust_new_data(self, X: np.ndarray, X_weather: np.ndarray) -> np.ndarray:
        """Apply adjustment to new data."""
        if X.ndim == 3:
            X = X.squeeze(-1)
        if X_weather.ndim == 3:
            X_weather = X_weather.squeeze(-1)
            
        # Create treatment assignment for new data
        weather_treatment = (X_weather > np.median(X_weather, axis=0)).astype(float)
        
        # Apply same adjustment
        return self._compute_dr_adjustment(X, weather_treatment)


class InstrumentalVariableClustering(BaseMultiModalClusteringMethod, LoggerMixin):
    """
    Instrumental Variable Clustering for causal inference.
    
    This method uses instrumental variables to identify causal effects
    and remove confounding before clustering.
    """
    
    def __init__(self,
                 n_clusters: int,
                 base_clusterer,
                 iv_strength_threshold: float = 0.1,
                 random_state: Optional[int] = None):
        """
        Initialize instrumental variable clustering.
        
        Args:
            n_clusters: Number of clusters
            base_clusterer: Base clustering method to apply after adjustment
            iv_strength_threshold: Minimum strength for valid instruments
            random_state: Random state for reproducibility
        """
        super().__init__(n_clusters, random_state=random_state)
        self.base_clusterer = base_clusterer
        self.iv_strength_threshold = iv_strength_threshold
        self.iv_estimator_ = None
        self.adjusted_data_ = None
        
    def fit(self, X_primary: np.ndarray, X_secondary: Optional[np.ndarray] = None,
            y: Optional[np.ndarray] = None) -> 'InstrumentalVariableClustering':
        """
        Fit instrumental variable clustering.
        
        Args:
            X_primary: Primary input data (load shapes)
            X_secondary: Secondary input data (weather data, used as instrument)
            y: Ignored, present for API consistency
            
        Returns:
            self: Fitted IV clustering object
        """
        if X_secondary is None:
            raise ValueError("IV clustering requires secondary data (weather as instrument)")
            
        self.log_info("Fitting instrumental variable clustering...")
        
        # Reshape data if needed
        if X_primary.ndim == 3:
            X_primary = X_primary.squeeze(-1)
        if X_secondary.ndim == 3:
            X_secondary = X_secondary.squeeze(-1)
            
        # Use weather as instrumental variable
        # Two-stage least squares (2SLS) approach
        self.log_info("Performing 2SLS estimation...")
        
        adjusted_load = np.zeros_like(X_primary)
        
        for t in range(X_primary.shape[1]):
            # Stage 1: Regress endogenous variable (weather effect) on instrument (weather)
            stage1_model = LinearRegression()
            
            # Use lagged weather as instrument (simple approach)
            if t > 0:
                instrument = X_secondary[:, t-1:t+1]  # Current and previous weather
            else:
                instrument = X_secondary[:, t:t+1]   # Just current weather
                
            stage1_model.fit(instrument, X_primary[:, t])
            predicted_weather_effect = stage1_model.predict(instrument)
            
            # Check instrument strength
            correlation = np.corrcoef(instrument.flatten(), X_primary[:, t])[0, 1]
            if abs(correlation) < self.iv_strength_threshold:
                self.log_warning(f"Weak instrument at timestep {t}, correlation: {correlation:.3f}")
                
            # Stage 2: Remove predicted weather effect
            adjusted_load[:, t] = X_primary[:, t] - predicted_weather_effect
            
        self.adjusted_data_ = adjusted_load
        self.iv_estimator_ = {'fitted': True}  # Placeholder
        
        # Apply base clustering to adjusted data
        self.log_info("Applying base clustering to IV-adjusted data...")
        self.base_clusterer.fit(self.adjusted_data_)
        self.labels_ = self.base_clusterer.labels_
        self.cluster_centers_ = getattr(self.base_clusterer, 'cluster_centers_', None)
        self.is_fitted_ = True
        
        self.log_info("Instrumental variable clustering completed")
        return self
        
    def predict(self, X_primary: np.ndarray, X_secondary: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """Predict cluster labels for new data."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        # For simplicity, apply base clusterer directly
        # In practice, you would apply the same IV adjustment
        return self.base_clusterer.predict(X_primary)


class DomainAdaptationClustering(BaseMultiModalClusteringMethod, LoggerMixin):
    """
    Domain Adaptation Clustering for handling weather distribution shifts.
    
    This method learns domain-invariant representations that are robust
    to changes in weather distributions across different seasons/regions.
    """
    
    def __init__(self,
                 n_clusters: int,
                 embedding_dim: int = 10,
                 domain_lambda: float = 0.1,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 random_state: Optional[int] = None):
        """
        Initialize domain adaptation clustering.
        
        Args:
            n_clusters: Number of clusters
            embedding_dim: Dimension of learned embeddings
            domain_lambda: Weight for domain adversarial loss
            device: Device to run the model on
            random_state: Random state for reproducibility
        """
        super().__init__(n_clusters, embedding_dim, random_state)
        self.domain_lambda = domain_lambda
        self.device = device
        self.feature_extractor_ = None
        self.domain_classifier_ = None
        self.cluster_classifier_ = None
        self.kmeans_ = None
        self.history_ = []
        
    def fit(self, X_primary: np.ndarray, X_secondary: Optional[np.ndarray] = None,
            y: Optional[np.ndarray] = None,
            epochs: int = 100, batch_size: int = 32,
            learning_rate: float = 1e-3) -> 'DomainAdaptationClustering':
        """
        Fit domain adaptation clustering.
        
        Args:
            X_primary: Primary input data (load shapes)
            X_secondary: Secondary input data (weather data for domain labels)
            y: Ignored, present for API consistency
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            
        Returns:
            self: Fitted domain adaptation clustering object
        """
        if X_secondary is None:
            raise ValueError("Domain adaptation clustering requires secondary data (weather)")
            
        self.log_info(f"Training domain adaptation clustering for {epochs} epochs...")
        
        # Reshape data if needed
        if X_primary.ndim == 3:
            X_primary = X_primary.squeeze(-1)
        if X_secondary.ndim == 3:
            X_secondary = X_secondary.squeeze(-1)
            
        # Create domain labels based on weather patterns
        # Simple approach: use weather season/regime as domain
        domain_labels = self._create_domain_labels(X_secondary)
        n_domains = len(np.unique(domain_labels))
        
        input_dim = X_primary.shape[1]
        
        # Create models
        self.feature_extractor_ = FeatureExtractor(input_dim, self.embedding_dim).to(self.device)
        self.domain_classifier_ = DomainClassifier(self.embedding_dim, n_domains).to(self.device)
        self.cluster_classifier_ = ClusterClassifier(self.embedding_dim, self.n_clusters).to(self.device)
        
        # Optimizers
        feature_optimizer = optim.Adam(self.feature_extractor_.parameters(), lr=learning_rate)
        domain_optimizer = optim.Adam(self.domain_classifier_.parameters(), lr=learning_rate)
        cluster_optimizer = optim.Adam(self.cluster_classifier_.parameters(), lr=learning_rate)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_primary).to(self.device)
        domain_tensor = torch.LongTensor(domain_labels).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, domain_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(epochs):
            epoch_feature_loss = 0
            epoch_domain_loss = 0
            epoch_cluster_loss = 0
            
            for batch_idx, (batch_x, batch_domain) in enumerate(dataloader):
                batch_x = batch_x.to(self.device)
                batch_domain = batch_domain.to(self.device)
                
                # Extract features
                features = self.feature_extractor_(batch_x)
                
                # Domain classification (adversarial)
                domain_pred = self.domain_classifier_(features.detach())
                domain_loss = F.cross_entropy(domain_pred, batch_domain)
                
                domain_optimizer.zero_grad()
                domain_loss.backward()
                domain_optimizer.step()
                
                # Feature learning (domain-adversarial)
                domain_pred_for_features = self.domain_classifier_(features)
                domain_adversarial_loss = -F.cross_entropy(domain_pred_for_features, batch_domain)
                
                # Clustering loss (pseudo-labels for simplicity)
                cluster_pred = self.cluster_classifier_(features)
                # Use entropy minimization as clustering objective
                cluster_loss = -torch.mean(torch.sum(cluster_pred * torch.log(cluster_pred + 1e-8), dim=1))
                
                # Total feature loss
                feature_loss = self.domain_lambda * domain_adversarial_loss + cluster_loss
                
                feature_optimizer.zero_grad()
                cluster_optimizer.zero_grad()
                feature_loss.backward()
                feature_optimizer.step()
                cluster_optimizer.step()
                
                epoch_feature_loss += feature_loss.item()
                epoch_domain_loss += domain_loss.item()
                epoch_cluster_loss += cluster_loss.item()
                
            if epoch % 20 == 0:
                avg_feature_loss = epoch_feature_loss / len(dataloader)
                avg_domain_loss = epoch_domain_loss / len(dataloader)
                avg_cluster_loss = epoch_cluster_loss / len(dataloader)
                self.log_info(f"Epoch {epoch}/{epochs} - Feature: {avg_feature_loss:.4f}, "
                            f"Domain: {avg_domain_loss:.4f}, Cluster: {avg_cluster_loss:.4f}")
                
            self.history_.append({
                'feature_loss': epoch_feature_loss / len(dataloader),
                'domain_loss': epoch_domain_loss / len(dataloader),
                'cluster_loss': epoch_cluster_loss / len(dataloader)
            })
            
        # Extract final embeddings and perform clustering
        self.log_info("Extracting embeddings and performing clustering...")
        embeddings = self.get_embeddings(X_primary, X_secondary)
        
        # Apply K-means clustering
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        self.labels_ = self.kmeans_.fit_predict(embeddings)
        self.cluster_centers_ = self.kmeans_.cluster_centers_
        self.embeddings_ = embeddings
        self.is_fitted_ = True
        
        self.log_info("Domain adaptation clustering completed")
        return self
        
    def predict(self, X_primary: np.ndarray, X_secondary: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """Predict cluster labels for new data."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        embeddings = self.get_embeddings(X_primary, X_secondary)
        return self.kmeans_.predict(embeddings)
        
    def get_embeddings(self, X_primary: np.ndarray, X_secondary: np.ndarray) -> np.ndarray:
        """Get learned domain-invariant embeddings."""
        if self.feature_extractor_ is None:
            raise ValueError("Feature extractor not available")
            
        if X_primary.ndim == 3:
            X_primary = X_primary.squeeze(-1)
            
        X_tensor = torch.FloatTensor(X_primary).to(self.device)
        
        self.feature_extractor_.eval()
        with torch.no_grad():
            embeddings = self.feature_extractor_(X_tensor)
            
        return embeddings.cpu().numpy()
        
    def _create_domain_labels(self, weather_data: np.ndarray) -> np.ndarray:
        """Create domain labels based on weather patterns."""
        # Simple approach: use temperature quantiles as domains
        mean_temp = np.mean(weather_data, axis=1)
        
        # Create 3 domains based on temperature terciles
        terciles = np.percentile(mean_temp, [33.33, 66.67])
        domain_labels = np.zeros(len(mean_temp), dtype=int)
        domain_labels[mean_temp > terciles[1]] = 2  # Hot
        domain_labels[(mean_temp > terciles[0]) & (mean_temp <= terciles[1])] = 1  # Medium
        domain_labels[mean_temp <= terciles[0]] = 0  # Cold
        
        return domain_labels


# Neural Network Components

class PropensityNetwork(nn.Module):
    """Neural network for propensity score estimation."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)


class OutcomeNetwork(nn.Module):
    """Neural network for outcome regression."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)


class FeatureExtractor(nn.Module):
    """Feature extractor for domain adaptation."""
    
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, embedding_dim)
        )
        
    def forward(self, x):
        return self.network(x)


class DomainClassifier(nn.Module):
    """Domain classifier for domain adaptation."""
    
    def __init__(self, input_dim: int, n_domains: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_domains)
        )
        
    def forward(self, x):
        return F.softmax(self.network(x), dim=1)


class ClusterClassifier(nn.Module):
    """Cluster classifier for domain adaptation."""
    
    def __init__(self, input_dim: int, n_clusters: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_clusters)
        )
        
    def forward(self, x):
        return F.softmax(self.network(x), dim=1)