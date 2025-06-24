"""
Weather normalization preprocessing methods for household energy segmentation.

This module implements various weather normalization techniques that can be used
as preprocessing steps before applying traditional clustering methods.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from typing import Optional, Tuple, Dict, Union
import warnings

from .base import BaseClusteringMethod
from ..utils.logging import LoggerMixin


class LinearWeatherNormalization(BaseClusteringMethod, LoggerMixin):
    """
    Linear weather normalization using regression-based approach.
    
    This method removes weather effects using linear regression models
    to predict weather-driven consumption, then clusters the residuals.
    """
    
    def __init__(self, 
                 n_clusters: int,
                 base_clusterer: BaseClusteringMethod,
                 regression_type: str = 'linear',
                 alpha: float = 1.0,
                 random_state: Optional[int] = None):
        """
        Initialize linear weather normalization.
        
        Args:
            n_clusters: Number of clusters
            base_clusterer: Base clustering method to apply after normalization
            regression_type: Type of regression ('linear', 'ridge', 'polynomial')
            alpha: Regularization strength for ridge regression
            random_state: Random state for reproducibility
        """
        super().__init__(n_clusters, random_state)
        self.base_clusterer = base_clusterer
        self.regression_type = regression_type
        self.alpha = alpha
        self.weather_models_ = None
        self.normalized_data_ = None
        
    def fit(self, X: np.ndarray, X_weather: np.ndarray, 
            y: Optional[np.ndarray] = None) -> 'LinearWeatherNormalization':
        """
        Fit weather normalization and clustering.
        
        Args:
            X: Load data of shape (n_samples, n_timesteps)
            X_weather: Weather data of shape (n_samples, n_timesteps, n_weather_features)
            y: Ignored, present for API consistency
            
        Returns:
            self: Fitted weather normalization object
        """
        self.log_info("Fitting linear weather normalization...")
        
        # Reshape data if needed
        if X.ndim == 3:
            X = X.squeeze(-1)
        if X_weather.ndim == 2:
            X_weather = X_weather.reshape(X_weather.shape[0], X_weather.shape[1], 1)
            
        n_samples, n_timesteps = X.shape
        n_weather_features = X_weather.shape[2]
        
        # Fit regression models for each timestep
        self.weather_models_ = []
        normalized_load = np.zeros_like(X)
        
        for t in range(n_timesteps):
            # Prepare data for this timestep
            y_load = X[:, t]  # Load at timestep t
            X_weather_t = X_weather[:, t, :]  # Weather at timestep t
            
            # Create regression model
            if self.regression_type == 'linear':
                model = LinearRegression()
            elif self.regression_type == 'ridge':
                model = Ridge(alpha=self.alpha)
            elif self.regression_type == 'polynomial':
                model = Pipeline([
                    ('poly', PolynomialFeatures(degree=2)),
                    ('linear', LinearRegression())
                ])
            else:
                raise ValueError(f"Unknown regression type: {self.regression_type}")
            
            # Fit model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_weather_t, y_load)
            
            # Predict weather-driven consumption
            weather_consumption = model.predict(X_weather_t)
            
            # Compute normalized (residual) load
            normalized_load[:, t] = y_load - weather_consumption
            
            self.weather_models_.append(model)
            
        self.normalized_data_ = normalized_load
        
        # Apply base clustering to normalized data
        self.log_info("Applying base clustering to normalized data...")
        self.base_clusterer.fit(normalized_load)
        self.labels_ = self.base_clusterer.labels_
        self.cluster_centers_ = getattr(self.base_clusterer, 'cluster_centers_', None)
        self.is_fitted_ = True
        
        self.log_info("Linear weather normalization completed")
        return self
        
    def predict(self, X: np.ndarray, X_weather: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Load data
            X_weather: Weather data
            
        Returns:
            Cluster labels
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        # Normalize new data
        normalized_load = self._normalize_data(X, X_weather)
        
        # Predict using base clusterer
        return self.base_clusterer.predict(normalized_load)
        
    def _normalize_data(self, X: np.ndarray, X_weather: np.ndarray) -> np.ndarray:
        """Normalize load data by removing weather effects."""
        if X.ndim == 3:
            X = X.squeeze(-1)
        if X_weather.ndim == 2:
            X_weather = X_weather.reshape(X_weather.shape[0], X_weather.shape[1], 1)
            
        n_samples, n_timesteps = X.shape
        normalized_load = np.zeros_like(X)
        
        for t in range(n_timesteps):
            y_load = X[:, t]
            X_weather_t = X_weather[:, t, :]
            
            # Predict weather consumption using fitted model
            weather_consumption = self.weather_models_[t].predict(X_weather_t)
            normalized_load[:, t] = y_load - weather_consumption
            
        return normalized_load


class NonlinearWeatherNormalization(BaseClusteringMethod, LoggerMixin):
    """
    Nonlinear weather normalization using neural networks.
    
    This method uses a neural network to model complex weather-load relationships
    and removes predicted weather effects before clustering.
    """
    
    def __init__(self,
                 n_clusters: int,
                 base_clusterer: BaseClusteringMethod,
                 hidden_dims: Tuple[int, ...] = (64, 32),
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 random_state: Optional[int] = None):
        """
        Initialize nonlinear weather normalization.
        
        Args:
            n_clusters: Number of clusters
            base_clusterer: Base clustering method to apply after normalization
            hidden_dims: Hidden layer dimensions for the neural network
            device: Device to run the model on
            random_state: Random state for reproducibility
        """
        super().__init__(n_clusters, random_state)
        self.base_clusterer = base_clusterer
        self.hidden_dims = hidden_dims
        self.device = device
        self.weather_model_ = None
        self.normalized_data_ = None
        
    def fit(self, X: np.ndarray, X_weather: np.ndarray,
            y: Optional[np.ndarray] = None,
            epochs: int = 100, batch_size: int = 32,
            learning_rate: float = 0.001) -> 'NonlinearWeatherNormalization':
        """
        Fit nonlinear weather normalization and clustering.
        
        Args:
            X: Load data of shape (n_samples, n_timesteps)
            X_weather: Weather data of shape (n_samples, n_timesteps, n_weather_features)
            y: Ignored, present for API consistency
            epochs: Number of training epochs for neural network
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            
        Returns:
            self: Fitted weather normalization object
        """
        self.log_info("Fitting nonlinear weather normalization...")
        
        # Reshape data if needed
        if X.ndim == 3:
            X = X.squeeze(-1)
        if X_weather.ndim == 2:
            X_weather = X_weather.reshape(X_weather.shape[0], X_weather.shape[1], 1)
            
        n_samples, n_timesteps = X.shape
        n_weather_features = X_weather.shape[2]
        
        # Create neural network for weather prediction
        self.weather_model_ = WeatherPredictionNetwork(
            input_dim=n_weather_features,
            output_dim=1,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        optimizer = optim.Adam(self.weather_model_.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Prepare training data
        X_weather_flat = X_weather.reshape(-1, n_weather_features)
        X_load_flat = X.reshape(-1, 1)
        
        # Convert to tensors
        X_weather_tensor = torch.FloatTensor(X_weather_flat).to(self.device)
        X_load_tensor = torch.FloatTensor(X_load_flat).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_weather_tensor, X_load_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Train neural network
        self.weather_model_.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_weather, batch_load in dataloader:
                optimizer.zero_grad()
                
                predicted_load = self.weather_model_(batch_weather)
                loss = criterion(predicted_load, batch_load)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            if epoch % 20 == 0:
                avg_loss = epoch_loss / len(dataloader)
                self.log_info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Generate normalized data
        self.weather_model_.eval()
        with torch.no_grad():
            predicted_weather_load = self.weather_model_(X_weather_tensor).cpu().numpy()
            
        # Reshape predictions back to original shape
        predicted_weather_load = predicted_weather_load.reshape(n_samples, n_timesteps)
        
        # Compute normalized load (residuals)
        self.normalized_data_ = X - predicted_weather_load
        
        # Apply base clustering to normalized data
        self.log_info("Applying base clustering to normalized data...")
        self.base_clusterer.fit(self.normalized_data_)
        self.labels_ = self.base_clusterer.labels_
        self.cluster_centers_ = getattr(self.base_clusterer, 'cluster_centers_', None)
        self.is_fitted_ = True
        
        self.log_info("Nonlinear weather normalization completed")
        return self
        
    def predict(self, X: np.ndarray, X_weather: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Load data
            X_weather: Weather data
            
        Returns:
            Cluster labels
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        # Normalize new data
        normalized_load = self._normalize_data(X, X_weather)
        
        # Predict using base clusterer
        return self.base_clusterer.predict(normalized_load)
        
    def _normalize_data(self, X: np.ndarray, X_weather: np.ndarray) -> np.ndarray:
        """Normalize load data by removing weather effects."""
        if X.ndim == 3:
            X = X.squeeze(-1)
        if X_weather.ndim == 2:
            X_weather = X_weather.reshape(X_weather.shape[0], X_weather.shape[1], 1)
            
        n_samples, n_timesteps = X.shape
        n_weather_features = X_weather.shape[2]
        
        # Prepare data for prediction
        X_weather_flat = X_weather.reshape(-1, n_weather_features)
        X_weather_tensor = torch.FloatTensor(X_weather_flat).to(self.device)
        
        # Predict weather-driven consumption
        self.weather_model_.eval()
        with torch.no_grad():
            predicted_weather_load = self.weather_model_(X_weather_tensor).cpu().numpy()
            
        # Reshape predictions back to original shape
        predicted_weather_load = predicted_weather_load.reshape(n_samples, n_timesteps)
        
        # Compute normalized load
        return X - predicted_weather_load


class WeatherPredictionNetwork(nn.Module):
    """
    Neural network for predicting weather-driven energy consumption.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int, ...]):
        """
        Initialize the weather prediction network.
        
        Args:
            input_dim: Input dimension (weather features)
            output_dim: Output dimension (predicted load)
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)


class CDDHDDWeatherNormalization(BaseClusteringMethod, LoggerMixin):
    """
    Weather normalization using Cooling Degree Days (CDD) and Heating Degree Days (HDD).
    
    This method implements the engineering approach commonly used in the energy industry
    for weather normalization based on degree days.
    """
    
    def __init__(self,
                 n_clusters: int,
                 base_clusterer: BaseClusteringMethod,
                 base_temp: float = 18.0,  # Base temperature in Celsius
                 random_state: Optional[int] = None):
        """
        Initialize CDD/HDD weather normalization.
        
        Args:
            n_clusters: Number of clusters
            base_clusterer: Base clustering method to apply after normalization
            base_temp: Base temperature for degree day calculation
            random_state: Random state for reproducibility
        """
        super().__init__(n_clusters, random_state)
        self.base_clusterer = base_clusterer
        self.base_temp = base_temp
        self.cdd_model_ = None
        self.hdd_model_ = None
        self.normalized_data_ = None
        
    def fit(self, X: np.ndarray, X_weather: np.ndarray,
            y: Optional[np.ndarray] = None) -> 'CDDHDDWeatherNormalization':
        """
        Fit CDD/HDD weather normalization and clustering.
        
        Args:
            X: Load data of shape (n_samples, n_timesteps)
            X_weather: Temperature data of shape (n_samples, n_timesteps) or (n_samples, n_timesteps, 1)
            y: Ignored, present for API consistency
            
        Returns:
            self: Fitted weather normalization object
        """
        self.log_info("Fitting CDD/HDD weather normalization...")
        
        # Reshape data if needed
        if X.ndim == 3:
            X = X.squeeze(-1)
        if X_weather.ndim == 3:
            X_weather = X_weather.squeeze(-1)
            
        # Compute degree days
        cdd, hdd = self._compute_degree_days(X_weather)
        
        # Aggregate daily load and degree days
        daily_load = np.sum(X, axis=1)  # Total daily consumption
        daily_cdd = np.sum(cdd, axis=1)  # Total daily CDD
        daily_hdd = np.sum(hdd, axis=1)  # Total daily HDD
        
        # Fit linear models for CDD and HDD effects
        # Load = Base_Load + CDD_coeff * CDD + HDD_coeff * HDD
        degree_day_features = np.column_stack([daily_cdd, daily_hdd])
        
        self.cdd_model_ = LinearRegression()
        self.cdd_model_.fit(degree_day_features, daily_load)
        
        # Predict weather-driven consumption
        predicted_weather_load = self.cdd_model_.predict(degree_day_features)
        
        # Compute base load (weather-normalized)
        base_daily_load = daily_load - predicted_weather_load + np.mean(predicted_weather_load)
        
        # Distribute base load back to hourly resolution (simplified approach)
        # Use original load shape but scale to match base daily totals
        original_daily_totals = np.sum(X, axis=1, keepdims=True)
        load_shapes = X / (original_daily_totals + 1e-8)  # Normalize to get shapes
        
        # Scale shapes by base daily load
        self.normalized_data_ = load_shapes * base_daily_load.reshape(-1, 1)
        
        # Apply base clustering to normalized data
        self.log_info("Applying base clustering to normalized data...")
        self.base_clusterer.fit(self.normalized_data_)
        self.labels_ = self.base_clusterer.labels_
        self.cluster_centers_ = getattr(self.base_clusterer, 'cluster_centers_', None)
        self.is_fitted_ = True
        
        self.log_info("CDD/HDD weather normalization completed")
        return self
        
    def predict(self, X: np.ndarray, X_weather: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Load data
            X_weather: Temperature data
            
        Returns:
            Cluster labels
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        # Normalize new data
        normalized_load = self._normalize_data(X, X_weather)
        
        # Predict using base clusterer
        return self.base_clusterer.predict(normalized_load)
        
    def _normalize_data(self, X: np.ndarray, X_weather: np.ndarray) -> np.ndarray:
        """Normalize load data using CDD/HDD approach."""
        if X.ndim == 3:
            X = X.squeeze(-1)
        if X_weather.ndim == 3:
            X_weather = X_weather.squeeze(-1)
            
        # Compute degree days
        cdd, hdd = self._compute_degree_days(X_weather)
        
        # Aggregate daily values
        daily_load = np.sum(X, axis=1)
        daily_cdd = np.sum(cdd, axis=1)
        daily_hdd = np.sum(hdd, axis=1)
        
        # Predict weather-driven consumption
        degree_day_features = np.column_stack([daily_cdd, daily_hdd])
        predicted_weather_load = self.cdd_model_.predict(degree_day_features)
        
        # Compute base load
        base_daily_load = daily_load - predicted_weather_load + np.mean(predicted_weather_load)
        
        # Distribute back to hourly resolution
        original_daily_totals = np.sum(X, axis=1, keepdims=True)
        load_shapes = X / (original_daily_totals + 1e-8)
        
        return load_shapes * base_daily_load.reshape(-1, 1)
        
    def _compute_degree_days(self, temperature: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Cooling Degree Days (CDD) and Heating Degree Days (HDD).
        
        Args:
            temperature: Temperature data
            
        Returns:
            Tuple of (CDD, HDD) arrays
        """
        # CDD: max(0, temp - base_temp)
        cdd = np.maximum(0, temperature - self.base_temp)
        
        # HDD: max(0, base_temp - temp)
        hdd = np.maximum(0, self.base_temp - temperature)
        
        return cdd, hdd