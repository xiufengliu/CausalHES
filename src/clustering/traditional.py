"""
Traditional clustering methods for household energy segmentation.

This module implements traditional clustering approaches including
SAX K-means and two-stage K-means from the original 2017 paper.
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin

from .base import BaseClusteringMethod
from ..utils.logging import LoggerMixin


class SAXKMeans(BaseClusteringMethod):
    """
    SAX K-means clustering implementation.
    
    This class implements Symbolic Aggregate approXimation (SAX) based
    K-means clustering for time series data.
    """
    
    def __init__(self, 
                 n_clusters: int,
                 word_size: int = 24,
                 alphabet_size: int = 20,
                 random_state: Optional[int] = None):
        """
        Initialize SAX K-means clustering.
        
        Args:
            n_clusters: Number of clusters
            word_size: SAX word size
            alphabet_size: SAX alphabet size
            random_state: Random state for reproducibility
        """
        super().__init__(n_clusters, random_state)
        self.word_size = word_size
        self.alphabet_size = alphabet_size
        self.sax_transformer = None
        self.kmeans = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'SAXKMeans':
        """
        Fit SAX K-means to the data.
        
        Args:
            X: Input time series data
            y: Ignored, present for API consistency
            
        Returns:
            self: Fitted SAX K-means object
        """
        self.log_info(f"Fitting SAX K-means with {self.n_clusters} clusters")
        
        # Transform data to SAX representation
        sax_data = self._transform_to_sax(X)
        
        # Apply K-means to SAX representations
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init='auto'
        )
        
        self.labels_ = self.kmeans.fit_predict(sax_data)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        self.is_fitted_ = True
        
        self.log_info("SAX K-means fitting completed")
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Input time series data
            
        Returns:
            Cluster labels
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        sax_data = self._transform_to_sax(X)
        return self.kmeans.predict(sax_data)
        
    def _transform_to_sax(self, X: np.ndarray) -> np.ndarray:
        """
        Transform time series data to SAX representation.
        
        Args:
            X: Input time series data
            
        Returns:
            SAX transformed data
        """
        # Simplified SAX transformation
        # In practice, you would use a proper SAX library
        
        # Normalize data
        X_normalized = self._normalize_data(X)
        
        # Piecewise Aggregate Approximation (PAA)
        paa_data = self._paa_transform(X_normalized)
        
        # Discretize to SAX symbols (simplified)
        sax_data = self._discretize_to_symbols(paa_data)
        
        return sax_data
        
    def _normalize_data(self, X: np.ndarray) -> np.ndarray:
        """Normalize time series data."""
        if X.ndim == 3:
            X = X.squeeze(-1)  # Remove feature dimension if present
            
        normalized = np.zeros_like(X)
        for i in range(len(X)):
            series = X[i]
            mean_val = np.mean(series)
            std_val = np.std(series)
            if std_val > 1e-8:
                normalized[i] = (series - mean_val) / std_val
            else:
                normalized[i] = series - mean_val
                
        return normalized
        
    def _paa_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply Piecewise Aggregate Approximation."""
        n_samples, n_timesteps = X.shape
        segment_size = n_timesteps // self.word_size
        
        paa_data = np.zeros((n_samples, self.word_size))
        
        for i in range(n_samples):
            for j in range(self.word_size):
                start_idx = j * segment_size
                end_idx = min((j + 1) * segment_size, n_timesteps)
                paa_data[i, j] = np.mean(X[i, start_idx:end_idx])
                
        return paa_data
        
    def _discretize_to_symbols(self, paa_data: np.ndarray) -> np.ndarray:
        """Discretize PAA data to SAX symbols."""
        # Simplified discretization using quantiles
        # In practice, you would use proper SAX breakpoints
        
        n_samples, word_size = paa_data.shape
        sax_data = np.zeros((n_samples, word_size))
        
        for j in range(word_size):
            feature_data = paa_data[:, j]
            quantiles = np.linspace(0, 1, self.alphabet_size + 1)[1:-1]
            breakpoints = np.quantile(feature_data, quantiles)
            sax_data[:, j] = np.digitize(feature_data, breakpoints)
            
        return sax_data


class TwoStageKMeans(BaseClusteringMethod):
    """
    Two-stage K-means clustering implementation.
    
    This class implements the two-stage clustering approach that first
    clusters by consumption patterns and then by peak timing.
    """
    
    def __init__(self,
                 n_clusters: int,
                 k_consumption: int = 3,
                 k_peaktime: int = 2,
                 random_state: Optional[int] = None):
        """
        Initialize two-stage K-means clustering.
        
        Args:
            n_clusters: Total number of clusters (should equal k_consumption * k_peaktime)
            k_consumption: Number of consumption-based clusters
            k_peaktime: Number of peak-time-based clusters
            random_state: Random state for reproducibility
        """
        super().__init__(n_clusters, random_state)
        self.k_consumption = k_consumption
        self.k_peaktime = k_peaktime
        
        if n_clusters != k_consumption * k_peaktime:
            self.log_warning(
                f"n_clusters ({n_clusters}) != k_consumption * k_peaktime "
                f"({k_consumption * k_peaktime}). Using calculated value."
            )
            self.n_clusters = k_consumption * k_peaktime
            
        self.consumption_kmeans = None
        self.peaktime_kmeans = {}
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'TwoStageKMeans':
        """
        Fit two-stage K-means to the data.
        
        Args:
            X: Input time series data
            y: Ignored, present for API consistency
            
        Returns:
            self: Fitted two-stage K-means object
        """
        self.log_info(f"Fitting two-stage K-means with {self.n_clusters} total clusters")
        
        # Stage 1: Cluster by consumption patterns (integral features)
        integral_features = self._compute_integral_features(X)
        
        self.consumption_kmeans = KMeans(
            n_clusters=self.k_consumption,
            random_state=self.random_state,
            n_init='auto'
        )
        
        consumption_labels = self.consumption_kmeans.fit_predict(integral_features)
        
        # Stage 2: Within each consumption cluster, cluster by peak timing
        final_labels = np.zeros(len(X), dtype=int)
        
        for consumption_cluster in range(self.k_consumption):
            # Get data points in this consumption cluster
            cluster_mask = consumption_labels == consumption_cluster
            cluster_data = X[cluster_mask]
            
            if len(cluster_data) == 0:
                continue
                
            # Extract peak timing features
            peak_features = self._extract_peak_features(cluster_data)
            
            # Cluster by peak timing
            peaktime_kmeans = KMeans(
                n_clusters=self.k_peaktime,
                random_state=self.random_state,
                n_init='auto'
            )
            
            peaktime_labels = peaktime_kmeans.fit_predict(peak_features)
            self.peaktime_kmeans[consumption_cluster] = peaktime_kmeans
            
            # Assign final labels
            base_label = consumption_cluster * self.k_peaktime
            final_labels[cluster_mask] = base_label + peaktime_labels
            
        self.labels_ = final_labels
        self.is_fitted_ = True
        
        self.log_info("Two-stage K-means fitting completed")
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Input time series data
            
        Returns:
            Cluster labels
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        # Stage 1: Predict consumption cluster
        integral_features = self._compute_integral_features(X)
        consumption_labels = self.consumption_kmeans.predict(integral_features)
        
        # Stage 2: Predict peak timing cluster within each consumption cluster
        final_labels = np.zeros(len(X), dtype=int)
        
        for consumption_cluster in range(self.k_consumption):
            cluster_mask = consumption_labels == consumption_cluster
            cluster_data = X[cluster_mask]
            
            if len(cluster_data) == 0:
                continue
                
            if consumption_cluster in self.peaktime_kmeans:
                peak_features = self._extract_peak_features(cluster_data)
                peaktime_labels = self.peaktime_kmeans[consumption_cluster].predict(peak_features)
                
                base_label = consumption_cluster * self.k_peaktime
                final_labels[cluster_mask] = base_label + peaktime_labels
                
        return final_labels
        
    def _compute_integral_features(self, X: np.ndarray) -> np.ndarray:
        """
        Compute integral-based features for consumption clustering.
        
        Args:
            X: Input time series data
            
        Returns:
            Integral features
        """
        if X.ndim == 3:
            X = X.squeeze(-1)  # Remove feature dimension if present
            
        # Normalize by total consumption
        total_consumption = np.sum(X, axis=1, keepdims=True)
        X_normalized = X / (total_consumption + 1e-8)
        
        # Compute cumulative integral
        integral_features = np.cumsum(X_normalized, axis=1)
        
        # Add max power as additional feature
        max_power = np.max(X, axis=1, keepdims=True)
        
        # Combine integral sequence with max power
        features = np.concatenate([integral_features, max_power], axis=1)
        
        return features
        
    def _extract_peak_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract peak timing features for the second stage clustering.
        
        Args:
            X: Input time series data
            
        Returns:
            Peak timing features
        """
        if X.ndim == 3:
            X = X.squeeze(-1)  # Remove feature dimension if present
            
        # Simple peak detection: find time of maximum value
        peak_times = np.argmax(X, axis=1)
        
        # Create features based on peak timing
        n_timesteps = X.shape[1]
        peak_features = np.zeros((len(X), n_timesteps))
        
        for i, peak_time in enumerate(peak_times):
            # Create a feature vector with peak timing information
            peak_features[i, peak_time] = 1.0
            
            # Add some context around the peak
            for offset in [-1, 1]:
                idx = peak_time + offset
                if 0 <= idx < n_timesteps:
                    peak_features[i, idx] = 0.5
                    
        return peak_features


class IntegralKMeans(BaseClusteringMethod):
    """
    Integral-based K-means clustering.
    
    This class implements clustering based on integral features
    of load shapes as described in the original paper.
    """
    
    def __init__(self,
                 n_clusters: int,
                 dx: float = 0.25,
                 random_state: Optional[int] = None):
        """
        Initialize integral K-means clustering.
        
        Args:
            n_clusters: Number of clusters
            dx: Time interval granularity
            random_state: Random state for reproducibility
        """
        super().__init__(n_clusters, random_state)
        self.dx = dx
        self.kmeans = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'IntegralKMeans':
        """
        Fit integral K-means to the data.
        
        Args:
            X: Input time series data
            y: Ignored, present for API consistency
            
        Returns:
            self: Fitted integral K-means object
        """
        self.log_info(f"Fitting integral K-means with {self.n_clusters} clusters")
        
        # Compute integral features
        integral_features = self._compute_integral_sequence(X)
        
        # Apply K-means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init='auto'
        )
        
        self.labels_ = self.kmeans.fit_predict(integral_features)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        self.is_fitted_ = True
        
        self.log_info("Integral K-means fitting completed")
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Input time series data
            
        Returns:
            Cluster labels
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        integral_features = self._compute_integral_sequence(X)
        return self.kmeans.predict(integral_features)
        
    def _compute_integral_sequence(self, X: np.ndarray) -> np.ndarray:
        """
        Compute integral sequence for each time series.
        
        Args:
            X: Input time series data
            
        Returns:
            Integral sequences
        """
        if X.ndim == 3:
            X = X.squeeze(-1)  # Remove feature dimension if present
            
        # Normalize by total power
        total_power = np.sum(X, axis=1, keepdims=True)
        X_normalized = X / (total_power + 1e-8)
        
        # Compute integral sequence using trapezoidal rule
        integral_sequences = np.zeros_like(X_normalized)
        
        for i in range(len(X)):
            integral_sequences[i, 0] = 0
            for j in range(1, X.shape[1]):
                # Trapezoidal integration
                integral_sequences[i, j] = (
                    integral_sequences[i, j-1] + 
                    0.5 * (X_normalized[i, j-1] + X_normalized[i, j]) * self.dx
                )
                
        # Add max power as additional feature
        max_power = np.max(X, axis=1, keepdims=True)
        
        # Combine integral sequence with max power
        features = np.concatenate([integral_sequences, max_power], axis=1)
        
        return features
