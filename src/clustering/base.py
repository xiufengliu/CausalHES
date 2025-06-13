"""
Base classes for clustering methods.

This module defines the common interface that all clustering methods should implement,
ensuring consistency across different approaches.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class BaseClusteringMethod(BaseEstimator, ClusterMixin, ABC):
    """
    Abstract base class for all clustering methods.
    
    This class defines the common interface that all clustering implementations
    should follow, ensuring consistency and interoperability.
    """
    
    def __init__(self, n_clusters: int, random_state: Optional[int] = None):
        """
        Initialize the clustering method.
        
        Args:
            n_clusters: Number of clusters to form
            random_state: Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None
        self.cluster_centers_ = None
        self.is_fitted_ = False
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> 'BaseClusteringMethod':
        """
        Fit the clustering model to the data.
        
        Args:
            X: Input data of shape (n_samples, n_features) or (n_samples, n_timesteps, n_features)
            y: Ignored, present for API consistency
            **kwargs: Additional arguments specific to the clustering method
            
        Returns:
            self: Fitted clustering object
        """
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Input data to predict clusters for
            **kwargs: Additional arguments specific to the clustering method
            
        Returns:
            Cluster labels for each sample
        """
        pass
        
    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Fit the model and predict cluster labels.
        
        Args:
            X: Input data
            y: Ignored, present for API consistency
            **kwargs: Additional arguments
            
        Returns:
            Cluster labels for each sample
        """
        return self.fit(X, y, **kwargs).predict(X, **kwargs)
        
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            'n_clusters': self.n_clusters,
            'random_state': self.random_state
        }
        
    def set_params(self, **params) -> 'BaseClusteringMethod':
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
        
    @property
    def n_features_in_(self) -> Optional[int]:
        """Number of features seen during fit."""
        return getattr(self, '_n_features_in', None)


class BaseDeepClusteringMethod(BaseClusteringMethod):
    """
    Base class for deep learning-based clustering methods.
    
    Extends the base clustering interface with deep learning specific functionality.
    """
    
    def __init__(self, n_clusters: int, embedding_dim: int = 10, 
                 random_state: Optional[int] = None):
        """
        Initialize the deep clustering method.
        
        Args:
            n_clusters: Number of clusters to form
            embedding_dim: Dimension of the learned embeddings
            random_state: Random state for reproducibility
        """
        super().__init__(n_clusters, random_state)
        self.embedding_dim = embedding_dim
        self.encoder_ = None
        self.embeddings_ = None
        
    @abstractmethod
    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """
        Get learned embeddings for the input data.
        
        Args:
            X: Input data
            
        Returns:
            Learned embeddings of shape (n_samples, embedding_dim)
        """
        pass
        
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        params = super().get_params(deep)
        params['embedding_dim'] = self.embedding_dim
        return params


class BaseMultiModalClusteringMethod(BaseDeepClusteringMethod):
    """
    Base class for multi-modal clustering methods.
    
    Handles clustering with multiple input modalities (e.g., load shapes + weather).
    """
    
    @abstractmethod
    def fit(self, X_primary: np.ndarray, X_secondary: Optional[np.ndarray] = None, 
            y: Optional[np.ndarray] = None, **kwargs) -> 'BaseMultiModalClusteringMethod':
        """
        Fit the clustering model to multi-modal data.
        
        Args:
            X_primary: Primary input data (e.g., load shapes)
            X_secondary: Secondary input data (e.g., weather data)
            y: Ignored, present for API consistency
            **kwargs: Additional arguments
            
        Returns:
            self: Fitted clustering object
        """
        pass
        
    @abstractmethod
    def predict(self, X_primary: np.ndarray, X_secondary: Optional[np.ndarray] = None, 
                **kwargs) -> np.ndarray:
        """
        Predict cluster labels for new multi-modal data.
        
        Args:
            X_primary: Primary input data
            X_secondary: Secondary input data
            **kwargs: Additional arguments
            
        Returns:
            Cluster labels for each sample
        """
        pass
        
    def fit_predict(self, X_primary: np.ndarray, X_secondary: Optional[np.ndarray] = None,
                    y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Fit the model and predict cluster labels for multi-modal data.
        
        Args:
            X_primary: Primary input data
            X_secondary: Secondary input data  
            y: Ignored, present for API consistency
            **kwargs: Additional arguments
            
        Returns:
            Cluster labels for each sample
        """
        return self.fit(X_primary, X_secondary, y, **kwargs).predict(X_primary, X_secondary, **kwargs)
