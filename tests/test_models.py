"""
Tests for PyTorch models in CausalHES framework.
"""

import pytest
import torch
import numpy as np
from src.models.base import BaseAutoencoder
from src.models.clustering_layers import ClusteringLayer
from src.losses import ReconstructionLoss, CSSAELoss


class TestBaseAutoencoder:
    """Test the base autoencoder class."""
    
    def test_initialization(self):
        """Test autoencoder initialization."""
        input_shape = (24, 1)
        embedding_dim = 32
        
        autoencoder = BaseAutoencoder(input_shape, embedding_dim)
        
        assert autoencoder.input_shape == input_shape
        assert autoencoder.embedding_dim == embedding_dim
        assert isinstance(autoencoder, torch.nn.Module)


class TestClusteringLayer:
    """Test the clustering layer."""
    
    def test_initialization(self):
        """Test clustering layer initialization."""
        n_clusters = 5
        embedding_dim = 32
        
        layer = ClusteringLayer(n_clusters, embedding_dim)
        
        assert layer.n_clusters == n_clusters
        assert layer.embedding_dim == embedding_dim
        assert layer.clusters.shape == (n_clusters, embedding_dim)
    
    def test_forward_pass(self):
        """Test forward pass through clustering layer."""
        n_clusters = 5
        embedding_dim = 32
        batch_size = 16
        
        layer = ClusteringLayer(n_clusters, embedding_dim)
        
        # Create dummy input
        x = torch.randn(batch_size, embedding_dim)
        
        # Forward pass
        output = layer(x)
        
        assert output.shape == (batch_size, n_clusters)
        
        # Check that outputs are probabilities (sum to 1)
        prob_sums = torch.sum(output, dim=1)
        assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-6)


class TestLossFunctions:
    """Test loss functions."""
    
    def test_reconstruction_loss(self):
        """Test reconstruction loss."""
        loss_fn = ReconstructionLoss()
        
        # Create dummy data
        y_true = torch.randn(16, 24, 1)
        y_pred = torch.randn(16, 24, 1)
        
        loss = loss_fn(y_true, y_pred)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0  # MSE loss should be non-negative
    
    def test_cssae_loss(self):
        """Test CSSAE loss function."""
        base_dim = 32
        weather_dim = 16
        
        loss_fn = CSSAELoss(base_dim, weather_dim)
        
        # Create dummy data
        y_true = torch.randn(16, 24, 1)
        outputs = {
            'total_reconstruction': torch.randn(16, 24, 1),
            'base_embedding': torch.randn(16, base_dim),
            'weather_embedding': torch.randn(16, weather_dim)
        }
        
        loss = loss_fn(y_true, outputs)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestDataGeneration:
    """Test data generation utilities."""
    
    def test_pecan_street_generator_import(self):
        """Test that Pecan Street generator can be imported."""
        try:
            from generate_pecan_street_style import PecanStreetStyleGenerator
            generator = PecanStreetStyleGenerator(random_state=42)
            assert generator.random_state == 42
        except ImportError:
            pytest.skip("Pecan Street generator not available")


if __name__ == "__main__":
    pytest.main([__file__])
