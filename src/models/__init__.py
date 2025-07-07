"""
PyTorch models for CausalHES framework.

This module provides:
- Base autoencoder architectures
- Clustering layers and components
- CausalHES model implementation
"""

# Note: PyTorch is required for these imports
try:
    import torch

    TORCH_AVAILABLE = True

    from .base import BaseAutoencoder
    from .clustering_layers import ClusteringLayer
    from .causal_hes_model import CausalHESModel

    __all__ = ["BaseAutoencoder", "ClusteringLayer", "CausalHESModel"]

except ImportError:
    TORCH_AVAILABLE = False
    __all__ = []

    def _torch_not_available(*args, **kwargs):
        raise ImportError(
            "PyTorch is required but not installed. Please install with: pip install torch torchvision torchaudio"
        )

    # Create placeholder classes that raise informative errors
    BaseAutoencoder = _torch_not_available
    ClusteringLayer = _torch_not_available
    CausalHESModel = _torch_not_available
