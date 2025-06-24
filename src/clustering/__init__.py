"""
Clustering algorithms for household energy segmentation.

This module provides various clustering approaches:
- Traditional methods (SAX K-means, Two-stage K-means, Integral K-means)
- Deep clustering methods (DEC, Weather-fused DEC)
- Weather normalization methods (Linear, Nonlinear, CDD/HDD)
- VAE-based methods (β-VAE, FactorVAE, Multi-modal VAE)
- Multi-modal clustering (Attention-fused, Contrastive, Graph-based)
- Causal inference methods (Doubly Robust, IV, Domain Adaptation)
- Novel approaches (MTFN, Bayesian clustering)
"""

from .traditional import SAXKMeans, TwoStageKMeans, IntegralKMeans
from .deep_clustering import DeepEmbeddedClustering, WeatherFusedDEC
from .weather_normalization import (
    LinearWeatherNormalization, 
    NonlinearWeatherNormalization,
    CDDHDDWeatherNormalization
)
from .vae_baselines import (
    BetaVAEClustering,
    FactorVAEClustering,
    MultiModalVAEClustering
)
from .multimodal_baselines import (
    AttentionFusedDEC,
    ContrastiveMVClustering,
    GraphMultiModalClustering
)
from .causal_baselines import (
    DoublyRobustClustering,
    InstrumentalVariableClustering,
    DomainAdaptationClustering
)
from .base import BaseClusteringMethod, BaseDeepClusteringMethod, BaseMultiModalClusteringMethod

__all__ = [
    # Base classes
    "BaseClusteringMethod",
    "BaseDeepClusteringMethod", 
    "BaseMultiModalClusteringMethod",
    
    # Traditional methods
    "SAXKMeans",
    "TwoStageKMeans", 
    "IntegralKMeans",
    
    # Deep clustering methods
    "DeepEmbeddedClustering",
    "WeatherFusedDEC",
    
    # Weather normalization methods
    "LinearWeatherNormalization",
    "NonlinearWeatherNormalization",
    "CDDHDDWeatherNormalization",
    
    # VAE-based methods
    "BetaVAEClustering",
    "FactorVAEClustering", 
    "MultiModalVAEClustering",
    
    # Multi-modal clustering methods
    "AttentionFusedDEC",
    "ContrastiveMVClustering",
    "GraphMultiModalClustering",
    
    # Causal inference methods
    "DoublyRobustClustering",
    "InstrumentalVariableClustering",
    "DomainAdaptationClustering"
]
