"""
Advanced Household Energy Segmentation with Multi-Modal Deep Learning

This package provides novel deep learning approaches for household energy consumption 
segmentation, extending traditional clustering methods with weather-aware neural 
networks and advanced attention mechanisms.
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from . import clustering
from . import evaluation
from . import utils
from . import data

# Note: models module requires PyTorch
try:
    from . import models

    __all__ = ["clustering", "evaluation", "utils", "data", "models"]
except ImportError:
    __all__ = ["clustering", "evaluation", "utils", "data"]
