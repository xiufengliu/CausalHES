"""
Advanced Household Energy Segmentation with Multi-Modal Deep Learning

This package provides novel deep learning approaches for household energy consumption 
segmentation, extending traditional clustering methods with weather-aware neural 
networks and advanced attention mechanisms.
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from . import models
from . import clustering  
from . import data
from . import evaluation
from . import utils

__all__ = [
    "models",
    "clustering", 
    "data",
    "evaluation",
    "utils"
]
