"""
Evaluation metrics and frameworks for clustering performance assessment.

This module provides:
- Internal clustering validation metrics
- External validation with ground truth
- Statistical significance testing
- Visualization utilities for cluster analysis
"""

from .metrics import ClusteringMetrics, UncertaintyMetrics
from .evaluator import ClusteringEvaluator
from .statistical_tests import StatisticalTester
from .visualization import ClusterVisualizer

__all__ = [
    "ClusteringMetrics",
    "UncertaintyMetrics",
    "ClusteringEvaluator",
    "StatisticalTester", 
    "ClusterVisualizer"
]
