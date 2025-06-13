"""
Comprehensive clustering evaluation framework.

This module provides a unified interface for evaluating clustering
performance using multiple metrics and statistical tests.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import json
from datetime import datetime

from .metrics import ClusteringMetrics, UncertaintyMetrics
from .statistical_tests import StatisticalTester
from ..utils.logging import LoggerMixin
from ..utils.helpers import save_json, ensure_dir


class ClusteringEvaluator(LoggerMixin):
    """
    Comprehensive clustering evaluation framework.
    
    This class provides a unified interface for evaluating clustering
    performance using internal metrics, external validation, uncertainty
    quantification, and statistical significance testing.
    """
    
    def __init__(self, 
                 include_uncertainty: bool = True,
                 include_stability: bool = True,
                 include_statistical_tests: bool = True):
        """
        Initialize clustering evaluator.
        
        Args:
            include_uncertainty: Whether to compute uncertainty metrics
            include_stability: Whether to compute stability metrics
            include_statistical_tests: Whether to perform statistical tests
        """
        self.include_uncertainty = include_uncertainty
        self.include_stability = include_stability
        self.include_statistical_tests = include_statistical_tests
        
        self.clustering_metrics = ClusteringMetrics()
        self.uncertainty_metrics = UncertaintyMetrics()
        self.statistical_tester = StatisticalTester() if include_statistical_tests else None
        
        self.results = {}
        
    def evaluate(self,
                 X: np.ndarray,
                 labels: np.ndarray,
                 true_labels: Optional[np.ndarray] = None,
                 probabilities: Optional[np.ndarray] = None,
                 cluster_centers: Optional[np.ndarray] = None,
                 clustering_method=None,
                 method_name: str = "clustering_method") -> Dict[str, Any]:
        """
        Perform comprehensive clustering evaluation.
        
        Args:
            X: Data points used for clustering
            labels: Predicted cluster labels
            true_labels: Ground truth labels (optional)
            probabilities: Cluster assignment probabilities (optional)
            cluster_centers: Cluster centroids (optional)
            clustering_method: Clustering method object for stability testing
            method_name: Name of the clustering method
            
        Returns:
            Dictionary containing all evaluation results
        """
        self.log_info(f"Starting comprehensive evaluation for {method_name}")
        
        results = {
            'method_name': method_name,
            'evaluation_timestamp': datetime.now().isoformat(),
            'data_shape': X.shape,
            'n_clusters': len(np.unique(labels)),
            'cluster_distribution': self._get_cluster_distribution(labels)
        }
        
        # Internal validation metrics
        self.log_info("Computing internal validation metrics...")
        internal_metrics = self.clustering_metrics.compute_internal_metrics(
            X, labels, cluster_centers
        )
        results['internal_metrics'] = internal_metrics
        
        # External validation metrics (if ground truth available)
        if true_labels is not None:
            self.log_info("Computing external validation metrics...")
            external_metrics = self.clustering_metrics.compute_external_metrics(
                true_labels, labels
            )
            results['external_metrics'] = external_metrics
        else:
            results['external_metrics'] = None
            
        # Uncertainty metrics (if probabilities available)
        if self.include_uncertainty and probabilities is not None:
            self.log_info("Computing uncertainty metrics...")
            uncertainty_results = self._evaluate_uncertainty(probabilities)
            results['uncertainty_metrics'] = uncertainty_results
        else:
            results['uncertainty_metrics'] = None
            
        # Stability metrics (if clustering method available)
        if self.include_stability and clustering_method is not None:
            self.log_info("Computing stability metrics...")
            try:
                stability_metrics = self.clustering_metrics.compute_stability_metrics(
                    X, clustering_method
                )
                results['stability_metrics'] = stability_metrics
            except Exception as e:
                self.log_warning(f"Could not compute stability metrics: {e}")
                results['stability_metrics'] = None
        else:
            results['stability_metrics'] = None
            
        # Statistical significance tests
        if self.include_statistical_tests and self.statistical_tester is not None:
            self.log_info("Performing statistical significance tests...")
            try:
                statistical_results = self.statistical_tester.test_clustering_significance(
                    X, labels
                )
                results['statistical_tests'] = statistical_results
            except Exception as e:
                self.log_warning(f"Could not perform statistical tests: {e}")
                results['statistical_tests'] = None
        else:
            results['statistical_tests'] = None
            
        # Store results
        self.results[method_name] = results
        
        self.log_info(f"Evaluation completed for {method_name}")
        return results
        
    def compare_methods(self, 
                       results_list: List[Dict[str, Any]],
                       primary_metric: str = 'silhouette_score') -> pd.DataFrame:
        """
        Compare multiple clustering methods.
        
        Args:
            results_list: List of evaluation results from different methods
            primary_metric: Primary metric for ranking methods
            
        Returns:
            DataFrame comparing methods across all metrics
        """
        self.log_info(f"Comparing {len(results_list)} clustering methods")
        
        comparison_data = []
        
        for result in results_list:
            method_data = {
                'method': result['method_name'],
                'n_clusters': result['n_clusters']
            }
            
            # Add internal metrics
            if result['internal_metrics']:
                for metric, value in result['internal_metrics'].items():
                    method_data[f'internal_{metric}'] = value
                    
            # Add external metrics
            if result['external_metrics']:
                for metric, value in result['external_metrics'].items():
                    method_data[f'external_{metric}'] = value
                    
            # Add uncertainty metrics
            if result['uncertainty_metrics']:
                for metric, value in result['uncertainty_metrics'].items():
                    method_data[f'uncertainty_{metric}'] = value
                    
            # Add stability metrics
            if result['stability_metrics']:
                for metric, value in result['stability_metrics'].items():
                    method_data[f'stability_{metric}'] = value
                    
            comparison_data.append(method_data)
            
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by primary metric (handle different metric directions)
        if primary_metric in ['davies_bouldin_score', 'inertia']:
            # Lower is better
            comparison_df = comparison_df.sort_values(
                f'internal_{primary_metric}', 
                ascending=True
            )
        else:
            # Higher is better
            comparison_df = comparison_df.sort_values(
                f'internal_{primary_metric}', 
                ascending=False
            )
            
        return comparison_df
        
    def _evaluate_uncertainty(self, probabilities: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate uncertainty in cluster assignments.
        
        Args:
            probabilities: Cluster assignment probabilities
            
        Returns:
            Dictionary of uncertainty metrics
        """
        uncertainty_results = {}
        
        # Entropy-based uncertainty
        entropy_uncertainty = self.uncertainty_metrics.compute_entropy_uncertainty(
            probabilities
        )
        uncertainty_results['mean_entropy_uncertainty'] = np.mean(entropy_uncertainty)
        uncertainty_results['std_entropy_uncertainty'] = np.std(entropy_uncertainty)
        uncertainty_results['max_entropy_uncertainty'] = np.max(entropy_uncertainty)
        
        # Confidence intervals
        lower_bounds, upper_bounds = self.uncertainty_metrics.compute_confidence_intervals(
            probabilities
        )
        
        # Average confidence interval width
        ci_widths = upper_bounds - lower_bounds
        uncertainty_results['mean_ci_width'] = np.mean(ci_widths)
        uncertainty_results['std_ci_width'] = np.std(ci_widths)
        
        # Proportion of high-confidence assignments (entropy < 0.5)
        high_confidence_prop = np.mean(entropy_uncertainty < 0.5)
        uncertainty_results['high_confidence_proportion'] = high_confidence_prop
        
        return uncertainty_results
        
    def _get_cluster_distribution(self, labels: np.ndarray) -> Dict[int, int]:
        """
        Get distribution of samples across clusters.
        
        Args:
            labels: Cluster labels
            
        Returns:
            Dictionary mapping cluster IDs to sample counts
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique_labels.tolist(), counts.tolist()))
        
    def generate_report(self, 
                       method_name: str,
                       output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            method_name: Name of the method to generate report for
            output_path: Path to save the report (optional)
            
        Returns:
            Report as string
        """
        if method_name not in self.results:
            raise ValueError(f"No results found for method: {method_name}")
            
        result = self.results[method_name]
        
        report_lines = [
            f"Clustering Evaluation Report",
            f"=" * 50,
            f"Method: {result['method_name']}",
            f"Evaluation Date: {result['evaluation_timestamp']}",
            f"Data Shape: {result['data_shape']}",
            f"Number of Clusters: {result['n_clusters']}",
            f"",
            f"Cluster Distribution:",
        ]
        
        for cluster_id, count in result['cluster_distribution'].items():
            report_lines.append(f"  Cluster {cluster_id}: {count} samples")
            
        report_lines.append("")
        
        # Internal metrics
        if result['internal_metrics']:
            report_lines.extend([
                "Internal Validation Metrics:",
                "-" * 30
            ])
            for metric, value in result['internal_metrics'].items():
                if not np.isnan(value):
                    report_lines.append(f"  {metric}: {value:.4f}")
                else:
                    report_lines.append(f"  {metric}: N/A")
            report_lines.append("")
            
        # External metrics
        if result['external_metrics']:
            report_lines.extend([
                "External Validation Metrics:",
                "-" * 30
            ])
            for metric, value in result['external_metrics'].items():
                if not np.isnan(value):
                    report_lines.append(f"  {metric}: {value:.4f}")
                else:
                    report_lines.append(f"  {metric}: N/A")
            report_lines.append("")
            
        # Uncertainty metrics
        if result['uncertainty_metrics']:
            report_lines.extend([
                "Uncertainty Metrics:",
                "-" * 20
            ])
            for metric, value in result['uncertainty_metrics'].items():
                if not np.isnan(value):
                    report_lines.append(f"  {metric}: {value:.4f}")
                else:
                    report_lines.append(f"  {metric}: N/A")
            report_lines.append("")
            
        # Stability metrics
        if result['stability_metrics']:
            report_lines.extend([
                "Stability Metrics:",
                "-" * 18
            ])
            for metric, value in result['stability_metrics'].items():
                if not np.isnan(value):
                    report_lines.append(f"  {metric}: {value:.4f}")
                else:
                    report_lines.append(f"  {metric}: N/A")
            report_lines.append("")
            
        # Statistical tests
        if result['statistical_tests']:
            report_lines.extend([
                "Statistical Significance Tests:",
                "-" * 32
            ])
            for test_name, test_result in result['statistical_tests'].items():
                if isinstance(test_result, dict):
                    report_lines.append(f"  {test_name}:")
                    for key, value in test_result.items():
                        report_lines.append(f"    {key}: {value}")
                else:
                    report_lines.append(f"  {test_name}: {test_result}")
            report_lines.append("")
            
        report = "\n".join(report_lines)
        
        # Save report if output path provided
        if output_path:
            output_path = Path(output_path)
            ensure_dir(output_path.parent)
            
            with open(output_path, 'w') as f:
                f.write(report)
                
            self.log_info(f"Report saved to: {output_path}")
            
        return report
        
    def save_results(self, output_path: Union[str, Path]) -> None:
        """
        Save all evaluation results to file.
        
        Args:
            output_path: Path to save results
        """
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        
        save_json(self.results, output_path)
        self.log_info(f"Results saved to: {output_path}")
        
    def load_results(self, input_path: Union[str, Path]) -> None:
        """
        Load evaluation results from file.
        
        Args:
            input_path: Path to load results from
        """
        input_path = Path(input_path)
        
        with open(input_path, 'r') as f:
            self.results = json.load(f)
            
        self.log_info(f"Results loaded from: {input_path}")
        
    def get_best_method(self, 
                       metric: str = 'silhouette_score',
                       higher_is_better: bool = True) -> Tuple[str, float]:
        """
        Get the best performing method based on a specific metric.
        
        Args:
            metric: Metric to use for comparison
            higher_is_better: Whether higher values are better for this metric
            
        Returns:
            Tuple of (best_method_name, best_metric_value)
        """
        if not self.results:
            raise ValueError("No evaluation results available")
            
        best_method = None
        best_value = float('-inf') if higher_is_better else float('inf')
        
        for method_name, result in self.results.items():
            # Look for metric in internal metrics first
            value = None
            if result['internal_metrics'] and metric in result['internal_metrics']:
                value = result['internal_metrics'][metric]
            elif result['external_metrics'] and metric in result['external_metrics']:
                value = result['external_metrics'][metric]
            elif result['uncertainty_metrics'] and metric in result['uncertainty_metrics']:
                value = result['uncertainty_metrics'][metric]
            elif result['stability_metrics'] and metric in result['stability_metrics']:
                value = result['stability_metrics'][metric]
                
            if value is not None and not np.isnan(value):
                if higher_is_better and value > best_value:
                    best_method = method_name
                    best_value = value
                elif not higher_is_better and value < best_value:
                    best_method = method_name
                    best_value = value
                    
        return best_method, best_value
