"""
Statistical significance testing for clustering evaluation.

This module provides statistical tests to assess the significance
of clustering results and compare different clustering methods.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import scipy.stats as stats
from sklearn.metrics import silhouette_score
from sklearn.utils import resample

from ..utils.logging import LoggerMixin


class StatisticalTester(LoggerMixin):
    """
    Statistical significance testing for clustering evaluation.
    
    This class provides various statistical tests to assess clustering
    quality and compare different clustering methods.
    """
    
    def __init__(self, 
                 significance_level: float = 0.05,
                 n_bootstrap: int = 1000):
        """
        Initialize statistical tester.
        
        Args:
            significance_level: Significance level for hypothesis tests
            n_bootstrap: Number of bootstrap samples for resampling tests
        """
        self.significance_level = significance_level
        self.n_bootstrap = n_bootstrap
        
    def test_clustering_significance(self, 
                                   X: np.ndarray, 
                                   labels: np.ndarray) -> Dict[str, Any]:
        """
        Test statistical significance of clustering results.
        
        Args:
            X: Data points
            labels: Cluster labels
            
        Returns:
            Dictionary of statistical test results
        """
        results = {}
        
        # Test against random clustering
        results['random_clustering_test'] = self._test_against_random_clustering(X, labels)
        
        # Test cluster separation
        results['cluster_separation_test'] = self._test_cluster_separation(X, labels)
        
        # Bootstrap confidence intervals for silhouette score
        results['silhouette_bootstrap'] = self._bootstrap_silhouette_score(X, labels)
        
        # Hopkins statistic for clustering tendency
        results['hopkins_statistic'] = self._compute_hopkins_statistic(X)
        
        return results
        
    def compare_clustering_methods(self,
                                 X: np.ndarray,
                                 labels_list: List[np.ndarray],
                                 method_names: List[str]) -> Dict[str, Any]:
        """
        Compare multiple clustering methods using statistical tests.
        
        Args:
            X: Data points
            labels_list: List of cluster labels from different methods
            method_names: Names of clustering methods
            
        Returns:
            Dictionary of comparison results
        """
        if len(labels_list) != len(method_names):
            raise ValueError("Number of label arrays must match number of method names")
            
        results = {
            'pairwise_comparisons': {},
            'overall_ranking': None,
            'friedman_test': None
        }
        
        # Pairwise comparisons
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                method1, method2 = method_names[i], method_names[j]
                labels1, labels2 = labels_list[i], labels_list[j]
                
                comparison_key = f"{method1}_vs_{method2}"
                results['pairwise_comparisons'][comparison_key] = self._compare_two_methods(
                    X, labels1, labels2, method1, method2
                )
                
        # Friedman test for overall comparison (if more than 2 methods)
        if len(method_names) > 2:
            results['friedman_test'] = self._friedman_test(X, labels_list, method_names)
            
        return results
        
    def _test_against_random_clustering(self, 
                                      X: np.ndarray, 
                                      labels: np.ndarray) -> Dict[str, Any]:
        """
        Test clustering quality against random clustering baseline.
        
        Args:
            X: Data points
            labels: Cluster labels
            
        Returns:
            Test results
        """
        n_clusters = len(np.unique(labels))
        actual_silhouette = silhouette_score(X, labels)
        
        # Generate random clusterings and compute silhouette scores
        random_silhouettes = []
        for _ in range(self.n_bootstrap):
            random_labels = np.random.randint(0, n_clusters, size=len(labels))
            try:
                random_sil = silhouette_score(X, random_labels)
                random_silhouettes.append(random_sil)
            except:
                continue
                
        if not random_silhouettes:
            return {'p_value': np.nan, 'significant': False, 'effect_size': np.nan}
            
        # Compute p-value (proportion of random scores >= actual score)
        p_value = np.mean(np.array(random_silhouettes) >= actual_silhouette)
        
        # Effect size (Cohen's d)
        effect_size = (actual_silhouette - np.mean(random_silhouettes)) / np.std(random_silhouettes)
        
        return {
            'actual_silhouette': actual_silhouette,
            'random_silhouette_mean': np.mean(random_silhouettes),
            'random_silhouette_std': np.std(random_silhouettes),
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'effect_size': effect_size
        }
        
    def _test_cluster_separation(self, 
                               X: np.ndarray, 
                               labels: np.ndarray) -> Dict[str, Any]:
        """
        Test statistical significance of cluster separation.
        
        Args:
            X: Data points
            labels: Cluster labels
            
        Returns:
            Test results
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            return {'p_value': np.nan, 'significant': False}
            
        # Perform ANOVA test for each feature
        feature_p_values = []
        
        for feature_idx in range(X.shape[1]):
            feature_data = X[:, feature_idx]
            groups = [feature_data[labels == label] for label in unique_labels]
            
            # Remove empty groups
            groups = [group for group in groups if len(group) > 0]
            
            if len(groups) >= 2:
                try:
                    f_stat, p_value = stats.f_oneway(*groups)
                    feature_p_values.append(p_value)
                except:
                    continue
                    
        if not feature_p_values:
            return {'p_value': np.nan, 'significant': False}
            
        # Use Bonferroni correction for multiple testing
        corrected_alpha = self.significance_level / len(feature_p_values)
        significant_features = np.sum(np.array(feature_p_values) < corrected_alpha)
        
        # Overall p-value using Fisher's method
        chi2_stat = -2 * np.sum(np.log(feature_p_values))
        overall_p_value = 1 - stats.chi2.cdf(chi2_stat, 2 * len(feature_p_values))
        
        return {
            'feature_p_values': feature_p_values,
            'significant_features': significant_features,
            'total_features': len(feature_p_values),
            'overall_p_value': overall_p_value,
            'significant': overall_p_value < self.significance_level
        }
        
    def _bootstrap_silhouette_score(self, 
                                  X: np.ndarray, 
                                  labels: np.ndarray) -> Dict[str, Any]:
        """
        Compute bootstrap confidence intervals for silhouette score.
        
        Args:
            X: Data points
            labels: Cluster labels
            
        Returns:
            Bootstrap results
        """
        bootstrap_scores = []
        
        for _ in range(self.n_bootstrap):
            # Bootstrap sample
            indices = resample(range(len(X)), n_samples=len(X), replace=True)
            X_boot = X[indices]
            labels_boot = labels[indices]
            
            try:
                score = silhouette_score(X_boot, labels_boot)
                bootstrap_scores.append(score)
            except:
                continue
                
        if not bootstrap_scores:
            return {'confidence_interval': (np.nan, np.nan), 'mean': np.nan, 'std': np.nan}
            
        # Compute confidence interval
        alpha = self.significance_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_scores, lower_percentile)
        ci_upper = np.percentile(bootstrap_scores, upper_percentile)
        
        return {
            'confidence_interval': (ci_lower, ci_upper),
            'mean': np.mean(bootstrap_scores),
            'std': np.std(bootstrap_scores),
            'bootstrap_scores': bootstrap_scores
        }
        
    def _compute_hopkins_statistic(self, X: np.ndarray, n_samples: int = None) -> Dict[str, Any]:
        """
        Compute Hopkins statistic for clustering tendency.
        
        Args:
            X: Data points
            n_samples: Number of samples for Hopkins test
            
        Returns:
            Hopkins statistic results
        """
        if n_samples is None:
            n_samples = min(int(0.1 * len(X)), 100)
            
        n_samples = min(n_samples, len(X) - 1)
        
        # Sample random points from data
        sample_indices = np.random.choice(len(X), size=n_samples, replace=False)
        sampled_points = X[sample_indices]
        
        # Generate uniform random points in the same space
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        random_points = np.random.uniform(min_vals, max_vals, size=(n_samples, X.shape[1]))
        
        # Compute distances to nearest neighbors
        from sklearn.neighbors import NearestNeighbors
        
        # For sampled points (distance to nearest neighbor in original data)
        nbrs_data = NearestNeighbors(n_neighbors=2).fit(X)
        distances_data, _ = nbrs_data.kneighbors(sampled_points)
        u_distances = distances_data[:, 1]  # Distance to nearest neighbor (excluding self)
        
        # For random points (distance to nearest neighbor in original data)
        distances_random, _ = nbrs_data.kneighbors(random_points)
        w_distances = distances_random[:, 0]  # Distance to nearest neighbor
        
        # Compute Hopkins statistic
        hopkins_stat = np.sum(w_distances) / (np.sum(u_distances) + np.sum(w_distances))
        
        # Interpretation
        if hopkins_stat < 0.3:
            tendency = "highly clusterable"
        elif hopkins_stat < 0.5:
            tendency = "moderately clusterable"
        elif hopkins_stat < 0.7:
            tendency = "weakly clusterable"
        else:
            tendency = "not clusterable (uniform)"
            
        return {
            'hopkins_statistic': hopkins_stat,
            'clustering_tendency': tendency,
            'n_samples_used': n_samples
        }
        
    def _compare_two_methods(self,
                           X: np.ndarray,
                           labels1: np.ndarray,
                           labels2: np.ndarray,
                           method1_name: str,
                           method2_name: str) -> Dict[str, Any]:
        """
        Compare two clustering methods using statistical tests.
        
        Args:
            X: Data points
            labels1: Labels from first method
            labels2: Labels from second method
            method1_name: Name of first method
            method2_name: Name of second method
            
        Returns:
            Comparison results
        """
        # Compute silhouette scores
        try:
            sil1 = silhouette_score(X, labels1)
            sil2 = silhouette_score(X, labels2)
        except:
            return {'error': 'Could not compute silhouette scores'}
            
        # Bootstrap test for difference in silhouette scores
        bootstrap_diffs = []
        
        for _ in range(self.n_bootstrap):
            indices = resample(range(len(X)), n_samples=len(X), replace=True)
            X_boot = X[indices]
            labels1_boot = labels1[indices]
            labels2_boot = labels2[indices]
            
            try:
                sil1_boot = silhouette_score(X_boot, labels1_boot)
                sil2_boot = silhouette_score(X_boot, labels2_boot)
                bootstrap_diffs.append(sil1_boot - sil2_boot)
            except:
                continue
                
        if not bootstrap_diffs:
            return {'error': 'Bootstrap test failed'}
            
        # Compute p-value for two-tailed test
        observed_diff = sil1 - sil2
        p_value = 2 * min(
            np.mean(np.array(bootstrap_diffs) >= observed_diff),
            np.mean(np.array(bootstrap_diffs) <= observed_diff)
        )
        
        # Effect size
        effect_size = observed_diff / np.std(bootstrap_diffs)
        
        return {
            f'{method1_name}_silhouette': sil1,
            f'{method2_name}_silhouette': sil2,
            'difference': observed_diff,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'effect_size': effect_size,
            'better_method': method1_name if sil1 > sil2 else method2_name
        }
        
    def _friedman_test(self,
                      X: np.ndarray,
                      labels_list: List[np.ndarray],
                      method_names: List[str]) -> Dict[str, Any]:
        """
        Perform Friedman test for comparing multiple clustering methods.
        
        Args:
            X: Data points
            labels_list: List of cluster labels from different methods
            method_names: Names of clustering methods
            
        Returns:
            Friedman test results
        """
        # Compute silhouette scores for all methods
        silhouette_scores = []
        
        for labels in labels_list:
            try:
                score = silhouette_score(X, labels)
                silhouette_scores.append(score)
            except:
                silhouette_scores.append(np.nan)
                
        # Check if we have valid scores
        if np.any(np.isnan(silhouette_scores)):
            return {'error': 'Some methods produced invalid silhouette scores'}
            
        # For Friedman test, we need multiple "blocks" (datasets)
        # Since we only have one dataset, we'll use bootstrap samples
        bootstrap_scores = []
        
        for _ in range(min(self.n_bootstrap, 100)):  # Limit for computational efficiency
            indices = resample(range(len(X)), n_samples=len(X), replace=True)
            X_boot = X[indices]
            
            boot_scores = []
            for labels in labels_list:
                labels_boot = labels[indices]
                try:
                    score = silhouette_score(X_boot, labels_boot)
                    boot_scores.append(score)
                except:
                    boot_scores.append(np.nan)
                    
            if not np.any(np.isnan(boot_scores)):
                bootstrap_scores.append(boot_scores)
                
        if len(bootstrap_scores) < 10:
            return {'error': 'Insufficient valid bootstrap samples for Friedman test'}
            
        # Perform Friedman test
        try:
            statistic, p_value = stats.friedmanchisquare(*zip(*bootstrap_scores))
            
            # Rank methods by average performance
            avg_scores = np.mean(bootstrap_scores, axis=0)
            ranking = sorted(zip(method_names, avg_scores), key=lambda x: x[1], reverse=True)
            
            return {
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < self.significance_level,
                'ranking': ranking,
                'average_scores': dict(zip(method_names, avg_scores))
            }
            
        except Exception as e:
            return {'error': f'Friedman test failed: {str(e)}'}
