"""
Evaluation metrics for clustering performance assessment.

This module provides comprehensive metrics for evaluating clustering
performance including internal validation, external validation,
and uncertainty quantification.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)
from sklearn.metrics.pairwise import pairwise_distances
import scipy.stats as stats

from ..utils.logging import LoggerMixin


class ClusteringMetrics(LoggerMixin):
    """
    Comprehensive clustering evaluation metrics.

    This class provides various metrics for evaluating clustering performance
    including both internal and external validation measures.
    """

    def __init__(self):
        """Initialize clustering metrics calculator."""
        self.internal_metrics = [
            "silhouette_score",
            "davies_bouldin_score",
            "calinski_harabasz_score",
            "inertia",
            "dunn_index",
        ]

        self.external_metrics = [
            "adjusted_rand_score",
            "normalized_mutual_info_score",
            "adjusted_mutual_info_score",
            "homogeneity_score",
            "completeness_score",
            "v_measure_score",
        ]

    def compute_internal_metrics(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        cluster_centers: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute internal validation metrics.

        Args:
            X: Data points used for clustering
            labels: Cluster labels
            cluster_centers: Cluster centroids (optional)

        Returns:
            Dictionary of internal validation metrics
        """
        metrics = {}

        # Check if we have enough clusters
        n_clusters = len(np.unique(labels))
        if n_clusters < 2:
            self.log_warning("Less than 2 clusters found, some metrics will be NaN")
            return {metric: np.nan for metric in self.internal_metrics}

        try:
            # Silhouette Score
            metrics["silhouette_score"] = silhouette_score(X, labels)

            # Davies-Bouldin Index
            metrics["davies_bouldin_score"] = davies_bouldin_score(X, labels)

            # Calinski-Harabasz Index
            metrics["calinski_harabasz_score"] = calinski_harabasz_score(X, labels)

            # Inertia (Within-cluster sum of squares)
            metrics["inertia"] = self._compute_inertia(X, labels, cluster_centers)

            # Dunn Index
            metrics["dunn_index"] = self._compute_dunn_index(X, labels)

        except Exception as e:
            self.log_error(f"Error computing internal metrics: {e}")
            metrics = {metric: np.nan for metric in self.internal_metrics}

        return metrics

    def compute_external_metrics(
        self, true_labels: np.ndarray, pred_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute external validation metrics.

        Args:
            true_labels: Ground truth cluster labels
            pred_labels: Predicted cluster labels

        Returns:
            Dictionary of external validation metrics
        """
        metrics = {}

        try:
            # Adjusted Rand Index
            metrics["adjusted_rand_score"] = adjusted_rand_score(
                true_labels, pred_labels
            )

            # Normalized Mutual Information
            metrics["normalized_mutual_info_score"] = normalized_mutual_info_score(
                true_labels, pred_labels
            )

            # Adjusted Mutual Information
            metrics["adjusted_mutual_info_score"] = adjusted_mutual_info_score(
                true_labels, pred_labels
            )

            # Homogeneity Score
            metrics["homogeneity_score"] = homogeneity_score(true_labels, pred_labels)

            # Completeness Score
            metrics["completeness_score"] = completeness_score(true_labels, pred_labels)

            # V-measure Score
            metrics["v_measure_score"] = v_measure_score(true_labels, pred_labels)

        except Exception as e:
            self.log_error(f"Error computing external metrics: {e}")
            metrics = {metric: np.nan for metric in self.external_metrics}

        return metrics

    def _compute_inertia(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        cluster_centers: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute within-cluster sum of squares (inertia).

        Args:
            X: Data points
            labels: Cluster labels
            cluster_centers: Cluster centroids

        Returns:
            Inertia value
        """
        if cluster_centers is None:
            # Compute centroids
            unique_labels = np.unique(labels)
            cluster_centers = np.array(
                [X[labels == label].mean(axis=0) for label in unique_labels]
            )

        inertia = 0.0
        for i, label in enumerate(np.unique(labels)):
            cluster_points = X[labels == label]
            centroid = cluster_centers[i]
            inertia += np.sum((cluster_points - centroid) ** 2)

        return inertia

    def _compute_dunn_index(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute Dunn index (ratio of minimum inter-cluster distance to maximum intra-cluster distance).

        Args:
            X: Data points
            labels: Cluster labels

        Returns:
            Dunn index value
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        if n_clusters < 2:
            return np.nan

        # Compute inter-cluster distances (minimum distance between clusters)
        inter_cluster_distances = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                cluster_i = X[labels == unique_labels[i]]
                cluster_j = X[labels == unique_labels[j]]

                # Minimum distance between any two points in different clusters
                distances = pairwise_distances(cluster_i, cluster_j)
                inter_cluster_distances.append(np.min(distances))

        min_inter_cluster_dist = np.min(inter_cluster_distances)

        # Compute intra-cluster distances (maximum distance within clusters)
        max_intra_cluster_dist = 0.0
        for label in unique_labels:
            cluster_points = X[labels == label]
            if len(cluster_points) > 1:
                distances = pairwise_distances(cluster_points)
                max_intra_cluster_dist = max(max_intra_cluster_dist, np.max(distances))

        if max_intra_cluster_dist == 0:
            return np.inf

        return min_inter_cluster_dist / max_intra_cluster_dist

    def compute_stability_metrics(
        self,
        X: np.ndarray,
        clustering_method,
        n_runs: int = 10,
        sample_fraction: float = 0.8,
    ) -> Dict[str, float]:
        """
        Compute clustering stability metrics using bootstrap sampling.

        Args:
            X: Data points
            clustering_method: Clustering method with fit_predict interface
            n_runs: Number of bootstrap runs
            sample_fraction: Fraction of data to sample in each run

        Returns:
            Dictionary of stability metrics
        """
        self.log_info(f"Computing stability metrics with {n_runs} bootstrap runs")

        n_samples = len(X)
        sample_size = int(n_samples * sample_fraction)

        all_labels = []
        all_metrics = []

        for run in range(n_runs):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, size=sample_size, replace=False)
            X_sample = X[indices]

            # Fit clustering
            try:
                labels = clustering_method.fit_predict(X_sample)
                all_labels.append(labels)

                # Compute internal metrics for this run
                metrics = self.compute_internal_metrics(X_sample, labels)
                all_metrics.append(metrics)

            except Exception as e:
                self.log_warning(f"Error in bootstrap run {run}: {e}")
                continue

        if not all_metrics:
            return {"stability_mean": np.nan, "stability_std": np.nan}

        # Compute stability statistics
        stability_metrics = {}

        # Mean and standard deviation of metrics across runs
        for metric_name in self.internal_metrics:
            values = [m.get(metric_name, np.nan) for m in all_metrics]
            values = [v for v in values if not np.isnan(v)]

            if values:
                stability_metrics[f"{metric_name}_mean"] = np.mean(values)
                stability_metrics[f"{metric_name}_std"] = np.std(values)
            else:
                stability_metrics[f"{metric_name}_mean"] = np.nan
                stability_metrics[f"{metric_name}_std"] = np.nan

        # Overall stability measure (coefficient of variation of silhouette scores)
        sil_scores = [m.get("silhouette_score", np.nan) for m in all_metrics]
        sil_scores = [s for s in sil_scores if not np.isnan(s)]

        if sil_scores and np.mean(sil_scores) != 0:
            stability_metrics["stability_coefficient"] = np.std(sil_scores) / np.mean(
                sil_scores
            )
        else:
            stability_metrics["stability_coefficient"] = np.nan

        return stability_metrics


class UncertaintyMetrics(LoggerMixin):
    """
    Metrics for quantifying uncertainty in clustering assignments.

    This class provides methods to assess the confidence and uncertainty
    of cluster assignments, particularly useful for Bayesian clustering methods.
    """

    def __init__(self):
        """Initialize uncertainty metrics calculator."""
        pass

    def compute_entropy_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Compute entropy-based uncertainty for cluster assignments.

        Args:
            probabilities: Cluster assignment probabilities (n_samples, n_clusters)

        Returns:
            Uncertainty scores for each sample
        """
        # Compute entropy
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)

        # Normalize by maximum possible entropy
        n_clusters = probabilities.shape[1]
        max_entropy = np.log(n_clusters)

        normalized_entropy = entropy / max_entropy
        return normalized_entropy

    def compute_confidence_intervals(
        self, probabilities: np.ndarray, confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute confidence intervals for cluster assignments.

        Args:
            probabilities: Cluster assignment probabilities
            confidence_level: Confidence level for intervals

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        alpha = 1 - confidence_level

        # Use beta distribution approximation for probability confidence intervals
        # This is a simplified approach - more sophisticated methods could be used

        lower_bounds = np.zeros_like(probabilities)
        upper_bounds = np.zeros_like(probabilities)

        for i in range(probabilities.shape[0]):
            for j in range(probabilities.shape[1]):
                p = probabilities[i, j]

                # Beta distribution parameters (simplified)
                alpha_param = p * 100 + 1  # Pseudo-count approach
                beta_param = (1 - p) * 100 + 1

                lower_bounds[i, j] = stats.beta.ppf(alpha / 2, alpha_param, beta_param)
                upper_bounds[i, j] = stats.beta.ppf(
                    1 - alpha / 2, alpha_param, beta_param
                )

        return lower_bounds, upper_bounds

    def compute_prediction_strength(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        test_fraction: float = 0.5,
        n_iterations: int = 100,
    ) -> float:
        """
        Compute prediction strength for assessing cluster stability.

        Args:
            X: Data points
            labels: Cluster labels
            test_fraction: Fraction of data to use for testing
            n_iterations: Number of iterations for stability assessment

        Returns:
            Prediction strength score
        """
        n_samples = len(X)
        test_size = int(n_samples * test_fraction)

        prediction_strengths = []

        for _ in range(n_iterations):
            # Split data
            indices = np.random.permutation(n_samples)
            train_indices = indices[:-test_size]
            test_indices = indices[-test_size:]

            X_train, X_test = X[train_indices], X[test_indices]
            labels_train, labels_test = labels[train_indices], labels[test_indices]

            # Compute prediction strength for this split
            strength = self._compute_single_prediction_strength(
                X_train, X_test, labels_train, labels_test
            )
            prediction_strengths.append(strength)

        return np.mean(prediction_strengths)

    def _compute_single_prediction_strength(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        labels_train: np.ndarray,
        labels_test: np.ndarray,
    ) -> float:
        """
        Compute prediction strength for a single train/test split.

        Args:
            X_train: Training data
            X_test: Test data
            labels_train: Training labels
            labels_test: Test labels

        Returns:
            Prediction strength for this split
        """
        # This is a simplified implementation
        # In practice, you would predict test labels based on training clusters
        # and measure how well the cluster structure is preserved

        unique_labels = np.unique(labels_train)
        n_clusters = len(unique_labels)

        if n_clusters < 2:
            return 0.0

        # Compute cluster centroids from training data
        centroids = np.array(
            [X_train[labels_train == label].mean(axis=0) for label in unique_labels]
        )

        # Predict test labels based on nearest centroids
        distances = pairwise_distances(X_test, centroids)
        predicted_labels = unique_labels[np.argmin(distances, axis=1)]

        # Compute agreement between predicted and actual test labels
        agreement = np.mean(predicted_labels == labels_test)

        return agreement


def calculate_clustering_metrics(
    true_labels: np.ndarray, pred_labels: np.ndarray, data: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive clustering metrics.

    This is a convenience function that computes both internal and external
    clustering metrics using the ClusteringMetrics class.

    Args:
        true_labels: Ground truth cluster labels
        pred_labels: Predicted cluster labels
        data: Optional data matrix for internal metrics

    Returns:
        Dictionary containing all clustering metrics
    """
    metrics_calculator = ClusteringMetrics()

    # Compute external metrics (require true labels)
    external_metrics = metrics_calculator.compute_external_metrics(
        true_labels, pred_labels
    )

    # Compute internal metrics if data is provided
    if data is not None:
        try:
            internal_metrics = metrics_calculator.compute_internal_metrics(
                data, pred_labels
            )
            external_metrics.update(internal_metrics)
        except Exception as e:
            # If internal metrics fail, just use a default silhouette score
            external_metrics["silhouette_score"] = 0.0

    # Add accuracy metric (using Hungarian algorithm for optimal assignment)
    from scipy.optimize import linear_sum_assignment

    # Create confusion matrix
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(pred_labels)

    confusion_matrix = np.zeros((len(unique_true), len(unique_pred)))
    for i, true_label in enumerate(unique_true):
        for j, pred_label in enumerate(unique_pred):
            confusion_matrix[i, j] = np.sum(
                (true_labels == true_label) & (pred_labels == pred_label)
            )

    # Use Hungarian algorithm to find optimal assignment
    row_indices, col_indices = linear_sum_assignment(-confusion_matrix)

    # Calculate accuracy
    accuracy = confusion_matrix[row_indices, col_indices].sum() / len(true_labels)
    external_metrics["accuracy"] = accuracy

    # Rename some metrics for consistency
    external_metrics["nmi"] = external_metrics["normalized_mutual_info_score"]
    external_metrics["ari"] = external_metrics["adjusted_rand_score"]

    return external_metrics
