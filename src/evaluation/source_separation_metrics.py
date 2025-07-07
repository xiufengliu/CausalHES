"""
Source Separation Evaluation Metrics for CausalHES.

This module provides specialized metrics for evaluating the quality of
causal source separation in the CSSAE model, including:
1. Independence measures between base and weather embeddings
2. Reconstruction quality metrics
3. Source separation quality indicators
4. Causal constraint satisfaction measures
"""

import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from typing import Dict, Tuple, Optional
import logging

from ..utils.logging import get_logger


def calculate_mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 50) -> float:
    """
    Calculate mutual information between two continuous variables.

    Args:
        x: First variable
        y: Second variable
        bins: Number of bins for discretization

    Returns:
        Mutual information estimate
    """
    # Discretize continuous variables
    x_discrete = np.digitize(x.flatten(), np.histogram(x.flatten(), bins=bins)[1])
    y_discrete = np.digitize(y.flatten(), np.histogram(y.flatten(), bins=bins)[1])

    # Calculate mutual information
    mi = mutual_info_score(x_discrete, y_discrete)
    return mi


def calculate_distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate distance correlation between two variables.

    Distance correlation is zero if and only if the variables are independent.

    Args:
        x: First variable [n_samples, n_features]
        y: Second variable [n_samples, n_features]

    Returns:
        Distance correlation
    """

    def _distance_matrix(data):
        """Compute pairwise distance matrix."""
        return squareform(pdist(data, metric="euclidean"))

    def _centered_distance_matrix(distances):
        """Compute centered distance matrix."""
        n = distances.shape[0]
        row_means = np.mean(distances, axis=1, keepdims=True)
        col_means = np.mean(distances, axis=0, keepdims=True)
        grand_mean = np.mean(distances)

        centered = distances - row_means - col_means + grand_mean
        return centered

    # Compute distance matrices
    dist_x = _distance_matrix(x)
    dist_y = _distance_matrix(y)

    # Center the distance matrices
    centered_x = _centered_distance_matrix(dist_x)
    centered_y = _centered_distance_matrix(dist_y)

    # Compute distance covariance and variances
    n = x.shape[0]
    dcov_xy = np.sqrt(np.sum(centered_x * centered_y) / (n * n))
    dcov_xx = np.sqrt(np.sum(centered_x * centered_x) / (n * n))
    dcov_yy = np.sqrt(np.sum(centered_y * centered_y) / (n * n))

    # Distance correlation
    if dcov_xx * dcov_yy > 0:
        dcor = dcov_xy / np.sqrt(dcov_xx * dcov_yy)
    else:
        dcor = 0.0

    return dcor


def calculate_pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate average Pearson correlation between embedding dimensions.

    Args:
        x: First embedding [n_samples, n_features]
        y: Second embedding [n_samples, n_features]

    Returns:
        Average absolute Pearson correlation
    """
    correlations = []

    for i in range(x.shape[1]):
        for j in range(y.shape[1]):
            corr, _ = stats.pearsonr(x[:, i], y[:, j])
            correlations.append(abs(corr))

    return np.mean(correlations)


def calculate_independence_score(
    base_embedding: np.ndarray, weather_embedding: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive independence score between embeddings.

    Args:
        base_embedding: Base load embeddings [n_samples, base_dim]
        weather_embedding: Weather embeddings [n_samples, weather_dim]

    Returns:
        Dictionary of independence metrics
    """
    logger = get_logger("IndependenceScore")

    try:
        # Mutual information (lower is better)
        mi_score = calculate_mutual_information(base_embedding, weather_embedding)

        # Distance correlation (lower is better)
        dcor_score = calculate_distance_correlation(base_embedding, weather_embedding)

        # Pearson correlation (lower is better)
        pearson_score = calculate_pearson_correlation(base_embedding, weather_embedding)

        # Composite independence score (higher is better)
        # Normalize and invert the individual scores
        independence_score = 1.0 / (1.0 + mi_score + dcor_score + pearson_score)

        return {
            "mutual_information": mi_score,
            "distance_correlation": dcor_score,
            "pearson_correlation": pearson_score,
            "independence_score": independence_score,
        }

    except Exception as e:
        logger.warning(f"Error calculating independence score: {e}")
        return {
            "mutual_information": 0.0,
            "distance_correlation": 0.0,
            "pearson_correlation": 0.0,
            "independence_score": 0.0,
        }


def calculate_reconstruction_metrics(
    original: np.ndarray, reconstructed: np.ndarray
) -> Dict[str, float]:
    """
    Calculate reconstruction quality metrics.

    Args:
        original: Original load profiles [n_samples, timesteps, features]
        reconstructed: Reconstructed load profiles [n_samples, timesteps, features]

    Returns:
        Dictionary of reconstruction metrics
    """
    # Mean Squared Error
    mse = np.mean((original - reconstructed) ** 2)

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Mean Absolute Error
    mae = np.mean(np.abs(original - reconstructed))

    # R-squared (coefficient of determination)
    ss_res = np.sum((original - reconstructed) ** 2)
    ss_tot = np.sum((original - np.mean(original)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    # Normalized RMSE
    nrmse = rmse / (np.max(original) - np.min(original) + 1e-8)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "nrmse": nrmse}


def calculate_source_separation_quality(
    original_load: np.ndarray,
    base_load: np.ndarray,
    weather_effect: np.ndarray,
    reconstructed_total: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate source separation quality metrics.

    Args:
        original_load: Original load profiles
        base_load: Separated base load component
        weather_effect: Separated weather effect component
        reconstructed_total: Total reconstruction (base + weather effect)

    Returns:
        Dictionary of separation quality metrics
    """
    # Reconstruction quality of total load
    total_recon_metrics = calculate_reconstruction_metrics(
        original_load, reconstructed_total
    )

    # Component additivity check
    component_sum = base_load + weather_effect
    additivity_error = np.mean((reconstructed_total - component_sum) ** 2)

    # Base load stability (should be less variable than original)
    original_std = np.std(original_load, axis=1).mean()
    base_std = np.std(base_load, axis=1).mean()
    stability_ratio = base_std / (original_std + 1e-8)

    # Weather effect responsiveness (should correlate with weather variations)
    weather_effect_std = np.std(weather_effect, axis=1).mean()
    responsiveness_ratio = weather_effect_std / (original_std + 1e-8)

    return {
        "total_reconstruction_r2": total_recon_metrics["r2"],
        "total_reconstruction_rmse": total_recon_metrics["rmse"],
        "additivity_error": additivity_error,
        "base_stability_ratio": stability_ratio,
        "weather_responsiveness_ratio": responsiveness_ratio,
        "separation_quality_score": total_recon_metrics["r2"]
        * (1.0 / (1.0 + additivity_error)),
    }


def calculate_source_separation_metrics(
    original_load: np.ndarray,
    base_embedding: np.ndarray,
    weather_embedding: np.ndarray,
    reconstructed_total: np.ndarray,
    reconstructed_base: np.ndarray,
    reconstructed_weather_effect: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate comprehensive source separation evaluation metrics.

    This is the main function that combines all evaluation aspects:
    1. Independence between base and weather embeddings
    2. Reconstruction quality
    3. Source separation quality

    Args:
        original_load: Original load profiles
        base_embedding: Base load embeddings
        weather_embedding: Weather embeddings
        reconstructed_total: Total reconstructed load
        reconstructed_base: Reconstructed base load component
        reconstructed_weather_effect: Reconstructed weather effect component

    Returns:
        Comprehensive dictionary of all metrics
    """
    logger = get_logger("SourceSeparationMetrics")
    logger.info("Calculating comprehensive source separation metrics...")

    # 1. Independence metrics
    independence_metrics = calculate_independence_score(
        base_embedding, weather_embedding
    )

    # 2. Reconstruction metrics
    reconstruction_metrics = calculate_reconstruction_metrics(
        original_load, reconstructed_total
    )

    # 3. Source separation quality
    separation_metrics = calculate_source_separation_quality(
        original_load,
        reconstructed_base,
        reconstructed_weather_effect,
        reconstructed_total,
    )

    # 4. Component-specific reconstruction metrics
    base_recon_metrics = calculate_reconstruction_metrics(
        original_load, reconstructed_base
    )
    weather_recon_metrics = calculate_reconstruction_metrics(
        np.zeros_like(original_load), reconstructed_weather_effect
    )

    # Combine all metrics
    all_metrics = {
        # Independence metrics
        **{f"independence_{k}": v for k, v in independence_metrics.items()},
        # Total reconstruction metrics
        **{f"total_recon_{k}": v for k, v in reconstruction_metrics.items()},
        # Separation quality metrics
        **{f"separation_{k}": v for k, v in separation_metrics.items()},
        # Component reconstruction metrics
        **{f"base_recon_{k}": v for k, v in base_recon_metrics.items()},
        **{f"weather_recon_{k}": v for k, v in weather_recon_metrics.items()},
    }

    # Overall quality score
    overall_score = (
        independence_metrics["independence_score"] * 0.4
        + reconstruction_metrics["r2"] * 0.3
        + separation_metrics["separation_quality_score"] * 0.3
    )
    all_metrics["overall_quality_score"] = overall_score

    logger.info(
        f"Source separation evaluation completed. Overall score: {overall_score:.4f}"
    )

    return all_metrics


def print_source_separation_report(metrics: Dict[str, float]):
    """
    Print a formatted report of source separation metrics.

    Args:
        metrics: Dictionary of metrics from calculate_source_separation_metrics
    """
    print("=" * 80)
    print("SOURCE SEPARATION EVALUATION REPORT")
    print("=" * 80)

    print("\n📊 INDEPENDENCE METRICS (Lower is Better)")
    print("-" * 50)
    print(f"Mutual Information:      {metrics['independence_mutual_information']:.6f}")
    print(
        f"Distance Correlation:    {metrics['independence_distance_correlation']:.6f}"
    )
    print(f"Pearson Correlation:     {metrics['independence_pearson_correlation']:.6f}")
    print(
        f"Independence Score:      {metrics['independence_independence_score']:.6f} ⭐"
    )

    print("\n🔧 RECONSTRUCTION QUALITY")
    print("-" * 50)
    print(f"Total R²:               {metrics['total_recon_r2']:.6f}")
    print(f"Total RMSE:             {metrics['total_recon_rmse']:.6f}")
    print(f"Total MAE:              {metrics['total_recon_mae']:.6f}")

    print("\n🎯 SOURCE SEPARATION QUALITY")
    print("-" * 50)
    print(f"Additivity Error:       {metrics['separation_additivity_error']:.6f}")
    print(f"Base Stability Ratio:   {metrics['separation_base_stability_ratio']:.6f}")
    print(
        f"Weather Responsiveness: {metrics['separation_weather_responsiveness_ratio']:.6f}"
    )
    print(
        f"Separation Score:       {metrics['separation_separation_quality_score']:.6f} ⭐"
    )

    print("\n🏆 OVERALL ASSESSMENT")
    print("-" * 50)
    print(f"Overall Quality Score:  {metrics['overall_quality_score']:.6f} ⭐⭐⭐")

    # Quality assessment
    if metrics["overall_quality_score"] > 0.8:
        assessment = "EXCELLENT - High quality source separation achieved"
    elif metrics["overall_quality_score"] > 0.6:
        assessment = "GOOD - Satisfactory source separation with room for improvement"
    elif metrics["overall_quality_score"] > 0.4:
        assessment = "FAIR - Basic separation achieved, significant improvement needed"
    else:
        assessment = "POOR - Source separation quality is insufficient"

    print(f"Assessment: {assessment}")
    print("=" * 80)
