#!/usr/bin/env python3
"""
Data preprocessing utilities for CausalHES.

This module provides common data preprocessing functions including
normalization, time window creation, and data validation.

Author: CausalHES Team
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List, Dict
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import warnings


def normalize_data(
    data: np.ndarray,
    method: str = "minmax",
    feature_range: Tuple[float, float] = (0, 1),
    axis: Optional[int] = None,
) -> np.ndarray:
    """
    Normalize data using specified method.

    Args:
        data: Input data array
        method: Normalization method ('minmax', 'standard', 'robust', 'none')
        feature_range: Range for minmax scaling
        axis: Axis along which to normalize (None for global normalization)

    Returns:
        Normalized data array
    """
    if method == "none":
        return data

    original_shape = data.shape

    # Reshape for sklearn scalers if needed
    if data.ndim > 2:
        data_reshaped = data.reshape(-1, data.shape[-1])
    else:
        data_reshaped = data

    if method == "minmax":
        scaler = MinMaxScaler(feature_range=feature_range)
        normalized = scaler.fit_transform(data_reshaped)
    elif method == "standard" or method == "zscore":
        scaler = StandardScaler()
        normalized = scaler.fit_transform(data_reshaped)
    elif method == "robust":
        scaler = RobustScaler()
        normalized = scaler.fit_transform(data_reshaped)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Reshape back to original shape
    return normalized.reshape(original_shape)


def create_time_windows(
    data: np.ndarray, window_size: int, step_size: int = 1, return_indices: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Create sliding time windows from time series data.

    Args:
        data: Input time series data (samples, timesteps, features)
        window_size: Size of each window
        step_size: Step size between windows
        return_indices: Whether to return window indices

    Returns:
        Windowed data array, optionally with indices
    """
    if data.ndim < 2:
        raise ValueError("Data must have at least 2 dimensions")

    n_samples, n_timesteps = data.shape[:2]

    if window_size > n_timesteps:
        raise ValueError(
            f"Window size ({window_size}) cannot be larger than timesteps ({n_timesteps})"
        )

    # Calculate number of windows
    n_windows = (n_timesteps - window_size) // step_size + 1

    windows = []
    indices = []

    for i in range(0, n_timesteps - window_size + 1, step_size):
        window = data[:, i : i + window_size]
        windows.append(window)
        if return_indices:
            indices.append((i, i + window_size))

    windowed_data = np.array(windows).transpose(1, 0, 2, *range(3, data.ndim + 1))

    if return_indices:
        return windowed_data, np.array(indices)
    return windowed_data


def validate_data_shapes(
    load_data: np.ndarray, weather_data: np.ndarray, labels: Optional[np.ndarray] = None
) -> None:
    """
    Validate that data arrays have compatible shapes.

    Args:
        load_data: Load time series data
        weather_data: Weather time series data
        labels: Optional cluster labels

    Raises:
        ValueError: If shapes are incompatible
    """
    if load_data.shape[0] != weather_data.shape[0]:
        raise ValueError(
            f"Load and weather data must have same number of samples: "
            f"{load_data.shape[0]} vs {weather_data.shape[0]}"
        )

    if load_data.shape[1] != weather_data.shape[1]:
        raise ValueError(
            f"Load and weather data must have same number of timesteps: "
            f"{load_data.shape[1]} vs {weather_data.shape[1]}"
        )

    if labels is not None and len(labels) != load_data.shape[0]:
        raise ValueError(
            f"Labels must have same length as data: "
            f"{len(labels)} vs {load_data.shape[0]}"
        )


def remove_outliers(
    data: np.ndarray, method: str = "iqr", threshold: float = 3.0, axis: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outliers from data.

    Args:
        data: Input data array
        method: Outlier detection method ('iqr', 'zscore')
        threshold: Threshold for outlier detection
        axis: Axis along which to detect outliers

    Returns:
        Tuple of (cleaned_data, outlier_mask)
    """
    if method == "iqr":
        q1 = np.percentile(data, 25, axis=axis, keepdims=True)
        q3 = np.percentile(data, 75, axis=axis, keepdims=True)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outlier_mask = (data < lower_bound) | (data > upper_bound)

    elif method == "zscore":
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        z_scores = np.abs((data - mean) / (std + 1e-8))
        outlier_mask = z_scores > threshold

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

    # For time series data, mark entire samples as outliers if any timestep is outlier
    if data.ndim > 1 and axis == 0:
        sample_outliers = np.any(outlier_mask, axis=tuple(range(1, data.ndim)))
        outlier_mask = sample_outliers[:, np.newaxis, ...]

    cleaned_data = data[~outlier_mask.any(axis=tuple(range(1, data.ndim)))]

    return cleaned_data, outlier_mask


def interpolate_missing_values(
    data: np.ndarray, method: str = "linear", limit: Optional[int] = None
) -> np.ndarray:
    """
    Interpolate missing values in time series data.

    Args:
        data: Input data with potential NaN values
        method: Interpolation method ('linear', 'forward', 'backward')
        limit: Maximum number of consecutive NaNs to interpolate

    Returns:
        Data with interpolated values
    """
    if not np.any(np.isnan(data)):
        return data

    result = data.copy()

    # Handle each sample separately for multi-dimensional data
    if data.ndim > 1:
        for i in range(data.shape[0]):
            sample = data[i]
            if data.ndim == 3:  # (samples, timesteps, features)
                for j in range(sample.shape[1]):
                    feature_series = sample[:, j]
                    result[i, :, j] = _interpolate_1d(feature_series, method, limit)
            else:  # (samples, timesteps)
                result[i] = _interpolate_1d(sample, method, limit)
    else:
        result = _interpolate_1d(data, method, limit)

    return result


def _interpolate_1d(
    series: np.ndarray, method: str, limit: Optional[int]
) -> np.ndarray:
    """Helper function for 1D interpolation."""
    if not np.any(np.isnan(series)):
        return series

    result = series.copy()

    if method == "linear":
        # Use pandas for convenient interpolation
        s = pd.Series(series)
        interpolated = s.interpolate(method="linear", limit=limit)
        result = interpolated.values

    elif method == "forward":
        result = pd.Series(series).fillna(method="ffill", limit=limit).values

    elif method == "backward":
        result = pd.Series(series).fillna(method="bfill", limit=limit).values

    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    return result


def split_data(
    data: Union[np.ndarray, List[np.ndarray]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple:
    """
    Split data into train/validation/test sets.

    Args:
        data: Data to split (single array or list of arrays)
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_state: Random seed

    Returns:
        Tuple of split data
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    np.random.seed(random_state)

    # Handle single array or list of arrays
    if isinstance(data, np.ndarray):
        arrays = [data]
        single_array = True
    else:
        arrays = data
        single_array = False

    n_samples = len(arrays[0])

    # Create random indices
    indices = np.random.permutation(n_samples)

    # Calculate split points
    train_end = int(train_ratio * n_samples)
    val_end = train_end + int(val_ratio * n_samples)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    # Split each array
    splits = []
    for array in arrays:
        train_split = array[train_idx]
        val_split = array[val_idx]
        test_split = array[test_idx]
        splits.extend([train_split, val_split, test_split])

    if single_array:
        return tuple(splits)
    else:
        # Reshape for multiple arrays
        n_arrays = len(arrays)
        result = []
        for i in range(3):  # train, val, test
            split_group = []
            for j in range(n_arrays):
                split_group.append(splits[j * 3 + i])
            result.append(tuple(split_group))
        return tuple(result)


def check_data_quality(
    data: np.ndarray, name: str = "data"
) -> Dict[str, Union[int, float, bool]]:
    """
    Check data quality and return statistics.

    Args:
        data: Input data array
        name: Name for reporting

    Returns:
        Dictionary with quality statistics
    """
    stats = {
        "name": name,
        "shape": data.shape,
        "dtype": str(data.dtype),
        "total_elements": data.size,
        "missing_values": int(np.sum(np.isnan(data))),
        "missing_percentage": float(np.sum(np.isnan(data)) / data.size * 100),
        "infinite_values": int(np.sum(np.isinf(data))),
        "min_value": float(np.nanmin(data)),
        "max_value": float(np.nanmax(data)),
        "mean_value": float(np.nanmean(data)),
        "std_value": float(np.nanstd(data)),
        "has_negative": bool(np.any(data < 0)),
        "all_finite": bool(np.all(np.isfinite(data[~np.isnan(data)]))),
    }

    return stats
