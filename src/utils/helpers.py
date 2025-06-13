"""
Helper utilities for the household segmentation project.

This module provides common utility functions used across the project.
"""

import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Union, Optional, Tuple, List
import json
import pickle


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, filepath: Union[str, Path]) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Union[str, Path]) -> dict:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(data, filepath: Union[str, Path]) -> None:
    """
    Save object to pickle file.
    
    Args:
        data: Object to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: Union[str, Path]):
    """
    Load object from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_memory_usage() -> float:
    """
    Get current memory usage in MB.
    
    Returns:
        Memory usage in MB
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.2f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"


def validate_input_shape(X: np.ndarray, expected_dims: int, 
                        name: str = "Input") -> None:
    """
    Validate input array shape.
    
    Args:
        X: Input array
        expected_dims: Expected number of dimensions
        name: Name of the input for error messages
        
    Raises:
        ValueError: If input shape is invalid
    """
    if X.ndim != expected_dims:
        raise ValueError(
            f"{name} must be {expected_dims}D array, got {X.ndim}D array with shape {X.shape}"
        )


def check_consistent_length(*arrays) -> None:
    """
    Check that all arrays have consistent first dimension.
    
    Args:
        *arrays: Arrays to check
        
    Raises:
        ValueError: If arrays have inconsistent lengths
    """
    lengths = [len(X) for X in arrays if X is not None]
    if len(set(lengths)) > 1:
        raise ValueError(f"Inconsistent numbers of samples: {lengths}")


def train_test_split_temporal(X: np.ndarray, test_size: float = 0.2, 
                            shuffle: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split temporal data into train and test sets.
    
    Args:
        X: Input data
        test_size: Proportion of data for testing
        shuffle: Whether to shuffle data before splitting
        
    Returns:
        Tuple of (X_train, X_test)
    """
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    if shuffle:
        indices = np.random.permutation(n_samples)
        X = X[indices]
    
    X_train = X[:-n_test] if n_test > 0 else X
    X_test = X[-n_test:] if n_test > 0 else np.array([])
    
    return X_train, X_test


def create_sliding_windows(data: np.ndarray, window_size: int, 
                         step_size: int = 1) -> np.ndarray:
    """
    Create sliding windows from time series data.
    
    Args:
        data: Input time series data
        window_size: Size of each window
        step_size: Step size between windows
        
    Returns:
        Array of sliding windows
    """
    if len(data) < window_size:
        raise ValueError(f"Data length ({len(data)}) must be >= window_size ({window_size})")
    
    n_windows = (len(data) - window_size) // step_size + 1
    windows = np.zeros((n_windows, window_size))
    
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        windows[i] = data[start_idx:end_idx]
    
    return windows


def normalize_array(X: np.ndarray, method: str = "minmax", 
                   axis: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """
    Normalize array using specified method.
    
    Args:
        X: Input array
        method: Normalization method ("minmax", "zscore", "robust")
        axis: Axis along which to normalize
        
    Returns:
        Tuple of (normalized_array, normalization_params)
    """
    if method == "minmax":
        min_val = np.min(X, axis=axis, keepdims=True)
        max_val = np.max(X, axis=axis, keepdims=True)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
        params = {"method": "minmax", "min": min_val, "max": max_val}
        
    elif method == "zscore":
        mean_val = np.mean(X, axis=axis, keepdims=True)
        std_val = np.std(X, axis=axis, keepdims=True)
        X_norm = (X - mean_val) / (std_val + 1e-8)
        params = {"method": "zscore", "mean": mean_val, "std": std_val}
        
    elif method == "robust":
        median_val = np.median(X, axis=axis, keepdims=True)
        q75 = np.percentile(X, 75, axis=axis, keepdims=True)
        q25 = np.percentile(X, 25, axis=axis, keepdims=True)
        iqr = q75 - q25
        X_norm = (X - median_val) / (iqr + 1e-8)
        params = {"method": "robust", "median": median_val, "iqr": iqr}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return X_norm, params


def denormalize_array(X_norm: np.ndarray, params: dict) -> np.ndarray:
    """
    Denormalize array using stored parameters.
    
    Args:
        X_norm: Normalized array
        params: Normalization parameters from normalize_array
        
    Returns:
        Denormalized array
    """
    method = params["method"]
    
    if method == "minmax":
        return X_norm * (params["max"] - params["min"]) + params["min"]
    elif method == "zscore":
        return X_norm * params["std"] + params["mean"]
    elif method == "robust":
        return X_norm * params["iqr"] + params["median"]
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def print_system_info() -> None:
    """Print system information for debugging."""
    print("System Information:")
    print(f"  Python version: {sys.version}")
    print(f"  NumPy version: {np.__version__}")
    print(f"  TensorFlow version: {tf.__version__}")
    print(f"  Memory usage: {get_memory_usage():.2f} MB")
    
    # GPU information
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  GPUs available: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"    GPU {i}: {gpu.name}")
    else:
        print("  No GPUs available")


import sys  # Add this import at the top
