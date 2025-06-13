"""
Constants and default parameters for the household segmentation project.
"""

import numpy as np

# Data processing constants
DEFAULT_TIMESTEPS = 24  # 24 hours
DEFAULT_LOAD_FEATURES = 1  # Univariate load data
DEFAULT_WEATHER_FEATURES = 2  # Temperature and humidity

# Model architecture constants
DEFAULT_EMBEDDING_DIM = 10
DEFAULT_N_CLUSTERS = 3
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001

# Training constants
DEFAULT_AUTOENCODER_EPOCHS = 50
DEFAULT_CLUSTERING_EPOCHS = 30
DEFAULT_PATIENCE = 10
DEFAULT_TOLERANCE = 0.001
DEFAULT_UPDATE_INTERVAL = 1

# Evaluation constants
DEFAULT_METRICS = [
    "silhouette_score",
    "davies_bouldin_score", 
    "calinski_harabasz_score"
]

# Statistical testing constants
DEFAULT_SIGNIFICANCE_LEVEL = 0.05
DEFAULT_N_BOOTSTRAP = 1000

# Data normalization methods
NORMALIZATION_METHODS = {
    "minmax": "Min-Max normalization to [0,1]",
    "zscore": "Z-score normalization (mean=0, std=1)",
    "robust": "Robust normalization using median and IQR"
}

# Model configurations for different scenarios
MODEL_CONFIGS = {
    "small": {
        "embedding_dim": 8,
        "n_clusters": 3,
        "batch_size": 16,
        "autoencoder_epochs": 30,
        "clustering_epochs": 20
    },
    "medium": {
        "embedding_dim": 10,
        "n_clusters": 5,
        "batch_size": 32,
        "autoencoder_epochs": 50,
        "clustering_epochs": 30
    },
    "large": {
        "embedding_dim": 16,
        "n_clusters": 8,
        "batch_size": 64,
        "autoencoder_epochs": 100,
        "clustering_epochs": 50
    }
}

# Default parameters dictionary
DEFAULT_PARAMS = {
    # Data parameters
    "n_timesteps": DEFAULT_TIMESTEPS,
    "n_load_features": DEFAULT_LOAD_FEATURES,
    "n_weather_features": DEFAULT_WEATHER_FEATURES,
    
    # Model parameters
    "embedding_dim": DEFAULT_EMBEDDING_DIM,
    "n_clusters": DEFAULT_N_CLUSTERS,
    "batch_size": DEFAULT_BATCH_SIZE,
    "learning_rate": DEFAULT_LEARNING_RATE,
    
    # Training parameters
    "autoencoder_epochs": DEFAULT_AUTOENCODER_EPOCHS,
    "clustering_epochs": DEFAULT_CLUSTERING_EPOCHS,
    "patience": DEFAULT_PATIENCE,
    "tolerance": DEFAULT_TOLERANCE,
    "update_interval": DEFAULT_UPDATE_INTERVAL,
    
    # Evaluation parameters
    "metrics": DEFAULT_METRICS,
    "significance_level": DEFAULT_SIGNIFICANCE_LEVEL,
    "n_bootstrap": DEFAULT_N_BOOTSTRAP
}

# Random seeds for reproducibility
RANDOM_SEEDS = {
    "data_generation": 42,
    "model_initialization": 123,
    "train_test_split": 456,
    "evaluation": 789
}

# File paths and directories
DEFAULT_PATHS = {
    "data_dir": "data",
    "models_dir": "models",
    "results_dir": "results",
    "logs_dir": "logs",
    "checkpoints_dir": "checkpoints"
}

# Attention mechanism parameters
ATTENTION_CONFIGS = {
    "single_head": {
        "num_heads": 1,
        "key_dim": 64,
        "dropout": 0.1
    },
    "multi_head": {
        "num_heads": 4,
        "key_dim": 64,
        "dropout": 0.1
    },
    "temporal": {
        "num_heads": 2,
        "key_dim": 32,
        "dropout": 0.2,
        "temporal_window": 6
    }
}

# Weather feature names and descriptions
WEATHER_FEATURES = {
    "temperature": {
        "index": 0,
        "unit": "°C",
        "description": "Ambient temperature"
    },
    "humidity": {
        "index": 1,
        "unit": "%",
        "description": "Relative humidity"
    }
}

# Clustering algorithm specific parameters
CLUSTERING_PARAMS = {
    "sax_kmeans": {
        "word_size": 24,
        "alphabet_size": 20,
        "max_iterations": 100
    },
    "two_stage_kmeans": {
        "k_consumption": 3,
        "k_peaktime": 2,
        "dx": 0.25
    },
    "dec": {
        "alpha": 1.0,
        "update_interval": 1,
        "tolerance": 0.001
    }
}

# Synthetic data generation parameters
SYNTHETIC_DATA_PARAMS = {
    "num_samples": 100,
    "noise_level": 0.1,
    "peak_probability": 0.8,
    "seasonal_amplitude": 0.3,
    "trend_strength": 0.1
}
