"""
Configuration management for the household segmentation project.

This module provides centralized configuration management with support for
different environments and model configurations.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    embedding_dim: int = 10
    n_clusters: int = 3
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 50
    patience: int = 10
    
    
@dataclass
class DataConfig:
    """Configuration for data processing."""
    n_timesteps: int = 24
    n_load_features: int = 1
    n_weather_features: int = 2
    normalization_method: str = "minmax"  # "minmax", "zscore", "robust"
    train_test_split: float = 0.8
    validation_split: float = 0.2
    
    
@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    autoencoder_epochs: int = 50
    clustering_epochs: int = 30
    update_interval: int = 1
    tolerance: float = 0.001
    verbose: int = 1
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    

@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters."""
    metrics: list = None
    statistical_tests: bool = True
    significance_level: float = 0.05
    n_bootstrap: int = 1000
    visualization: bool = True
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["silhouette", "davies_bouldin", "calinski_harabasz"]


class Config:
    """
    Main configuration class that combines all configuration components.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        # Set default configurations
        self.model = ModelConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()
        self.evaluation = EvaluationConfig()
        
        # Load from file if provided
        if config_path:
            self.load_from_file(config_path)
            
        # Override with environment variables if present
        self._load_from_env()
        
    def load_from_file(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to configuration file
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
                
        self._update_from_dict(config_dict)
        
    def _update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        if 'model' in config_dict:
            self.model = ModelConfig(**config_dict['model'])
        if 'data' in config_dict:
            self.data = DataConfig(**config_dict['data'])
        if 'training' in config_dict:
            self.training = TrainingConfig(**config_dict['training'])
        if 'evaluation' in config_dict:
            self.evaluation = EvaluationConfig(**config_dict['evaluation'])
            
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Model configuration
        if os.getenv('EMBEDDING_DIM'):
            self.model.embedding_dim = int(os.getenv('EMBEDDING_DIM'))
        if os.getenv('N_CLUSTERS'):
            self.model.n_clusters = int(os.getenv('N_CLUSTERS'))
        if os.getenv('BATCH_SIZE'):
            self.model.batch_size = int(os.getenv('BATCH_SIZE'))
        if os.getenv('LEARNING_RATE'):
            self.model.learning_rate = float(os.getenv('LEARNING_RATE'))
            
        # Data configuration
        if os.getenv('N_TIMESTEPS'):
            self.data.n_timesteps = int(os.getenv('N_TIMESTEPS'))
        if os.getenv('NORMALIZATION_METHOD'):
            self.data.normalization_method = os.getenv('NORMALIZATION_METHOD')
            
    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """
        Save configuration to a file.
        
        Args:
            config_path: Path to save configuration file
        """
        config_path = Path(config_path)
        config_dict = self.to_dict()
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False)
            elif config_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': asdict(self.model),
            'data': asdict(self.data),
            'training': asdict(self.training),
            'evaluation': asdict(self.evaluation)
        }
        
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(\n  model={self.model},\n  data={self.data},\n  training={self.training},\n  evaluation={self.evaluation}\n)"


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from file or create default configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration object
    """
    return Config(config_path)


# Default configuration for quick access
DEFAULT_CONFIG = Config()
