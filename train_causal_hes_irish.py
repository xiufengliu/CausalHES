#!/usr/bin/env python3
"""
Train CausalHES Model on Irish Dataset.

This script trains the complete CausalHES model on the processed Irish
household energy dataset and saves the trained model for generating
real source separation results.

Usage:
    python train_causal_hes_irish.py
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import yaml
import json
from datetime import datetime

# Add src to path
sys.path.append('src')

from models.causal_hes_model import CausalHESModel
from trainers.causal_hes_trainer import CausalHESTrainer
from data.irish_dataset_processor import IrishDatasetProcessor
from utils.logging import setup_logging, get_logger


def load_processed_irish_data():
    """Load processed Irish dataset."""
    print("Loading processed Irish dataset...")
    
    data_dir = Path("data/processed_irish")
    
    # Load data files
    load_data = np.load(data_dir / "load_profiles.npy")
    weather_data = np.load(data_dir / "weather_data.npy") 
    labels = np.load(data_dir / "cluster_labels.npy")
    
    print(f"Loaded data shapes:")
    print(f"  Load profiles: {load_data.shape}")
    print(f"  Weather data: {weather_data.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Unique labels: {np.unique(labels)}")
    
    return {
        'load_data': load_data,
        'weather_data': weather_data,
        'labels': labels
    }


def split_data(data, train_ratio=0.8, val_ratio=0.1):
    """Split data into train/validation/test sets."""
    n_samples = len(data['load_data'])
    
    # Calculate split indices
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    # Create indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    splits = {}
    for split_name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        splits[split_name] = {
            'load_data': data['load_data'][idx],
            'weather_data': data['weather_data'][idx],
            'labels': data['labels'][idx]
        }
        print(f"{split_name.capitalize()} set: {len(idx)} samples")
    
    return splits


def create_causal_hes_model(config):
    """Create CausalHES model from configuration."""
    print("Creating CausalHES model...")
    
    model = CausalHESModel(
        n_clusters=config['n_clusters'],
        load_input_shape=tuple(config['load_input_shape']),
        weather_input_shape=tuple(config['weather_input_shape']),
        load_embedding_dim=config['cssae']['load_embedding_dim'],
        weather_embedding_dim=config['cssae']['weather_embedding_dim'],
        base_dim=config['cssae']['base_dim'],
        weather_effect_dim=config['cssae']['weather_effect_dim'],
        separation_method=config['cssae']['separation_method'],
        clustering_alpha=config['dec']['alpha'],
        dropout_rate=0.1
    )
    
    model.summary()
    return model


def train_model(model, data_splits, config, output_dir):
    """Train the CausalHES model."""
    print("\n" + "="*80)
    print("TRAINING CAUSALHES MODEL")
    print("="*80)
    
    # Setup trainer
    trainer = CausalHESTrainer(
        model=model,
        log_dir=str(output_dir / "training_logs"),
        save_checkpoints=True
    )
    
    # Training parameters
    training_params = {
        'cssae_epochs': 50,  # Reduced for faster training
        'cssae_batch_size': 64,
        'cssae_learning_rate': 0.001,
        'joint_epochs': 30,  # Reduced for faster training
        'joint_batch_size': 64,
        'joint_learning_rate': 0.0005,
        'evaluate_separation': True,
        'evaluate_clustering': True,
        'verbose': 1
    }
    
    print(f"Training parameters: {training_params}")
    
    # Train on training set
    train_data = data_splits['train']
    
    training_history = trainer.train(
        load_data=train_data['load_data'],
        weather_data=train_data['weather_data'],
        true_labels=train_data['labels'],
        **training_params
    )
    
    # Save training history
    history_file = output_dir / "training_history.json"
    with open(history_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_history = {}
        for key, value in training_history.items():
            if isinstance(value, dict):
                json_history[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        json_history[key][k] = v.tolist()
                    elif isinstance(v, (np.float32, np.float64)):
                        json_history[key][k] = float(v)
                    else:
                        json_history[key][k] = v
            else:
                json_history[key] = value
        
        json.dump(json_history, f, indent=2)
    
    print(f"Training history saved to {history_file}")
    
    return training_history


def evaluate_model(model, data_splits, output_dir):
    """Evaluate the trained model on test set."""
    print("\n" + "="*80)
    print("EVALUATING TRAINED MODEL")
    print("="*80)
    
    test_data = data_splits['test']
    
    # Convert to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_tensor = torch.from_numpy(test_data['load_data']).float().to(device)
    weather_tensor = torch.from_numpy(test_data['weather_data']).float().to(device)
    
    # Get predictions
    predicted_labels = model.predict(load_tensor, weather_tensor)
    
    # Get source separation results
    separation_results = model.get_source_separation_results(load_tensor, weather_tensor)
    
    # Calculate metrics
    from evaluation.metrics import calculate_clustering_metrics
    clustering_metrics = calculate_clustering_metrics(test_data['labels'], predicted_labels)
    
    # Save evaluation results
    eval_results = {
        'timestamp': datetime.now().isoformat(),
        'test_samples': len(test_data['load_data']),
        'clustering_metrics': clustering_metrics,
        'predicted_labels': predicted_labels.tolist(),
        'separation_results_shapes': {k: v.shape for k, v in separation_results.items()}
    }
    
    eval_file = output_dir / "evaluation_results.json"
    with open(eval_file, 'w') as f:
        json.dump(eval_results, f, indent=2, default=str)
    
    print(f"Evaluation results saved to {eval_file}")
    
    # Print summary
    print(f"\nTest Set Evaluation:")
    print(f"  Accuracy: {clustering_metrics['accuracy']:.4f}")
    print(f"  NMI: {clustering_metrics['nmi']:.4f}")
    print(f"  ARI: {clustering_metrics['ari']:.4f}")
    
    return eval_results, separation_results


def save_model(model, output_dir):
    """Save the trained model."""
    model_path = output_dir / "causal_hes_trained.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'n_clusters': model.n_clusters,
            'load_input_shape': model.load_input_shape,
            'weather_input_shape': model.weather_input_shape,
            'base_dim': model.base_dim
        }
    }, model_path)
    
    print(f"Trained model saved to {model_path}")
    return model_path


def main():
    """Main training function."""
    print("="*80)
    print("CAUSALHES IRISH DATASET TRAINING")
    print("="*80)
    
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
    # Load configuration
    with open('configs/causal_hes_config.yaml', 'r') as f:
        config = yaml.safe_load(f)['model']
    
    # Create output directory
    output_dir = Path("experiments/results/causal_hes_training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        data = load_processed_irish_data()
        
        # Split data
        data_splits = split_data(data)
        
        # Create model
        model = create_causal_hes_model(config)
        
        # Train model
        training_history = train_model(model, data_splits, config, output_dir)
        
        # Evaluate model
        eval_results, separation_results = evaluate_model(model, data_splits, output_dir)
        
        # Save trained model
        model_path = save_model(model, output_dir)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Results saved to: {output_dir}")
        print(f"Trained model: {model_path}")
        print(f"Test accuracy: {eval_results['clustering_metrics']['accuracy']:.4f}")
        
        return model, separation_results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    model, separation_results = main()
