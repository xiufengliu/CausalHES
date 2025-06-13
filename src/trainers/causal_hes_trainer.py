"""
CausalHES Trainer for complete framework training.

This module provides the main training logic for the complete CausalHES framework,
orchestrating the training of both CSSAE and clustering components.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

from ..models.causal_hes_model import CausalHESModel
from ..models.cssae import CSSAE
from .cssae_trainer import CSSAETrainer
from ..evaluation.source_separation_metrics import (
    calculate_source_separation_metrics,
    print_source_separation_report
)
from ..evaluation.metrics import calculate_clustering_metrics
from ..utils.logging import get_logger


class CausalHESTrainer:
    """
    Main trainer for the complete CausalHES framework.
    
    This trainer orchestrates the complete training process:
    1. CSSAE pre-training for source separation
    2. Clustering initialization using base embeddings
    3. Joint fine-tuning of separation and clustering
    4. Comprehensive evaluation and reporting
    """
    
    def __init__(self,
                 model: CausalHESModel,
                 log_dir: Optional[str] = None,
                 save_checkpoints: bool = True):
        """
        Initialize CausalHES trainer.
        
        Args:
            model: CausalHES model to train
            log_dir: Directory for logging and checkpoints
            save_checkpoints: Whether to save model checkpoints
        """
        self.model = model
        self.save_checkpoints = save_checkpoints
        
        self.logger = get_logger(self.__class__.__name__)
        
        # Setup logging directory
        if log_dir:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir = self.log_dir / "checkpoints"
            self.checkpoint_dir.mkdir(exist_ok=True)
            self.results_dir = self.log_dir / "results"
            self.results_dir.mkdir(exist_ok=True)
        else:
            self.log_dir = None
            self.checkpoint_dir = None
            self.results_dir = None
        
        # Initialize CSSAE trainer
        self.cssae_trainer = CSSAETrainer(
            model=self.model.cssae,
            reconstruction_weight=self.model.reconstruction_weight,
            causal_weight=self.model.causal_weight,
            log_dir=str(self.log_dir / "cssae") if self.log_dir else None
        )
        
        # Training history
        self.training_history = {
            'cssae_pretraining': None,
            'joint_training': None,
            'evaluation_metrics': {}
        }
        
        self.logger.info("CausalHES Trainer initialized")
    
    def train(self,
              load_data: np.ndarray,
              weather_data: np.ndarray,
              true_labels: Optional[np.ndarray] = None,
              # CSSAE pre-training parameters
              cssae_epochs: int = 100,
              cssae_batch_size: int = 32,
              cssae_learning_rate: float = 0.001,
              # Joint training parameters
              joint_epochs: int = 50,
              joint_batch_size: int = 32,
              joint_learning_rate: float = 0.0005,
              # Training strategy
              skip_cssae_pretraining: bool = False,
              skip_joint_training: bool = False,
              # Evaluation
              evaluate_separation: bool = True,
              evaluate_clustering: bool = True,
              verbose: int = 1) -> Dict:
        """
        Train the complete CausalHES model.
        
        Args:
            load_data: Load time series data [n_samples, timesteps, features]
            weather_data: Weather time series data [n_samples, timesteps, features]
            true_labels: True cluster labels for evaluation (optional)
            cssae_epochs: Epochs for CSSAE pre-training
            cssae_batch_size: Batch size for CSSAE pre-training
            cssae_learning_rate: Learning rate for CSSAE pre-training
            joint_epochs: Epochs for joint training
            joint_batch_size: Batch size for joint training
            joint_learning_rate: Learning rate for joint training
            skip_cssae_pretraining: Skip CSSAE pre-training (use existing weights)
            skip_joint_training: Skip joint training (only do CSSAE pre-training)
            evaluate_separation: Whether to evaluate source separation quality
            evaluate_clustering: Whether to evaluate clustering quality
            verbose: Verbosity level
            
        Returns:
            Complete training history and evaluation results
        """
        self.logger.info("="*80)
        self.logger.info("STARTING CAUSAL HOUSEHOLD ENERGY SEGMENTATION TRAINING")
        self.logger.info("="*80)
        self.logger.info(f"Data shapes - Load: {load_data.shape}, Weather: {weather_data.shape}")
        self.logger.info(f"Training strategy - CSSAE epochs: {cssae_epochs}, Joint epochs: {joint_epochs}")
        
        # Phase 1: CSSAE Pre-training
        if not skip_cssae_pretraining:
            self.logger.info("\n" + "="*60)
            self.logger.info("PHASE 1: CSSAE PRE-TRAINING FOR SOURCE SEPARATION")
            self.logger.info("="*60)
            
            cssae_history = self.cssae_trainer.fit(
                load_data=load_data,
                weather_data=weather_data,
                epochs=cssae_epochs,
                batch_size=cssae_batch_size,
                learning_rate=cssae_learning_rate,
                verbose=verbose
            )
            
            self.training_history['cssae_pretraining'] = cssae_history
            
            # Save CSSAE checkpoint
            if self.save_checkpoints and self.checkpoint_dir:
                cssae_path = self.checkpoint_dir / "cssae_pretrained.h5"
                self.model.cssae.model.save_weights(str(cssae_path))
                self.logger.info(f"CSSAE checkpoint saved to {cssae_path}")
        
        else:
            self.logger.info("Skipping CSSAE pre-training (using existing weights)")
        
        # Phase 2: Joint Training
        if not skip_joint_training:
            self.logger.info("\n" + "="*60)
            self.logger.info("PHASE 2: JOINT TRAINING OF SEPARATION AND CLUSTERING")
            self.logger.info("="*60)
            
            joint_history = self.model.fit(
                load_data=load_data,
                weather_data=weather_data,
                pretrain_epochs=0,  # Skip pre-training since we already did it
                clustering_epochs=joint_epochs,
                batch_size=joint_batch_size,
                learning_rate=joint_learning_rate,
                verbose=verbose
            )
            
            self.training_history['joint_training'] = joint_history
            
            # Save final model checkpoint
            if self.save_checkpoints and self.checkpoint_dir:
                final_path = self.checkpoint_dir / "causal_hes_final.h5"
                self.model.full_model.save_weights(str(final_path))
                self.logger.info(f"Final model checkpoint saved to {final_path}")
        
        else:
            self.logger.info("Skipping joint training")
            # Still need to initialize clustering for evaluation
            self.model.initialize_clustering(load_data, weather_data)
        
        # Phase 3: Comprehensive Evaluation
        self.logger.info("\n" + "="*60)
        self.logger.info("PHASE 3: COMPREHENSIVE EVALUATION")
        self.logger.info("="*60)
        
        evaluation_results = {}
        
        # Evaluate source separation quality
        if evaluate_separation:
            self.logger.info("Evaluating source separation quality...")
            separation_results = self._evaluate_source_separation(load_data, weather_data)
            evaluation_results['source_separation'] = separation_results
            
            # Print separation report
            if verbose:
                print_source_separation_report(separation_results)
        
        # Evaluate clustering quality
        if evaluate_clustering:
            self.logger.info("Evaluating clustering quality...")
            clustering_results = self._evaluate_clustering(load_data, weather_data, true_labels)
            evaluation_results['clustering'] = clustering_results
            
            # Print clustering report
            if verbose:
                self._print_clustering_report(clustering_results)
        
        self.training_history['evaluation_metrics'] = evaluation_results
        
        # Save evaluation results
        if self.results_dir:
            self._save_evaluation_results(evaluation_results)
        
        self.logger.info("\n" + "="*80)
        self.logger.info("CAUSAL HOUSEHOLD ENERGY SEGMENTATION TRAINING COMPLETED")
        self.logger.info("="*80)
        
        return self.training_history
    
    def _evaluate_source_separation(self, 
                                   load_data: np.ndarray, 
                                   weather_data: np.ndarray) -> Dict[str, float]:
        """Evaluate source separation quality."""
        # Get source separation results
        separation_results = self.model.get_source_separation_results(load_data, weather_data)
        
        # Calculate comprehensive metrics
        metrics = calculate_source_separation_metrics(
            original_load=separation_results['original_load'],
            base_embedding=separation_results['base_embedding'],
            weather_embedding=separation_results['weather_embedding'],
            reconstructed_total=separation_results['reconstructed_total'],
            reconstructed_base=separation_results['base_load'],
            reconstructed_weather_effect=separation_results['weather_effect']
        )
        
        return metrics
    
    def _evaluate_clustering(self,
                           load_data: np.ndarray,
                           weather_data: np.ndarray,
                           true_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate clustering quality."""
        # Get cluster predictions
        predicted_labels = self.model.predict(load_data, weather_data)
        
        # Calculate clustering metrics
        if true_labels is not None:
            metrics = calculate_clustering_metrics(true_labels, predicted_labels)
        else:
            # Calculate unsupervised metrics only
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            
            # Get base embeddings for unsupervised evaluation
            base_embeddings = self.model.cssae.get_base_embeddings(load_data, weather_data)
            
            metrics = {
                'silhouette_score': silhouette_score(base_embeddings, predicted_labels),
                'calinski_harabasz_score': calinski_harabasz_score(base_embeddings, predicted_labels),
                'davies_bouldin_score': davies_bouldin_score(base_embeddings, predicted_labels),
                'n_clusters': len(np.unique(predicted_labels))
            }
        
        metrics['predicted_labels'] = predicted_labels
        return metrics
    
    def _print_clustering_report(self, clustering_results: Dict):
        """Print clustering evaluation report."""
        print("\n" + "="*80)
        print("CLUSTERING EVALUATION REPORT")
        print("="*80)
        
        if 'accuracy' in clustering_results:
            print(f"Clustering Accuracy:     {clustering_results['accuracy']:.4f}")
            print(f"Normalized Mutual Info:  {clustering_results['nmi']:.4f}")
            print(f"Adjusted Rand Index:     {clustering_results['ari']:.4f}")
        
        print(f"Silhouette Score:        {clustering_results['silhouette_score']:.4f}")
        print(f"Calinski-Harabasz:       {clustering_results['calinski_harabasz_score']:.4f}")
        print(f"Davies-Bouldin:          {clustering_results['davies_bouldin_score']:.4f}")
        print(f"Number of Clusters:      {clustering_results['n_clusters']}")
        
        # Cluster distribution
        predicted_labels = clustering_results['predicted_labels']
        unique_labels, counts = np.unique(predicted_labels, return_counts=True)
        print(f"\nCluster Distribution:")
        for label, count in zip(unique_labels, counts):
            percentage = count / len(predicted_labels) * 100
            print(f"  Cluster {label}: {count:4d} samples ({percentage:5.1f}%)")
    
    def _save_evaluation_results(self, evaluation_results: Dict):
        """Save evaluation results to files."""
        import json
        
        # Save as JSON
        results_file = self.results_dir / "evaluation_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in evaluation_results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        json_results[key][k] = v.tolist()
                    else:
                        json_results[key][k] = float(v) if isinstance(v, (np.float32, np.float64)) else v
            else:
                json_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Evaluation results saved to {results_file}")
    
    def get_training_summary(self) -> Dict:
        """Get a summary of the training process."""
        summary = {
            'model_config': {
                'n_clusters': self.model.n_clusters,
                'base_dim': self.model.base_dim,
                'reconstruction_weight': self.model.reconstruction_weight,
                'causal_weight': self.model.causal_weight,
                'clustering_weight': self.model.clustering_weight
            },
            'training_completed': {
                'cssae_pretraining': self.training_history['cssae_pretraining'] is not None,
                'joint_training': self.training_history['joint_training'] is not None,
                'evaluation': bool(self.training_history['evaluation_metrics'])
            }
        }
        
        # Add key metrics if available
        if self.training_history['evaluation_metrics']:
            eval_metrics = self.training_history['evaluation_metrics']
            
            if 'source_separation' in eval_metrics:
                summary['source_separation_quality'] = eval_metrics['source_separation']['overall_quality_score']
            
            if 'clustering' in eval_metrics:
                clustering = eval_metrics['clustering']
                if 'accuracy' in clustering:
                    summary['clustering_accuracy'] = clustering['accuracy']
                summary['silhouette_score'] = clustering['silhouette_score']
        
        return summary
