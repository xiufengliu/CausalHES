"""
CSSAE Trainer for CausalHES framework.

This module provides specialized training logic for the Causal Source Separation
Autoencoder (CSSAE), including:
1. Pre-training with reconstruction objectives
2. Causal independence constraint enforcement
3. Advanced training strategies (curriculum learning, adversarial training)
4. Monitoring and evaluation of source separation quality
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Callable
import logging
from pathlib import Path
import matplotlib.pyplot as plt

from ..models.cssae import CSSAE
from ..losses import CSSAELoss
from ..utils.logging import get_logger
from ..evaluation.source_separation_metrics import calculate_source_separation_metrics


class CSSAETrainer:
    """
    Specialized trainer for CSSAE models.
    
    This trainer implements sophisticated training strategies for learning
    causal source separation, including:
    - Multi-stage training (reconstruction → independence)
    - Curriculum learning for causal constraints
    - Adversarial training for independence
    - Monitoring of separation quality
    """
    
    def __init__(self,
                 model: CSSAE,
                 reconstruction_weight: float = 1.0,
                 causal_weight: float = 0.1,
                 mi_weight: float = 1.0,
                 adversarial_weight: float = 0.5,
                 dcor_weight: float = 0.3,
                 curriculum_learning: bool = True,
                 warmup_epochs: int = 10,
                 log_dir: Optional[str] = None):
        """
        Initialize CSSAE trainer.
        
        Args:
            model: CSSAE model to train
            reconstruction_weight: Weight for reconstruction loss
            causal_weight: Weight for causal independence loss
            mi_weight: Weight for mutual information component
            adversarial_weight: Weight for adversarial component
            dcor_weight: Weight for distance correlation component
            curriculum_learning: Whether to use curriculum learning
            warmup_epochs: Number of warmup epochs for curriculum learning
            log_dir: Directory for logging and checkpoints
        """
        self.model = model
        self.reconstruction_weight = reconstruction_weight
        self.causal_weight = causal_weight
        self.curriculum_learning = curriculum_learning
        self.warmup_epochs = warmup_epochs
        
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize loss function
        # Get embedding dimensions from model
        base_dim = model.base_dim
        weather_dim = model.weather_embedding_dim

        self.loss_function = CSSAELoss(
            base_dim=base_dim,
            weather_dim=weather_dim,
            reconstruction_weight=reconstruction_weight,
            causal_weight=causal_weight,
            mi_weight=mi_weight,
            adversarial_weight=adversarial_weight,
            dcor_weight=dcor_weight
        )
        
        # Training history
        self.history = {
            'epoch': [],
            'total_loss': [],
            'reconstruction_loss': [],
            'causal_loss': [],
            'mi_loss': [],
            'adversarial_loss': [],
            'dcor_loss': [],
            'separation_quality': []
        }
        
        # Setup logging
        if log_dir:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir = self.log_dir / "checkpoints"
            self.checkpoint_dir.mkdir(exist_ok=True)
        else:
            self.log_dir = None
            self.checkpoint_dir = None
        
        self.logger.info(f"CSSAE Trainer initialized")
        self.logger.info(f"Loss weights - Reconstruction: {reconstruction_weight}, Causal: {causal_weight}")
        self.logger.info(f"Curriculum learning: {curriculum_learning}, Warmup epochs: {warmup_epochs}")
    
    def _get_current_causal_weight(self, epoch: int, total_epochs: int) -> float:
        """
        Get current causal weight for curriculum learning.
        
        During curriculum learning, we gradually increase the causal weight
        to allow the model to first learn good reconstructions before
        enforcing strong independence constraints.
        """
        if not self.curriculum_learning:
            return self.causal_weight
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            return self.causal_weight * (epoch / self.warmup_epochs)
        else:
            return self.causal_weight
    
    def _compute_losses(self, 
                       load_data: np.ndarray,
                       weather_data: np.ndarray,
                       outputs: Dict[str, torch.Tensor],
                       epoch: int,
                       total_epochs: int) -> Dict[str, torch.Tensor]:
        """Compute all loss components."""
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(
            load_data, 
            outputs['total_reconstruction']
        )
        
        # Causal independence loss
        causal_loss_value = self.causal_loss(
            outputs['base_embedding'],
            outputs['weather_embedding']
        )
        
        # Get current causal weight (for curriculum learning)
        current_causal_weight = self._get_current_causal_weight(epoch, total_epochs)
        
        # Total loss
        total_loss = (
            self.reconstruction_weight * recon_loss +
            current_causal_weight * causal_loss_value
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'causal_loss': causal_loss_value,
            'current_causal_weight': current_causal_weight
        }
    
    def train_step(self,
                   load_batch: torch.Tensor,
                   weather_batch: torch.Tensor,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   total_epochs: int) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            load_batch: Batch of load data
            weather_batch: Batch of weather data
            optimizer: Optimizer
            epoch: Current epoch
            total_epochs: Total number of epochs
            
        Returns:
            Dictionary of loss values
        """
        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self.model([load_batch, weather_batch], training=True)
            
            # Compute losses
            losses = self._compute_losses(
                load_batch, weather_batch, outputs, epoch, total_epochs
            )
            
            total_loss = losses['total_loss']
        
        # Compute gradients and update weights
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return {k: float(v.numpy()) for k, v in losses.items()}
    
    def evaluate_separation_quality(self,
                                   load_data: np.ndarray,
                                   weather_data: np.ndarray,
                                   sample_size: int = 1000) -> Dict[str, float]:
        """
        Evaluate the quality of source separation.
        
        Args:
            load_data: Load time series data
            weather_data: Weather time series data
            sample_size: Number of samples to use for evaluation
            
        Returns:
            Dictionary of separation quality metrics
        """
        # Sample data for evaluation
        if len(load_data) > sample_size:
            indices = np.random.choice(len(load_data), sample_size, replace=False)
            load_sample = load_data[indices]
            weather_sample = weather_data[indices]
        else:
            load_sample = load_data
            weather_sample = weather_data
        
        # Get embeddings and reconstructions
        embeddings = self.model.get_embeddings(load_sample, weather_sample)
        reconstructions = self.model.reconstruct(load_sample, weather_sample)
        
        # Calculate separation metrics
        metrics = calculate_source_separation_metrics(
            original_load=load_sample,
            base_embedding=embeddings['base_embedding'],
            weather_embedding=embeddings['weather_embedding'],
            reconstructed_total=reconstructions['total'],
            reconstructed_base=reconstructions['base_load'],
            reconstructed_weather_effect=reconstructions['weather_effect']
        )
        
        return metrics
    
    def fit(self,
            load_data: np.ndarray,
            weather_data: np.ndarray,
            epochs: int = 100,
            batch_size: int = 32,
            learning_rate: float = 0.001,
            validation_split: float = 0.1,
            patience: int = 10,
            verbose: int = 1,
            save_best: bool = True) -> Dict:
        """
        Train the CSSAE model.

        Args:
            load_data: Load time series data
            weather_data: Weather time series data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            validation_split: Fraction of data to use for validation
            patience: Early stopping patience
            verbose: Verbosity level
            save_best: Whether to save the best model

        Returns:
            Training history
        """
        self.logger.info(f"Starting CSSAE training for {epochs} epochs...")
        self.logger.info(f"Data shapes - Load: {load_data.shape}, Weather: {weather_data.shape}")

        # TODO: Complete PyTorch training implementation
        # This is a simplified placeholder - full implementation needed

        # Convert to PyTorch tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        load_tensor = torch.FloatTensor(load_data).to(device)
        weather_tensor = torch.FloatTensor(weather_data).to(device)

        # Setup optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Simple training loop
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            # Forward pass
            outputs = self.model([load_tensor, weather_tensor])

            # Compute loss
            loss = self.loss_function(load_tensor, outputs)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update history
            self.history['epoch'].append(epoch)
            self.history['total_loss'].append(loss.item())

            if verbose and epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch:3d}/{epochs}: Loss={loss.item():.4f}")

        self.logger.info("CSSAE training completed")
        return self.history
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        if not self.history['epoch']:
            self.logger.warning("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(self.history['epoch'], self.history['total_loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Reconstruction vs Causal loss
        axes[0, 1].plot(self.history['epoch'], self.history['reconstruction_loss'], label='Reconstruction')
        axes[0, 1].plot(self.history['epoch'], self.history['causal_loss'], label='Causal')
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Separation quality
        axes[1, 0].plot(self.history['epoch'], self.history['separation_quality'])
        axes[1, 0].set_title('Separation Quality')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Independence Score')
        axes[1, 0].grid(True)
        
        # Loss ratio
        if len(self.history['reconstruction_loss']) > 0 and len(self.history['causal_loss']) > 0:
            loss_ratio = np.array(self.history['causal_loss']) / (np.array(self.history['reconstruction_loss']) + 1e-8)
            axes[1, 1].plot(self.history['epoch'], loss_ratio)
            axes[1, 1].set_title('Causal/Reconstruction Loss Ratio')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Ratio')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
