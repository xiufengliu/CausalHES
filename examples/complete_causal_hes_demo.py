#!/usr/bin/env python3
"""
Complete CausalHES Demonstration Script.

This script demonstrates the complete CausalHES methodology as described in the paper,
including both training stages and evaluation.

Usage:
    python examples/complete_causal_hes_demo.py

Author: CausalHES Team
Date: 2024
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from pathlib import Path

# CausalHES imports
from src.models.causal_hes_model import CausalHESModel
from src.losses.cssae_loss import CSSAELoss
from src.trainers.causal_hes_complete_trainer import CausalHESTrainer
from src.data.irish_dataset_processor import IrishDatasetProcessor
from src.utils.logging import setup_logging, get_logger


def create_synthetic_dataset(n_samples=1000, n_timesteps=24, n_clusters=4):
    """
    Create synthetic dataset for demonstration.

    Following the paper's data model:
    x^(l) = s_base + s_weather + ε
    """
    np.random.seed(42)

    # Generate base load patterns (weather-independent)
    base_patterns = []
    cluster_labels = []

    for cluster_id in range(n_clusters):
        n_cluster_samples = n_samples // n_clusters

        if cluster_id == 0:  # Morning peak
            pattern = 0.3 + 0.7 * np.exp(-((np.arange(n_timesteps) - 7) ** 2) / 8)
        elif cluster_id == 1:  # Evening peak
            pattern = 0.2 + 0.8 * np.exp(-((np.arange(n_timesteps) - 19) ** 2) / 8)
        elif cluster_id == 2:  # Flat consumption
            pattern = 0.4 * np.ones(n_timesteps)
        else:  # Night shift
            pattern = 0.2 + 0.6 * np.exp(-((np.arange(n_timesteps) - 2) ** 2) / 12)

        for _ in range(n_cluster_samples):
            # Add household-specific variation
            noise = 0.1 * np.random.randn(n_timesteps)
            base_patterns.append(pattern + noise)
            cluster_labels.append(cluster_id)

    base_patterns = np.array(base_patterns)
    cluster_labels = np.array(cluster_labels)

    # Generate weather data (temperature and humidity)
    weather_data = []
    weather_effects = []

    for i in range(n_samples):
        # Daily temperature cycle
        hours = np.arange(n_timesteps)
        temp = (
            0.5
            + 0.3 * np.sin(2 * np.pi * hours / 24 - np.pi / 2)
            + 0.1 * np.random.randn(n_timesteps)
        )
        temp = np.clip(temp, 0, 1)

        # Humidity (inversely correlated with temperature)
        humidity = 0.8 - 0.4 * temp + 0.1 * np.random.randn(n_timesteps)
        humidity = np.clip(humidity, 0, 1)

        weather_data.append(np.stack([temp, humidity], axis=1))

        # Weather effect on load (heating/cooling)
        weather_effect = (
            0.2 * (temp - 0.5) ** 2
        )  # U-shaped: high at extreme temperatures
        weather_effects.append(weather_effect)

    weather_data = np.array(weather_data)
    weather_effects = np.array(weather_effects)

    # Combine base load and weather effects
    observed_load = (
        base_patterns + weather_effects + 0.05 * np.random.randn(n_samples, n_timesteps)
    )
    observed_load = np.maximum(observed_load, 0.01)  # Ensure positive

    # Normalize
    observed_load = (observed_load - observed_load.min()) / (
        observed_load.max() - observed_load.min()
    )

    # Reshape for model input: [batch_size, timesteps, features]
    load_profiles = observed_load.reshape(n_samples, n_timesteps, 1)

    return load_profiles, weather_data, cluster_labels


def create_data_loaders(
    load_profiles, weather_data, cluster_labels, batch_size=32, train_split=0.8
):
    """Create train and validation data loaders."""
    n_samples = len(load_profiles)
    n_train = int(n_samples * train_split)

    # Split data
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(load_profiles[train_indices]),
        torch.FloatTensor(weather_data[train_indices]),
    )

    val_dataset = TensorDataset(
        torch.FloatTensor(load_profiles[val_indices]),
        torch.FloatTensor(weather_data[val_indices]),
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_indices, val_indices


def evaluate_clustering(true_labels, predicted_labels):
    """Evaluate clustering performance."""
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)

    return {"adjusted_rand_index": ari, "normalized_mutual_info": nmi}


def visualize_source_separation(model, data_loader, device, save_path=None):
    """Visualize source separation results."""
    model.eval()

    # Get a batch of data
    load_data, weather_data = next(iter(data_loader))
    load_data = load_data.to(device)
    weather_data = weather_data.to(device)

    with torch.no_grad():
        outputs = model(load_data, weather_data)

    # Convert to numpy
    original = load_data.cpu().numpy()
    reconstructed = outputs["load_reconstruction"].cpu().numpy()
    base_embedding = outputs["base_embedding"].cpu().numpy()
    weather_embedding = outputs["weather_embedding"].cpu().numpy()

    # Plot first 5 samples
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))

    for i in range(5):
        # Original vs reconstructed
        axes[0, i].plot(original[i, :, 0], label="Original", alpha=0.7)
        axes[0, i].plot(reconstructed[i, :, 0], label="Reconstructed", alpha=0.7)
        axes[0, i].set_title(f"Sample {i+1}")
        axes[0, i].legend()
        axes[0, i].grid(True)

        # Embeddings visualization (first 2 dimensions)
        axes[1, i].scatter(
            base_embedding[i, 0], base_embedding[i, 1], c="blue", label="Base", s=50
        )
        if weather_embedding is not None:
            axes[1, i].scatter(
                weather_embedding[i, 0],
                weather_embedding[i, 1],
                c="red",
                label="Weather",
                s=50,
            )
        axes[1, i].set_title(f"Embeddings {i+1}")
        axes[1, i].legend()
        axes[1, i].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")

    plt.show()


def main():
    """Main demonstration function."""
    print("🚀 CausalHES Complete Demonstration")
    print("=" * 50)

    # Setup logging
    setup_logging(level="INFO")
    logger = get_logger("CausalHESDemo")

    # Configuration
    config = {
        "n_samples": 1000,
        "n_timesteps": 24,
        "n_clusters": 4,
        "batch_size": 32,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "pretrain_epochs": 30,
        "joint_epochs": 20,
        "learning_rate": 1e-3,
        "lambda_causal": 0.1,
        "lambda_cluster": 0.5,
    }

    logger.info(f"Using device: {config['device']}")

    # 1. Create synthetic dataset
    logger.info("Creating synthetic dataset...")
    load_profiles, weather_data, true_labels = create_synthetic_dataset(
        n_samples=config["n_samples"],
        n_timesteps=config["n_timesteps"],
        n_clusters=config["n_clusters"],
    )

    logger.info(
        f"Dataset created: {load_profiles.shape} load profiles, {weather_data.shape} weather data"
    )

    # 2. Create data loaders
    train_loader, val_loader, train_indices, val_indices = create_data_loaders(
        load_profiles, weather_data, true_labels, batch_size=config["batch_size"]
    )

    # 3. Initialize model and loss function
    logger.info("Initializing CausalHES model...")

    model = CausalHESModel(
        load_input_shape=(config["n_timesteps"], 1),
        weather_input_shape=(config["n_timesteps"], 2),
        n_clusters=config["n_clusters"],
        base_dim=32,
        weather_effect_dim=16,
        embedding_dim=64,
    )

    loss_fn = CSSAELoss(
        base_dim=32,
        weather_dim=16,
        lambda_causal=config["lambda_causal"],
        lambda_cluster=config["lambda_cluster"],
    )

    # 4. Initialize trainer
    trainer = CausalHESTrainer(
        model=model,
        loss_fn=loss_fn,
        device=config["device"],
        learning_rate=config["learning_rate"],
    )

    # 5. Stage 1: CSSAE Pre-training
    logger.info("Starting Stage 1: CSSAE Pre-training...")

    pretrain_results = trainer.stage1_pretrain(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["pretrain_epochs"],
        patience=10,
        save_path="checkpoints/causal_hes_pretrain.pth",
    )

    logger.info(
        f"Pre-training completed. Best validation loss: {pretrain_results['best_val_loss']:.6f}"
    )

    # 6. Stage 2: Joint Training with Clustering
    logger.info("Starting Stage 2: Joint Training with Clustering...")

    joint_results = trainer.stage2_joint_training(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["joint_epochs"],
        patience=10,
        save_path="checkpoints/causal_hes_joint.pth",
    )

    logger.info(
        f"Joint training completed. Best validation loss: {joint_results['best_val_loss']:.6f}"
    )

    # 7. Evaluate clustering performance
    logger.info("Evaluating clustering performance...")

    predicted_labels = joint_results["cluster_assignments"]
    true_train_labels = true_labels[train_indices]

    clustering_metrics = evaluate_clustering(true_train_labels, predicted_labels)

    logger.info("Clustering Results:")
    logger.info(
        f"  Adjusted Rand Index: {clustering_metrics['adjusted_rand_index']:.4f}"
    )
    logger.info(
        f"  Normalized Mutual Info: {clustering_metrics['normalized_mutual_info']:.4f}"
    )

    # 8. Visualize source separation
    logger.info("Generating visualizations...")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    visualize_source_separation(
        model=trainer.model,
        data_loader=val_loader,
        device=config["device"],
        save_path=output_dir / "source_separation_demo.png",
    )

    # 9. Summary
    print("\n" + "=" * 50)
    print("🎉 CausalHES Demonstration Completed!")
    print("=" * 50)
    print(f"✅ Pre-training Loss: {pretrain_results['best_val_loss']:.6f}")
    print(f"✅ Joint Training Loss: {joint_results['best_val_loss']:.6f}")
    print(f"✅ Clustering ARI: {clustering_metrics['adjusted_rand_index']:.4f}")
    print(f"✅ Clustering NMI: {clustering_metrics['normalized_mutual_info']:.4f}")
    print(f"📁 Outputs saved to: {output_dir.absolute()}")
    print("\n🔬 The model successfully demonstrates:")
    print("  • Causal source separation (base load vs weather effects)")
    print("  • Weather-independent clustering")
    print("  • Two-stage training methodology")
    print("  • Composite independence loss optimization")


if __name__ == "__main__":
    main()
