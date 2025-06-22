"""
Visualization utilities for clustering analysis and results presentation.

This module provides comprehensive visualization capabilities for:
- Cluster analysis and validation
- Source separation results
- t-SNE embeddings visualization
- Attention mechanism visualization
- Performance comparison plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


class ClusterVisualizer:
    """
    Comprehensive visualization utilities for clustering analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size
            dpi: Resolution for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
        
    def plot_tsne_embeddings(self, 
                           embeddings: np.ndarray,
                           labels: np.ndarray,
                           true_labels: Optional[np.ndarray] = None,
                           title: str = "t-SNE Visualization",
                           save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create t-SNE visualization of embeddings.
        
        Args:
            embeddings: High-dimensional embeddings to visualize
            labels: Predicted cluster labels
            true_labels: Ground truth labels (optional)
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        from sklearn.manifold import TSNE
        
        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create figure
        if true_labels is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Plot predicted clusters
            scatter1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                 c=labels, cmap='tab10', alpha=0.7, s=50)
            ax1.set_title(f'{title} - Predicted Clusters')
            ax1.set_xlabel('t-SNE 1')
            ax1.set_ylabel('t-SNE 2')
            ax1.grid(True, alpha=0.3)
            
            # Plot true clusters
            scatter2 = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                 c=true_labels, cmap='tab10', alpha=0.7, s=50)
            ax2.set_title(f'{title} - True Clusters')
            ax2.set_xlabel('t-SNE 1')
            ax2.set_ylabel('t-SNE 2')
            ax2.grid(True, alpha=0.3)
            
        else:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                               c=labels, cmap='tab10', alpha=0.7, s=50)
            ax.set_title(title)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path:
            # Ensure PDF format
            if not str(save_path).endswith('.pdf'):
                save_path = str(save_path).replace('.png', '.pdf')
            plt.savefig(save_path, bbox_inches='tight')
            
        return fig
        
    def plot_cluster_profiles(self,
                            data: np.ndarray,
                            labels: np.ndarray,
                            title: str = "Cluster Profiles",
                            save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Plot average profiles for each cluster.
        
        Args:
            data: Time series data (n_samples, n_timesteps, n_features)
            labels: Cluster labels
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        fig, axes = plt.subplots(2, (n_clusters + 1) // 2, figsize=(15, 8))
        if n_clusters == 1:
            axes = [axes]
        elif n_clusters <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
            
        for i, label in enumerate(unique_labels):
            if i >= len(axes):
                break
                
            mask = labels == label
            cluster_data = data[mask]
            
            # Plot individual profiles with transparency
            for j in range(min(20, len(cluster_data))):  # Show max 20 profiles
                axes[i].plot(cluster_data[j, :, 0], alpha=0.3, color='gray', linewidth=0.5)
            
            # Plot mean profile
            mean_profile = np.mean(cluster_data, axis=0)
            axes[i].plot(mean_profile[:, 0], linewidth=3, color=f'C{i}', 
                        label=f'Cluster {label} (n={np.sum(mask)})')
            
            axes[i].set_title(f'Cluster {label}')
            axes[i].set_xlabel('Hour of Day')
            axes[i].set_ylabel('Load')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
            
        # Hide unused subplots
        for i in range(n_clusters, len(axes)):
            axes[i].set_visible(False)
            
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            # Ensure PDF format
            if not str(save_path).endswith('.pdf'):
                save_path = str(save_path).replace('.png', '.pdf')
            plt.savefig(save_path, bbox_inches='tight')

        return fig
        
    def plot_source_separation(self,
                             original_load: np.ndarray,
                             base_component: np.ndarray,
                             weather_component: np.ndarray,
                             reconstructed_load: np.ndarray,
                             n_examples: int = 5,
                             save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Visualize source separation results.
        
        Args:
            original_load: Original load profiles
            base_component: Separated base load component
            weather_component: Separated weather effect component
            reconstructed_load: Reconstructed total load
            n_examples: Number of examples to show
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(n_examples, 4, figsize=(20, 4*n_examples))
        
        for i in range(n_examples):
            # Original load
            axes[i, 0].plot(original_load[i, :, 0], 'b-', linewidth=2, label='Original')
            axes[i, 0].set_title('Original Load')
            axes[i, 0].set_ylabel('Load')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Base component
            axes[i, 1].plot(base_component[i, :, 0], 'g-', linewidth=2, label='Base')
            axes[i, 1].set_title('Base Component')
            axes[i, 1].set_ylabel('Load')
            axes[i, 1].grid(True, alpha=0.3)
            
            # Weather component
            axes[i, 2].plot(weather_component[i, :, 0], 'r-', linewidth=2, label='Weather')
            axes[i, 2].set_title('Weather Effect')
            axes[i, 2].set_ylabel('Load')
            axes[i, 2].grid(True, alpha=0.3)
            
            # Reconstruction comparison
            axes[i, 3].plot(original_load[i, :, 0], 'b-', linewidth=2, label='Original')
            axes[i, 3].plot(reconstructed_load[i, :, 0], 'r--', linewidth=2, label='Reconstructed')
            axes[i, 3].set_title('Reconstruction Quality')
            axes[i, 3].set_ylabel('Load')
            axes[i, 3].legend()
            axes[i, 3].grid(True, alpha=0.3)
            
            # Set x-axis labels for bottom row
            if i == n_examples - 1:
                for j in range(4):
                    axes[i, j].set_xlabel('Hour of Day')
        
        plt.suptitle('Source Separation Results', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            # Ensure PDF format
            if not str(save_path).endswith('.pdf'):
                save_path = str(save_path).replace('.png', '.pdf')
            plt.savefig(save_path, bbox_inches='tight')

        return fig
        
    def plot_performance_comparison(self,
                                  results_dict: Dict[str, Dict],
                                  metrics: List[str] = ['accuracy', 'nmi', 'ari'],
                                  title: str = "Performance Comparison",
                                  save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create performance comparison bar plot.
        
        Args:
            results_dict: Dictionary of method results
            metrics: List of metrics to compare
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        # Prepare data for plotting
        methods = list(results_dict.keys())
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
            
        for i, metric in enumerate(metrics):
            values = []
            method_names = []
            
            for method, results in results_dict.items():
                if 'clustering_metrics' in results and metric in results['clustering_metrics']:
                    values.append(results['clustering_metrics'][metric])
                    method_names.append(method)
            
            # Create bar plot
            bars = axes[i].bar(range(len(values)), values, alpha=0.7)
            axes[i].set_title(f'{metric.upper()}')
            axes[i].set_ylabel(metric.upper())
            axes[i].set_xticks(range(len(method_names)))
            axes[i].set_xticklabels(method_names, rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for j, (bar, value) in enumerate(zip(bars, values)):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            # Ensure PDF format
            if not str(save_path).endswith('.pdf'):
                save_path = str(save_path).replace('.png', '.pdf')
            plt.savefig(save_path, bbox_inches='tight')

        return fig

    def plot_attention_gates(self,
                           gate_values: np.ndarray,
                           weather_data: np.ndarray,
                           title: str = "Attention Gate Visualization",
                           save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Visualize attention gate values alongside weather data.

        Args:
            gate_values: Attention gate values over time
            weather_data: Weather data (temperature, humidity)
            title: Plot title
            save_path: Path to save the figure

        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        hours = np.arange(len(gate_values))

        # Plot gate values
        ax1.plot(hours, gate_values, 'b-', linewidth=2, label='Gate Values')
        ax1.set_title('Learned Gate Values')
        ax1.set_ylabel('Gate Value')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot weather data
        ax2.plot(hours, weather_data[:, 0], 'r-', linewidth=2, label='Temperature')
        if weather_data.shape[1] > 1:
            ax2_twin = ax2.twinx()
            ax2_twin.plot(hours, weather_data[:, 1], 'g-', linewidth=2, label='Humidity')
            ax2_twin.set_ylabel('Humidity (%)', color='g')
            ax2_twin.legend(loc='upper right')

        ax2.set_title('Weather Conditions')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Temperature (°C)', color='r')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_path:
            # Ensure PDF format
            if not str(save_path).endswith('.pdf'):
                save_path = str(save_path).replace('.png', '.pdf')
            plt.savefig(save_path, bbox_inches='tight')

        return fig

    def plot_sensitivity_analysis(self,
                                param_values: np.ndarray,
                                performance_values: np.ndarray,
                                param_name: str,
                                metric_name: str = "Accuracy",
                                title: str = "Sensitivity Analysis",
                                save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Plot sensitivity analysis results.

        Args:
            param_values: Parameter values tested
            performance_values: Corresponding performance values
            param_name: Name of the parameter
            metric_name: Name of the performance metric
            title: Plot title
            save_path: Path to save the figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        ax.plot(param_values, performance_values, 'bo-', linewidth=2, markersize=8)
        ax.set_title(f'{title}: {metric_name} vs {param_name}')
        ax.set_xlabel(param_name)
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)

        # Highlight best performance
        best_idx = np.argmax(performance_values)
        ax.plot(param_values[best_idx], performance_values[best_idx],
               'ro', markersize=12, label=f'Best: {param_values[best_idx]:.3f}')
        ax.legend()

        plt.tight_layout()

        if save_path:
            # Ensure PDF format
            if not str(save_path).endswith('.pdf'):
                save_path = str(save_path).replace('.png', '.pdf')
            plt.savefig(save_path, bbox_inches='tight')

        return fig

    def plot_failure_cases(self,
                         load_profiles: np.ndarray,
                         uncertainty_scores: np.ndarray,
                         predicted_labels: np.ndarray,
                         true_labels: Optional[np.ndarray] = None,
                         n_cases: int = 3,
                         title: str = "Failure Case Analysis",
                         save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Visualize failure cases with high uncertainty.

        Args:
            load_profiles: Load profile data
            uncertainty_scores: Uncertainty scores for each sample
            predicted_labels: Predicted cluster labels
            true_labels: True cluster labels (optional)
            n_cases: Number of failure cases to show
            title: Plot title
            save_path: Path to save the figure

        Returns:
            matplotlib Figure object
        """
        # Find samples with highest uncertainty
        high_uncertainty_idx = np.argsort(uncertainty_scores)[-n_cases:]

        fig, axes = plt.subplots(n_cases, 1, figsize=(12, 4*n_cases))
        if n_cases == 1:
            axes = [axes]

        for i, idx in enumerate(high_uncertainty_idx):
            # Plot the load profile
            axes[i].plot(load_profiles[idx, :, 0], 'b-', linewidth=2, alpha=0.7)

            title_text = f'Sample {idx}: Uncertainty = {uncertainty_scores[idx]:.3f}'
            if true_labels is not None:
                title_text += f', True: {true_labels[idx]}, Pred: {predicted_labels[idx]}'
            else:
                title_text += f', Predicted: {predicted_labels[idx]}'

            axes[i].set_title(title_text)
            axes[i].set_ylabel('Load')
            axes[i].grid(True, alpha=0.3)

            if i == n_cases - 1:
                axes[i].set_xlabel('Hour of Day')

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_path:
            # Ensure PDF format
            if not str(save_path).endswith('.pdf'):
                save_path = str(save_path).replace('.png', '.pdf')
            plt.savefig(save_path, bbox_inches='tight')

        return fig

    def create_comprehensive_report(self,
                                  results_dict: Dict[str, Any],
                                  output_dir: Union[str, Path],
                                  dataset_name: str = "Dataset") -> None:
        """
        Create a comprehensive visualization report.

        Args:
            results_dict: Dictionary containing all experiment results
            output_dir: Directory to save all figures
            dataset_name: Name of the dataset for titles
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Creating comprehensive visualization report in {output_dir}")

        # Performance comparison
        if 'baselines' in results_dict or 'causal_hes' in results_dict:
            all_results = {}
            if 'causal_hes' in results_dict:
                all_results['CausalHES'] = results_dict['causal_hes']
            if 'baselines' in results_dict:
                all_results.update(results_dict['baselines'])
            if 'ablations' in results_dict:
                all_results.update(results_dict['ablations'])

            self.plot_performance_comparison(
                all_results,
                title=f"{dataset_name} Performance Comparison",
                save_path=output_dir / "performance_comparison.pdf"
            )

        print("Comprehensive visualization report created successfully!")
