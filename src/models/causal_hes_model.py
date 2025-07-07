"""
CausalHES model implementation - Causal Source Separation Autoencoder.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from .base import BaseAutoencoder
    from .clustering_layers import ClusteringLayer

    class CausalHESModel(nn.Module):
        """
        CausalHES: Causal Source Separation Autoencoder for household energy segmentation.

        This model disentangles weather-independent base consumption from weather-dependent effects
        using causal inference principles.
        """

        def __init__(
            self,
            load_input_shape,
            weather_input_shape,
            n_clusters,
            base_dim=32,
            weather_effect_dim=16,
            embedding_dim=64,
        ):
            """
            Initialize CausalHES model.

            Args:
                load_input_shape: Shape of load profile input (timesteps, features)
                weather_input_shape: Shape of weather input (timesteps, features)
                n_clusters: Number of clusters for segmentation
                base_dim: Dimension of base load embedding
                weather_effect_dim: Dimension of weather effect embedding
                embedding_dim: Total embedding dimension
            """
            super(CausalHESModel, self).__init__()

            self.load_input_shape = load_input_shape
            self.weather_input_shape = weather_input_shape
            self.n_clusters = n_clusters
            self.base_dim = base_dim
            self.weather_effect_dim = weather_effect_dim
            self.embedding_dim = embedding_dim

            # Load encoder (mixed load -> base + weather effect)
            # Using 1D CNNs as specified in paper for temporal pattern capture
            self.load_encoder = nn.Sequential(
                nn.Conv1d(load_input_shape[1], 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(128, embedding_dim),
            )

            # Weather encoder
            # Using 1D CNNs as specified in paper for temporal pattern capture
            self.weather_encoder = nn.Sequential(
                nn.Conv1d(weather_input_shape[1], 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, weather_effect_dim),
            )

            # Source separation layers (MLPs as specified in paper)
            # Base separator: z_base = S_b(z_mixed) - independent of weather
            self.base_separator = nn.Sequential(
                nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, base_dim)
            )

            # Weather effect separator: z_weather_effect = S_we([z_mixed, z_weather])
            # Conditions on both mixed embedding and weather embedding
            self.weather_separator = nn.Sequential(
                nn.Linear(embedding_dim + weather_effect_dim, 64),
                nn.ReLU(),
                nn.Linear(64, weather_effect_dim),
            )

            # Decoders
            load_input_size = load_input_shape[0] * load_input_shape[1]
            self.load_decoder = nn.Sequential(
                nn.Linear(base_dim + weather_effect_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, load_input_size),
            )

            # Clustering layer (operates on base embeddings only)
            self.clustering_layer = ClusteringLayer(n_clusters, base_dim)

        def encode_load(self, x_load):
            """Encode load profiles to mixed embedding."""
            # x_load shape: [batch_size, timesteps, features]
            # Conv1d expects: [batch_size, features, timesteps]
            x_load_transposed = x_load.transpose(1, 2)
            return self.load_encoder(x_load_transposed)

        def encode_weather(self, x_weather):
            """Encode weather data."""
            # x_weather shape: [batch_size, timesteps, features]
            # Conv1d expects: [batch_size, features, timesteps]
            x_weather_transposed = x_weather.transpose(1, 2)
            return self.weather_encoder(x_weather_transposed)

        def separate_sources(self, mixed_embedding, weather_embedding=None):
            """
            Separate base load and weather effects from mixed embedding.

            Following paper equations 4-5:
            z_base = S_b(z_mixed)  - independent of weather
            z_weather_effect = S_we([z_mixed, z_weather])  - conditioned on weather
            """
            # Base embedding: independent of weather (Eq. 4)
            base_embedding = self.base_separator(mixed_embedding)

            # Weather effect embedding: conditioned on weather (Eq. 5)
            if weather_embedding is not None:
                # Concatenate mixed and weather embeddings as specified in paper
                weather_input = torch.cat([mixed_embedding, weather_embedding], dim=1)
                weather_effect = self.weather_separator(weather_input)
            else:
                # Fallback: use only mixed embedding (for inference without weather)
                # Pad with zeros to match expected input size
                batch_size = mixed_embedding.size(0)
                zero_weather = torch.zeros(
                    batch_size, self.weather_effect_dim, device=mixed_embedding.device
                )
                weather_input = torch.cat([mixed_embedding, zero_weather], dim=1)
                weather_effect = self.weather_separator(weather_input)

            return base_embedding, weather_effect

        def decode_load(self, base_embedding, weather_effect):
            """Reconstruct load from base and weather components."""
            combined = torch.cat([base_embedding, weather_effect], dim=1)
            x_recon = self.load_decoder(combined)
            return x_recon.view(-1, *self.load_input_shape)

        def cluster_assignment(self, base_embedding):
            """Compute cluster assignments from base embeddings."""
            return self.clustering_layer(base_embedding)

        def forward(self, x_load, x_weather=None):
            """
            Forward pass through CausalHES model.

            Args:
                x_load: Load profile data
                x_weather: Weather data (optional)

            Returns:
                Dictionary containing all model outputs
            """
            # Encode load to mixed embedding
            mixed_embedding = self.encode_load(x_load)

            # Encode actual weather if provided
            weather_embedding = None
            if x_weather is not None:
                weather_embedding = self.encode_weather(x_weather)

            # Separate sources (with proper weather conditioning)
            base_embedding, weather_effect_pred = self.separate_sources(
                mixed_embedding, weather_embedding
            )

            # Reconstruct load
            load_reconstruction = self.decode_load(base_embedding, weather_effect_pred)

            # Cluster assignments (based on base embeddings only)
            cluster_probs = self.cluster_assignment(base_embedding)

            return {
                "load_reconstruction": load_reconstruction,
                "base_embedding": base_embedding,
                "weather_effect_pred": weather_effect_pred,
                "weather_embedding": weather_embedding,
                "cluster_probs": cluster_probs,
                "mixed_embedding": mixed_embedding,
            }

except ImportError:
    # PyTorch not available, create placeholder
    class CausalHESModel:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required but not installed. Please install with: pip install torch torchvision torchaudio"
            )
