"""
Base autoencoder architectures for CausalHES framework.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class BaseAutoencoder(nn.Module):
        """
        Base autoencoder class for household energy segmentation.
        """

        def __init__(self, input_shape, embedding_dim):
            """
            Initialize the base autoencoder.

            Args:
                input_shape: Shape of input data (timesteps, features)
                embedding_dim: Dimension of the learned embeddings
            """
            super(BaseAutoencoder, self).__init__()
            self.input_shape = input_shape
            self.embedding_dim = embedding_dim

            # Calculate input size
            self.input_size = input_shape[0] * input_shape[1]

            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(self.input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, embedding_dim),
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, self.input_size),
            )

        def encode(self, x):
            """Encode input to embedding space."""
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
            return self.encoder(x)

        def decode(self, z):
            """Decode embedding back to input space."""
            x = self.decoder(z)
            return x.view(-1, *self.input_shape)

        def forward(self, x):
            """Forward pass through autoencoder."""
            z = self.encode(x)
            x_recon = self.decode(z)
            return x_recon, z

except ImportError:
    # PyTorch not available, create placeholder
    class BaseAutoencoder:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required but not installed. Please install with: pip install torch torchvision torchaudio"
            )
