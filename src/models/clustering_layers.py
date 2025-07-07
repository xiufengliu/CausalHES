"""
Clustering layers for deep embedded clustering.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ClusteringLayer(nn.Module):
        """
        Clustering layer for Deep Embedded Clustering (DEC).
        """

        def __init__(self, n_clusters, embedding_dim, alpha=1.0):
            """
            Initialize clustering layer.

            Args:
                n_clusters: Number of clusters
                embedding_dim: Dimension of input embeddings
                alpha: Degrees of freedom parameter for Student's t-distribution
            """
            super(ClusteringLayer, self).__init__()
            self.n_clusters = n_clusters
            self.embedding_dim = embedding_dim
            self.alpha = alpha

            # Initialize cluster centers
            self.clusters = nn.Parameter(torch.randn(n_clusters, embedding_dim))

        def forward(self, x):
            """
            Compute soft cluster assignments using Student's t-distribution.

            Args:
                x: Input embeddings of shape (batch_size, embedding_dim)

            Returns:
                Soft cluster assignments of shape (batch_size, n_clusters)
            """
            # Compute squared distances to cluster centers
            # x: (batch_size, embedding_dim)
            # clusters: (n_clusters, embedding_dim)
            distances = torch.sum(
                (x.unsqueeze(1) - self.clusters.unsqueeze(0)) ** 2, dim=2
            )

            # Apply Student's t-distribution
            q = 1.0 / (1.0 + distances / self.alpha)
            q = q ** ((self.alpha + 1.0) / 2.0)
            q = q / torch.sum(q, dim=1, keepdim=True)

            return q

        def target_distribution(self, q):
            """
            Compute target distribution P from soft assignments Q.

            Args:
                q: Soft cluster assignments

            Returns:
                Target distribution P
            """
            weight = q**2 / torch.sum(q, dim=0)
            p = weight / torch.sum(weight, dim=1, keepdim=True)
            return p

except ImportError:
    # PyTorch not available, create placeholder
    class ClusteringLayer:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required but not installed. Please install with: pip install torch torchvision torchaudio"
            )
