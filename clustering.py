
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import numpy as np
from typing import Tuple


class ContextClusteringModule(nn.Module):
    """
    Learnable context clustering module.

    Features:
    - FAISS k-means initialization (epoch 0)
    - Learnable cluster centers (gradient-based updates)
    - Hard cluster assignment via L2 distance
    - Compactness + separation loss
    """

    def __init__(self, args):
        """
        Initialize clustering module.

        Args:
            args: Configuration object with:
                - num_clusters: Number of clusters
                - hidden_size: Dimension of embeddings
                - device: PyTorch device
                - compactness_weight: Weight for compactness loss (default: 1.0)
                - separation_weight: Weight for separation loss (default: 1.0)
        """
        super().__init__()
        self.num_clusters = args.num_clusters
        self.hidden_size = args.hidden_size
        self.device = args.device

        # Loss weights
        self.compactness_weight = getattr(args, 'compactness_weight', 1.0)
        self.separation_weight = getattr(args, 'separation_weight', 1.0)

        # Learnable cluster centers
        self.cluster_centers = nn.Parameter(
            torch.randn(self.num_clusters, self.hidden_size)
        )
        nn.init.normal_(self.cluster_centers, mean=0.0, std=0.02)

    def initialize_from_faiss(self, dataloader, model):
        """
        Initialize learnable centers using FAISS k-means (epoch 0 only).

        This method collects context embeddings from the training data,
        runs FAISS k-means to find good initial cluster centers, and
        copies them to the learnable parameters.

        Args:
            dataloader: Training data loader
            model: The model instance (to extract embeddings)
        """
        print("Initializing clustering")
        model.eval()

        # Collect context embeddings
        all_embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                        for k, v in batch.items()}

                # Get context representation
                outputs = model(**batch, return_all_outputs=True)
                if isinstance(outputs, dict) and 'context_repr' in outputs:
                    context_repr = outputs['context_repr']
                    all_embeddings.append(context_repr)

        # Convert to numpy for FAISS
        embeddings_np = torch.cat(all_embeddings, dim=0).cpu().numpy().astype(np.float32)

        # Normalize embeddings
        faiss.normalize_L2(embeddings_np)

        # Run FAISS k-means
        kmeans = faiss.Clustering(self.hidden_size, self.num_clusters)
        kmeans.niter = 20  # K-means iterations
        kmeans.seed = 42  # For reproducibility
        kmeans.verbose = False

        # Create index for k-means
        index = faiss.IndexFlatL2(self.hidden_size)
        kmeans.train(embeddings_np, index)

        # Extract centroids
        centroids = faiss.vector_to_array(kmeans.centroids).reshape(
            self.num_clusters, self.hidden_size
        )

        # Set learnable parameters
        self.cluster_centers.data = torch.tensor(
            centroids, device=self.device, dtype=torch.float32
        )
        self.cluster_centers.data = F.normalize(self.cluster_centers.data, p=2, dim=1)

        print(f"✓ Initialized {self.num_clusters} clusters with FAISS")
        model.train()

    def get_cluster_assignment(
        self, context_repr: torch.Tensor
    ) -> torch.Tensor:
        """
        Get hard cluster assignment via L2 distance.

        Args:
            context_repr: [batch_size, hidden_size] - Context representation

        Returns:
            cluster_embedding: [batch_size, hidden_size]
        """
        # Normalize for stable L2 distance
        context_norm = F.normalize(context_repr, p=2, dim=1)
        centers_norm = F.normalize(self.cluster_centers, p=2, dim=1)

        # Compute L2 distances to all clusters
        distances = torch.cdist(context_norm, centers_norm, p=2)

        # Hard assignment (argmin)
        hard_assignments = torch.argmin(distances, dim=-1)

        # One-hot encoding
        cluster_assignment_one_hot = F.one_hot(
            hard_assignments, num_classes=self.num_clusters
        ).float()

        # Get cluster embeddings (nearest cluster center for each sample)
        cluster_embedding = self.cluster_centers[hard_assignments]

        return cluster_embedding

    def compute_clustering_loss(self, context_repr: torch.Tensor) -> torch.Tensor:
        """
        Compute distance-based clustering loss.

        Loss = compactness_loss + separation_loss
        - Compactness: minimize distance to nearest cluster
        - Separation: maximize distance between cluster centers

        Args:
            context_repr: [batch_size, hidden_size]

        Returns:
            loss: Scalar tensor
        """
        # Normalize
        context_norm = F.normalize(context_repr, p=2, dim=1)
        centers_norm = F.normalize(self.cluster_centers, p=2, dim=1)

        # 1. Compactness loss: minimize distance to nearest cluster
        distances = torch.cdist(context_norm, centers_norm, p=2)
        min_distances = torch.min(distances, dim=-1)[0]
        compactness_loss = torch.mean(min_distances)

        # 2. Separation loss: maximize inter-cluster distances
        center_distances = torch.cdist(centers_norm, centers_norm, p=2)

        # Mask out diagonal
        mask = ~torch.eye(self.num_clusters, dtype=torch.bool,
                         device=center_distances.device)
        separation_loss = -torch.mean(center_distances[mask])

        # Combined loss
        total_loss = (self.compactness_weight * compactness_loss +
                     self.separation_weight * separation_loss)

        return total_loss
