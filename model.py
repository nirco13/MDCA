"""
Model classes for MDCA.

MDCA = Multi-modal Attention-based Temporal Context-Aware Recommendation

This module contains three model variants:
1. MDCASimple: POI-only baseline (item sequences only)
2. MDCAContext: Context-aware model (items + context, no clustering)
3. MDCA: Full model (items + context + learnable clustering)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from embeddings import ItemEmbedding, ContextEmbedding
from clustering import ContextClusteringModule


class MDCABase(nn.Module):
    """
    Base class for MDCA models.

    Provides shared functionality including:
    - Transformer encoder building
    - Causal attention mask creation
    - Weight initialization
    """

    def __init__(self, args):
        """
        Initialize base model.

        Args:
            args: Configuration object with:
                - hidden_size: Dimension of hidden representations
                - max_seq_length: Maximum sequence length
                - num_attention_heads: Number of attention heads
                - num_hidden_layers: Number of transformer layers
                - hidden_dropout_prob: Dropout probability
                - item_size: Number of unique items
        """
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.max_seq_length = args.max_seq_length
        self.num_attention_heads = args.num_attention_heads
        self.num_layers = args.num_hidden_layers
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        # Item embeddings
        self.item_embedding = ItemEmbedding(
            args.item_size, args.hidden_size, args.max_seq_length
        )

        # Item transformer
        self.item_transformer = self._build_transformer(args)

    def _build_transformer(self, args):
        """Build transformer encoder."""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_size,
            nhead=args.num_attention_heads,
            dim_feedforward=args.hidden_size * 4,
            dropout=args.hidden_dropout_prob,
            activation='gelu',
            batch_first=True
        )
        return nn.TransformerEncoder(
            encoder_layer,
            num_layers=args.num_hidden_layers
        )

    def _create_causal_mask(self, seq_length, device):
        """
        Create causal attention mask.

        Returns a mask where True = mask out (block), False = allow.
        Uses upper triangular to block future positions.
        """
        # Lower triangular (including diagonal) = allowed positions
        allowed_positions = torch.tril(
            torch.ones(seq_length, seq_length, dtype=torch.bool, device=device)
        )
        # Invert: True = mask out, False = allow
        causal_mask = ~allowed_positions
        return causal_mask

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_embedding_regularization_loss(self):
        """L2 regularization for embedding layers."""
        embed_reg = getattr(self.args, 'embed_reg', 1e-6)
        reg_loss = 0.0

        # Item and position embeddings
        reg_loss += torch.norm(self.item_embedding.item_embedding.weight, p=2) ** 2
        reg_loss += torch.norm(self.item_embedding.position_embedding.weight, p=2) ** 2

        return embed_reg * reg_loss

    def _init_context_aware_components(self):
        """
        Initialize shared context-aware components used by both MDCAContext and MDCA.

        This includes:
        - Context embeddings
        - Context transformer
        - Cross-attention layers (item-to-context and context-to-item)
        - Layer normalization layers
        """
        # Context embeddings
        self.context_embedding = ContextEmbedding(self.args)

        # Context transformer
        self.context_transformer = self._build_transformer(self.args)

        # Cross-attention layers
        self.cross_attention_i2c = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_attention_heads,
            dropout=self.args.hidden_dropout_prob,
            batch_first=True
        )

        self.cross_attention_c2i = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_attention_heads,
            dropout=self.args.hidden_dropout_prob,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm_1 = nn.LayerNorm(self.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(self.hidden_size)


class MDCASimple(MDCABase):
    """
    MDCA - Simple POI-only version.

    Uses only item sequences without context information.
    Suitable as a baseline for comparison.
    """

    def __init__(self, args):
        super().__init__(args)

        # Final representation size
        self.final_repr_size = self.hidden_size

        # Output layers
        self.output_layer = nn.Linear(self.hidden_size, self.final_repr_size)
        self.item_projection = nn.Linear(self.hidden_size, self.final_repr_size)

        # Initialize weights
        self.apply(self._init_weights)

    def forward(self, input_ids, return_all_outputs=False, **kwargs):
        """
        Simple forward pass without context.

        Args:
            input_ids: [batch_size, seq_length] - Item IDs
            return_all_outputs: If True, return dict with all outputs; if False, return only final_user_repr
            **kwargs: Additional arguments (ignored)

        Returns:
            If return_all_outputs=True:
                dict with:
                    - final_user_repr: [batch_size, final_repr_size]
                    - item_repr: [batch_size, hidden_size]
            If return_all_outputs=False:
                final_user_repr: [batch_size, final_repr_size]
        """
        # Get item embeddings with positional encoding
        item_embeddings = self.item_embedding(input_ids)
        item_embeddings = self.dropout(item_embeddings)

        # Create causal mask
        seq_length = input_ids.size(1)
        mask = self._create_causal_mask(seq_length, input_ids.device)

        # Encode with transformer
        encoded_items = self.item_transformer(item_embeddings, mask=mask)

        # Get final representation
        item_repr = encoded_items[:, -1, :]
        final_user_repr = self.output_layer(item_repr)

        if return_all_outputs:
            return {
                'final_user_repr': final_user_repr,
                'item_repr': item_repr
            }
        else:
            return final_user_repr


class MDCAContext(MDCABase):
    """
    MDCA - Context-aware model without clustering.

    Uses item sequences + temporal/categorical/geospatial/weather context.
    Includes cross-attention between items and context.
    """

    def __init__(self, args):
        super().__init__(args)

        # Initialize shared context-aware components
        self._init_context_aware_components()

        # Final representation size
        self.final_repr_size = self.hidden_size

        # Output layers (item + context)
        self.output_layer = nn.Linear(2 * self.hidden_size, self.final_repr_size)
        self.item_projection = nn.Linear(self.hidden_size, self.final_repr_size)

        # Initialize weights
        self.apply(self._init_weights)

    def forward(self, input_ids, time_features=None, category_features=None,
                geospatial_features=None, condition_features=None,
                temperature_features=None, wind_features=None, precip=None,
                return_all_outputs=False, **kwargs):
        """
        Forward pass with context.

        Args:
            input_ids: [batch_size, seq_length] - Item IDs
            time_features: [batch_size, seq_length, time_dim] - One-hot time features
            category_features: [batch_size, seq_length, category_dim] - One-hot category features
            geospatial_features: [batch_size, seq_length, 2] - Normalized [lat, lon]
            condition_features: [batch_size, seq_length, conditions_dim] - One-hot conditions
            temperature_features: [batch_size, seq_length, temp_dim] - One-hot temp bins
            wind_features: [batch_size, seq_length, wind_dim] - One-hot wind bins
            precip: [batch_size, seq_length, 1] - Continuous precipitation
            return_all_outputs: If True, return dict with all outputs; if False, return only final_user_repr
            **kwargs: Additional arguments (ignored)

        Returns:
            If return_all_outputs=True:
                dict with:
                    - final_user_repr: [batch_size, final_repr_size]
                    - item_repr: [batch_size, hidden_size]
                    - context_repr: [batch_size, hidden_size]
            If return_all_outputs=False:
                final_user_repr: [batch_size, final_repr_size]
        """
        # Encode items
        item_embeddings = self.item_embedding(input_ids)
        item_embeddings = self.dropout(item_embeddings)

        seq_length = input_ids.size(1)
        mask = self._create_causal_mask(seq_length, input_ids.device)

        encoded_items = self.item_transformer(item_embeddings, mask=mask)

        # Encode context
        context_repr = self.context_embedding(
            time_features=time_features,
            category_features=category_features,
            geospatial_features=geospatial_features,
            condition_features=condition_features,
            temperature_features=temperature_features,
            wind_features=wind_features,
            precip=precip
        )

        if context_repr is None:
            # If no context available - fallback to POI-only
            item_repr = encoded_items[:, -1, :]
            final_user_repr = self.output_layer(
                torch.cat([item_repr, torch.zeros_like(item_repr)], dim=-1)
            )
            if return_all_outputs:
                return {
                    'final_user_repr': final_user_repr,
                    'item_repr': item_repr,
                    'context_repr': None
                }
            else:
                return final_user_repr

        context_repr = self.dropout(context_repr)
        context_encoded = self.context_transformer(context_repr, mask=mask)

        # Cross-attention: items attend to context
        item_attended, _ = self.cross_attention_i2c(
            query=encoded_items,
            key=context_encoded,
            value=context_encoded,
            attn_mask=mask
        )
        encoded_items = self.layer_norm_1(encoded_items + item_attended)

        # Cross-attention: context attends to items
        context_attended, _ = self.cross_attention_c2i(
            query=context_encoded,
            key=encoded_items,
            value=encoded_items,
            attn_mask=mask
        )
        context_encoded = self.layer_norm_2(context_encoded + context_attended)

        # Final representation
        item_repr = encoded_items[:, -1, :]
        context_repr_final = context_encoded[:, -1, :]

        combined = torch.cat([item_repr, context_repr_final], dim=-1)
        final_user_repr = self.output_layer(combined)

        if return_all_outputs:
            return {
                'final_user_repr': final_user_repr,
                'item_repr': item_repr,
                'context_repr': context_repr_final
            }
        else:
            return final_user_repr

    def get_embedding_regularization_loss(self):
        """L2 regularization including context embeddings."""
        reg_loss = super().get_embedding_regularization_loss()
        embed_reg = getattr(self.args, 'embed_reg', 1e-6)

        # Context embeddings
        if hasattr(self.context_embedding, 'time_projection'):
            reg_loss += embed_reg * torch.norm(self.context_embedding.time_projection.weight, p=2) ** 2
        if hasattr(self.context_embedding, 'category_projection'):
            reg_loss += embed_reg * torch.norm(self.context_embedding.category_projection.weight, p=2) ** 2
        if hasattr(self.context_embedding, 'conditions_projection'):
            reg_loss += embed_reg * torch.norm(self.context_embedding.conditions_projection.weight, p=2) ** 2
        if hasattr(self.context_embedding, 'geo_projection'):
            reg_loss += embed_reg * torch.norm(self.context_embedding.geo_projection.weight, p=2) ** 2

        return reg_loss


class MDCA(MDCABase):
    """
    MDCA - Full model with context and learnable clustering.

    Multi-modal Attention-based Temporal Context-Aware Recommendation
    with learnable context clustering initialized by FAISS k-means.
    """

    def __init__(self, args):
        super().__init__(args)

        self.use_clustering = True

        # Initialize shared context-aware components
        self._init_context_aware_components()

        # Clustering module
        self.clustering = ContextClusteringModule(args)

        # Final representation size
        self.final_repr_size = self.hidden_size

        # Output layers (item + context + cluster)
        self.output_layer = nn.Linear(3 * self.hidden_size, self.final_repr_size)
        self.item_projection = nn.Linear(self.hidden_size, self.final_repr_size)

        # Initialize weights
        self.apply(self._init_weights)

    def forward(self, input_ids, time_features=None, category_features=None,
                geospatial_features=None, condition_features=None,
                temperature_features=None, wind_features=None, precip=None,
                return_all_outputs=False, **kwargs):
        """
        Forward pass with context and clustering.

        Args:
            input_ids: [batch_size, seq_length] - Item IDs
            time_features: [batch_size, seq_length, time_dim] - One-hot time features
            category_features: [batch_size, seq_length, category_dim] - One-hot category features
            geospatial_features: [batch_size, seq_length, 2] - Normalized [lat, lon]
            condition_features: [batch_size, seq_length, conditions_dim] - One-hot conditions
            temperature_features: [batch_size, seq_length, temp_dim] - One-hot temp bins
            wind_features: [batch_size, seq_length, wind_dim] - One-hot wind bins
            precip: [batch_size, seq_length, 1] - Continuous precipitation
            return_all_outputs: If True, return dict with all outputs; if False, return only final_user_repr
            **kwargs: Additional arguments (ignored)

        Returns:
            If return_all_outputs=True:
                dict with:
                    - final_user_repr: [batch_size, final_repr_size] - Combined representation
                    - item_repr: [batch_size, hidden_size] - Item representation
                    - context_repr: [batch_size, hidden_size] - Context representation
                    - cluster_embedding: [batch_size, hidden_size] - Cluster center embedding
            If return_all_outputs=False:
                final_user_repr: [batch_size, final_repr_size]
        """
        # Encode items
        item_embeddings = self.item_embedding(input_ids)
        item_embeddings = self.dropout(item_embeddings)

        seq_length = input_ids.size(1)
        mask = self._create_causal_mask(seq_length, input_ids.device)

        encoded_items = self.item_transformer(item_embeddings, mask=mask)

        # Encode context
        context_repr = self.context_embedding(
            time_features=time_features,
            category_features=category_features,
            geospatial_features=geospatial_features,
            condition_features=condition_features,
            temperature_features=temperature_features,
            wind_features=wind_features,
            precip=precip
        )

        if context_repr is None:
            # No context available - should not happen in this mode
            raise ValueError("Context features required for clustering model")

        context_repr = self.dropout(context_repr)
        context_encoded = self.context_transformer(context_repr, mask=mask)

        # Cross-attention: items attend to context
        item_attended, _ = self.cross_attention_i2c(
            query=encoded_items,
            key=context_encoded,
            value=context_encoded,
            attn_mask=mask
        )
        encoded_items = self.layer_norm_1(encoded_items + item_attended)

        # Cross-attention: context attends to items
        context_attended, _ = self.cross_attention_c2i(
            query=context_encoded,
            key=encoded_items,
            value=encoded_items,
            attn_mask=mask
        )
        context_encoded = self.layer_norm_2(context_encoded + context_attended)

        # Extract final representations
        item_repr = encoded_items[:, -1, :]
        context_repr_final = context_encoded[:, -1, :]

        # Get cluster embedding
        cluster_embedding = self.clustering.get_cluster_assignment(
            context_repr_final
        )

        # Combine all representations
        combined = torch.cat(
            [item_repr, context_repr_final, cluster_embedding], dim=-1
        )
        final_user_repr = self.output_layer(combined)

        if return_all_outputs:
            return {
                'final_user_repr': final_user_repr,
                'item_repr': item_repr,
                'context_repr': context_repr_final,
                'cluster_embedding': cluster_embedding
            }
        else:
            return final_user_repr

    def initialize_clustering(self, train_dataloader):
        """
        Initialize clustering with FAISS k-means (epoch 0).

        Args:
            train_dataloader: Training data loader
        """
        self.clustering.initialize_from_faiss(train_dataloader, self)

    def get_clustering_loss(self, context_repr):
        """
        Compute clustering loss.

        Args:
            context_repr: [batch_size, hidden_size] - Context representation (last position)

        Returns:
            loss: Scalar tensor
        """
        return self.clustering.compute_clustering_loss(context_repr)

    def get_embedding_regularization_loss(self):
        """L2 regularization including cluster centers."""
        reg_loss = super().get_embedding_regularization_loss()
        embed_reg = getattr(self.args, 'embed_reg', 1e-6)

        # Cluster centers
        if hasattr(self.clustering, 'cluster_centers'):
            reg_loss += embed_reg * torch.norm(self.clustering.cluster_centers, p=2) ** 2

        return reg_loss


def create_mdca_model(config: str, args):
    """
    Factory function to create the appropriate MDCA model variant.

    Args:
        config: One of 'poi_only', 'context', 'context_clustering'
        args: Configuration object

    Returns:
        model: Instance of appropriate MDCA model class
    """
    if config == 'poi_only':
        return MDCASimple(args)
    elif config == 'context':
        return MDCAContext(args)
    elif config == 'context_clustering':
        return MDCA(args)
    else:
        raise ValueError(
            f"Unknown config: {config}. "
            f"Choose from: poi_only, context, context_clustering"
        )
