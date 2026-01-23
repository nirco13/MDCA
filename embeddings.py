
import torch
import torch.nn as nn


class ItemEmbedding(nn.Module):
    """
    Item embeddings with positional encoding.

    Combines item embeddings and position embeddings to create
    sequence-aware item representations.
    """

    def __init__(self, item_size: int, hidden_size: int, max_seq_length: int):
        """
        Initialize item embedding layer.

        Args:
            item_size: Number of unique items in vocabulary
            hidden_size: Dimension of embedding vectors
            max_seq_length: Maximum sequence length for positional encoding
        """
        super().__init__()
        self.item_embedding = nn.Embedding(
            item_size, hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(
            max_seq_length, hidden_size
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for item embeddings.

        Args:
            input_ids: [batch_size, seq_length] - Item IDs

        Returns:
            embeddings: [batch_size, seq_length, hidden_size] - Item + position embeddings
        """
        # Get item embeddings
        item_emb = self.item_embedding(input_ids)

        # Get position embeddings
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=input_ids.device)
        position_emb = self.position_embedding(position_ids)

        # Combine item and position embeddings
        return item_emb + position_emb


class ContextEmbedding(nn.Module):
    """
    Context embeddings for multi-modal one-hot encoded features.

    Handles all context features with linear projections:
    - Temporal: one-hot encoded time features (hour, day, time-of-day, etc.)
    - Categorical: one-hot encoded POI categories
    - Geospatial: normalized latitude, longitude [0, 1]
    - Weather: one-hot encoded conditions, temperature bins, wind bins + continuous precip
    """

    def __init__(self, args):
        """
        Initialize context embedding layer for one-hot features.

        Args:
            args: Configuration object with dynamically computed dimensions:
                - hidden_size: Dimension of embedding vectors
                - use_time: Whether to use temporal features
                - use_category: Whether to use category features
                - use_geospatial: Whether to use geospatial features
                - use_weather: Whether to use weather features
                - time_dim: Dimension of one-hot time features
                - category_dim: Dimension of one-hot category features
                - conditions_dim: Dimension of one-hot weather conditions
                - temp_dim: Dimension of one-hot temperature bins
                - wind_dim: Dimension of one-hot wind bins
                - geo_dim: Dimension of geospatial features (2)
                - precip_dim: Dimension of continuous precipitation (1)
        """
        super().__init__()
        self.hidden_size = args.hidden_size
        self.use_time = getattr(args, 'use_time', False)
        self.use_category = getattr(args, 'use_category', False)
        self.use_geospatial = getattr(args, 'use_geospatial', False)
        self.use_weather = getattr(args, 'use_weather', False)

        # Temporal projection
        if self.use_time:
            time_dim = getattr(args, 'time_dim', 0)
            if time_dim > 0:
                self.time_projection = nn.Linear(time_dim, self.hidden_size)

        # Category projection
        if self.use_category:
            category_dim = getattr(args, 'category_dim', 0)
            if category_dim > 0:
                self.category_projection = nn.Linear(category_dim, self.hidden_size)

        # Geospatial projection
        if self.use_geospatial:
            geo_dim = getattr(args, 'geo_dim', 2)
            if geo_dim > 0:
                self.geo_projection = nn.Linear(geo_dim, self.hidden_size)

        # Weather projections
        if self.use_weather:
            # Weather conditions
            conditions_dim = getattr(args, 'conditions_dim', 0)
            if conditions_dim > 0:
                self.conditions_projection = nn.Linear(conditions_dim, self.hidden_size)

            # Temperature bins
            temp_dim = getattr(args, 'temp_dim', 0)
            if temp_dim > 0:
                self.temp_projection = nn.Linear(temp_dim, self.hidden_size)

            # Wind bins
            wind_dim = getattr(args, 'wind_dim', 0)
            if wind_dim > 0:
                self.wind_projection = nn.Linear(wind_dim, self.hidden_size)

            # Precipitation
            precip_dim = getattr(args, 'precip_dim', 0)
            if precip_dim > 0:
                self.precip_projection = nn.Linear(precip_dim, self.hidden_size)

    def forward(self, time_features=None, category_features=None,
                geospatial_features=None, condition_features=None,
                temperature_features=None, wind_features=None, precip=None,
                ):
        """
        Forward pass for context embeddings with one-hot encoded features.

        Creates context representation by fusing all enabled context features.
        Features are combined additively after projection to hidden_size.

        Args:
            time_features: [batch_size, seq_length, time_dim] - One-hot time features
            category_features: [batch_size, seq_length, category_dim] - One-hot category features
            geospatial_features: [batch_size, seq_length, 2] - Normalized [lat, lon]
            condition_features: [batch_size, seq_length, conditions_dim] - One-hot conditions
            temperature_features: [batch_size, seq_length, temp_dim] - One-hot temp bins
            wind_features: [batch_size, seq_length, wind_dim] - One-hot wind bins
            precip: [batch_size, seq_length, 1] - Continuous precipitation

            Legacy parameters (for backward compatibility - ignored):
                time_ids, weekday_ids, category_ids, lat, lon, temp, windspeed,
                temp_category, weather_condition

        Returns:
            context_repr: [batch_size, seq_length, hidden_size] - Fused context representation
                         Returns None if no context features are enabled
        """
        context_features = []

        # Temporal features
        if self.use_time and time_features is not None:
            if hasattr(self, 'time_projection'):
                context_features.append(self.time_projection(time_features))

        # Category features
        if self.use_category and category_features is not None:
            if hasattr(self, 'category_projection'):
                context_features.append(self.category_projection(category_features))

        # Geospatial features
        if self.use_geospatial and geospatial_features is not None:
            if hasattr(self, 'geo_projection'):
                context_features.append(self.geo_projection(geospatial_features))

        # Weather condition features
        if self.use_weather and condition_features is not None:
            if hasattr(self, 'conditions_projection'):
                context_features.append(self.conditions_projection(condition_features))

        # Temperature features
        if self.use_weather and temperature_features is not None:
            if hasattr(self, 'temp_projection'):
                context_features.append(self.temp_projection(temperature_features))

        # Wind features
        if self.use_weather and wind_features is not None:
            if hasattr(self, 'wind_projection'):
                context_features.append(self.wind_projection(wind_features))

        # Precipitation
        if self.use_weather and precip is not None:
            if hasattr(self, 'precip_projection'):
                context_features.append(self.precip_projection(precip))

        # Fuse all features
        if context_features:
            return sum(context_features)
        else:
            return None
