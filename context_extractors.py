
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple


class ContextExtractor:
    """
    Unified context extraction from user sequences.

    Extracts all context features (temporal, categorical, geospatial, weather, etc..)
    from user interaction sequences with consistent padding and alignment.
    """

    def __init__(self, args):
        """
        Initialize context extractor.

        Args:
            args: Configuration object with:
                - max_seq_length: Maximum sequence length
                - use_time: Whether to extract temporal features
                - use_category: Whether to extract category features
                - use_geospatial: Whether to extract geospatial features
                - use_weather: Whether to extract weather features
        """
        self.args = args
        self.max_len = args.max_seq_length
        self.use_time = getattr(args, 'use_time', False)
        self.use_category = getattr(args, 'use_category', False)
        self.use_geospatial = getattr(args, 'use_geospatial', False)
        self.use_weather = getattr(args, 'use_weather', False)

        # Column mapping for normalization
        self.column_mapping = {
            'userid': ['userid', 'user_id'],
            'placeid': ['placeid', 'venue_id', 'place_id', 'fsq_id'],
            'datetime': ['datetime', 'timestamp'],
            'lat': ['lat', 'latitude'],
            'lon': ['lon', 'longitude'],
        }

        # Scalers for geospatial normalization (fitted during preprocess)
        self.lat_min = None
        self.lat_max = None
        self.lon_min = None
        self.lon_max = None

    def _extract_feature_group(self, user_seq: dict, seq_len: int,
                               feature_columns: Dict[str, List[str]],
                               feature_type: str) -> Optional[torch.Tensor]:
        """
        Generic helper method to extract and stack a group of features.

        Args:
            user_seq: User sequence dictionary
            seq_len: Sequence length to extract
            feature_columns: Dictionary mapping feature types to column names
            feature_type: Type of feature group (e.g., 'time', 'category', 'conditions')

        Returns:
            Stacked tensor of shape [max_len, num_features] or None if no features found
        """
        if feature_type not in feature_columns:
            return None

        feature_matrix = []
        for col in feature_columns[feature_type]:
            if col in user_seq:
                values = user_seq[col][:seq_len]
                padded = self._align_and_pad(values, dtype=torch.float, pad_value=0.0)
                feature_matrix.append(padded)

        if feature_matrix:
            return torch.stack(feature_matrix, dim=1)
        return None

    def extract(self, user_seq: dict, seq_len: int,
                feature_columns: Dict[str, List[str]] = None,
                target_index: int = None):
        """
        Extract all context features from user sequence.

        Args:
            user_seq: User sequence dictionary with one-hot column values
            seq_len: Length of sequence to extract
            feature_columns: Dictionary mapping feature types to column names (required for one-hot mode)
            target_index: Optional target index for evaluation (not used currently)

        Returns:
            context_data: Dictionary of tensors, e.g.:
                {'time_features': [max_len, time_dim],
                 'category_features': [max_len, category_dim],
                 'geospatial_features': [max_len, 2],
                 ...}
        """

        context_data = {}

        # Extract time features (one-hot + hour/day_of_week)
        if self.use_time:
            time_features = self._extract_feature_group(user_seq, seq_len, feature_columns, 'time')
            if time_features is not None:
                context_data['time_features'] = time_features

        # Extract category features
        if self.use_category:
            category_features = self._extract_feature_group(user_seq, seq_len, feature_columns, 'category')
            if category_features is not None:
                context_data['category_features'] = category_features

        # Extract geospatial features (SCALED to [0, 1])
        if self.use_geospatial:
            if 'lat' in user_seq and 'lon' in user_seq:
                lats = user_seq['lat'][:seq_len]
                lons = user_seq['lon'][:seq_len]

                lat_padded = self._align_and_pad(lats, dtype=torch.float, pad_value=0.0)
                lon_padded = self._align_and_pad(lons, dtype=torch.float, pad_value=0.0)

                # Stack into [max_len, 2] tensor
                context_data['geospatial_features'] = torch.stack([lat_padded, lon_padded], dim=1)

        # Extract weather condition features
        if self.use_weather:
            condition_features = self._extract_feature_group(user_seq, seq_len, feature_columns, 'conditions')
            if condition_features is not None:
                context_data['condition_features'] = condition_features

            # Extract temperature features
            temp_features = self._extract_feature_group(user_seq, seq_len, feature_columns, 'temperature')
            if temp_features is not None:
                context_data['temperature_features'] = temp_features

            # Extract wind features
            wind_features = self._extract_feature_group(user_seq, seq_len, feature_columns, 'wind')
            if wind_features is not None:
                context_data['wind_features'] = wind_features

        # Extract precipitation
        if self.use_weather and 'precip' in user_seq:
            precip_values = user_seq['precip'][:seq_len]
            precip_padded = self._align_and_pad(precip_values, dtype=torch.float, pad_value=0.0)
            context_data['precip'] = precip_padded.unsqueeze(1)  # [max_len, 1]

        return context_data

    def _align_and_pad(self, sequence: List, dtype=torch.long,
                       pad_value=0) -> torch.Tensor:
        """
        Align sequence to max_len with left padding.

        Sequences longer than max_len are truncated from the left (keeping recent items).
        Sequences shorter than max_len are left-padded.

        Args:
            sequence: List of values
            dtype: Target tensor dtype (torch.long or torch.float)
            pad_value: Value for padding (0 for most cases)

        Returns:
            padded_tensor: [max_len] tensor with left-padded values
        """
        if len(sequence) == 0:
            return torch.full((self.max_len,), pad_value, dtype=dtype)

        # Truncate from left if too long (keep recent items)
        if len(sequence) > self.max_len:
            aligned = sequence[-self.max_len:]
        else:
            aligned = sequence

        # Create padded tensor
        padded = torch.full((self.max_len,), pad_value, dtype=dtype)

        # Fill from right (left-padded)
        start_idx = self.max_len - len(aligned)

        if dtype == torch.long:
            padded[start_idx:] = torch.LongTensor(aligned)
        elif dtype == torch.float:
            padded[start_idx:] = torch.FloatTensor(aligned)
        else:
            padded[start_idx:] = torch.tensor(aligned, dtype=dtype)

        return padded

    # ========================================================================
    # PREPROCESSING METHODS (ONE-HOT ENCODING)
    # ========================================================================

    def preprocess(self,
                   data_source: Union[str, pd.DataFrame],
                   min_interactions: int = 5) -> Tuple[List[dict], int, dict]:
        """
        Preprocess raw dataset into user sequences with one-hot encoded features.

        Pipeline:
        1. Load DataFrame (from file or validate existing)
        2. Normalize column names (venue_id→placeid, timestamp→datetime, etc.)
        3. Apply one-hot encoding transformations:
           - Timestamp → time-of-day and weekday/weekend binary features
           - Category → one-hot dummy columns
           - Conditions → one-hot dummy columns
           - Temperature → binned one-hot columns
           - Wind speed → binned one-hot columns
        4. Normalize lat/lon to [0, 1] range
        5. Group by user and sort by timestamp
        6. Filter users with minimum interactions

        Args:
            data_source: File path (str) or DataFrame
            min_interactions: Minimum interactions per user and item (default: 5)

        Returns:
            user_seq_list: List of user sequence dictionaries
            max_item_id: Maximum item ID
            feature_columns: Dictionary mapping feature types to their one-hot column names
        """
        # Step 1: Load data
        df = self._load_dataframe(data_source)

        # Step 2: Normalize column names
        df = self._normalize_column_names(df)
        print(f"Raw dataset: {len(df)} interactions, {df['userid'].nunique()} users, {df['placeid'].nunique()} places")

        # Step 2.5: Filter places with minimum interactions
        place_counts = df['placeid'].value_counts()
        valid_places = place_counts[place_counts >= min_interactions].index
        df = df[df['placeid'].isin(valid_places)]
        print(f"After filtering places (>={min_interactions} interactions): {len(df)} interactions, {df['placeid'].nunique()} places")

        # Step 2.6: Filter users with minimum interactions
        user_counts = df['userid'].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        df = df[df['userid'].isin(valid_users)]
        print(f"After filtering users (>={min_interactions} interactions): {len(df)} interactions, {df['userid'].nunique()} users")

        # Step 3: Apply one-hot encoding transformations
        feature_columns = {}

        # Time features
        df, time_cols = self._create_time_features(df)
        if time_cols:
            feature_columns['time'] = time_cols

        # Category one-hot encoding
        df, cat_cols = self._create_category_dummies(df)
        if cat_cols:
            feature_columns['category'] = cat_cols

        # Weather condition one-hot encoding
        df, cond_cols = self._create_conditions_dummies(df)
        if cond_cols:
            feature_columns['conditions'] = cond_cols

        # Temperature binning and one-hot encoding
        df, temp_cols = self._create_temperature_dummies(df)
        if temp_cols:
            feature_columns['temperature'] = temp_cols

        # Wind speed binning and one-hot encoding
        df, wind_cols = self._create_wind_dummies(df)
        if wind_cols:
            feature_columns['wind'] = wind_cols

        # Step 4: Normalize lat/lon to [0, 1]
        if self.use_geospatial:
            df = self._normalize_geospatial(df)

        # Step 5: Group by user
        user_seq_list = self._group_by_user(df, feature_columns)

        # Step 6: Compute max item ID
        all_items = set()
        for user_seq in user_seq_list:
            all_items.update(user_seq['items'])
        max_item_id = max(all_items) if all_items else 0

        print(f"Final dataset: {len(user_seq_list)} users")
        print(f"Max item ID: {max_item_id}")
        print(f"Feature groups: {list(feature_columns.keys())}")
        for feature_type, columns in feature_columns.items():
            print(f"  {feature_type}: {len(columns)} columns")

        return user_seq_list, max_item_id, feature_columns

    def _load_dataframe(self, data_source: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """Load DataFrame from file path or validate existing DataFrame."""
        if isinstance(data_source, str):
            return pd.read_csv(data_source)
        elif isinstance(data_source, pd.DataFrame):
            return data_source.copy()
        else:
            raise ValueError(f"data_source must be str or DataFrame, got {type(data_source)}")

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to standard format.

        Maps variations like:
        - venue_id → placeid
        - timestamp → datetime
        - latitude → lat, longitude → lon
        """
        for standard_name, alternatives in self.column_mapping.items():
            for alt_name in alternatives:
                if alt_name in df.columns and standard_name not in df.columns:
                    df.rename(columns={alt_name: standard_name}, inplace=True)
                    break

        # Validate required columns
        required_cols = ['userid', 'placeid', 'datetime']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df

    def _create_time_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Parse timestamp and create time-of-day binary features.

        Creates columns:
        - hour (0-23)
        - day_of_week (0-6, Monday=0)
        - is_morning, is_noon, is_afternoon, is_evening, is_night (binary)
        - is_weekday, is_weekend (binary)
        """
        if not self.use_time:
            return df, []

        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek

        # Time of day categories
        morning_hours = [6, 7, 8, 9, 10, 11]
        noon_hours = [12, 13]
        afternoon_hours = [14, 15, 16, 17]
        evening_hours = [18, 19, 20]
        night_hours = [21, 22, 23, 0, 1, 2, 3, 4, 5]

        df['is_morning'] = np.where(df['hour'].isin(morning_hours), 1, 0)
        df['is_noon'] = np.where(df['hour'].isin(noon_hours), 1, 0)
        df['is_afternoon'] = np.where(df['hour'].isin(afternoon_hours), 1, 0)
        df['is_evening'] = np.where(df['hour'].isin(evening_hours), 1, 0)
        df['is_night'] = np.where(df['hour'].isin(night_hours), 1, 0)
        df['is_weekday'] = np.where(df['day_of_week'] < 5, 1, 0)
        df['is_weekend'] = np.where(df['day_of_week'] >= 5, 1, 0)

        time_columns = ['hour', 'day_of_week', 'is_morning', 'is_noon',
                        'is_afternoon', 'is_evening', 'is_night',
                        'is_weekday', 'is_weekend']
        return df, time_columns

    def _create_category_dummies(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        One-hot encode category column.

        Creates columns: cat_<CategoryName> for each unique category
        """
        if not self.use_category or 'category' not in df.columns:
            return df, []

        ohe_df = df['category'].str.get_dummies(sep=', ').add_prefix('cat_')
        df = pd.concat([df, ohe_df], axis=1)

        category_columns = ohe_df.columns.tolist()
        return df, category_columns

    def _create_conditions_dummies(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        One-hot encode weather conditions column.

        Creates columns: condition_<ConditionName> for each unique condition
        Handles comma-separated multiple conditions.
        """
        if not self.use_weather or 'conditions' not in df.columns:
            return df, []

        ohe_df = df['conditions'].str.get_dummies(sep=', ').add_prefix('condition_')
        df = pd.concat([df, ohe_df], axis=1)

        condition_columns = ohe_df.columns.tolist()
        return df, condition_columns

    def _create_temperature_dummies(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Bin temperature into categories and one-hot encode.

        Bins: Freezing (<0°C), Cold (0-10°C), Mild (10-20°C), Warm (20-30°C), Hot (>30°C)
        Creates columns: temp_Freezing, temp_Cold, temp_Mild, temp_Warm, temp_Hot
        """
        if not self.use_weather or 'temp' not in df.columns:
            return df, []

        bins = [-np.inf, 0, 10, 20, 30, np.inf]
        labels = ['Freezing', 'Cold', 'Mild', 'Warm', 'Hot']

        temp_category = pd.cut(df['temp'], bins=bins, labels=labels)

        temp_columns = []
        for label in labels:
            col_name = f"temp_{label}"
            df[col_name] = (temp_category == label).astype(int)
            temp_columns.append(col_name)

        return df, temp_columns

    def _create_wind_dummies(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Bin wind speed into categories and one-hot encode.

        Bins: Very slow (<18 km/h), Slow (18-36), Considerable (36-72), Very windy (>72)
        Creates columns: wind_VerySlow, wind_Slow, wind_Considerable, wind_VeryWindy
        """
        if not self.use_weather or 'windspeed' not in df.columns:
            return df, []

        bins = [-np.inf, 18, 36, 72, np.inf]
        labels = ['VerySlow', 'Slow', 'Considerable', 'VeryWindy']

        wind_category = pd.cut(df['windspeed'], bins=bins, labels=labels)

        wind_columns = []
        for label in labels:
            col_name = f"wind_{label}"
            df[col_name] = (wind_category == label).astype(int)
            wind_columns.append(col_name)

        return df, wind_columns

    def _normalize_geospatial(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize latitude and longitude to [0, 1] range using min-max scaling.

        Stores min/max values in self for consistent normalization at inference time.
        Creates columns: lat_scaled, lon_scaled

        Formula: scaled = (value - min) / (max - min)
        """
        if 'lat' not in df.columns or 'lon' not in df.columns:
            print("Warning: Geospatial columns (lat/lon) not found. Skipping normalization.")
            return df

        # Compute and store min/max for later use (inference time)
        self.lat_min = df['lat'].min()
        self.lat_max = df['lat'].max()
        self.lon_min = df['lon'].min()
        self.lon_max = df['lon'].max()

        # Min-max normalization to [0, 1]
        # Handle edge case where all values are the same (avoid division by zero)
        if self.lat_max == self.lat_min:
            df['lat_scaled'] = 0.5  # Set to middle of range
        else:
            df['lat_scaled'] = (df['lat'] - self.lat_min) / (self.lat_max - self.lat_min)

        if self.lon_max == self.lon_min:
            df['lon_scaled'] = 0.5
        else:
            df['lon_scaled'] = (df['lon'] - self.lon_min) / (self.lon_max - self.lon_min)

        print(f"Geospatial normalization:")
        print(f"  Latitude: [{self.lat_min:.4f}, {self.lat_max:.4f}] → [0, 1]")
        print(f"  Longitude: [{self.lon_min:.4f}, {self.lon_max:.4f}] → [0, 1]")

        return df

    def _group_by_user(self, df: pd.DataFrame, feature_columns: Dict[str, List[str]]) -> List[dict]:
        """
        Group interactions by user, sort by timestamp, extract all one-hot features.

        Args:
            df: Preprocessed DataFrame with all one-hot columns
            feature_columns: Dict mapping feature types to column names

        Returns:
            List of user sequence dictionaries
        """
        user_seq_list = []

        for user_id in df['userid'].unique():
            user_data = df[df['userid'] == user_id].sort_values('datetime')

            seq_data = {
                'user_id': user_id,
                'items': user_data['placeid'].tolist()
            }

            # Add time features (as lists of values)
            if 'time' in feature_columns:
                for col in feature_columns['time']:
                    seq_data[col] = user_data[col].tolist()

            # Add category one-hot features
            if 'category' in feature_columns:
                for col in feature_columns['category']:
                    seq_data[col] = user_data[col].tolist()

            # Add geospatial features (SCALED)
            if self.use_geospatial:
                if 'lat_scaled' in user_data.columns and 'lon_scaled' in user_data.columns:
                    seq_data['lat'] = user_data['lat_scaled'].tolist()
                    seq_data['lon'] = user_data['lon_scaled'].tolist()

            # Add condition one-hot features
            if 'conditions' in feature_columns:
                for col in feature_columns['conditions']:
                    seq_data[col] = user_data[col].tolist()

            # Add temperature one-hot features
            if 'temperature' in feature_columns:
                for col in feature_columns['temperature']:
                    seq_data[col] = user_data[col].tolist()

            # Add wind one-hot features
            if 'wind' in feature_columns:
                for col in feature_columns['wind']:
                    seq_data[col] = user_data[col].tolist()

            # Keep raw continuous values if needed
            if self.use_weather:
                if 'precip' in user_data.columns:
                    seq_data['precip'] = user_data['precip'].tolist()

            user_seq_list.append(seq_data)

        return user_seq_list
