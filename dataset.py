import torch
import random
import numpy as np
from torch.utils.data import Dataset
from utils import get_percentage_split_indices
from context_extractors import ContextExtractor


class ContextAwareDataset(Dataset):
    """
    Clean dataset for context-aware recommendation

    Supports three configurations:
    1. POI-Only: Only item sequences
    2. Context-Aware: Item sequences + contextual features
    3. Context + Clustering: Context-aware with learnable clustering
    """
    
    def __init__(self, args, user_seq, data_type="train", feature_columns=None):
        self.args = args
        self.user_seq = user_seq
        self.data_type = data_type
        self.max_len = args.max_seq_length
        self.config = args.config
        self.feature_columns = feature_columns or {}  # Store feature columns for one-hot mode

        # Create ContextExtractor instance
        self.context_extractor = ContextExtractor(args)

        # Context feature flags
        self.use_context = self.config in ['context', 'context_clustering']

        # Negative sampling
        self.num_negatives = getattr(args, 'num_negatives', 99)

        # Create a separate Random instance for deterministic candidate generation
        self._candidate_rng = random.Random()



    def __len__(self):
        return len(self.user_seq)
    
    def __getitem__(self, index):
        user_seq = self.user_seq[index]
        
        if self.data_type == "train":
            return self._get_train_item(user_seq)
        elif self.data_type == "valid":
            return self._get_percentage_eval_item(user_seq, "valid")  # Use validation target
        elif self.data_type == "test":
            return self._get_percentage_eval_item(user_seq, "test")  # Use test target
        else:
            raise ValueError(f"Unknown data_type: {self.data_type}")
    
    def _get_train_item(self, user_seq):
        """Get training item with percentage-based splitting"""
        # Extract sequences
        items = user_seq['items']
        seq_len = len(items)
        
        # Use percentage-based splitting: 70% for training
        if seq_len >= 5:  # minimum 5 items
            split_info = get_percentage_split_indices(seq_len)
            train_end = split_info['train_end']
            
            # For training: use first 70%, predict last item of training set
            train_items = items[:train_end-1]
            target_item = items[train_end-1]

        if len(train_items) > self.max_len:
            input_items = torch.LongTensor(train_items[-self.max_len:])
        else:
            input_items = torch.zeros(self.max_len, dtype=torch.long)
            if len(train_items) > 0:
                input_items[-len(train_items):] = torch.LongTensor(train_items)
        
        # Generate negative samples (exclude all user's historical items)
        user_items = set(items)  # All items user has seen
        negatives = self._generate_negatives(user_items)
        
        # Create candidate items (positive + negatives)
        candidates = torch.LongTensor([target_item] + negatives)
        labels = torch.zeros(len(candidates))  # First item (index 0) is positive
        
        # Prepare context data (use same temporal split as items)
        if seq_len >= 5:
            # Use context data up to train_end-1 (matching the temporal split)
            context_data = self._extract_context_data(user_seq, train_end-1)
        else:
            # Fallback: use context up to last item
            context_data = self._extract_context_data(user_seq, None)

        # Prepare return data
        return_data = {
            'input_ids': input_items,
            'candidates': candidates,
            'labels': labels,
            'user_id': torch.LongTensor([user_seq.get('user_id', 0)]),
            **context_data
        }
        
        return return_data

    def _get_percentage_eval_item(self, user_seq, split_type):
        """Get evaluation item using percentage-based splitting"""
        items = user_seq['items']
        seq_len = len(items)
        
        if seq_len >= 5:
            split_info = get_percentage_split_indices(seq_len)
            
            if split_type == "train":
                # Same target as training: use first 70% to predict next item
                input_end = split_info['train_end']
                target_idx = split_info['train_end']
            elif split_type == "valid":
                # Validation: use all items up to last validation item as input
                valid_end = split_info['valid_end']
                input_end = valid_end - 1  # All items up to (but not including) last validation item
                target_idx = valid_end - 1  # Last validation item as target
            elif split_type == "test":
                # Test: use all items up to last item as input
                input_end = seq_len - 1  # All items up to (but not including) last item
                target_idx = seq_len - 1  # Last item as target
            else:
                raise ValueError(f"Unknown split_type: {split_type}")
            
            # Ensure indices are valid
            input_end = max(1, min(input_end, seq_len-1))
            target_idx = max(0, min(target_idx, seq_len-1))
            
            input_items_seq = items[:input_end]
            target_item = items[target_idx]
        else:
            # Fallback for short sequences
            raise ValueError(
                f"User {user_seq.get('user_id', 'unknown')} has only {seq_len} interactions. "
                f"Users with fewer than 5 interactions should have been filtered during preprocessing."
            )

        # Prepare input tensor
        if len(input_items_seq) > self.max_len:
            input_items = torch.LongTensor(input_items_seq[-self.max_len:])
        else:
            input_items = torch.zeros(self.max_len, dtype=torch.long)
            if len(input_items_seq) > 0:
                input_items[-len(input_items_seq):] = torch.LongTensor(input_items_seq)
        
        # Extract context data aligned with input sequence
        context_data = self._extract_context_data(user_seq, len(input_items_seq))
        
        # Generate fixed candidates for evaluation (1 positive + 99 negatives)
        candidates = self._generate_fixed_eval_candidates(target_item, user_seq)
        
        # Prepare return data
        return_data = {
            'input_ids': input_items,
            'target_item': torch.LongTensor([target_item]),
            'candidates': candidates,
            'user_id': torch.LongTensor([user_seq.get('user_id', 0)]),
            **context_data
        }
        
        # For validation split, add only the last validation target
        if split_type == "valid":
            if seq_len >= 5:
                split_info = get_percentage_split_indices(seq_len)
                valid_end = split_info['valid_end']
                all_valid_items = [items[valid_end - 1]]
            else:
                # For short sequences, use second-to-last item as validation target
                all_valid_items = [items[-2]] if len(items) > 1 else [0]

            return_data['all_test_targets'] = torch.LongTensor(all_valid_items)
        
        # For test split, add only the last test target
        elif split_type == "test":
            all_test_items = [items[-1]] if len(items) > 0 else [0]

            return_data['all_test_targets'] = torch.LongTensor(all_test_items)
        
        return return_data
    

    def _extract_context_data(self, user_seq, target_index=None):
        """
        Extract context features using ContextExtractor.

        Args:
            user_seq: User sequence dictionary with one-hot features
            target_index: Optional target index for slicing sequences

        Returns:
            context_data: Dictionary of context tensors
        """
        if not self.use_context:
            return {}

        # Use ContextExtractor to extract features with one-hot encoding
        return self.context_extractor.extract(
            user_seq,
            self.max_len,
            self.feature_columns,
            target_index
        )
    
    def _generate_negatives(self, user_items):
        """Generate negative samples - exclude all items user has interacted with"""
        negatives = []
        seen = set(user_items)
        while len(negatives) < self.num_negatives:
            neg_item = random.randint(1, self.args.item_size - 1)  # Avoid padding token 0
            if neg_item not in user_items:
                negatives.append(neg_item)
                seen.add(neg_item)  # Avoid sampling same negative twice
        
        return negatives
    
    def _generate_fixed_eval_candidates(self, target_item, user_seq):
        """Generate fixed candidates for evaluation (1 positive + 99 negatives)
        
        Uses a deterministic seed based on user and target item to ensure 
        reproducible candidate sets across runs.
        """
        # Create a deterministic seed based on user_id and target_item for reproducibility
        user_id = user_seq.get('user_id', 0)
        seed = hash((user_id, target_item, self.data_type)) % (2**31)
        
        # Use separate Random instance with deterministic seed for candidate generation
        self._candidate_rng.seed(seed)

        # Get all user's historical items to exclude from negatives
        user_items = set(user_seq['items'])

        # Generate negative samples
        negatives = []
        while len(negatives) < self.num_negatives:
            neg_item = self._candidate_rng.randint(1, self.args.item_size - 1)  # Avoid padding token 0
            if neg_item not in user_items and neg_item != target_item:
                negatives.append(neg_item)

        # Create candidates: [positive_item] + [negative_items]
        candidates = [target_item] + negatives
        
        return torch.LongTensor(candidates)


def collate_fn(batch):
    """Custom collate function for batching"""
    batch_data = {}

    # Extract common fields
    for key in batch[0].keys():
        
        values = [item[key] for item in batch]
        
        if key == 'all_test_targets':
            # Handle variable-length test targets - store as list
            batch_data[key] = [v for v in values if v is not None]
        elif isinstance(values[0], torch.Tensor):
            batch_data[key] = torch.stack(values)
        else:
            batch_data[key] = values

    return batch_data