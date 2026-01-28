import os
import math
import torch
import random
import numpy as np
from typing import List, Tuple


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def check_path(path):
    """Create path if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)



def get_percentage_split_indices(seq_length, train_pct=0.7, valid_pct=0.1, test_pct=0.2):
    """
    Calculate split indices for percentage-based sequence splitting
    
    Args:
        seq_length: Length of the sequence
        train_pct: Percentage for training (default 0.7 = 70%)
        valid_pct: Percentage for validation (default 0.1 = 10%) 
        test_pct: Percentage for test (default 0.2 = 20%)
        
    Returns:
        dict with split indices: {'train_end': int, 'valid_end': int}
    """
    # Ensure percentages sum to 1.0
    total_pct = train_pct + valid_pct + test_pct
    if abs(total_pct - 1.0) > 1e-6:
        train_pct = train_pct / total_pct
        valid_pct = valid_pct / total_pct
        test_pct = test_pct / total_pct
    
    # Calculate split points
    train_end = int(seq_length * train_pct)
    valid_end = int(seq_length * (train_pct + valid_pct))
    
    # Ensure we have at least 1 item for each split
    train_end = max(1, train_end)
    valid_end = max(train_end + 1, valid_end)
    valid_end = min(seq_length - 1, valid_end)  # Leave at least 1 for test
    
    return {
        'train_end': train_end,
        'valid_end': valid_end,
        'seq_length': seq_length
    }


def hit_rate_at_k(actual, predicted, topk):
    """Calculate hit rate at k for all samples
    
    Hit rate measures the fraction of users for whom at least one relevant item 
    appears in the top-k recommendations.
    """
    hits = 0
    num_users = len(predicted)
    
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])

        if len(act_set & pred_set) > 0:
            hits += 1
    
    return hits / float(num_users) if num_users > 0 else 0.0


def ndcg_k(actual, predicted, topk):
    """Calculate NDCG at k for all samples"""
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in set(actual[user_id])) / math.log(j + 2, 2) for j in range(topk)])
        res += dcg_k / idcg if idcg > 0 else 0
    return res / float(len(actual))


def idcg_k(k):
    """Calculate ideal DCG at k"""
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def mrr_at_k(actual_items, predicted_items, k):
    """
    Calculate Mean Reciprocal Rank (MRR) at k for recommendation evaluation.
    
    MRR measures the average reciprocal rank of the first relevant item in the recommendations.
    
    Args:
        actual_items: List of lists, each containing relevant items for a user
        predicted_items: List of lists, each containing predicted items for a user (ranked by score)
        k: Maximum number of predictions to consider
        
    Returns:
        float: MRR@k score (0.0 to 1.0, higher is better)
        
    Example:
        actual = [[5], [2], [8]]  # One positive item per user
        predicted = [[1, 5, 3], [4, 6, 2], [8, 9, 1]]  # Top predictions per user
        mrr_at_k(actual, predicted, 3) = (1/2 + 1/3 + 1/1) / 3 = 0.61
    """
    if not actual_items or not predicted_items:
        return 0.0
    
    reciprocal_ranks = []
    
    for actual_list, predicted_list in zip(actual_items, predicted_items):
        if not actual_list or not predicted_list:
            reciprocal_ranks.append(0.0)
            continue
            
        # Consider only top-k predictions
        top_k_predictions = predicted_list[:k]
        
        # Find the rank of the first relevant item
        reciprocal_rank = 0.0
        for rank, predicted_item in enumerate(top_k_predictions, start=1):
            if predicted_item in actual_list:
                reciprocal_rank = 1.0 / rank
                break
        
        reciprocal_ranks.append(reciprocal_rank)
    
    # Return mean reciprocal rank
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0