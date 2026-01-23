"""
Main training script for MDCA.

Uses YAML configuration files for reproducible experiments.
"""

import os
# Fix OpenMP conflict with FAISS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import argparse
from config import load_config, save_config
from model import create_mdca_model
from trainer import ContextAwareTrainer
from dataset import ContextAwareDataset, collate_fn
from context_extractors import ContextExtractor
from utils import set_seed, check_path
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MDCA Training')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config file)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (overrides config file)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (overrides config file)')
    cmd_args = parser.parse_args()

    # Load config from YAML
    print(f"Loading configuration from {cmd_args.config}")
    args = load_config(cmd_args.config)

    # Override with command line args if provided
    if cmd_args.seed is not None:
        args.seed = cmd_args.seed
    if cmd_args.output_dir is not None:
        args.output_dir = cmd_args.output_dir
    if cmd_args.device is not None:
        args.device = cmd_args.device

    # Ensure output directory exists
    check_path(args.output_dir)

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Save configuration to output directory
    config_save_path = os.path.join(args.output_dir, 'config.yaml')
    save_config(args, config_save_path)

    # Set device
    args.device = torch.device(args.device)
    print(f"Using device: {args.device}")

    # Load dataset with one-hot encoding preprocessing
    print(f"Loading dataset from {args.dataset_path}")
    context_extractor = ContextExtractor(args)
    user_seq_list, max_item_id, feature_columns = context_extractor.preprocess(
        args.dataset_path,
        min_interactions=5
    )

    # Update config with computed values
    args.item_size = max_item_id + 1

    # Store feature column mappings for model initialization
    args.feature_columns = feature_columns
    print(f"\nOne-hot encoded feature groups:")
    for feature_type, columns in feature_columns.items():
        print(f"  {feature_type}: {len(columns)} columns")

    # Calculate dimensions for each feature type
    args.time_dim = len(feature_columns.get('time', []))
    args.category_dim = len(feature_columns.get('category', []))
    args.conditions_dim = len(feature_columns.get('conditions', []))
    args.temp_dim = len(feature_columns.get('temperature', []))
    args.wind_dim = len(feature_columns.get('wind', []))
    args.geo_dim = 2 if args.use_geospatial else 0
    args.precip_dim = 1 if args.use_weather else 0

    # Total context dimension
    args.context_dim = (args.time_dim + args.category_dim + args.conditions_dim +
                        args.temp_dim + args.wind_dim + args.geo_dim + args.precip_dim)
    print(f"Total context dimension: {args.context_dim}")

    print(f"Total users: {len(user_seq_list)} (all users in train/val/test)")
    print(f"Temporal split per sequence: 70% train, 10% val, 20% test")

    # Create datasets with feature_columns parameter
    train_dataset = ContextAwareDataset(args, user_seq_list, data_type="train", feature_columns=feature_columns)
    val_dataset = ContextAwareDataset(args, user_seq_list, data_type="valid", feature_columns=feature_columns)
    test_dataset = ContextAwareDataset(args, user_seq_list, data_type="test", feature_columns=feature_columns)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    # Create model
    print(f"Creating {args.config} model")
    model = create_mdca_model(args.config, args)
    model = model.to(args.device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Initialize clustering
    if args.config == 'context_clustering':
        model.initialize_clustering(train_dataloader)

    # Create trainer
    trainer = ContextAwareTrainer(model, train_dataloader, val_dataloader, test_dataloader, args)

    # Train model
    trainer.train()

    # Test best model
    print("\nEvaluating best model on test set...")
    test_metrics = trainer.test()

    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best validation NDCG@10: {trainer.best_metric:.4f}")
    print(f"\nTest Set Results:")
    for k in [1, 3, 5, 10, 20]:
        hit_k = test_metrics.get(f'Hit@{k}', 0.0)
        ndcg_k = test_metrics.get(f'NDCG@{k}', 0.0)
        mrr_k = test_metrics.get(f'MRR@{k}', 0.0)
        print(f"  k={k:2d}: Hit@{k}={hit_k:.4f}, NDCG@{k}={ndcg_k:.4f}, MRR@{k}={mrr_k:.4f}")
    print(f"\nResults saved to: {args.output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
