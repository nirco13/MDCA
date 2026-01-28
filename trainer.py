import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils import hit_rate_at_k, ndcg_k, mrr_at_k


class FinalRepresentationLogger:
    """Saves final model representations for downstream analysis."""

    def __init__(self, output_dir, model_config, model_args=None):
        self.output_dir = output_dir
        self.model_config = model_config
        self.repr_dir = os.path.join(output_dir, 'representations')
        os.makedirs(self.repr_dir, exist_ok=True)

    def save_split_representations(self, model, dataloader, split_name, device):
        """Save representations for a specific data split."""
        model.eval()
        all_representations = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Processing {split_name}"):
                input_ids = batch['input_ids'].to(device)

                try:
                    if hasattr(model, 'get_sequence_representation'):
                        representations = model.get_sequence_representation(batch)
                    elif hasattr(model, 'item_embeddings'):
                        item_embs = model.item_embeddings(input_ids)
                        representations = item_embs.mean(dim=1)
                    else:
                        representations = input_ids.float().mean(dim=1, keepdim=True)

                    all_representations.append(representations.cpu().numpy())
                except Exception as e:
                    print(f"Warning: Could not extract representations: {e}")
                    continue

        if all_representations:
            representations = np.concatenate(all_representations, axis=0)
            split_dir = os.path.join(self.repr_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)

            repr_path = os.path.join(split_dir, 'representations.npy')
            np.save(repr_path, representations)
            print(f"Saved {len(representations)} {split_name} representations to: {split_dir}")
            return split_dir
        return None


class ContextAwareTrainer:
    """
    Trainer for context-aware recommendation with optional clustering

    Supports three training modes based on model configuration:
    1. POI-Only: Standard recommendation loss
    2. Context-Aware: Recommendation loss with context
    3. Context + Clustering: Recommendation + clustering loss
    """
    
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.args = args
        
        # Training configuration
        self.device = args.device
        self.config = args.config
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Loss weights
        self.rec_weight = getattr(args, 'rec_weight', 1.0)
        self.cluster_weight = getattr(args, 'cluster_weight', 0.1)

        # Training tracking
        self.num_epochs = args.num_epochs
        self.best_metric = 0.0  # Track best NDCG@10
        self.best_epoch = 0

    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.num_epochs} epochs...")

        for epoch in range(self.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"{'='*60}")

            # Train one epoch
            train_losses = self.train_epoch(epoch)

            # Evaluate on validation set
            print(f"\nEvaluating on validation set...")
            val_metrics = self.evaluate(self.eval_dataloader)

            # Track best model (using NDCG@10 as guiding metric)
            current_metric = val_metrics.get('NDCG@10', 0.0)
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = epoch

                # Save best model
                best_model_path = os.path.join(self.args.output_dir, 'best_model.pt')
                self.save_model(best_model_path)
                print(f"New best model saved, NDCG@10: {self.best_metric:.4f}")

            # Print epoch summary with all k values
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_losses['total_loss']:.4f} (Rec: {train_losses['rec_loss']:.4f})")
            print(f"  Validation Metrics:")
            for k in [1, 3, 5, 10, 20]:
                hit_k = val_metrics.get(f'Hit@{k}', 0.0)
                ndcg_k = val_metrics.get(f'NDCG@{k}', 0.0)
                mrr_k = val_metrics.get(f'MRR@{k}', 0.0)
                print(f"    k={k:2d}: Hit@{k}={hit_k:.4f}, NDCG@{k}={ndcg_k:.4f}, MRR@{k}={mrr_k:.4f}")
            print(f"  Best NDCG@10: {self.best_metric:.4f} (Epoch {self.best_epoch + 1})")

        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation NDCG@10: {self.best_metric:.4f} at epoch {self.best_epoch + 1}")
        print(f"{'='*60}")

    def test(self):
        """Evaluate on test set using best model"""
        # Load best model
        best_model_path = os.path.join(self.args.output_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            self.load_model(best_model_path)
            print(f"Loaded best model from epoch {self.best_epoch + 1}")

        # Evaluate
        print(f"\nEvaluating on test set...")
        test_metrics = self.evaluate(self.test_dataloader)

        return test_metrics

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()

        total_rec_loss = 0.0
        total_reg_loss = 0.0
        total_loss = 0.0

        for batch_idx, batch in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()

            # Move batch to device
            batch = self._move_to_device(batch)

            # Compute losses
            rec_loss = self._compute_recommendation_loss(batch)
            cluster_loss = self._compute_clustering_loss(batch)
            reg_loss = self.model.get_embedding_regularization_loss()

            # Combined loss
            loss = (self.rec_weight * rec_loss +
                   self.cluster_weight * cluster_loss +
                   reg_loss)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Accumulate losses
            total_rec_loss += rec_loss.item()
            total_reg_loss += reg_loss.item()
            total_loss += loss.item()

        # Return average losses
        num_batches = len(self.train_dataloader)
        return {
            'rec_loss': total_rec_loss / num_batches,
            'reg_loss': total_reg_loss / num_batches,
            'total_loss': total_loss / num_batches
        }
    
    def _compute_recommendation_loss(self, batch):
        """Compute recommendation loss using BPR"""
        input_ids = batch['input_ids']
        candidates = batch['candidates']
        
        # Extract context data
        context_data = self._extract_context_from_batch(batch)
        
        # Get final representation from model
        outputs = self.model(input_ids, return_all_outputs=True, **context_data)
        final_user_repr = outputs['final_user_repr']  # [B, final_repr_size]
        final_user_repr = F.normalize(final_user_repr, p=2, dim=-1)

        # Get candidate embeddings using raw item embeddings
        candidate_item_emb = self.model.item_embedding.item_embedding(candidates)  # [B, num_candidates, hidden_size]

        # Project to final_repr_size using learned projection
        candidate_embeddings = self.model.item_projection(candidate_item_emb)  # [B, num_candidates, final_repr_size]
        candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=-1)
        
        # Compute similarity scores: [B, 1, final_repr_size] @ [B, num_candidates, final_repr_size] -> [B, num_candidates]
        candidate_scores = torch.bmm(
            final_user_repr.unsqueeze(1), 
            candidate_embeddings.transpose(1, 2)
        ).squeeze(1)  # [B, num_candidates]
        
        # BPR loss: first candidate is positive, rest are negative
        pos_scores = candidate_scores[:, 0]  # [B]
        neg_scores = candidate_scores[:, 1:]  # [B, num_negatives]
        
        # Compute BPR loss
        bpr_loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-8).mean()
        
        return bpr_loss
    
    def _compute_clustering_loss(self, batch):
        """Compute clustering loss """
        if self.config != 'context_clustering':
            return torch.tensor(0.0, device=self.device)
        
        input_ids = batch['input_ids']
        context_data = self._extract_context_from_batch(batch)
        
        # Get model outputs with all components
        outputs = self.model(input_ids, return_all_outputs=True, **context_data)
        context_repr = outputs.get('context_repr')

        # Compute clustering loss using context representation
        cluster_loss = self.model.get_clustering_loss(context_repr)
        
        return cluster_loss

    def _extract_context_from_batch(self, batch):
        """Extract context data from batch"""
        context_data = {}
        # Extract one-hot encoded features
        context_keys = [
            'time_features',
            'category_features',
            'geospatial_features',
            'condition_features',
            'temperature_features',
            'wind_features',
            'precip'
        ]

        for key in context_keys:
            if key in batch:
                context_data[key] = batch.get(key)

        return context_data
    
    
    def _move_to_device(self, batch):
        """Move batch data to device"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        
        return device_batch

    def evaluate(self, dataloader):
        """Evaluate model performance using candidate-based approach (1 positive + 99 negatives)"""
        self.model.eval()
        
        all_predictions = []
        all_actual = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_to_device(batch)
                
                input_ids = batch['input_ids']
                candidates = batch['candidates']  # [B, num_candidates] - 1 positive + 99 negatives
                context_data = self._extract_context_from_batch(batch)
                
                # Get final user representation
                outputs = self.model(input_ids, return_all_outputs=True, **context_data)
                final_user_repr = outputs['final_user_repr']  # [B, final_repr_size]
                final_user_repr = F.normalize(final_user_repr, p=2, dim=-1)

                # Score against candidate items using raw item embeddings
                candidate_item_emb = self.model.item_embedding.item_embedding(candidates)  # [B, num_candidates, hidden_size]
                candidate_embeddings = self.model.item_projection(candidate_item_emb)  # [B, num_candidates, final_repr_size]
                candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=-1)
                
                # Compute similarity scores: [B, 1, final_repr_size] @ [B, num_candidates, final_repr_size] -> [B, num_candidates]
                candidate_scores = torch.bmm(
                    final_user_repr.unsqueeze(1), 
                    candidate_embeddings.transpose(1, 2)
                ).squeeze(1)  # [B, num_candidates]
                
                # Get top-k predictions for each user
                batch_size = candidate_scores.size(0)
                for i in range(batch_size):
                    user_scores = candidate_scores[i]  # [num_candidates]
                    user_candidates = candidates[i]  # [num_candidates]

                    # Sort by score (descending) and get item IDs
                    _, sorted_indices = torch.sort(user_scores, descending=True)
                    ranked_items = user_candidates[sorted_indices].cpu().numpy().tolist()

                    # First candidate is always the positive item
                    positive_item = candidates[i][0].item()

                    all_predictions.append(ranked_items)
                    all_actual.append([positive_item])  # Single positive item
        
        # Compute metrics using existing utility functions
        metrics = {}
        for k in [1, 3, 5, 10, 20]:
            if len(all_predictions) > 0 and k <= len(all_predictions[0]):  # Check if k is valid
                metrics[f'Hit@{k}'] = hit_rate_at_k(all_actual, all_predictions, k)
                metrics[f'NDCG@{k}'] = ndcg_k(all_actual, all_predictions, k)
                metrics[f'MRR@{k}'] = mrr_at_k(all_actual, all_predictions, k)
        
        return metrics
    
    
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        """Load model checkpoint"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
    
    def save_final_representations(self, model_config_str, save_validation=True, save_test=True):
        """
        Save final model representations for evaluation and test sets.
        
        Args:
            model_config_str: Configuration string for file naming
            save_validation: Whether to save validation representations
            save_test: Whether to save test representations
        """
        # Initialize representation logger
        repr_logger = FinalRepresentationLogger(
            output_dir=getattr(self.args, 'output_dir', './output'),
            model_config=model_config_str,
            model_args=vars(self.args)
        )
        

        # Save validation representations
        if save_validation and hasattr(self, 'eval_dataloader'):
            try:
                validation_path = repr_logger.save_split_representations(
                    model=self.model,
                    dataloader=self.eval_dataloader,
                    split_name='validation',
                    device=self.device
                )
                print(f"Validation representations saved to: {validation_path}")
            except Exception as e:
                print(f"Error saving validation representations: {e}")
        
        # Save test representations
        if save_test and hasattr(self, 'test_dataloader'):
            try:
                test_path = repr_logger.save_split_representations(
                    model=self.model,
                    dataloader=self.test_dataloader,
                    split_name='test',
                    device=self.device
                )
                print(f"Test representations saved to: {test_path}")
            except Exception as e:
                print(f"Error saving test representations: {e}")
        
        return repr_logger