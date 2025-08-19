"""
Training utilities for deep learning models.

This module provides training loops, loss functions, optimizers,
and other utilities for training segmentation and detection models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable, Any
import time
import logging
import os
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class SegmentationLoss(nn.Module):
    """
    Combined loss function for segmentation tasks.
    
    Combines multiple loss functions including Cross Entropy,
    Dice loss, and Focal loss for robust training.
    """
    
    def __init__(self, weight_ce: float = 1.0, weight_dice: float = 1.0,
                 weight_focal: float = 0.0, class_weights: Optional[torch.Tensor] = None,
                 smooth: float = 1e-6):
        """
        Initialize segmentation loss.
        
        Args:
            weight_ce: Weight for cross entropy loss
            weight_dice: Weight for dice loss
            weight_focal: Weight for focal loss
            class_weights: Class weights for cross entropy
            smooth: Smoothing factor for dice loss
        """
        super(SegmentationLoss, self).__init__()
        
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.smooth = smooth
        
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss for segmentation.
        
        Args:
            pred: Predicted probabilities (B, C, H, W)
            target: Ground truth labels (B, H, W)
            
        Returns:
            Dice loss value
        """
        pred_soft = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.size(1))
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Compute intersection and union
        intersection = (pred_soft * target_one_hot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Average over classes and batch
        dice_loss = 1.0 - dice.mean()
        
        return dice_loss
    
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor,
                   alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
        """
        Compute Focal loss for handling class imbalance.
        
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Ground truth labels (B, H, W)
            alpha: Weighting factor
            gamma: Focusing parameter
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        
        return focal_loss.mean()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Ground truth labels (B, H, W)
            
        Returns:
            Combined loss value
        """
        total_loss = 0.0
        
        if self.weight_ce > 0:
            ce_loss = self.ce_loss(pred, target)
            total_loss += self.weight_ce * ce_loss
        
        if self.weight_dice > 0:
            dice_loss = self.dice_loss(pred, target)
            total_loss += self.weight_dice * dice_loss
        
        if self.weight_focal > 0:
            focal_loss = self.focal_loss(pred, target)
            total_loss += self.weight_focal * focal_loss
        
        return total_loss


class IoUMetric:
    """
    Intersection over Union metric for segmentation evaluation.
    """
    
    def __init__(self, num_classes: int, ignore_index: int = -1):
        """
        Initialize IoU metric.
        
        Args:
            num_classes: Number of classes
            ignore_index: Index to ignore in computation
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset metric state."""
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metric with new predictions.
        
        Args:
            pred: Predicted labels (B, H, W)
            target: Ground truth labels (B, H, W)
        """
        # Move to CPU for computation
        pred = pred.cpu()
        target = target.cpu()
        
        # Mask out ignore index
        valid_mask = target != self.ignore_index
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        for cls in range(self.num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)
            
            intersection = (pred_cls & target_cls).sum().float()
            union = (pred_cls | target_cls).sum().float()
            
            self.intersection[cls] += intersection
            self.union[cls] += union
    
    def compute(self) -> Dict[str, float]:
        """
        Compute IoU metrics.
        
        Returns:
            Dictionary with IoU metrics
        """
        iou_per_class = self.intersection / (self.union + 1e-8)
        mean_iou = iou_per_class.mean().item()
        
        results = {
            'mIoU': mean_iou,
            'IoU_per_class': iou_per_class.tolist()
        }
        
        return results


class DiceMetric:
    """
    Dice coefficient metric for segmentation evaluation.
    """
    
    def __init__(self, num_classes: int, ignore_index: int = -1):
        """
        Initialize Dice metric.
        
        Args:
            num_classes: Number of classes
            ignore_index: Index to ignore in computation
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset metric state."""
        self.intersection = torch.zeros(self.num_classes)
        self.total = torch.zeros(self.num_classes)
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metric with new predictions.
        
        Args:
            pred: Predicted labels (B, H, W)
            target: Ground truth labels (B, H, W)
        """
        pred = pred.cpu()
        target = target.cpu()
        
        valid_mask = target != self.ignore_index
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        for cls in range(self.num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)
            
            intersection = (pred_cls & target_cls).sum().float()
            total = pred_cls.sum().float() + target_cls.sum().float()
            
            self.intersection[cls] += intersection
            self.total[cls] += total
    
    def compute(self) -> Dict[str, float]:
        """
        Compute Dice metrics.
        
        Returns:
            Dictionary with Dice metrics
        """
        dice_per_class = (2.0 * self.intersection) / (self.total + 1e-8)
        mean_dice = dice_per_class.mean().item()
        
        results = {
            'mDice': mean_dice,
            'Dice_per_class': dice_per_class.tolist()
        }
        
        return results


class Trainer:
    """
    Training manager for deep learning models.
    
    Handles training loops, validation, checkpointing, and logging.
    """
    
    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 optimizer: Optional[optim.Optimizer] = None,
                 scheduler: Optional[Any] = None,
                 loss_fn: Optional[nn.Module] = None,
                 device: Optional[torch.device] = None,
                 checkpoint_dir: str = 'checkpoints',
                 log_interval: int = 10):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            loss_fn: Loss function
            device: Device for training
            checkpoint_dir: Directory for saving checkpoints
            log_interval: Interval for logging
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer
        
        # Setup scheduler
        self.scheduler = scheduler
        
        # Setup loss function
        if loss_fn is None:
            self.loss_fn = SegmentationLoss()
        else:
            self.loss_fn = loss_fn
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        running_loss = 0.0
        num_batches = 0
        
        # Initialize metrics
        iou_metric = IoUMetric(num_classes=self._get_num_classes())
        dice_metric = DiceMetric(num_classes=self._get_num_classes())
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            num_batches += 1
            
            # Get predictions for metrics
            with torch.no_grad():
                pred_labels = torch.argmax(outputs, dim=1)
                iou_metric.update(pred_labels, targets)
                dice_metric.update(pred_labels, targets)
            
            # Log progress
            if batch_idx % self.log_interval == 0:
                logger.info(f'Train Epoch: {self.epoch} [{batch_idx}/{len(self.train_loader)}] '
                           f'Loss: {loss.item():.6f}')
        
        # Compute epoch metrics
        avg_loss = running_loss / num_batches
        iou_results = iou_metric.compute()
        dice_results = dice_metric.compute()
        
        metrics = {
            'loss': avg_loss,
            'mIoU': iou_results['mIoU'],
            'mDice': dice_results['mDice']
        }
        
        return metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary with validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        running_loss = 0.0
        num_batches = 0
        
        # Initialize metrics
        iou_metric = IoUMetric(num_classes=self._get_num_classes())
        dice_metric = DiceMetric(num_classes=self._get_num_classes())
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
                
                # Update metrics
                running_loss += loss.item()
                num_batches += 1
                
                # Get predictions for metrics
                pred_labels = torch.argmax(outputs, dim=1)
                iou_metric.update(pred_labels, targets)
                dice_metric.update(pred_labels, targets)
        
        # Compute epoch metrics
        avg_loss = running_loss / num_batches
        iou_results = iou_metric.compute()
        dice_results = dice_metric.compute()
        
        metrics = {
            'loss': avg_loss,
            'mIoU': iou_results['mIoU'],
            'mDice': dice_results['mDice']
        }
        
        return metrics
    
    def train(self, num_epochs: int, save_best: bool = True,
              save_interval: int = 10) -> Dict[str, List[float]]:
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_best: Whether to save best model
            save_interval: Interval for saving checkpoints
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                else:
                    self.scheduler.step()
            
            # Log epoch results
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch}/{num_epochs-1} completed in {epoch_time:.2f}s")
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"mIoU: {train_metrics['mIoU']:.4f}, "
                       f"mDice: {train_metrics['mDice']:.4f}")
            
            if val_metrics:
                logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                           f"mIoU: {val_metrics['mIoU']:.4f}, "
                           f"mDice: {val_metrics['mDice']:.4f}")
            
            # Save history
            for key, value in train_metrics.items():
                self.train_history[f'train_{key}'].append(value)
            
            for key, value in val_metrics.items():
                self.val_history[f'val_{key}'].append(value)
            
            # Save checkpoints
            if save_best and val_metrics and val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model.pth')
                logger.info("Saved new best model")
            
            if epoch % save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
        
        # Combine histories
        history = {**self.train_history, **self.val_history}
        
        logger.info("Training completed")
        return history
    
    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history)
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {filepath}")
    
    def _get_num_classes(self) -> int:
        """Get number of classes from model output."""
        # This is a simple heuristic - in practice you'd determine this differently
        dummy_input = torch.randn(1, 3, 64, 64).to(self.device)
        with torch.no_grad():
            output = self.model(dummy_input)
            if isinstance(output, dict):
                # For models that return dictionaries (like Mask R-CNN)
                return 3  # Default
            else:
                return output.size(1)


def create_optimizer(model: nn.Module, optimizer_type: str = 'adam',
                    learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                    **kwargs) -> optim.Optimizer:
    """
    Create optimizer for model training.
    
    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')
        learning_rate: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    if optimizer_type.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, 
                         weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, 
                        weight_decay=weight_decay, momentum=0.9, **kwargs)
    elif optimizer_type.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, 
                          weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(optimizer: optim.Optimizer, scheduler_type: str = 'cosine',
                    **kwargs) -> Any:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ('cosine', 'step', 'plateau')
        **kwargs: Additional scheduler arguments
        
    Returns:
        Scheduler instance
    """
    if scheduler_type.lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_type.lower() == 'step':
        return optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_type.lower() == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


if __name__ == "__main__":
    # Example usage
    from torch.utils.data import TensorDataset
    
    # Create dummy data
    dummy_images = torch.randn(100, 3, 64, 64)
    dummy_targets = torch.randint(0, 2, (100, 64, 64))
    
    train_dataset = TensorDataset(dummy_images, dummy_targets)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 2, 3, padding=1)
        
        def forward(self, x):
            return self.conv(x)
    
    model = DummyModel()
    
    # Create trainer
    optimizer = create_optimizer(model, 'adam', learning_rate=1e-3)
    scheduler = create_scheduler(optimizer, 'cosine', T_max=10)
    
    trainer = Trainer(model, train_loader, optimizer=optimizer, scheduler=scheduler)
    
    # Train for a few epochs
    history = trainer.train(num_epochs=3)
    
    print("Training history:", history)
