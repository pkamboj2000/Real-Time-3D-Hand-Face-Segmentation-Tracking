"""
Training pipeline for hand/face segmentation models.

This module provides a complete training infrastructure including data loading,
augmentation, training loops, validation, checkpointing, and metrics tracking.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from core_models import UNet

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for tracking training metrics."""
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_ious: list[float] = field(default_factory=list)
    val_dice: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, list]:
        """Convert metrics to dictionary for serialization."""
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_ious": self.val_ious,
            "val_dice": self.val_dice,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, list]) -> "TrainingMetrics":
        """Create metrics from dictionary."""
        return cls(**data)


class SegmentationDataset(Dataset):
    """
    Dataset for semantic segmentation of hands and faces.
    
    Supports loading from standard directory structure or generating
    synthetic data for testing and demonstration purposes.
    
    Args:
        data_dir: Root directory containing the dataset
        split: Dataset split ('train', 'val', 'test')
        transform: Optional transform to apply to images
        target_size: Target image dimensions (H, W)
        create_synthetic: Whether to create synthetic data if real data not found
    """
    
    SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp"}
    
    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_size: tuple[int, int] = (512, 512),
        create_synthetic: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        
        self.image_paths: list[Path] = []
        self.mask_paths: list[Path] = []
        
        self._load_dataset(create_synthetic)
        
        logger.info(
            f"Loaded {len(self)} samples for {split} split "
            f"(target size: {target_size})"
        )
    
    def _load_dataset(self, create_synthetic: bool) -> None:
        """Load dataset from disk or create synthetic data."""
        images_dir = self.data_dir / "images" / self.split
        masks_dir = self.data_dir / "masks" / self.split
        
        if images_dir.exists() and masks_dir.exists():
            self._load_from_directory(images_dir, masks_dir)
        elif create_synthetic:
            logger.warning(f"Dataset not found at {self.data_dir}, creating synthetic data")
            self._create_synthetic_data()
        else:
            raise FileNotFoundError(
                f"Dataset not found at {self.data_dir} and synthetic creation disabled"
            )
    
    def _load_from_directory(self, images_dir: Path, masks_dir: Path) -> None:
        """Load image-mask pairs from standard directory structure."""
        for ext in self.SUPPORTED_IMAGE_FORMATS:
            for img_path in images_dir.glob(f"*{ext}"):
                # Try multiple mask extensions
                for mask_ext in [".png", ".jpg", ".bmp"]:
                    mask_path = masks_dir / f"{img_path.stem}{mask_ext}"
                    if mask_path.exists():
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)
                        break
    
    def _create_synthetic_data(self) -> None:
        """Generate synthetic training data for demonstration."""
        images_dir = self.data_dir / "images" / self.split
        masks_dir = self.data_dir / "masks" / self.split
        
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        num_samples = 100 if self.split == "train" else 20
        
        for i in range(num_samples):
            # Create synthetic image with gradient background
            image = self._generate_synthetic_image()
            mask = self._generate_synthetic_mask()
            
            img_path = images_dir / f"synthetic_{i:04d}.png"
            mask_path = masks_dir / f"synthetic_{i:04d}.png"
            
            cv2.imwrite(str(img_path), image)
            cv2.imwrite(str(mask_path), mask)
            
            self.image_paths.append(img_path)
            self.mask_paths.append(mask_path)
        
        logger.info(f"Created {num_samples} synthetic samples for {self.split}")
    
    def _generate_synthetic_image(self) -> np.ndarray:
        """Generate a synthetic training image."""
        h, w = self.target_size
        image = np.random.randint(100, 200, (h, w, 3), dtype=np.uint8)
        
        # Add some noise and variation
        noise = np.random.randn(h, w, 3) * 20
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def _generate_synthetic_mask(self) -> np.ndarray:
        """Generate a synthetic segmentation mask."""
        h, w = self.target_size
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Add random hand regions (class 1)
        for _ in range(np.random.randint(0, 3)):
            cx = np.random.randint(80, w - 80)
            cy = np.random.randint(80, h - 80)
            radius = np.random.randint(40, 100)
            cv2.circle(mask, (cx, cy), radius, 1, -1)
        
        # Add random face regions (class 2)
        for _ in range(np.random.randint(0, 2)):
            cx = np.random.randint(60, w - 60)
            cy = np.random.randint(60, h - 60)
            radius = np.random.randint(30, 70)
            cv2.circle(mask, (cx, cy), radius, 2, -1)
        
        return mask
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and preprocess a single sample."""
        # Load image and mask
        image = cv2.imread(str(self.image_paths[idx]))
        if image is None:
            raise IOError(f"Failed to load image: {self.image_paths[idx]}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Failed to load mask: {self.mask_paths[idx]}")
        
        # Resize to target size
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        mask = cv2.resize(
            mask,
            (self.target_size[1], self.target_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        mask = torch.from_numpy(mask).long()
        
        return image, mask


class SegmentationTrainer:
    """
    Training manager for segmentation models.
    
    Provides a complete training loop with validation, checkpointing,
    learning rate scheduling, mixed precision training, and metrics tracking.
    
    Args:
        model: PyTorch model to train
        device: Training device ('auto', 'cpu', 'cuda', 'mps')
        num_classes: Number of segmentation classes
        use_amp: Whether to use automatic mixed precision
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        num_classes: int = 3,
        use_amp: bool = True
    ):
        self.device = self._resolve_device(device)
        self.model = model.to(self.device)
        self.num_classes = num_classes
        self.use_amp = use_amp and self.device.type == "cuda"
        
        # Loss function with class weighting support
        self.criterion = nn.CrossEntropyLoss()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Metrics tracking
        self.metrics = TrainingMetrics()
        
        logger.info(
            f"Trainer initialized: device={self.device}, "
            f"num_classes={num_classes}, amp={self.use_amp}"
        )
    
    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve the training device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> dict[str, float]:
        """
        Compute segmentation metrics.
        
        Args:
            predictions: Model logits (B, C, H, W)
            targets: Ground truth masks (B, H, W)
            
        Returns:
            Dictionary with IoU and Dice scores
        """
        pred_labels = torch.argmax(predictions, dim=1)
        
        ious = []
        dice_scores = []
        
        for class_id in range(self.num_classes):
            pred_mask = (pred_labels == class_id)
            target_mask = (targets == class_id)
            
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            pred_sum = pred_mask.sum().float()
            target_sum = target_mask.sum().float()
            
            # IoU
            iou = (intersection / (union + 1e-8)).item()
            ious.append(iou)
            
            # Dice
            dice = (2 * intersection / (pred_sum + target_sum + 1e-8)).item()
            dice_scores.append(dice)
        
        return {
            "mean_iou": np.mean(ious),
            "mean_dice": np.mean(dice_scores),
            "per_class_iou": ious,
            "per_class_dice": dice_scores,
        }
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        gradient_clip: Optional[float] = None
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer instance
            gradient_clip: Optional gradient clipping value
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass with optional AMP
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                self.scaler.scale(loss).backward()
                
                if gradient_clip:
                    self.scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                
                if gradient_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.debug(
                    f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}"
                )
        
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> tuple[float, float, float]:
        """
        Validate model on validation set.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Tuple of (loss, IoU, Dice score)
        """
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        num_batches = 0
        
        for images, masks in dataloader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            
            metrics = self.compute_metrics(outputs, masks)
            
            total_loss += loss.item()
            total_iou += metrics["mean_iou"]
            total_dice += metrics["mean_dice"]
            num_batches += 1
        
        n = max(num_batches, 1)
        return total_loss / n, total_iou / n, total_dice / n
    
    def save_checkpoint(
        self,
        path: Path,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: dict[str, Any]
    ) -> None:
        """Save a training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
        }
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved: {path}")
    
    def load_checkpoint(
        self,
        path: Path,
        optimizer: Optional[optim.Optimizer] = None
    ) -> dict[str, Any]:
        """Load a training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        logger.info(f"Checkpoint loaded: {path}")
        return checkpoint
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        save_dir: str | Path = "checkpoints",
        gradient_clip: Optional[float] = 1.0,
        early_stopping_patience: int = 10,
        resume_from: Optional[Path] = None
    ) -> TrainingMetrics:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
            save_dir: Directory for saving checkpoints
            gradient_clip: Gradient clipping value
            early_stopping_patience: Epochs without improvement before stopping
            resume_from: Optional checkpoint path to resume from
            
        Returns:
            TrainingMetrics with complete training history
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5, verbose=True
        )
        
        start_epoch = 0
        best_iou = 0.0
        patience_counter = 0
        
        # Resume from checkpoint if specified
        if resume_from and resume_from.exists():
            checkpoint = self.load_checkpoint(resume_from, optimizer)
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_iou = checkpoint.get("metrics", {}).get("val_iou", 0.0)
            logger.info(f"Resuming from epoch {start_epoch}")
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.perf_counter()
            
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, gradient_clip)
            
            # Validation
            val_loss, val_iou, val_dice = self.validate(val_loader)
            
            # Update scheduler
            scheduler.step(val_iou)
            
            epoch_time = time.perf_counter() - epoch_start
            current_lr = optimizer.param_groups[0]["lr"]
            
            # Track metrics
            self.metrics.train_losses.append(train_loss)
            self.metrics.val_losses.append(val_loss)
            self.metrics.val_ious.append(val_iou)
            self.metrics.val_dice.append(val_dice)
            self.metrics.learning_rates.append(current_lr)
            self.metrics.epoch_times.append(epoch_time)
            
            # Logging
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val IoU: {val_iou:.4f} | "
                f"Val Dice: {val_dice:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Save best model
            if val_iou > best_iou:
                best_iou = val_iou
                patience_counter = 0
                self.save_checkpoint(
                    save_dir / "best_model.pth",
                    optimizer,
                    epoch,
                    {"val_iou": val_iou, "val_dice": val_dice, "val_loss": val_loss}
                )
                logger.info(f"New best model! IoU: {val_iou:.4f}")
            else:
                patience_counter += 1
            
            # Save latest checkpoint
            self.save_checkpoint(
                save_dir / "latest_checkpoint.pth",
                optimizer,
                epoch,
                {"val_iou": val_iou, "val_dice": val_dice, "val_loss": val_loss}
            )
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save training history
        history_path = save_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.metrics.to_dict(), f, indent=2)
        
        logger.info(f"Training completed! Best IoU: {best_iou:.4f}")
        
        return self.metrics


def create_data_loaders(
    data_dir: str | Path,
    batch_size: int = 4,
    num_workers: int = 4,
    target_size: tuple[int, int] = (512, 512),
    pin_memory: bool = True
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        target_size: Target image dimensions (H, W)
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = SegmentationDataset(
        data_dir, split="train", target_size=target_size
    )
    val_dataset = SegmentationDataset(
        data_dir, split="val", target_size=target_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    logger.info(f"Data loaders created: train={len(train_dataset)}, val={len(val_dataset)}")
    
    return train_loader, val_loader


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train segmentation models for hand/face detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--image-size", type=int, default=512,
        help="Input image size (assumes square)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model", type=str, default="unet", choices=["unet"],
        help="Model architecture"
    )
    parser.add_argument(
        "--num-classes", type=int, default=3,
        help="Number of segmentation classes"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr", "--learning-rate", type=float, default=1e-4,
        dest="learning_rate", help="Initial learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5,
        help="Weight decay for regularization"
    )
    parser.add_argument(
        "--gradient-clip", type=float, default=1.0,
        help="Gradient clipping value (0 to disable)"
    )
    parser.add_argument(
        "--early-stopping", type=int, default=10,
        help="Early stopping patience (epochs)"
    )
    
    # Runtime arguments
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Data loading workers"
    )
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable automatic mixed precision"
    )
    
    # Output arguments
    parser.add_argument(
        "--save-dir", type=str, default="checkpoints",
        help="Directory for saving checkpoints"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    
    # Logging
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main training entry point."""
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    logger.info("Starting training pipeline")
    logger.info(f"Arguments: {vars(args)}")
    
    # Create data loaders
    target_size = (args.image_size, args.image_size)
    train_loader, val_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=target_size,
    )
    
    # Create model
    if args.model == "unet":
        model = UNet(in_channels=3, out_channels=args.num_classes)
    else:
        raise ValueError(f"Unknown model architecture: {args.model}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {args.model} | Total params: {total_params:,} | Trainable: {trainable_params:,}")
    
    # Create trainer
    trainer = SegmentationTrainer(
        model=model,
        device=args.device,
        num_classes=args.num_classes,
        use_amp=not args.no_amp,
    )
    
    # Train
    resume_path = Path(args.resume) if args.resume else None
    gradient_clip = args.gradient_clip if args.gradient_clip > 0 else None
    
    metrics = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        gradient_clip=gradient_clip,
        early_stopping_patience=args.early_stopping,
        resume_from=resume_path,
    )
    
    logger.info("Training pipeline completed")


# Backwards compatibility alias
HandFaceDataset = SegmentationDataset


if __name__ == "__main__":
    main()
