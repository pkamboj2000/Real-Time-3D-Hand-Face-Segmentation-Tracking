"""
Training script for hand/face segmentation models.
Includes data loading, training loops, and model evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import time
import json
from core_models import UNet, MaskRCNN
from collections import defaultdict


class HandFaceDataset(Dataset):
    """
    Dataset class for hand/face segmentation.
    Works with multiple datasets: EgoHands, FreiHAND, CelebAMask-HQ, WFLW.
    """
    
    def __init__(self, data_dir, split='train', transform=None, target_size=(512, 512)):
        """
        Set up dataset.
        
        Args:
            data_dir: Root directory containing dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Image transformations
            target_size: Target image size (H, W)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        
        # Find image and mask files
        self.image_files = []
        self.mask_files = []
        
        # Check for standard dataset structure
        images_dir = self.data_dir / 'images' / split
        masks_dir = self.data_dir / 'masks' / split
        
        if images_dir.exists() and masks_dir.exists():
            self._load_from_standard_structure(images_dir, masks_dir)
        else:
            # Create dummy data for demo purposes
            self._create_dummy_data()
    
    def _load_from_standard_structure(self, images_dir, masks_dir):
        """Load data from standard directory structure."""
        for img_file in images_dir.glob('*.jpg'):
            mask_file = masks_dir / (img_file.stem + '.png')
            if mask_file.exists():
                self.image_files.append(img_file)
                self.mask_files.append(mask_file)
        
        for img_file in images_dir.glob('*.png'):
            mask_file = masks_dir / img_file.name
            if mask_file.exists():
                self.image_files.append(img_file)
                self.mask_files.append(mask_file)
    
    def _create_dummy_data(self):
        """Create dummy data for demonstration."""
        print("Creating dummy training data...")
        
        # Create directories
        os.makedirs(self.data_dir / 'images' / self.split, exist_ok=True)
        os.makedirs(self.data_dir / 'masks' / self.split, exist_ok=True)
        
        num_samples = 100 if self.split == 'train' else 20
        
        for i in range(num_samples):
            # Create dummy image
            image = np.random.randint(0, 255, (*self.target_size, 3), dtype=np.uint8)
            
            # Create dummy mask with hand/face regions
            mask = np.zeros(self.target_size, dtype=np.uint8)
            
            # Add some random hand regions (class 1)
            for _ in range(np.random.randint(0, 3)):
                center = (np.random.randint(50, self.target_size[1]-50), 
                         np.random.randint(50, self.target_size[0]-50))
                radius = np.random.randint(30, 80)
                cv2.circle(mask, center, radius, 1, -1)
            
            # Add some random face regions (class 2)
            for _ in range(np.random.randint(0, 2)):
                center = (np.random.randint(50, self.target_size[1]-50), 
                         np.random.randint(50, self.target_size[0]-50))
                radius = np.random.randint(40, 70)
                cv2.circle(mask, center, radius, 2, -1)
            
            # Save files
            img_path = self.data_dir / 'images' / self.split / f'sample_{i:04d}.jpg'
            mask_path = self.data_dir / 'masks' / self.split / f'sample_{i:04d}.png'
            
            cv2.imwrite(str(img_path), image)
            cv2.imwrite(str(mask_path), mask)
            
            self.image_files.append(img_path)
            self.mask_files.append(mask_path)
        
        print(f"Created {num_samples} dummy samples for {self.split} split")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(str(self.image_files[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(str(self.mask_files[idx]), cv2.IMREAD_GRAYSCALE)
        
        # Resize
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), 
                         interpolation=cv2.INTER_NEAREST)
        
        # Apply transforms
        if self.transform:
            # Convert to PIL or apply tensor transforms
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        mask = torch.from_numpy(mask).long()
        
        return image, mask


class SegmentationTrainer:
    """
    Trainer for segmentation models.
    """
    
    def __init__(self, model, device='auto', num_classes=3):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            device: Training device
            num_classes: Number of segmentation classes
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.num_classes = num_classes
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
        
        print(f"Trainer initialized on device: {self.device}")
    
    def calculate_iou(self, predictions, targets, num_classes):
        """
        Calculate IoU metric.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            num_classes: Number of classes
            
        Returns:
            Mean IoU across classes
        """
        pred_labels = torch.argmax(predictions, dim=1)
        
        ious = []
        for class_id in range(num_classes):
            pred_mask = (pred_labels == class_id)
            target_mask = (targets == class_id)
            
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            
            if union > 0:
                iou = intersection / union
                ious.append(iou.item())
        
        return np.mean(ious) if ious else 0.0
    
    def train_epoch(self, dataloader, optimizer):
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def validate(self, dataloader):
        """
        Validate model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Validation loss and IoU
        """
        self.model.eval()
        total_loss = 0
        total_iou = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, masks in dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                iou = self.calculate_iou(outputs, masks, self.num_classes)
                
                total_loss += loss.item()
                total_iou += iou
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches
        
        return avg_loss, avg_iou
    
    def train(self, train_loader, val_loader, num_epochs=10, lr=1e-4, save_dir='checkpoints'):
        """
        Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            lr: Learning rate
            save_dir: Directory to save checkpoints
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_iou = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer)
            
            # Validate
            val_loss, val_iou = self.validate(val_loader)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_ious.append(val_iou)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val IoU: {val_iou:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save(self.model.state_dict(), 
                          os.path.join(save_dir, 'best_model.pth'))
                print(f"  New best model saved! IoU: {val_iou:.4f}")
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_iou': val_iou,
            }
            torch.save(checkpoint, os.path.join(save_dir, 'last_checkpoint.pth'))
            
            print("-" * 50)
        
        print(f"Training completed! Best validation IoU: {best_val_iou:.4f}")
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_ious': self.val_ious
        }
        
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)


def create_data_loaders(data_dir, batch_size=4, num_workers=2, target_size=(512, 512)):
    """
    Create data loaders for training and validation.
    
    Args:
        data_dir: Data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        target_size: Target image size
        
    Returns:
        Train and validation data loaders
    """
    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Datasets
    train_dataset = HandFaceDataset(data_dir, 'train', train_transform, target_size)
    val_dataset = HandFaceDataset(data_dir, 'val', val_transform, target_size)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Hand/Face Segmentation Models')
    
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Path to dataset directory')
    parser.add_argument('--model', type=str, default='unet',
                       choices=['unet'],
                       help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Training device')
    parser.add_argument('--image-size', type=int, default=512,
                       help='Input image size')
    parser.add_argument('--num-classes', type=int, default=3,
                       help='Number of segmentation classes')
    
    args = parser.parse_args()
    
    # Create data loaders
    target_size = (args.image_size, args.image_size)
    train_loader, val_loader = create_data_loaders(
        args.data_dir, 
        args.batch_size, 
        target_size=target_size
    )
    
    # Create model
    if args.model == 'unet':
        model = UNet(in_channels=3, out_channels=args.num_classes)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Create trainer
    trainer = SegmentationTrainer(model, args.device, args.num_classes)
    
    # Train model
    trainer.train(
        train_loader, 
        val_loader, 
        num_epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
