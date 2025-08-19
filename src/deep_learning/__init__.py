"""
Deep learning module initialization.

This module provides deep learning models and training utilities for
hand and face segmentation and detection tasks.
"""

from .unet import UNet, UNetWithBackbone, create_unet_model
from .mask_rcnn import MaskRCNN, create_mask_rcnn
from .training import (
    Trainer, SegmentationLoss, IoUMetric, DiceMetric,
    create_optimizer, create_scheduler
)

__all__ = [
    'UNet',
    'UNetWithBackbone', 
    'create_unet_model',
    'MaskRCNN',
    'create_mask_rcnn',
    'Trainer',
    'SegmentationLoss',
    'IoUMetric',
    'DiceMetric',
    'create_optimizer',
    'create_scheduler'
]
