"""
Core segmentation models for real-time hand/face tracking.

This module provides the primary model architectures for semantic and instance
segmentation, including U-Net, Mask R-CNN, classical CV baselines, and depth
estimation capabilities.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from src.utils.exceptions import ModelInferenceError, ModelLoadError

logger = logging.getLogger(__name__)


class SegmentationClass(Enum):
    """Enumeration of segmentation classes."""
    BACKGROUND = 0
    HAND = 1
    FACE = 2


@dataclass
class InferenceResult:
    """Container for model inference results."""
    mask: np.ndarray
    confidence: Optional[np.ndarray] = None
    boxes: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    processing_time_ms: float = 0.0


class BaseSegmentationModel(ABC):
    """Abstract base class for all segmentation models."""
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> InferenceResult:
        """
        Run inference on input image.
        
        Args:
            image: Input BGR image as numpy array (H, W, 3)
            
        Returns:
            InferenceResult containing segmentation mask and metadata
        """
        pass
    
    @abstractmethod
    def load_weights(self, path: Path) -> None:
        """Load model weights from checkpoint."""
        pass


def _get_device() -> torch.device:
    """Determine the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ConvBlock(nn.Module):
    """
    Double convolution block with batch normalization.
    
    Standard U-Net building block consisting of two 3x3 convolutions,
    each followed by batch normalization and ReLU activation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        
        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ])
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation.
    
    Implements the encoder-decoder architecture with skip connections
    for precise localization in segmentation tasks.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        out_channels: Number of output classes
        base_features: Number of features in first encoder layer
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_features: int = 64,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        features = [base_features * (2 ** i) for i in range(5)]
        
        # Encoder path
        self.encoder1 = ConvBlock(in_channels, features[0], dropout_rate)
        self.encoder2 = ConvBlock(features[0], features[1], dropout_rate)
        self.encoder3 = ConvBlock(features[1], features[2], dropout_rate)
        self.encoder4 = ConvBlock(features[2], features[3], dropout_rate)
        
        # Bottleneck
        self.bottleneck = ConvBlock(features[3], features[4], dropout_rate)
        
        # Decoder path
        self.upconv4 = nn.ConvTranspose2d(features[4], features[3], kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(features[4], features[3], dropout_rate)
        
        self.upconv3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(features[3], features[2], dropout_rate)
        
        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(features[2], features[1], dropout_rate)
        
        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(features[1], features[0], dropout_rate)
        
        # Output layer
        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize model weights using Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder with skip connections
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = self._center_crop_and_concat(dec4, enc4)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = self._center_crop_and_concat(dec3, enc3)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = self._center_crop_and_concat(dec2, enc2)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = self._center_crop_and_concat(dec1, enc1)
        dec1 = self.decoder1(dec1)
        
        return self.output_conv(dec1)
    
    @staticmethod
    def _center_crop_and_concat(
        upsampled: torch.Tensor,
        skip: torch.Tensor
    ) -> torch.Tensor:
        """Crop skip connection to match upsampled size and concatenate."""
        diff_h = skip.size(2) - upsampled.size(2)
        diff_w = skip.size(3) - upsampled.size(3)
        
        if diff_h != 0 or diff_w != 0:
            skip = skip[
                :, :,
                diff_h // 2 : skip.size(2) - (diff_h - diff_h // 2),
                diff_w // 2 : skip.size(3) - (diff_w - diff_w // 2)
            ]
        
        return torch.cat([upsampled, skip], dim=1)


class MaskRCNNWrapper(BaseSegmentationModel):
    """
    Mask R-CNN wrapper for instance segmentation.
    
    Provides instance-level segmentation with bounding boxes and per-instance
    masks for hand and face detection.
    
    Args:
        num_classes: Number of classes including background
        score_threshold: Minimum confidence score for detections
        pretrained: Whether to load pretrained backbone weights
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        score_threshold: float = 0.5,
        pretrained: bool = True
    ):
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.device = _get_device()
        
        logger.info(f"Initializing Mask R-CNN on {self.device}")
        
        try:
            # Load pre-trained Mask R-CNN
            self.model = maskrcnn_resnet50_fpn(
                weights="DEFAULT" if pretrained else None
            )
            
            # Replace box predictor head
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes
            )
            
            # Replace mask predictor head
            in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
            self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
                in_features_mask, 256, num_classes
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Mask R-CNN initialized successfully")
            
        except Exception as e:
            raise ModelLoadError("MaskRCNN", cause=e)
    
    def predict(self, image: np.ndarray) -> InferenceResult:
        """
        Run instance segmentation on input image.
        
        Args:
            image: Input BGR image (H, W, 3)
            
        Returns:
            InferenceResult with combined semantic mask from instances
        """
        import time
        start_time = time.perf_counter()
        
        try:
            # Preprocess
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(image_tensor)[0]
            
            # Convert to semantic mask
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            confidence = np.zeros((h, w), dtype=np.float32)
            
            if "masks" in predictions and len(predictions["masks"]) > 0:
                masks = predictions["masks"].cpu().numpy()
                labels = predictions["labels"].cpu().numpy()
                scores = predictions["scores"].cpu().numpy()
                
                # Sort by score (process low confidence first, high confidence overwrites)
                sorted_indices = np.argsort(scores)
                
                for idx in sorted_indices:
                    if scores[idx] >= self.score_threshold:
                        instance_mask = masks[idx, 0] > 0.5
                        mask[instance_mask] = labels[idx]
                        confidence[instance_mask] = scores[idx]
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            return InferenceResult(
                mask=mask,
                confidence=confidence,
                boxes=predictions.get("boxes", torch.tensor([])).cpu().numpy(),
                labels=predictions.get("labels", torch.tensor([])).cpu().numpy(),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            raise ModelInferenceError(
                "MaskRCNN",
                input_shape=image.shape,
                cause=e
            )
    
    def load_weights(self, path: Path) -> None:
        """Load model weights from checkpoint."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded Mask R-CNN weights from {path}")
        except Exception as e:
            raise ModelLoadError("MaskRCNN", str(path), cause=e)


class ClassicalSegmenter(BaseSegmentationModel):
    """
    Classical computer vision baseline using skin detection and morphological operations.
    
    Provides fast, lightweight segmentation using color-space analysis
    without requiring GPU acceleration.
    
    Args:
        hsv_lower: Lower HSV threshold for skin detection
        hsv_upper: Upper HSV threshold for skin detection
        ycrcb_lower: Lower YCrCb threshold for skin detection
        ycrcb_upper: Upper YCrCb threshold for skin detection
        min_contour_area: Minimum contour area to consider
        hand_area_threshold: Area threshold to distinguish hands from faces
    """
    
    DEFAULT_HSV_LOWER = np.array([0, 20, 70], dtype=np.uint8)
    DEFAULT_HSV_UPPER = np.array([20, 255, 255], dtype=np.uint8)
    DEFAULT_YCRCB_LOWER = np.array([0, 135, 85], dtype=np.uint8)
    DEFAULT_YCRCB_UPPER = np.array([255, 180, 135], dtype=np.uint8)
    
    def __init__(
        self,
        hsv_lower: Optional[np.ndarray] = None,
        hsv_upper: Optional[np.ndarray] = None,
        ycrcb_lower: Optional[np.ndarray] = None,
        ycrcb_upper: Optional[np.ndarray] = None,
        min_contour_area: int = 1000,
        hand_area_threshold: int = 5000
    ):
        self.hsv_lower = hsv_lower if hsv_lower is not None else self.DEFAULT_HSV_LOWER
        self.hsv_upper = hsv_upper if hsv_upper is not None else self.DEFAULT_HSV_UPPER
        self.ycrcb_lower = ycrcb_lower if ycrcb_lower is not None else self.DEFAULT_YCRCB_LOWER
        self.ycrcb_upper = ycrcb_upper if ycrcb_upper is not None else self.DEFAULT_YCRCB_UPPER
        self.min_contour_area = min_contour_area
        self.hand_area_threshold = hand_area_threshold
        
        # Morphological kernel
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Optical flow state
        self._prev_frame: Optional[np.ndarray] = None
        self._prev_points: Optional[np.ndarray] = None
        
        self._lk_params = {
            "winSize": (15, 15),
            "maxLevel": 2,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }
        
        self._feature_params = {
            "maxCorners": 100,
            "qualityLevel": 0.3,
            "minDistance": 7,
            "blockSize": 7
        }
        
        logger.info("Classical segmenter initialized")
    
    def detect_skin(self, image: np.ndarray) -> np.ndarray:
        """
        Detect skin regions using multi-colorspace analysis.
        
        Combines HSV and YCrCb color space thresholding for robust
        skin detection across different lighting conditions.
        
        Args:
            image: Input BGR image
            
        Returns:
            Binary mask of detected skin regions
        """
        # Convert to color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Threshold in each color space
        mask_hsv = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask_ycrcb = cv2.inRange(ycrcb, self.ycrcb_lower, self.ycrcb_upper)
        
        # Combine masks
        skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        
        # Morphological cleanup
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, self._kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, self._kernel)
        
        return skin_mask
    
    def track_features(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Track features using Lucas-Kanade optical flow.
        
        Args:
            image: Current frame
            
        Returns:
            Tuple of (new_points, old_points) arrays
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if self._prev_frame is None:
            self._prev_frame = gray
            self._prev_points = cv2.goodFeaturesToTrack(
                gray, mask=None, **self._feature_params
            )
            return np.array([]), np.array([])
        
        if self._prev_points is not None and len(self._prev_points) > 0:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_frame, gray, self._prev_points, None, **self._lk_params
            )
            
            good_new = next_points[status.flatten() == 1]
            good_old = self._prev_points[status.flatten() == 1]
            
            self._prev_points = good_new.reshape(-1, 1, 2)
            self._prev_frame = gray
            
            return good_new, good_old
        
        self._prev_frame = gray
        return np.array([]), np.array([])
    
    def predict(self, image: np.ndarray) -> InferenceResult:
        """
        Segment hands and faces using classical methods.
        
        Args:
            image: Input BGR image
            
        Returns:
            InferenceResult with segmentation mask
        """
        return self.segment_hands_faces(image)
    
    def segment_hands_faces(self, image: np.ndarray) -> InferenceResult:
        """
        Segment hands and faces based on skin detection and contour analysis.
        
        Args:
            image: Input BGR image
            
        Returns:
            InferenceResult with class assignments based on contour area
        """
        import time
        start_time = time.perf_counter()
        
        skin_mask = self.detect_skin(image)
        
        contours, _ = cv2.findContours(
            skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area >= self.min_contour_area:
                # Classify based on area heuristic
                class_id = (
                    SegmentationClass.HAND.value
                    if area >= self.hand_area_threshold
                    else SegmentationClass.FACE.value
                )
                cv2.fillPoly(mask, [contour], class_id)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return InferenceResult(
            mask=mask,
            processing_time_ms=processing_time
        )
    
    def load_weights(self, path: Path) -> None:
        """No weights to load for classical methods."""
        logger.warning("Classical segmenter has no weights to load")
    
    def reset_tracking(self) -> None:
        """Reset optical flow tracking state."""
        self._prev_frame = None
        self._prev_points = None


class DepthEstimator:
    """
    Monocular depth estimation using MiDaS.
    
    Provides relative depth maps from single RGB images for 3D
    scene understanding and mesh generation.
    
    Args:
        model_type: MiDaS model variant ("MiDaS_small", "DPT_Large", etc.)
        enable_optimization: Whether to enable model optimization
    """
    
    SUPPORTED_MODELS = ["MiDaS_small", "DPT_Large", "DPT_Hybrid"]
    
    def __init__(
        self,
        model_type: str = "MiDaS_small",
        enable_optimization: bool = True
    ):
        self.model_type = model_type
        self.device = _get_device()
        self.available = False
        self.model = None
        self.transform = None
        
        self._load_model(enable_optimization)
    
    def _load_model(self, enable_optimization: bool) -> None:
        """Load the MiDaS model and transforms."""
        try:
            logger.info(f"Loading MiDaS model: {self.model_type}")
            
            self.model = torch.hub.load(
                "intel-isl/MiDaS",
                self.model_type,
                trust_repo=True
            )
            
            transforms = torch.hub.load(
                "intel-isl/MiDaS",
                "transforms",
                trust_repo=True
            )
            
            if self.model_type == "MiDaS_small":
                self.transform = transforms.small_transform
            else:
                self.transform = transforms.dpt_transform
            
            self.model.to(self.device)
            self.model.eval()
            
            if enable_optimization and self.device.type == "cuda":
                self.model = torch.jit.optimize_for_inference(
                    torch.jit.script(self.model)
                ) if hasattr(torch.jit, "optimize_for_inference") else self.model
            
            self.available = True
            logger.info(f"MiDaS loaded successfully on {self.device}")
            
        except Exception as e:
            logger.warning(f"MiDaS initialization failed: {e}")
            self.available = False
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from a single RGB image.
        
        Args:
            image: Input BGR image
            
        Returns:
            Depth map normalized to input image dimensions
        """
        if not self.available:
            logger.debug("MiDaS not available, returning uniform depth")
            return np.ones(image.shape[:2], dtype=np.float32)
        
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(rgb).to(self.device)
            
            with torch.no_grad():
                depth = self.model(input_tensor)
                depth = F.interpolate(
                    depth.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            depth_np = depth.cpu().numpy()
            
            # Normalize to [0, 1] range
            depth_min, depth_max = depth_np.min(), depth_np.max()
            if depth_max - depth_min > 1e-6:
                depth_np = (depth_np - depth_min) / (depth_max - depth_min)
            
            return depth_np.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return np.ones(image.shape[:2], dtype=np.float32)
    
    @property
    def is_available(self) -> bool:
        """Check if depth estimation is available."""
        return self.available


class ModelFactory:
    """
    Factory class for creating and managing model instances.
    
    Provides centralized model creation with caching and configuration.
    """
    
    _instances: dict[str, Any] = {}
    
    @classmethod
    def create_unet(
        cls,
        in_channels: int = 3,
        out_channels: int = 3,
        checkpoint_path: Optional[Path] = None,
        device: Optional[torch.device] = None
    ) -> UNet:
        """Create a U-Net model instance."""
        model = UNet(in_channels=in_channels, out_channels=out_channels)
        
        if device is None:
            device = _get_device()
        model = model.to(device)
        
        if checkpoint_path and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded U-Net weights from {checkpoint_path}")
        
        return model
    
    @classmethod
    def create_maskrcnn(
        cls,
        num_classes: int = 3,
        checkpoint_path: Optional[Path] = None
    ) -> MaskRCNNWrapper:
        """Create a Mask R-CNN model instance."""
        model = MaskRCNNWrapper(num_classes=num_classes)
        
        if checkpoint_path and checkpoint_path.exists():
            model.load_weights(checkpoint_path)
        
        return model
    
    @classmethod
    def create_classical(cls) -> ClassicalSegmenter:
        """Create a classical segmenter instance."""
        return ClassicalSegmenter()
    
    @classmethod
    def create_depth_estimator(
        cls,
        model_type: str = "MiDaS_small"
    ) -> DepthEstimator:
        """Create a depth estimator instance."""
        return DepthEstimator(model_type=model_type)
    
    @classmethod
    def create_all(
        cls,
        unet_checkpoint: Optional[Path] = None,
        maskrcnn_checkpoint: Optional[Path] = None
    ) -> dict[str, Any]:
        """
        Create all available models.
        
        Args:
            unet_checkpoint: Optional path to U-Net weights
            maskrcnn_checkpoint: Optional path to Mask R-CNN weights
            
        Returns:
            Dictionary containing all model instances
        """
        logger.info("Creating all segmentation models...")
        
        models = {
            "unet": cls.create_unet(checkpoint_path=unet_checkpoint),
            "maskrcnn": cls.create_maskrcnn(checkpoint_path=maskrcnn_checkpoint),
            "classical": cls.create_classical(),
            "depth": cls.create_depth_estimator(),
        }
        
        logger.info("All models created successfully")
        return models


# Backwards compatibility alias
def create_models() -> dict[str, Any]:
    """
    Create all segmentation models.
    
    Returns:
        Dictionary with model instances keyed by name
    """
    return ModelFactory.create_all()


# Legacy class aliases for backwards compatibility
MaskRCNN = MaskRCNNWrapper
ClassicalBaseline = ClassicalSegmenter


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing model creation...")
    
    # Test model factory
    models = create_models()
    print(f"Created {len(models)} models: {list(models.keys())}")
    
    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test classical method
    result = models["classical"].predict(dummy_image)
    print(f"Classical segmentation: shape={result.mask.shape}, time={result.processing_time_ms:.2f}ms")
    
    # Test depth estimation
    depth_map = models["depth"].estimate_depth(dummy_image)
    print(f"Depth map: shape={depth_map.shape}, range=[{depth_map.min():.3f}, {depth_map.max():.3f}]")
    
    print("\nAll tests passed!")
