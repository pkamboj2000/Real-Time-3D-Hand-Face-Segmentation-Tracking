"""
MiDaS depth estimation implementation.

This module provides depth estimation capabilities using MiDaS (Monocular Depth estimation)
for 3D hand and face tracking applications. It includes model loading, inference,
and post-processing utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
import logging
import os

logger = logging.getLogger(__name__)


class MiDaSModel(nn.Module):
    """
    MiDaS model wrapper for depth estimation.
    
    This class provides a simplified interface to the MiDaS depth estimation
    model with preprocessing and postprocessing capabilities.
    """
    
    def __init__(self, model_type: str = 'MiDaS_small', device: Optional[torch.device] = None):
        """
        Initialize MiDaS model.
        
        Args:
            model_type: Type of MiDaS model ('MiDaS', 'MiDaS_small', 'DPT_Large')
            device: Device for inference
        """
        super(MiDaSModel, self).__init__()
        
        self.model_type = model_type
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model configuration
        self.input_size = self._get_input_size()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Move to device
        self.to(self.device)
        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)
    
    def _get_input_size(self) -> Tuple[int, int]:
        """Get input size for the model type."""
        if self.model_type == 'MiDaS_small':
            return (256, 256)
        elif self.model_type == 'MiDaS':
            return (384, 384)
        elif self.model_type == 'DPT_Large':
            return (384, 384)
        else:
            return (256, 256)
    
    def _load_model(self) -> nn.Module:
        """Load MiDaS model."""
        try:
            # Try to load from torch hub
            model = torch.hub.load('intel-isl/MiDaS', self.model_type, pretrained=True)
            logger.info(f"Loaded {self.model_type} from torch hub")
            return model
        except Exception as e:
            logger.warning(f"Failed to load from torch hub: {e}")
            # Fallback to simplified model
            return self._create_simple_depth_model()
    
    def _create_simple_depth_model(self) -> nn.Module:
        """Create a simple depth estimation model as fallback."""
        class SimpleDepthModel(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
                
                # Decoder
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, 2, stride=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    
                    nn.ConvTranspose2d(128, 64, 2, stride=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    
                    nn.ConvTranspose2d(64, 32, 2, stride=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 1, 3, padding=1),
                )
            
            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x
        
        model = SimpleDepthModel()
        logger.info("Created simple depth model as fallback")
        return model
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for depth estimation.
        
        Args:
            image: Input image as numpy array (H, W, 3) in BGR format
            
        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize to model input size
        h, w = self.input_size
        resized = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor and normalize
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0)
        tensor = tensor / 255.0
        
        # Normalize with ImageNet stats
        tensor = (tensor - self.mean) / self.std
        
        return tensor.to(self.device)
    
    def postprocess(self, depth_map: torch.Tensor, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess depth map to target size.
        
        Args:
            depth_map: Raw depth map from model (1, 1, H, W)
            target_size: Target size (height, width)
            
        Returns:
            Processed depth map as numpy array
        """
        # Remove batch and channel dimensions
        if len(depth_map.shape) == 4:
            depth_map = depth_map.squeeze(0).squeeze(0)
        elif len(depth_map.shape) == 3:
            depth_map = depth_map.squeeze(0)
        
        # Convert to numpy
        depth_np = depth_map.cpu().numpy()
        
        # Resize to target size
        depth_resized = cv2.resize(depth_np, (target_size[1], target_size[0]), 
                                  interpolation=cv2.INTER_LINEAR)
        
        # Normalize to 0-255 range for visualization
        depth_normalized = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        return depth_normalized
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MiDaS model.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Depth map tensor (B, 1, H, W)
        """
        with torch.no_grad():
            depth = self.model(x)
            
            # Ensure output has correct dimensions
            if len(depth.shape) == 3:
                depth = depth.unsqueeze(1)
            
            return depth
    
    def estimate_depth(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate depth from input image.
        
        Args:
            image: Input image (H, W, 3) in BGR format
            
        Returns:
            Tuple of (raw_depth, normalized_depth)
        """
        original_size = (image.shape[0], image.shape[1])
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Inference
        with torch.no_grad():
            depth_tensor = self.forward(input_tensor)
        
        # Postprocess
        depth_map = self.postprocess(depth_tensor, original_size)
        
        # Also return raw depth values
        raw_depth = depth_tensor.squeeze().cpu().numpy()
        raw_depth_resized = cv2.resize(raw_depth, (original_size[1], original_size[0]), 
                                      interpolation=cv2.INTER_LINEAR)
        
        return raw_depth_resized, depth_map


class DepthProcessor:
    """
    Utility class for processing depth maps and extracting 3D information.
    """
    
    def __init__(self, camera_intrinsics: Optional[Dict[str, float]] = None):
        """
        Initialize depth processor.
        
        Args:
            camera_intrinsics: Dictionary with 'fx', 'fy', 'cx', 'cy' keys
        """
        if camera_intrinsics is None:
            # Default camera intrinsics (approximate values)
            self.camera_intrinsics = {
                'fx': 500.0,  # Focal length in x
                'fy': 500.0,  # Focal length in y
                'cx': 320.0,  # Principal point x
                'cy': 240.0   # Principal point y
            }
        else:
            self.camera_intrinsics = camera_intrinsics
    
    def depth_to_pointcloud(self, depth_map: np.ndarray, 
                           color_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert depth map to 3D point cloud.
        
        Args:
            depth_map: Depth map (H, W)
            color_image: Optional color image (H, W, 3)
            
        Returns:
            Point cloud array (N, 3) or (N, 6) if color provided
        """
        height, width = depth_map.shape
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to 3D coordinates
        fx, fy = self.camera_intrinsics['fx'], self.camera_intrinsics['fy']
        cx, cy = self.camera_intrinsics['cx'], self.camera_intrinsics['cy']
        
        # Scale depth values (MiDaS outputs relative depth)
        depth_scaled = depth_map * 0.01  # Scale factor
        
        x = (u - cx) * depth_scaled / fx
        y = (v - cy) * depth_scaled / fy
        z = depth_scaled
        
        # Stack coordinates
        points_3d = np.stack([x, y, z], axis=-1)
        
        # Reshape to point cloud format
        points = points_3d.reshape(-1, 3)
        
        # Filter out invalid points
        valid_mask = (points[:, 2] > 0) & (points[:, 2] < 10.0)  # Reasonable depth range
        points = points[valid_mask]
        
        # Add color if provided
        if color_image is not None:
            colors = color_image.reshape(-1, 3)[valid_mask]
            points = np.concatenate([points, colors], axis=1)
        
        return points
    
    def estimate_hand_pose(self, depth_map: np.ndarray, 
                          hand_mask: np.ndarray) -> Dict[str, Any]:
        """
        Estimate hand pose from depth map and segmentation mask.
        
        Args:
            depth_map: Depth map (H, W)
            hand_mask: Binary hand segmentation mask (H, W)
            
        Returns:
            Dictionary with hand pose information
        """
        # Extract hand region depth
        hand_depth = depth_map[hand_mask > 0]
        
        if len(hand_depth) == 0:
            return {}
        
        # Get hand centroid in 3D
        hand_coords = np.where(hand_mask > 0)
        hand_points_2d = np.column_stack([hand_coords[1], hand_coords[0]])  # (x, y)
        
        # Convert to 3D
        fx, fy = self.camera_intrinsics['fx'], self.camera_intrinsics['fy']
        cx, cy = self.camera_intrinsics['cx'], self.camera_intrinsics['cy']
        
        depth_values = depth_map[hand_coords]
        x_3d = (hand_points_2d[:, 0] - cx) * depth_values / fx
        y_3d = (hand_points_2d[:, 1] - cy) * depth_values / fy
        z_3d = depth_values
        
        hand_points_3d = np.column_stack([x_3d, y_3d, z_3d])
        
        # Calculate hand properties
        centroid_3d = np.mean(hand_points_3d, axis=0)
        
        # Estimate hand orientation using PCA
        centered_points = hand_points_3d - centroid_3d
        cov_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Primary direction (largest eigenvalue)
        primary_direction = eigenvectors[:, 0]
        
        # Estimate hand size
        hand_size = np.sqrt(eigenvalues[0]) * 2  # Approximate hand length
        
        return {
            'centroid_3d': centroid_3d,
            'primary_direction': primary_direction,
            'hand_size': hand_size,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'num_points': len(hand_points_3d)
        }
    
    def create_depth_colormap(self, depth_map: np.ndarray, 
                             colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Create colored depth visualization.
        
        Args:
            depth_map: Depth map (H, W)
            colormap: OpenCV colormap
            
        Returns:
            Colored depth map (H, W, 3)
        """
        # Normalize depth map
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, colormap)
        
        return depth_colored
    
    def filter_depth_by_mask(self, depth_map: np.ndarray, 
                            mask: np.ndarray, 
                            background_value: float = 0.0) -> np.ndarray:
        """
        Filter depth map using segmentation mask.
        
        Args:
            depth_map: Input depth map (H, W)
            mask: Binary mask (H, W)
            background_value: Value to set for background pixels
            
        Returns:
            Filtered depth map
        """
        filtered_depth = depth_map.copy()
        filtered_depth[mask == 0] = background_value
        
        return filtered_depth


class RealTimeDepthEstimator:
    """
    Real-time depth estimation for video streams.
    
    Optimized for real-time performance with temporal smoothing
    and efficient processing.
    """
    
    def __init__(self, model_type: str = 'MiDaS_small', 
                 smoothing_factor: float = 0.7,
                 target_fps: int = 30):
        """
        Initialize real-time depth estimator.
        
        Args:
            model_type: MiDaS model type
            smoothing_factor: Temporal smoothing factor (0.0 to 1.0)
            target_fps: Target frame rate
        """
        self.model = MiDaSModel(model_type)
        self.processor = DepthProcessor()
        self.smoothing_factor = smoothing_factor
        self.target_fps = target_fps
        
        # State for temporal smoothing
        self.previous_depth = None
        
        # Performance tracking
        self.frame_times = []
        self.max_frame_times = 100
    
    def process_frame(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process single frame for depth estimation.
        
        Args:
            image: Input image (H, W, 3)
            
        Returns:
            Dictionary with processing results
        """
        import time
        start_time = time.time()
        
        # Estimate depth
        raw_depth, depth_map = self.model.estimate_depth(image)
        
        # Apply temporal smoothing
        if self.previous_depth is not None:
            smoothed_depth = (self.smoothing_factor * raw_depth + 
                            (1 - self.smoothing_factor) * self.previous_depth)
        else:
            smoothed_depth = raw_depth
        
        self.previous_depth = smoothed_depth.copy()
        
        # Create visualizations
        depth_colored = self.processor.create_depth_colormap(smoothed_depth)
        
        # Track performance
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_frame_times:
            self.frame_times.pop(0)
        
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        average_fps = len(self.frame_times) / sum(self.frame_times) if self.frame_times else 0
        
        return {
            'depth_raw': raw_depth,
            'depth_smoothed': smoothed_depth,
            'depth_colored': depth_colored,
            'depth_normalized': depth_map,
            'fps': current_fps,
            'average_fps': average_fps,
            'frame_time': frame_time
        }
    
    def reset(self):
        """Reset temporal state."""
        self.previous_depth = None
        self.frame_times = []


if __name__ == "__main__":
    # Test depth estimation
    depth_estimator = RealTimeDepthEstimator()
    
    # Test with webcam if available
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("Testing real-time depth estimation. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            results = depth_estimator.process_frame(frame)
            
            # Display results
            depth_colored = results['depth_colored']
            fps = results['fps']
            
            # Add FPS text
            cv2.putText(depth_colored, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show original and depth
            combined = np.hstack([frame, depth_colored])
            cv2.imshow('Original | Depth', combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("No webcam available for testing")
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = depth_estimator.process_frame(dummy_image)
        
        print(f"Processed dummy image:")
        print(f"Depth shape: {results['depth_raw'].shape}")
        print(f"FPS: {results['fps']:.2f}")
        print(f"Frame time: {results['frame_time']:.3f}s")
