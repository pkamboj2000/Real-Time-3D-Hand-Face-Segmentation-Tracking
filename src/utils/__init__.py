"""
Utility functions for the Hand/Face Segmentation project
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import Tuple, List, Dict, Optional, Union
import time
import logging
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger("hand_face_segmentation")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class Timer:
    """Context manager for timing operations"""
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time


def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (256, 256),
                    normalize: bool = True, to_tensor: bool = True) -> Union[np.ndarray, torch.Tensor]:
    """
    Preprocess image for model input
    
    Args:
        image: Input image in BGR format
        target_size: Target size (height, width)
        normalize: Whether to normalize to [0, 1]
        to_tensor: Whether to convert to torch tensor
    
    Returns:
        Preprocessed image
    """
    # Convert BGR to RGB
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Resize image
    resized = cv2.resize(image_rgb, (target_size[1], target_size[0]))
    
    if normalize:
        resized = resized.astype(np.float32) / 255.0
    
    if to_tensor:
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
        return tensor
    
    return resized


def postprocess_mask(mask: np.ndarray, original_size: Tuple[int, int],
                    threshold: float = 0.5) -> np.ndarray:
    """
    Postprocess segmentation mask to original size
    
    Args:
        mask: Predicted mask
        original_size: Original image size (height, width)
        threshold: Threshold for binary mask
    
    Returns:
        Processed mask
    """
    if len(mask.shape) == 4:  # Remove batch dimension
        mask = mask.squeeze(0)
    if len(mask.shape) == 3:  # Remove channel dimension or get first channel
        mask = mask[0] if mask.shape[0] > 1 else mask.squeeze(0)
    
    # Apply threshold
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # Resize to original size
    resized_mask = cv2.resize(binary_mask, (original_size[1], original_size[0]))
    
    return resized_mask


def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two masks
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
    
    Returns:
        IoU score
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(intersection) / float(union)


def calculate_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calculate Dice coefficient between two masks
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
    
    Returns:
        Dice coefficient
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return 2.0 * float(intersection) / float(total)


def draw_bounding_box(image: np.ndarray, bbox: Tuple[int, int, int, int],
                     color: Tuple[int, int, int] = (0, 255, 0),
                     thickness: int = 2, label: str = None) -> np.ndarray:
    """
    Draw bounding box on image
    
    Args:
        image: Input image
        bbox: Bounding box (x, y, width, height)
        color: Box color in BGR format
        thickness: Line thickness
        label: Optional text label
    
    Returns:
        Image with bounding box drawn
    """
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    
    if label:
        font_scale = 0.6
        text_thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                            font_scale, text_thickness)
        
        cv2.rectangle(image, (x, y - text_h - 10), (x + text_w, y), color, -1)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, (255, 255, 255), text_thickness)
    
    return image


def draw_keypoints(image: np.ndarray, keypoints: List[Tuple[int, int]],
                  color: Tuple[int, int, int] = (255, 0, 0),
                  radius: int = 3) -> np.ndarray:
    """
    Draw keypoints on image
    
    Args:
        image: Input image
        keypoints: List of (x, y) coordinates
        color: Keypoint color in BGR format
        radius: Keypoint radius
    
    Returns:
        Image with keypoints drawn
    """
    for x, y in keypoints:
        cv2.circle(image, (int(x), int(y)), radius, color, -1)
    
    return image


def apply_mask_overlay(image: np.ndarray, mask: np.ndarray,
                      color: Tuple[int, int, int] = (0, 255, 0),
                      alpha: float = 0.5) -> np.ndarray:
    """
    Apply colored mask overlay on image
    
    Args:
        image: Input image
        mask: Binary mask
        color: Overlay color in BGR format
        alpha: Transparency factor
    
    Returns:
        Image with mask overlay
    """
    overlay = image.copy()
    overlay[mask > 0] = color
    
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)


class FPSCounter:
    """
    Frame rate counter for real-time applications
    """
    
    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = []
        self.last_time = time.time()
    
    def update(self) -> float:
        """
        Update FPS counter and return current FPS
        
        Returns:
            Current FPS
        """
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.last_time = current_time
        
        self.frame_times.append(frame_time)
        
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
        return 0.0
    
    def reset(self):
        """Reset FPS counter"""
        self.frame_times = []
        self.last_time = time.time()


def create_video_writer(output_path: str, fps: int, frame_size: Tuple[int, int],
                       fourcc: str = 'mp4v') -> cv2.VideoWriter:
    """
    Create video writer for saving processed video
    
    Args:
        output_path: Output video file path
        fps: Frames per second
        frame_size: Frame size (width, height)
        fourcc: Video codec fourcc code
    
    Returns:
        OpenCV VideoWriter object
    """
    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    return cv2.VideoWriter(output_path, fourcc_code, fps, frame_size)


def save_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         epoch: int, loss: float, filepath: str):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Save path
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': time.time()
    }
    
    torch.save(checkpoint, filepath)


def load_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         filepath: str, device: str = 'cpu') -> Dict:
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        filepath: Checkpoint file path
        device: Device to load model on
    
    Returns:
        Checkpoint information
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss'],
        'timestamp': checkpoint.get('timestamp', None)
    }


def export_to_onnx(model: torch.nn.Module, dummy_input: torch.Tensor,
                  output_path: str, opset_version: int = 11):
    """
    Export PyTorch model to ONNX format
    
    Args:
        model: PyTorch model
        dummy_input: Sample input tensor
        output_path: Output ONNX file path
        opset_version: ONNX opset version
    """
    model.eval()
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )


def ensure_directory_exists(directory_path: Union[str, Path]):
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory_path: Path to directory
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)


def get_git_commit_hash() -> str:
    """
    Get current git commit hash for experiment tracking
    
    Returns:
        Git commit hash or 'unknown' if not available
    """
    try:
        import subprocess
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    
    return 'unknown'


# Export all utility functions
__all__ = [
    'setup_logging', 'Timer', 'preprocess_image', 'postprocess_mask',
    'calculate_iou', 'calculate_dice', 'draw_bounding_box', 'draw_keypoints',
    'apply_mask_overlay', 'FPSCounter', 'create_video_writer',
    'save_model_checkpoint', 'load_model_checkpoint', 'export_to_onnx',
    'ensure_directory_exists', 'get_git_commit_hash'
]
        

@property
def elapsed_ms(self) -> float:
    """Get elapsed time in milliseconds"""
    return self.elapsed * 1000 if self.elapsed else 0

class FPSCounter:
    """FPS counter for real-time applications"""
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = []
        self.last_time = time.time()
    
    def update(self) -> float:
        """Update FPS counter and return current FPS"""
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.last_time = current_time
        
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        return 0

def preprocess_image(
    image: np.ndarray, 
    target_size: Tuple[int, int] = (512, 512),
    normalize: bool = True
) -> torch.Tensor:
    """
    Preprocess image for model inference
    
    Args:
        image: Input image as numpy array (H, W, C)
        target_size: Target size (height, width)
        normalize: Whether to normalize pixel values
    
    Returns:
        Preprocessed image tensor (1, C, H, W)
    """
    # Resize image
    if image.shape[:2] != target_size:
        image = cv2.resize(image, target_size[::-1])  # cv2 uses (W, H)
    
    # Convert BGR to RGB if necessary
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor
    transform_list = [transforms.ToTensor()]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        )
    
    transform = transforms.Compose(transform_list)
    tensor = transform(image)
    
    # Add batch dimension
    return tensor.unsqueeze(0)

def postprocess_mask(
    mask: torch.Tensor, 
    original_size: Tuple[int, int],
    threshold: float = 0.5
) -> np.ndarray:
    """
    Postprocess segmentation mask
    
    Args:
        mask: Model output tensor (1, C, H, W) or (C, H, W)
        original_size: Original image size (height, width)
        threshold: Threshold for binary mask
    
    Returns:
        Binary mask as numpy array (H, W)
    """
    # Remove batch dimension if present
    if mask.dim() == 4:
        mask = mask.squeeze(0)
    
    # Take the first channel if multi-channel
    if mask.dim() == 3:
        mask = mask[0]
    
    # Convert to numpy
    mask_np = mask.cpu().numpy()
    
    # Apply threshold
    binary_mask = (mask_np > threshold).astype(np.uint8)
    
    # Resize to original size
    if binary_mask.shape != original_size:
        binary_mask = cv2.resize(
            binary_mask, 
            original_size[::-1],  # cv2 uses (W, H)
            interpolation=cv2.INTER_NEAREST
        )
    
    return binary_mask

def draw_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5
) -> np.ndarray:
    """
    Draw segmentation mask overlay on image
    
    Args:
        image: Original image (H, W, C)
        mask: Binary mask (H, W)
        color: Overlay color (R, G, B)
        alpha: Transparency factor
    
    Returns:
        Image with overlay
    """
    overlay = image.copy()
    
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    
    # Blend with original image
    result = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    
    return result

def draw_bounding_box(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    label: Optional[str] = None
) -> np.ndarray:
    """
    Draw bounding box on image
    
    Args:
        image: Input image
        bbox: Bounding box (x1, y1, x2, y2)
        color: Box color (R, G, B)
        thickness: Line thickness
        label: Optional label text
    
    Returns:
        Image with bounding box
    """
    x1, y1, x2, y2 = bbox
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label if provided
    if label:
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(
            image, 
            (x1, y1 - label_size[1] - 10), 
            (x1 + label_size[0], y1), 
            color, 
            -1
        )
        cv2.putText(
            image, 
            label, 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
    
    return image

def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two masks
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
    
    Returns:
        IoU score
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def calculate_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate Dice coefficient between two masks
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
    
    Returns:
        Dice coefficient
    """
    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return 2.0 * intersection / total

def get_largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Get the largest contour from a binary mask
    
    Args:
        mask: Binary mask
    
    Returns:
        Largest contour points or None if no contours found
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return None
    
    # Find largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    return largest_contour

def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Convert binary mask to bounding box
    
    Args:
        mask: Binary mask
    
    Returns:
        Bounding box (x1, y1, x2, y2) or None if no objects found
    """
    contour = get_largest_contour(mask)
    
    if contour is None:
        return None
    
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, x + w, y + h

def smooth_tracking(
    current_bbox: Tuple[int, int, int, int],
    previous_bbox: Optional[Tuple[int, int, int, int]],
    smoothing_factor: float = 0.7
) -> Tuple[int, int, int, int]:
    """
    Smooth bounding box tracking using exponential moving average
    
    Args:
        current_bbox: Current frame bounding box
        previous_bbox: Previous frame bounding box
        smoothing_factor: Smoothing factor (0-1)
    
    Returns:
        Smoothed bounding box
    """
    if previous_bbox is None:
        return current_bbox
    
    smoothed = []
    for curr, prev in zip(current_bbox, previous_bbox):
        smoothed_val = smoothing_factor * prev + (1 - smoothing_factor) * curr
        smoothed.append(int(smoothed_val))
    
    return tuple(smoothed)

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_device() -> torch.device:
    """Get the best available device for PyTorch"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    else:
        return torch.device("cpu")

class PerformanceProfiler:
    """Simple performance profiler for tracking bottlenecks"""
    
    def __init__(self):
        self.timings = {}
        self.counts = {}
    
    def start_timer(self, name: str):
        """Start timing an operation"""
        self.timings[f"{name}_start"] = time.time()
    
    def end_timer(self, name: str):
        """End timing an operation and record the duration"""
        start_time = self.timings.get(f"{name}_start")
        if start_time is None:
            return
        
        duration = time.time() - start_time
        
        if name not in self.timings:
            self.timings[name] = []
            self.counts[name] = 0
        
        self.timings[name].append(duration)
        self.counts[name] += 1
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a named operation"""
        if name not in self.timings:
            return {}
        
        times = self.timings[name]
        return {
            "count": self.counts[name],
            "total_time": sum(times),
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "avg_fps": 1.0 / (sum(times) / len(times)) if times else 0
        }
    
    def print_stats(self):
        """Print all collected statistics"""
        print("\n=== Performance Statistics ===")
        for name in self.timings:
            if not name.endswith("_start"):
                stats = self.get_stats(name)
                print(f"{name}:")
                print(f"  Count: {stats['count']}")
                print(f"  Avg Time: {stats['avg_time']*1000:.2f}ms")
                print(f"  Min Time: {stats['min_time']*1000:.2f}ms")
                print(f"  Max Time: {stats['max_time']*1000:.2f}ms")
                print(f"  Avg FPS: {stats['avg_fps']:.2f}")
                print()
