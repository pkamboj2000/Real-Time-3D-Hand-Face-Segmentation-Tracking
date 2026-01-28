#!/usr/bin/env python3
"""
Real-time live demo for hand/face segmentation and tracking.

This module provides a production-ready live demo application for real-time
hand and face segmentation with depth visualization and FPS monitoring.

Features:
    - Multiple segmentation methods (Classical, U-Net, Mask R-CNN)
    - Real-time depth estimation
    - FPS and latency monitoring
    - Interactive method switching
    - Configurable camera selection

Usage:
    python live_demo.py --method classical --camera-id 0
    python live_demo.py --method unet --no-depth
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import cv2
import numpy as np

from src.utils.exceptions import CameraError, ModelInferenceError
from src.utils.logging_utils import LoggerFactory, log_execution_time

if TYPE_CHECKING:
    import torch

# Lazy import for torch to speed up startup
_torch: Optional[Any] = None


def _get_torch() -> Any:
    """Lazy load torch module."""
    global _torch
    if _torch is None:
        import torch as _torch_module
        _torch = _torch_module
    return _torch


# Module-level logger
logger = LoggerFactory.get_logger(__name__)


class SegmentationMethod(Enum):
    """Available segmentation methods."""
    CLASSICAL = auto()
    UNET = auto()
    MASKRCNN = auto()
    
    @classmethod
    def from_string(cls, method: str) -> SegmentationMethod:
        """
        Create enum from string representation.
        
        Args:
            method: String name of method
            
        Returns:
            Corresponding enum value
            
        Raises:
            ValueError: If method string is invalid
        """
        method_map = {
            'classical': cls.CLASSICAL,
            'unet': cls.UNET,
            'maskrcnn': cls.MASKRCNN,
        }
        method_lower = method.lower()
        if method_lower not in method_map:
            valid_methods = ', '.join(method_map.keys())
            raise ValueError(f"Invalid method '{method}'. Valid: {valid_methods}")
        return method_map[method_lower]
    
    def __str__(self) -> str:
        """Return human-readable string."""
        return self.name.lower()


class SegmentationClass(Enum):
    """Segmentation class identifiers."""
    BACKGROUND = 0
    HAND = 1
    FACE = 2


@dataclass(frozen=True)
class ClassColor:
    """Color definition for segmentation classes."""
    blue: int
    green: int
    red: int
    
    def to_bgr_tuple(self) -> tuple[int, int, int]:
        """Convert to BGR tuple for OpenCV."""
        return (self.blue, self.green, self.red)


@dataclass(frozen=True)
class ColorPalette:
    """Color palette for segmentation visualization."""
    background: ClassColor = field(default_factory=lambda: ClassColor(0, 0, 0))
    hand: ClassColor = field(default_factory=lambda: ClassColor(0, 255, 0))
    face: ClassColor = field(default_factory=lambda: ClassColor(255, 0, 0))
    
    def get_color(self, class_id: int) -> tuple[int, int, int]:
        """Get BGR color for a class ID."""
        color_map = {
            SegmentationClass.BACKGROUND.value: self.background,
            SegmentationClass.HAND.value: self.hand,
            SegmentationClass.FACE.value: self.face,
        }
        return color_map.get(class_id, self.background).to_bgr_tuple()


@dataclass
class FrameMetrics:
    """Metrics computed for a single frame."""
    hand_coverage: float
    face_coverage: float
    total_detection: float
    processing_time_ms: float
    fps: float
    
    @classmethod
    def compute(
        cls,
        mask: np.ndarray,
        processing_time: float,
        fps: float
    ) -> FrameMetrics:
        """
        Compute metrics from segmentation mask.
        
        Args:
            mask: Segmentation mask array
            processing_time: Processing time in seconds
            fps: Current FPS value
            
        Returns:
            Computed frame metrics
        """
        total_pixels = mask.size
        hand_pixels = int(np.sum(mask == SegmentationClass.HAND.value))
        face_pixels = int(np.sum(mask == SegmentationClass.FACE.value))
        
        return cls(
            hand_coverage=hand_pixels / total_pixels if total_pixels > 0 else 0.0,
            face_coverage=face_pixels / total_pixels if total_pixels > 0 else 0.0,
            total_detection=(hand_pixels + face_pixels) / total_pixels if total_pixels > 0 else 0.0,
            processing_time_ms=processing_time * 1000,
            fps=fps
        )


@dataclass
class DemoConfig:
    """Configuration for live demo."""
    method: SegmentationMethod
    show_depth: bool
    camera_id: int
    frame_width: int = 640
    frame_height: int = 480
    fps_window_size: int = 30
    overlay_alpha: float = 0.3
    confidence_threshold: float = 0.5
    mirror_frame: bool = True
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> DemoConfig:
        """Create config from parsed arguments."""
        return cls(
            method=SegmentationMethod.from_string(args.method),
            show_depth=not args.no_depth,
            camera_id=args.camera_id,
        )


@dataclass
class CameraInfo:
    """Information about a camera device."""
    device_id: int
    is_available: bool
    frame_shape: Optional[tuple[int, int, int]] = None
    
    @property
    def resolution_str(self) -> str:
        """Get resolution as string."""
        if self.frame_shape:
            return f"{self.frame_shape[1]}x{self.frame_shape[0]}"
        return "unknown"


class CameraManager:
    """Manages camera device discovery and capture."""
    
    MAX_SCAN_IDS: int = 5
    
    def __init__(self, preferred_id: int = 0):
        """
        Initialize camera manager.
        
        Args:
            preferred_id: Preferred camera device ID
        """
        self._preferred_id = preferred_id
        self._capture: Optional[cv2.VideoCapture] = None
        self._device_id: Optional[int] = None
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
    
    def scan_cameras(self) -> list[CameraInfo]:
        """
        Scan for available camera devices.
        
        Returns:
            List of camera information objects
        """
        cameras: list[CameraInfo] = []
        scan_ids = [self._preferred_id] + [
            i for i in range(self.MAX_SCAN_IDS) if i != self._preferred_id
        ]
        
        self._logger.info("Starting camera device scan...")
        
        for cam_id in scan_ids:
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cameras.append(CameraInfo(
                        device_id=cam_id,
                        is_available=True,
                        frame_shape=frame.shape
                    ))
                    self._logger.info(
                        f"Camera {cam_id}: Available ({frame.shape[1]}x{frame.shape[0]})"
                    )
                else:
                    self._logger.debug(f"Camera {cam_id}: Opened but no frame")
                cap.release()
            else:
                self._logger.debug(f"Camera {cam_id}: Not available")
        
        self._logger.info(f"Found {len(cameras)} available camera(s)")
        return cameras
    
    def open(
        self,
        width: int = 640,
        height: int = 480
    ) -> cv2.VideoCapture:
        """
        Open best available camera.
        
        Args:
            width: Desired frame width
            height: Desired frame height
            
        Returns:
            OpenCV VideoCapture object
            
        Raises:
            CameraError: If no camera could be opened
        """
        cameras = self.scan_cameras()
        
        if not cameras:
            raise CameraError("No camera devices available")
        
        # Prefer the requested camera if available
        selected = next(
            (c for c in cameras if c.device_id == self._preferred_id),
            cameras[0]
        )
        
        self._capture = cv2.VideoCapture(selected.device_id)
        if not self._capture.isOpened():
            raise CameraError(f"Failed to open camera {selected.device_id}")
        
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._device_id = selected.device_id
        
        self._logger.info(f"Opened camera {selected.device_id} at {width}x{height}")
        return self._capture
    
    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read frame from camera."""
        if self._capture is None:
            return False, None
        return self._capture.read()
    
    def release(self) -> None:
        """Release camera resources."""
        if self._capture is not None:
            self._capture.release()
            self._logger.debug(f"Released camera {self._device_id}")
            self._capture = None
            self._device_id = None
    
    @property
    def is_open(self) -> bool:
        """Check if camera is open."""
        return self._capture is not None and self._capture.isOpened()


class FPSTracker:
    """Tracks and smooths FPS measurements."""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize FPS tracker.
        
        Args:
            window_size: Number of samples for moving average
        """
        self._window_size = window_size
        self._fps_queue: deque[float] = deque(maxlen=window_size)
        self._last_time: Optional[float] = None
    
    def update(self, processing_time: float) -> float:
        """
        Update FPS with new measurement.
        
        Args:
            processing_time: Time taken to process frame in seconds
            
        Returns:
            Smoothed FPS value
        """
        fps = 1.0 / max(processing_time, 1e-6)
        self._fps_queue.append(fps)
        return self.current_fps
    
    @property
    def current_fps(self) -> float:
        """Get current smoothed FPS."""
        if not self._fps_queue:
            return 0.0
        return float(np.mean(self._fps_queue))
    
    def reset(self) -> None:
        """Reset FPS tracker."""
        self._fps_queue.clear()
        self._last_time = None


class OverlayRenderer:
    """Renders visualization overlays on frames."""
    
    def __init__(self, palette: Optional[ColorPalette] = None):
        """
        Initialize renderer.
        
        Args:
            palette: Color palette for visualization
        """
        self._palette = palette or ColorPalette()
    
    def create_mask_overlay(
        self,
        mask: np.ndarray,
        alpha: float = 0.6
    ) -> np.ndarray:
        """
        Create colored overlay from segmentation mask.
        
        Args:
            mask: Segmentation mask (H, W)
            alpha: Transparency factor (unused, for API compatibility)
            
        Returns:
            Colored overlay image (H, W, 3)
        """
        h, w = mask.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id in [c.value for c in SegmentationClass]:
            overlay[mask == class_id] = self._palette.get_color(class_id)
        
        return overlay
    
    def draw_info_panel(
        self,
        frame: np.ndarray,
        method: SegmentationMethod,
        metrics: FrameMetrics
    ) -> None:
        """
        Draw information panel on frame.
        
        Args:
            frame: Frame to draw on (modified in-place)
            method: Current segmentation method
            metrics: Frame metrics to display
        """
        # Info panel background
        cv2.rectangle(frame, (10, 10), (300, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 150), (255, 255, 255), 2)
        
        info_lines = [
            f"Method: {method.name}",
            f"FPS: {metrics.fps:.1f}",
            f"Latency: {metrics.processing_time_ms:.1f}ms",
            f"Hand Coverage: {metrics.hand_coverage * 100:.1f}%",
            f"Face Coverage: {metrics.face_coverage * 100:.1f}%",
        ]
        
        for i, text in enumerate(info_lines):
            cv2.putText(
                frame, text, (20, 35 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
    
    def draw_legend(self, frame: np.ndarray) -> None:
        """
        Draw legend on frame.
        
        Args:
            frame: Frame to draw on (modified in-place)
        """
        h = frame.shape[0]
        legend_y = h - 100
        
        # Legend background
        cv2.rectangle(frame, (10, legend_y), (200, h - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, legend_y), (200, h - 10), (255, 255, 255), 2)
        
        cv2.putText(
            frame, "Legend:", (20, legend_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        # Hand color box
        cv2.rectangle(
            frame, (20, legend_y + 30), (35, legend_y + 45),
            self._palette.hand.to_bgr_tuple(), -1
        )
        cv2.putText(
            frame, "Hand", (45, legend_y + 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        # Face color box
        cv2.rectangle(
            frame, (20, legend_y + 50), (35, legend_y + 65),
            self._palette.face.to_bgr_tuple(), -1
        )
        cv2.putText(
            frame, "Face", (45, legend_y + 62),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    
    def draw_controls(self, frame: np.ndarray) -> None:
        """
        Draw control instructions on frame.
        
        Args:
            frame: Frame to draw on (modified in-place)
        """
        h = frame.shape[0]
        controls = "Press 'q' to quit, 'c' classical, 'u' U-Net, 'm' Mask R-CNN, 'd' depth"
        cv2.putText(
            frame, controls, (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )


class SegmentationProcessor:
    """Processes frames through segmentation models."""
    
    def __init__(self, models: dict[str, Any]):
        """
        Initialize processor.
        
        Args:
            models: Dictionary of loaded models
        """
        self._models = models
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
    
    def segment(
        self,
        frame: np.ndarray,
        method: SegmentationMethod
    ) -> np.ndarray:
        """
        Segment frame using specified method.
        
        Args:
            frame: Input BGR frame
            method: Segmentation method to use
            
        Returns:
            Segmentation mask
            
        Raises:
            ModelInferenceError: If segmentation fails
        """
        try:
            if method == SegmentationMethod.CLASSICAL:
                return self._segment_classical(frame)
            elif method == SegmentationMethod.UNET:
                return self._segment_unet(frame)
            elif method == SegmentationMethod.MASKRCNN:
                return self._segment_maskrcnn(frame)
            else:
                raise ValueError(f"Unknown method: {method}")
        except Exception as e:
            self._logger.warning(f"{method.name} failed: {e}, falling back to classical")
            return self._segment_classical(frame)
    
    def _segment_classical(self, frame: np.ndarray) -> np.ndarray:
        """Segment using classical computer vision."""
        return self._models['classical'].segment_hands_faces(frame)
    
    def _segment_unet(self, frame: np.ndarray) -> np.ndarray:
        """Segment using U-Net model."""
        torch = _get_torch()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0)
        
        with torch.no_grad():
            output = self._models['unet'](frame_tensor)
            mask = torch.argmax(output, dim=1).squeeze().numpy()
        
        return mask.astype(np.uint8)
    
    def _segment_maskrcnn(self, frame: np.ndarray) -> np.ndarray:
        """Segment using Mask R-CNN model."""
        predictions = self._models['maskrcnn'].predict(frame)
        
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if 'masks' in predictions and len(predictions['masks']) > 0:
            masks = predictions['masks'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()
            
            for instance_mask, label, score in zip(masks, labels, scores):
                if score > 0.5:
                    mask[instance_mask[0] > 0.5] = label
        
        return mask
    
    def estimate_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth for frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Depth map or None if estimation fails
        """
        try:
            return self._models['depth'].estimate_depth(frame)
        except Exception as e:
            self._logger.debug(f"Depth estimation failed: {e}")
            return None


class LiveDemo:
    """
    Production-ready live demo for real-time hand/face segmentation.
    
    This class orchestrates camera capture, segmentation, visualization,
    and user interaction for a real-time demo application.
    
    Attributes:
        config: Demo configuration
        
    Example:
        >>> config = DemoConfig(
        ...     method=SegmentationMethod.CLASSICAL,
        ...     show_depth=True,
        ...     camera_id=0
        ... )
        >>> demo = LiveDemo(config)
        >>> demo.run()
    """
    
    WINDOW_MAIN = 'Hand/Face Segmentation Demo'
    WINDOW_DEPTH = 'Depth Map'
    WINDOW_MASK = 'Segmentation Mask'
    
    def __init__(self, config: DemoConfig):
        """
        Initialize live demo.
        
        Args:
            config: Demo configuration
            
        Raises:
            CameraError: If camera cannot be opened
        """
        self._config = config
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._running = False
        
        # Initialize components
        self._camera = CameraManager(config.camera_id)
        self._fps_tracker = FPSTracker(config.fps_window_size)
        self._renderer = OverlayRenderer()
        
        # Load models
        self._logger.info("Loading segmentation models...")
        from core_models import create_models
        models = create_models()
        self._processor = SegmentationProcessor(models)
        
        # Open camera
        self._camera.open(config.frame_width, config.frame_height)
        
        self._logger.info(
            f"Demo initialized with method={config.method}, "
            f"depth={'enabled' if config.show_depth else 'disabled'}"
        )
    
    @property
    def current_method(self) -> SegmentationMethod:
        """Get current segmentation method."""
        return self._config.method
    
    @current_method.setter
    def current_method(self, method: SegmentationMethod) -> None:
        """Set current segmentation method."""
        if method != self._config.method:
            self._config.method = method
            self._logger.info(f"Switched to {method.name} method")
    
    @property
    def show_depth(self) -> bool:
        """Get depth display state."""
        return self._config.show_depth
    
    @show_depth.setter
    def show_depth(self, value: bool) -> None:
        """Set depth display state."""
        if value != self._config.show_depth:
            self._config.show_depth = value
            self._logger.info(f"Depth display: {'ON' if value else 'OFF'}")
            if not value:
                cv2.destroyWindow(self.WINDOW_DEPTH)
    
    def _process_frame(
        self,
        frame: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], FrameMetrics]:
        """
        Process a single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (result_frame, mask, depth_map, metrics)
        """
        start_time = time.perf_counter()
        
        # Mirror frame if configured
        if self._config.mirror_frame:
            frame = cv2.flip(frame, 1)
        
        # Segment frame
        mask = self._processor.segment(frame, self._config.method)
        
        # Get depth if enabled
        depth_map = None
        if self._config.show_depth:
            depth_map = self._processor.estimate_depth(frame)
        
        # Create visualization
        overlay = self._renderer.create_mask_overlay(mask)
        result = cv2.addWeighted(
            frame, 1.0 - self._config.overlay_alpha,
            overlay, self._config.overlay_alpha,
            0
        )
        
        # Calculate metrics
        processing_time = time.perf_counter() - start_time
        fps = self._fps_tracker.update(processing_time)
        metrics = FrameMetrics.compute(mask, processing_time, fps)
        
        # Draw overlays
        self._renderer.draw_info_panel(result, self._config.method, metrics)
        self._renderer.draw_legend(result)
        self._renderer.draw_controls(result)
        
        return result, mask, depth_map, metrics
    
    def _handle_keypress(self, key: int) -> bool:
        """
        Handle keyboard input.
        
        Args:
            key: Key code
            
        Returns:
            True if demo should continue, False to quit
        """
        if key == ord('q'):
            return False
        elif key == ord('c'):
            self.current_method = SegmentationMethod.CLASSICAL
        elif key == ord('u'):
            self.current_method = SegmentationMethod.UNET
        elif key == ord('m'):
            self.current_method = SegmentationMethod.MASKRCNN
        elif key == ord('d'):
            self.show_depth = not self.show_depth
        return True
    
    def _display_results(
        self,
        result: np.ndarray,
        mask: np.ndarray,
        depth_map: Optional[np.ndarray]
    ) -> None:
        """
        Display visualization windows.
        
        Args:
            result: Main result frame
            mask: Segmentation mask
            depth_map: Optional depth map
        """
        cv2.imshow(self.WINDOW_MAIN, result)
        
        # Mask visualization
        mask_vis = (mask * 127).astype(np.uint8)
        cv2.imshow(self.WINDOW_MASK, mask_vis)
        
        # Depth visualization
        if self._config.show_depth and depth_map is not None:
            depth_max = depth_map.max()
            if depth_max > 0:
                depth_normalized = cv2.convertScaleAbs(
                    depth_map, alpha=255 / depth_max
                )
                depth_vis = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                cv2.imshow(self.WINDOW_DEPTH, depth_vis)
    
    def run(self) -> None:
        """
        Run the live demo main loop.
        
        This method captures frames, processes them, and displays results
        until the user quits or an error occurs.
        """
        self._print_controls()
        self._running = True
        
        # Setup signal handler for graceful shutdown
        original_handler = signal.signal(signal.SIGINT, self._signal_handler)
        
        try:
            while self._running:
                # Capture frame
                ret, frame = self._camera.read()
                if not ret or frame is None:
                    self._logger.error("Failed to capture frame")
                    break
                
                # Process and display
                try:
                    result, mask, depth_map, metrics = self._process_frame(frame)
                    self._display_results(result, mask, depth_map)
                except Exception as e:
                    self._logger.error(f"Frame processing error: {e}")
                    continue
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_keypress(key):
                    break
        
        finally:
            signal.signal(signal.SIGINT, original_handler)
            self.cleanup()
    
    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle interrupt signal."""
        self._logger.info("Interrupt received, shutting down...")
        self._running = False
    
    def _print_controls(self) -> None:
        """Print control instructions to console."""
        print("\n" + "=" * 50)
        print("Hand/Face Segmentation Live Demo")
        print("=" * 50)
        print("\nControls:")
        print("  'c' - Switch to Classical method")
        print("  'u' - Switch to U-Net")
        print("  'm' - Switch to Mask R-CNN")
        print("  'd' - Toggle depth display")
        print("  'q' - Quit")
        print("\n" + "=" * 50 + "\n")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self._camera.release()
        cv2.destroyAllWindows()
        self._logger.info("Demo finished, resources released")


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create command-line argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Real-time Hand/Face Segmentation Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --method classical
    %(prog)s --method unet --camera-id 1
    %(prog)s --method maskrcnn --no-depth
        """
    )
    
    parser.add_argument(
        '--method', '-m',
        type=str,
        default='classical',
        choices=['classical', 'unet', 'maskrcnn'],
        help='Segmentation method to use (default: classical)'
    )
    
    parser.add_argument(
        '--no-depth',
        action='store_true',
        help='Disable depth estimation display'
    )
    
    parser.add_argument(
        '--camera-id', '-c',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def main() -> int:
    """
    Main entry point for live demo.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        config = DemoConfig.from_args(args)
        demo = LiveDemo(config)
        demo.run()
        return 0
        
    except CameraError as e:
        logger.error(f"Camera error: {e}")
        print(f"\nError: {e}")
        print("Make sure you have a camera connected and accessible.")
        return 1
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
        return 0
        
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"\nError starting demo: {e}")
        print("Check the logs for more details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
