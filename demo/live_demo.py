"""
Live demo application for real-time hand and face segmentation.

This script provides a complete demonstration of the hand and face segmentation
system with real-time webcam input, showing classical and deep learning approaches
along with 3D depth estimation.
"""

import cv2
import numpy as np
import torch
import argparse
import time
import logging
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.classical import SkinDetector, OpticalFlowTracker, ContourTracker
from src.deep_learning import create_unet_model, create_mask_rcnn
from src.depth_estimation import RealTimeDepthEstimator
from src.utils import PerformanceMonitor, setup_logging, visualize_segmentation
from config.config import MODEL_CONFIGS, REALTIME_CONFIG, VISUALIZATION_CONFIG

logger = setup_logging()


class LiveDemo:
    """
    Live demonstration system for hand and face segmentation.
    
    Combines multiple approaches and provides real-time visualization
    of segmentation, tracking, and depth estimation results.
    """
    
    def __init__(self, use_classical: bool = True, use_deep_learning: bool = True,
                 use_depth_estimation: bool = True, device: str = 'auto'):
        """
        Initialize live demo system.
        
        Args:
            use_classical: Enable classical computer vision methods
            use_deep_learning: Enable deep learning models
            use_depth_estimation: Enable depth estimation
            device: Device for inference ('cpu', 'cuda', 'auto')
        """
        self.use_classical = use_classical
        self.use_deep_learning = use_deep_learning
        self.use_depth_estimation = use_depth_estimation
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize classical methods
        if self.use_classical:
            self.skin_detector = SkinDetector('combined')
            self.optical_flow_tracker = OpticalFlowTracker()
            self.contour_tracker = ContourTracker()
            logger.info("Classical methods initialized")
        
        # Initialize deep learning models
        if self.use_deep_learning:
            self._load_deep_learning_models()
            logger.info("Deep learning models initialized")
        
        # Initialize depth estimation
        if self.use_depth_estimation:
            self.depth_estimator = RealTimeDepthEstimator(
                model_type='MiDaS_small',
                target_fps=REALTIME_CONFIG['target_fps']
            )
            logger.info("Depth estimation initialized")
        
        # Camera setup
        self.camera_config = REALTIME_CONFIG['camera']
        
        # Visualization settings
        self.colors = VISUALIZATION_CONFIG['colors']
        self.overlay_alpha = VISUALIZATION_CONFIG['overlay']['mask_alpha']
        
        # Demo state
        self.current_method = 'classical'  # 'classical', 'deep_learning', 'depth'
        self.show_fps = True
        self.save_output = False
        self.output_writer = None
    
    def _load_deep_learning_models(self):
        """Load pre-trained deep learning models."""
        try:
            # Load U-Net model
            self.unet_model = create_unet_model(
                'standard',
                in_channels=3,
                out_channels=3,  # background, hand, face
                **MODEL_CONFIGS['unet']
            )
            self.unet_model = self.unet_model.to(self.device)
            self.unet_model.eval()
            
            # Load Mask R-CNN model
            self.maskrcnn_model = create_mask_rcnn(
                num_classes=3,  # background, hand, face
                **MODEL_CONFIGS['mask_rcnn']
            )
            self.maskrcnn_model = self.maskrcnn_model.to(self.device)
            self.maskrcnn_model.eval()
            
        except Exception as e:
            logger.warning(f"Could not load deep learning models: {e}")
            logger.warning("Running without deep learning models")
            self.use_deep_learning = False
    
    def process_frame_classical(self, frame: np.ndarray) -> dict:
        """
        Process frame using classical computer vision methods.
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary with processing results
        """
        results = {}
        
        # Skin detection
        skin_mask = self.skin_detector.detect_skin(frame)
        skin_mask_cleaned = self.skin_detector.post_process_mask(skin_mask)
        
        # Optical flow tracking
        flow_results = self.optical_flow_tracker.update(frame)
        
        # Contour tracking
        contour_results = self.contour_tracker.update(frame)
        
        # Create visualization
        visualization = frame.copy()
        
        # Overlay skin mask
        skin_colored = cv2.applyColorMap(skin_mask_cleaned, cv2.COLORMAP_JET)
        visualization = cv2.addWeighted(visualization, 0.7, skin_colored, 0.3, 0)
        
        # Draw tracked points
        if len(flow_results['points']) > 0:
            for point in flow_results['points']:
                x, y = point.ravel().astype(int)
                cv2.circle(visualization, (x, y), 3, self.colors['hand'], -1)
        
        # Draw contours and gestures
        for result in contour_results:
            contour = result['contour']
            gesture = result['gesture']
            properties = result['properties']
            
            # Draw contour
            cv2.drawContours(visualization, [contour], -1, self.colors['face'], 2)
            
            # Draw gesture label
            centroid = properties['centroid']
            cv2.putText(visualization, gesture, centroid,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        results.update({
            'skin_mask': skin_mask_cleaned,
            'tracked_points': flow_results['points'],
            'contours': contour_results,
            'visualization': visualization
        })
        
        return results
    
    def process_frame_deep_learning(self, frame: np.ndarray) -> dict:
        """
        Process frame using deep learning models.
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary with processing results
        """
        results = {}
        
        if not self.use_deep_learning:
            return {'visualization': frame.copy()}
        
        # Preprocess frame for neural networks
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # U-Net segmentation
            try:
                unet_output = self.unet_model(frame_tensor)
                unet_pred = torch.softmax(unet_output, dim=1)
                unet_mask = torch.argmax(unet_pred, dim=1).squeeze().cpu().numpy()
                
                # Create colored segmentation
                segmentation_colored = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
                segmentation_colored[unet_mask == 1] = self.colors['hand']  # Hand
                segmentation_colored[unet_mask == 2] = self.colors['face']  # Face
                
                # Create visualization
                visualization = cv2.addWeighted(
                    frame, 1 - self.overlay_alpha, 
                    segmentation_colored, self.overlay_alpha, 0)
                
                results.update({
                    'unet_mask': unet_mask,
                    'segmentation_colored': segmentation_colored,
                    'visualization': visualization
                })
                
            except Exception as e:
                logger.error(f"U-Net inference error: {e}")
                results['visualization'] = frame.copy()
        
        return results
    
    def process_frame_depth(self, frame: np.ndarray) -> dict:
        """
        Process frame using depth estimation.
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary with processing results
        """
        results = {}
        
        if not self.use_depth_estimation:
            return {'visualization': frame.copy()}
        
        # Estimate depth
        depth_results = self.depth_estimator.process_frame(frame)
        
        # Create side-by-side visualization
        depth_colored = depth_results['depth_colored']
        
        # Resize to match frame height
        h, w = frame.shape[:2]
        depth_resized = cv2.resize(depth_colored, (w, h))
        
        # Combine original and depth
        visualization = np.hstack([frame, depth_resized])
        
        results.update({
            'depth_raw': depth_results['depth_raw'],
            'depth_colored': depth_colored,
            'depth_fps': depth_results['fps'],
            'visualization': visualization
        })
        
        return results
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process frame using the currently selected method.
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary with processing results
        """
        self.performance_monitor.start_frame()
        
        if self.current_method == 'classical':
            results = self.process_frame_classical(frame)
        elif self.current_method == 'deep_learning':
            results = self.process_frame_deep_learning(frame)
        elif self.current_method == 'depth':
            results = self.process_frame_depth(frame)
        else:
            results = {'visualization': frame.copy()}
        
        self.performance_monitor.end_frame()
        
        return results
    
    def add_overlay_info(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """
        Add overlay information to the frame.
        
        Args:
            frame: Input frame
            results: Processing results
            
        Returns:
            Frame with overlay information
        """
        overlay_frame = frame.copy()
        
        # Add FPS counter
        if self.show_fps:
            fps = self.performance_monitor.get_fps()
            cv2.putText(overlay_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add current method
        method_text = f"Method: {self.current_method.replace('_', ' ').title()}"
        cv2.putText(overlay_frame, method_text, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add controls info
        controls_text = "Controls: 1-Classical, 2-Deep Learning, 3-Depth, Q-Quit, S-Save"
        cv2.putText(overlay_frame, controls_text, (10, overlay_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay_frame
    
    def handle_key_input(self, key: int) -> bool:
        """
        Handle keyboard input for demo control.
        
        Args:
            key: Key code
            
        Returns:
            False if should exit, True otherwise
        """
        if key & 0xFF == ord('q'):
            return False
        elif key & 0xFF == ord('1') and self.use_classical:
            self.current_method = 'classical'
            logger.info("Switched to classical methods")
        elif key & 0xFF == ord('2') and self.use_deep_learning:
            self.current_method = 'deep_learning'
            logger.info("Switched to deep learning")
        elif key & 0xFF == ord('3') and self.use_depth_estimation:
            self.current_method = 'depth'
            logger.info("Switched to depth estimation")
        elif key & 0xFF == ord('s'):
            self.save_output = not self.save_output
            logger.info(f"Output saving: {'enabled' if self.save_output else 'disabled'}")
        elif key & 0xFF == ord('f'):
            self.show_fps = not self.show_fps
        
        return True
    
    def run(self, camera_id: int = 0):
        """
        Run the live demo.
        
        Args:
            camera_id: Camera device ID
        """
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Could not open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_config['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_config['height'])
        cap.set(cv2.CAP_PROP_FPS, self.camera_config['fps'])
        
        logger.info("Starting live demo. Press 'q' to quit.")
        logger.info("Controls:")
        logger.info("  1 - Classical methods")
        logger.info("  2 - Deep learning")
        logger.info("  3 - Depth estimation")
        logger.info("  s - Toggle output saving")
        logger.info("  f - Toggle FPS display")
        logger.info("  q - Quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read from camera")
                    break
                
                # Process frame
                results = self.process_frame(frame)
                
                # Get visualization
                visualization = results.get('visualization', frame)
                
                # Add overlay information
                final_frame = self.add_overlay_info(visualization, results)
                
                # Save output if enabled
                if self.save_output and self.output_writer is not None:
                    self.output_writer.write(final_frame)
                
                # Display frame
                cv2.imshow('Hand/Face Segmentation Demo', final_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1)
                if not self.handle_key_input(key):
                    break
        
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if self.output_writer:
                self.output_writer.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            stats = self.performance_monitor.get_statistics()
            logger.info("Demo Statistics:")
            logger.info(f"Average FPS: {stats['fps']:.2f}")
            logger.info(f"Average Frame Time: {stats['avg_frame_time']:.3f}s")


def main():
    """Main entry point for the demo application."""
    parser = argparse.ArgumentParser(description='Live Hand/Face Segmentation Demo')
    
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--no-classical', action='store_true',
                       help='Disable classical methods')
    parser.add_argument('--no-deep-learning', action='store_true',
                       help='Disable deep learning models')
    parser.add_argument('--no-depth', action='store_true',
                       help='Disable depth estimation')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device for inference (default: auto)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Create and run demo
    demo = LiveDemo(
        use_classical=not args.no_classical,
        use_deep_learning=not args.no_deep_learning,
        use_depth_estimation=not args.no_depth,
        device=args.device
    )
    
    demo.run(camera_id=args.camera)


if __name__ == "__main__":
    main()
