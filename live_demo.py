"""
Real-time live demo for hand/face segmentation and tracking.
Displays segmentation masks, FPS, and 3D depth visualization.
"""

import cv2
import numpy as np
import torch
import time
import argparse
from core_models import create_models
import threading
from collections import deque


class LiveDemo:
    """
    Live demo application for real-time hand/face segmentation.
    """
    
    def __init__(self, method='classical', show_depth=True):
        """
        Initialize demo.
        
        Args:
            method: Segmentation method ('classical', 'unet', 'maskrcnn')
            show_depth: Whether to show depth estimation
        """
        self.method = method
        self.show_depth = show_depth
        
        # Performance tracking
        self.fps_queue = deque(maxlen=30)
        self.current_fps = 0
        
        # Load models
        print("Loading models...")
        self.models = create_models()
        print(f"Using {method} method")
        
        # Colors for visualization
        self.colors = {
            0: (0, 0, 0),      # Background - black
            1: (0, 255, 0),    # Hand - green
            2: (255, 0, 0),    # Face - blue
        }
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            raise ValueError("Could not open camera")
    
    def create_overlay_mask(self, mask, alpha=0.6):
        """
        Create colored overlay from segmentation mask.
        
        Args:
            mask: Segmentation mask (H, W)
            alpha: Transparency factor
            
        Returns:
            Colored overlay image
        """
        h, w = mask.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in self.colors.items():
            overlay[mask == class_id] = color
        
        return overlay
    
    def segment_frame(self, frame):
        """
        Segment frame using selected method.
        
        Args:
            frame: Input frame
            
        Returns:
            Segmentation mask
        """
        if self.method == 'classical':
            return self.models['classical'].segment_hands_faces(frame)
        
        elif self.method == 'unet':
            # Preprocess for U-Net
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor.unsqueeze(0)
            
            with torch.no_grad():
                output = self.models['unet'](frame_tensor)
                mask = torch.argmax(output, dim=1).squeeze().numpy()
            
            return mask.astype(np.uint8)
        
        elif self.method == 'maskrcnn':
            predictions = self.models['maskrcnn'].predict(frame)
            
            # Convert instance masks to semantic mask
            h, w = frame.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            if 'masks' in predictions and len(predictions['masks']) > 0:
                masks = predictions['masks'].cpu().numpy()
                labels = predictions['labels'].cpu().numpy()
                scores = predictions['scores'].cpu().numpy()
                
                for i, (instance_mask, label, score) in enumerate(zip(masks, labels, scores)):
                    if score > 0.5:  # Confidence threshold
                        mask[instance_mask[0] > 0.5] = label
            
            return mask
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def calculate_metrics(self, mask):
        """
        Calculate real-time metrics.
        
        Args:
            mask: Segmentation mask
            
        Returns:
            Dictionary with metrics
        """
        total_pixels = mask.size
        hand_pixels = np.sum(mask == 1)
        face_pixels = np.sum(mask == 2)
        
        return {
            'hand_coverage': hand_pixels / total_pixels,
            'face_coverage': face_pixels / total_pixels,
            'total_detection': (hand_pixels + face_pixels) / total_pixels
        }
    
    def draw_info(self, frame, mask, depth_map=None, processing_time=0):
        """
        Draw information overlay on frame.
        
        Args:
            frame: Input frame
            mask: Segmentation mask
            depth_map: Optional depth map
            processing_time: Processing time in seconds
        """
        h, w = frame.shape[:2]
        
        # Calculate metrics
        metrics = self.calculate_metrics(mask)
        
        # Draw text information
        info_text = [
            f"Method: {self.method.upper()}",
            f"FPS: {self.current_fps:.1f}",
            f"Latency: {processing_time*1000:.1f}ms",
            f"Hand Coverage: {metrics['hand_coverage']*100:.1f}%",
            f"Face Coverage: {metrics['face_coverage']*100:.1f}%"
        ]
        
        # Draw semi-transparent background for text
        cv2.rectangle(frame, (10, 10), (300, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 150), (255, 255, 255), 2)
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (20, 35 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw legend
        legend_y = h - 100
        cv2.rectangle(frame, (10, legend_y), (200, h - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, legend_y), (200, h - 10), (255, 255, 255), 2)
        
        cv2.putText(frame, "Legend:", (20, legend_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (20, legend_y + 30), (35, legend_y + 45), (0, 255, 0), -1)
        cv2.putText(frame, "Hand", (45, legend_y + 42), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (20, legend_y + 50), (35, legend_y + 65), (255, 0, 0), -1)
        cv2.putText(frame, "Face", (45, legend_y + 62), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw controls
        controls_text = "Press 'q' to quit, 'c' for classical, 'u' for U-Net, 'm' for Mask R-CNN"
        cv2.putText(frame, controls_text, (10, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run(self):
        """
        Run the live demo.
        """
        print("Starting live demo... Press 'q' to quit")
        print("Controls:")
        print("  'c' - Switch to classical method")
        print("  'u' - Switch to U-Net")
        print("  'm' - Switch to Mask R-CNN")
        print("  'd' - Toggle depth display")
        print("  'q' - Quit")
        
        try:
            while True:
                start_time = time.time()
                
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Segment frame
                try:
                    mask = self.segment_frame(frame)
                except Exception as e:
                    print(f"Segmentation failed: {e}")
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                
                # Get depth map if enabled
                depth_map = None
                if self.show_depth:
                    try:
                        depth_map = self.models['depth'].estimate_depth(frame)
                    except Exception as e:
                        print(f"Depth estimation failed: {e}")
                
                # Create visualization
                overlay = self.create_overlay_mask(mask, alpha=0.4)
                result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                
                # Calculate processing time and FPS
                processing_time = time.time() - start_time
                self.fps_queue.append(1.0 / max(processing_time, 1e-6))
                self.current_fps = np.mean(self.fps_queue)
                
                # Draw information
                self.draw_info(result, mask, depth_map, processing_time)
                
                # Show main result
                cv2.imshow('Hand/Face Segmentation Demo', result)
                
                # Show depth map if enabled
                if self.show_depth and depth_map is not None:
                    depth_vis = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_map, alpha=255/depth_map.max()), 
                        cv2.COLORMAP_JET
                    )
                    cv2.imshow('Depth Map', depth_vis)
                
                # Show segmentation mask
                mask_vis = (mask * 127).astype(np.uint8)  # Scale for visibility
                cv2.imshow('Segmentation Mask', mask_vis)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.method = 'classical'
                    print("Switched to classical method")
                elif key == ord('u'):
                    self.method = 'unet'
                    print("Switched to U-Net")
                elif key == ord('m'):
                    self.method = 'maskrcnn'
                    print("Switched to Mask R-CNN")
                elif key == ord('d'):
                    self.show_depth = not self.show_depth
                    print(f"Depth display: {'ON' if self.show_depth else 'OFF'}")
                    if not self.show_depth:
                        cv2.destroyWindow('Depth Map')
        
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        print("Demo finished")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Real-time Hand/Face Segmentation Demo')
    
    parser.add_argument('--method', type=str, default='classical',
                       choices=['classical', 'unet', 'maskrcnn'],
                       help='Segmentation method to use')
    parser.add_argument('--no-depth', action='store_true',
                       help='Disable depth estimation')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='Camera device ID')
    
    args = parser.parse_args()
    
    try:
        demo = LiveDemo(
            method=args.method,
            show_depth=not args.no_depth
        )
        demo.run()
        
    except Exception as e:
        print(f"Error starting demo: {e}")
        print("Make sure you have a camera connected and the required dependencies installed.")


if __name__ == "__main__":
    main()
