"""
Skin detection module using classical computer vision techniques.

This module implements various skin detection algorithms including
HSV-based filtering, YCrCb color space analysis, and statistical
skin models for real-time hand and face detection.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class SkinDetector:
    """
    A comprehensive skin detection class that implements multiple
    algorithms for robust skin segmentation in various lighting conditions.
    """
    
    def __init__(self, method: str = 'hsv'):
        """
        Initialize the skin detector with specified method.
        
        Args:
            method: Detection method ('hsv', 'ycrcb', 'rgb', 'combined')
        """
        self.method = method
        self.hsv_lower = np.array([0, 20, 70], dtype=np.uint8)
        self.hsv_upper = np.array([20, 255, 255], dtype=np.uint8)
        
        # YCrCb thresholds work well for various skin tones
        self.ycrcb_lower = np.array([0, 133, 77], dtype=np.uint8)
        self.ycrcb_upper = np.array([255, 173, 127], dtype=np.uint8)
        
        # RGB thresholds for basic skin detection
        self.rgb_lower = np.array([45, 34, 30], dtype=np.uint8)
        self.rgb_upper = np.array([255, 255, 255], dtype=np.uint8)
        
    def detect_skin_hsv(self, image: np.ndarray) -> np.ndarray:
        """
        Detect skin using HSV color space thresholding.
        
        HSV is particularly good for handling lighting variations
        since it separates chrominance from luminance.
        
        Args:
            image: Input BGR image
            
        Returns:
            Binary mask where skin pixels are white (255)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def detect_skin_ycrcb(self, image: np.ndarray) -> np.ndarray:
        """
        Detect skin using YCrCb color space.
        
        YCrCb is often more robust for skin detection as the Cr and Cb
        components provide good separation of skin from non-skin pixels.
        
        Args:
            image: Input BGR image
            
        Returns:
            Binary mask where skin pixels are white (255)
        """
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, self.ycrcb_lower, self.ycrcb_upper)
        
        # Clean up noise with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def detect_skin_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Basic RGB-based skin detection.
        
        This method uses simple RGB thresholds and additional rules
        based on the relationship between R, G, B values.
        
        Args:
            image: Input BGR image
            
        Returns:
            Binary mask where skin pixels are white (255)
        """
        # Convert BGR to RGB for processing
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r, g, b = cv2.split(rgb)
        
        # Apply RGB range constraints
        mask1 = cv2.inRange(rgb, self.rgb_lower, self.rgb_upper)
        
        # Additional constraints for better skin detection
        # R > G and R > B (skin typically has higher red component)
        mask2 = (r > g) & (r > b)
        
        # Combine masks
        mask = cv2.bitwise_and(mask1, mask2.astype(np.uint8) * 255)
        
        return mask
    
    def detect_skin_combined(self, image: np.ndarray) -> np.ndarray:
        """
        Combined skin detection using multiple color spaces.
        
        This method combines results from HSV and YCrCb detection
        for more robust skin segmentation.
        
        Args:
            image: Input BGR image
            
        Returns:
            Binary mask where skin pixels are white (255)
        """
        hsv_mask = self.detect_skin_hsv(image)
        ycrcb_mask = self.detect_skin_ycrcb(image)
        
        # Combine masks using bitwise OR for broader coverage
        combined_mask = cv2.bitwise_or(hsv_mask, ycrcb_mask)
        
        # Apply additional filtering to reduce false positives
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask
    
    def detect_skin(self, image: np.ndarray) -> np.ndarray:
        """
        Main skin detection method that routes to the specified algorithm.
        
        Args:
            image: Input BGR image
            
        Returns:
            Binary mask where skin pixels are white (255)
        """
        if self.method == 'hsv':
            return self.detect_skin_hsv(image)
        elif self.method == 'ycrcb':
            return self.detect_skin_ycrcb(image)
        elif self.method == 'rgb':
            return self.detect_skin_rgb(image)
        elif self.method == 'combined':
            return self.detect_skin_combined(image)
        else:
            logger.warning(f"Unknown method {self.method}, using HSV")
            return self.detect_skin_hsv(image)
    
    def post_process_mask(self, mask: np.ndarray, 
                         min_area: int = 500) -> np.ndarray:
        """
        Post-process the skin mask to remove small noise regions.
        
        Args:
            mask: Binary skin mask
            min_area: Minimum area for connected components to keep
            
        Returns:
            Cleaned binary mask
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        
        # Create output mask
        output_mask = np.zeros_like(mask)
        
        # Keep components larger than minimum area
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                output_mask[labels == i] = 255
                
        return output_mask
    
    def get_skin_regions(self, image: np.ndarray, 
                        min_area: int = 500) -> Tuple[np.ndarray, list]:
        """
        Detect skin regions and return both mask and bounding boxes.
        
        Args:
            image: Input BGR image
            min_area: Minimum area for regions to consider
            
        Returns:
            Tuple of (skin_mask, list of bounding boxes)
        """
        mask = self.detect_skin(image)
        mask = self.post_process_mask(mask, min_area)
        
        # Find contours for bounding boxes
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bounding_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append((x, y, w, h))
        
        return mask, bounding_boxes


def adaptive_skin_detection(image: np.ndarray, 
                          adaptation_rate: float = 0.1) -> np.ndarray:
    """
    Adaptive skin detection that adjusts thresholds based on image statistics.
    
    This function analyzes the image histogram and lighting conditions
    to dynamically adjust skin detection parameters.
    
    Args:
        image: Input BGR image
        adaptation_rate: Rate at which to adapt thresholds (0.0 to 1.0)
        
    Returns:
        Binary skin mask
    """
    # Convert to different color spaces for analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Analyze lighting conditions using L channel statistics
    l_channel = lab[:, :, 0]
    mean_brightness = np.mean(l_channel)
    std_brightness = np.std(l_channel)
    
    # Create adaptive skin detector
    detector = SkinDetector('combined')
    
    # Adjust HSV thresholds based on brightness
    if mean_brightness < 100:  # Dark image
        detector.hsv_lower[2] = max(30, detector.hsv_lower[2] - 20)
        detector.hsv_upper[2] = min(255, detector.hsv_upper[2] + 20)
    elif mean_brightness > 180:  # Bright image
        detector.hsv_lower[1] = max(10, detector.hsv_lower[1] - 10)
        detector.hsv_upper[1] = min(255, detector.hsv_upper[1] + 10)
    
    return detector.detect_skin(image)


if __name__ == "__main__":
    # Example usage and testing
    detector = SkinDetector('combined')
    
    # Test with webcam if available
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            mask = detector.detect_skin(frame)
            
            # Apply mask to original image
            result = cv2.bitwise_and(frame, frame, mask=mask)
            
            # Display results
            combined = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), result])
            cv2.imshow('Skin Detection: Original | Mask | Result', combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("No webcam available for testing")
