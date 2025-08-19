"""
Contour detection and tracking module for hand and face segmentation.

This module provides robust contour detection, filtering, and tracking
capabilities specifically designed for hand and face shapes. It includes
shape analysis, gesture recognition features, and temporal smoothing.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import math
import logging

logger = logging.getLogger(__name__)


class ContourTracker:
    """
    Advanced contour tracking system for hands and faces.
    
    This class implements contour detection with shape analysis,
    temporal smoothing, and gesture recognition capabilities.
    """
    
    def __init__(self, min_area: int = 1000, max_area: int = 50000,
                 smoothing_factor: float = 0.3):
        """
        Initialize the contour tracker.
        
        Args:
            min_area: Minimum contour area to consider
            max_area: Maximum contour area to consider
            smoothing_factor: Temporal smoothing factor (0.0 to 1.0)
        """
        self.min_area = min_area
        self.max_area = max_area
        self.smoothing_factor = smoothing_factor
        
        # Tracking state
        self.previous_contours = []
        self.contour_history = {}
        self.next_id = 0
        self.max_history_length = 10
        
        # Shape analysis parameters
        self.aspect_ratio_range = (0.3, 3.0)  # Valid aspect ratios
        self.circularity_threshold = 0.1  # Minimum circularity
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better contour detection.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Preprocessed binary image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding for better edge detection
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def detect_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """
        Detect and filter contours based on area and shape constraints.
        
        Args:
            binary_image: Binary input image
            
        Returns:
            List of valid contours
        """
        # Find contours
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Calculate bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # Filter by aspect ratio
            if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                continue
            
            # Calculate circularity (shape compactness)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * math.pi * area / (perimeter * perimeter)
                if circularity < self.circularity_threshold:
                    continue
            
            valid_contours.append(contour)
        
        return valid_contours
    
    def analyze_contour_shape(self, contour: np.ndarray) -> Dict[str, Any]:
        """
        Analyze geometric properties of a contour.
        
        Args:
            contour: Input contour
            
        Returns:
            Dictionary with shape analysis results
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        
        # Minimum enclosing circle
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        
        # Convex hull and defects
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # Convexity defects (useful for finger detection)
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        if len(hull_indices) > 3:
            defects = cv2.convexityDefects(contour, hull_indices)
            num_defects = len(defects) if defects is not None else 0
        else:
            defects = None
            num_defects = 0
        
        # Moments for centroid calculation
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            centroid_x = int(moments['m10'] / moments['m00'])
            centroid_y = int(moments['m01'] / moments['m00'])
        else:
            centroid_x, centroid_y = int(center_x), int(center_y)
        
        # Circularity
        circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Ellipse fitting (if enough points)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            ellipse_area = math.pi * ellipse[1][0] * ellipse[1][1] / 4
            ellipse_ratio = area / ellipse_area if ellipse_area > 0 else 0
        else:
            ellipse = None
            ellipse_ratio = 0
        
        return {
            'area': area,
            'perimeter': perimeter,
            'centroid': (centroid_x, centroid_y),
            'bounding_rect': (x, y, w, h),
            'aspect_ratio': aspect_ratio,
            'center': (center_x, center_y),
            'radius': radius,
            'solidity': solidity,
            'circularity': circularity,
            'num_defects': num_defects,
            'defects': defects,
            'hull': hull,
            'ellipse': ellipse,
            'ellipse_ratio': ellipse_ratio
        }
    
    def match_contours(self, current_contours: List[np.ndarray]) -> List[Tuple[int, np.ndarray, Dict]]:
        """
        Match current contours with previous ones for tracking.
        
        Args:
            current_contours: List of current frame contours
            
        Returns:
            List of (track_id, contour, properties) tuples
        """
        if not self.previous_contours:
            # First frame - assign new IDs to all contours
            matched_contours = []
            for contour in current_contours:
                properties = self.analyze_contour_shape(contour)
                matched_contours.append((self.next_id, contour, properties))
                self.next_id += 1
            
            self.previous_contours = [(track_id, contour, props) 
                                    for track_id, contour, props in matched_contours]
            return matched_contours
        
        # Match contours based on distance and shape similarity
        matched_contours = []
        used_previous = set()
        
        for curr_contour in current_contours:
            curr_props = self.analyze_contour_shape(curr_contour)
            curr_centroid = curr_props['centroid']
            
            best_match_id = None
            best_distance = float('inf')
            best_prev_idx = -1
            
            # Find best match among previous contours
            for i, (prev_id, prev_contour, prev_props) in enumerate(self.previous_contours):
                if i in used_previous:
                    continue
                
                prev_centroid = prev_props['centroid']
                
                # Calculate distance between centroids
                distance = math.sqrt(
                    (curr_centroid[0] - prev_centroid[0]) ** 2 +
                    (curr_centroid[1] - prev_centroid[1]) ** 2
                )
                
                # Calculate shape similarity
                area_ratio = abs(curr_props['area'] - prev_props['area']) / max(curr_props['area'], prev_props['area'])
                aspect_ratio_diff = abs(curr_props['aspect_ratio'] - prev_props['aspect_ratio'])
                
                # Combined score (lower is better)
                shape_penalty = area_ratio * 100 + aspect_ratio_diff * 50
                total_score = distance + shape_penalty
                
                # Distance threshold for matching
                max_distance = 100  # pixels
                if distance < max_distance and total_score < best_distance:
                    best_distance = total_score
                    best_match_id = prev_id
                    best_prev_idx = i
            
            if best_match_id is not None:
                # Existing contour - apply temporal smoothing
                used_previous.add(best_prev_idx)
                prev_props = self.previous_contours[best_prev_idx][2]
                
                # Smooth centroid position
                smooth_centroid = (
                    int(self.smoothing_factor * curr_centroid[0] + 
                        (1 - self.smoothing_factor) * prev_props['centroid'][0]),
                    int(self.smoothing_factor * curr_centroid[1] + 
                        (1 - self.smoothing_factor) * prev_props['centroid'][1])
                )
                curr_props['centroid'] = smooth_centroid
                
                matched_contours.append((best_match_id, curr_contour, curr_props))
            else:
                # New contour - assign new ID
                matched_contours.append((self.next_id, curr_contour, curr_props))
                self.next_id += 1
        
        # Update previous contours
        self.previous_contours = matched_contours
        
        # Update contour history
        for track_id, contour, props in matched_contours:
            if track_id not in self.contour_history:
                self.contour_history[track_id] = []
            
            self.contour_history[track_id].append({
                'contour': contour,
                'properties': props,
                'timestamp': len(self.contour_history[track_id])
            })
            
            # Limit history length
            if len(self.contour_history[track_id]) > self.max_history_length:
                self.contour_history[track_id].pop(0)
        
        return matched_contours
    
    def detect_fingers(self, contour: np.ndarray, properties: Dict[str, Any]) -> List[Tuple[int, int]]:
        """
        Detect fingertips using convexity defects analysis.
        
        Args:
            contour: Hand contour
            properties: Contour properties from analyze_contour_shape
            
        Returns:
            List of fingertip coordinates
        """
        if properties['defects'] is None or len(properties['defects']) < 1:
            return []
        
        fingertips = []
        defects = properties['defects']
        
        # Filter defects based on depth and angle
        for defect in defects:
            start_idx, end_idx, far_idx, depth = defect[0]
            
            if depth > 1000:  # Minimum depth threshold
                start_point = tuple(contour[start_idx][0])
                end_point = tuple(contour[end_idx][0])
                far_point = tuple(contour[far_idx][0])
                
                # Calculate angle at the defect point
                a = math.sqrt((end_point[0] - far_point[0]) ** 2 + (end_point[1] - far_point[1]) ** 2)
                b = math.sqrt((start_point[0] - far_point[0]) ** 2 + (start_point[1] - far_point[1]) ** 2)
                c = math.sqrt((start_point[0] - end_point[0]) ** 2 + (start_point[1] - end_point[1]) ** 2)
                
                if a > 0 and b > 0:
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                    
                    # Filter based on angle (fingers typically have acute angles)
                    if angle <= math.pi / 2:
                        # Add the start point as a potential fingertip
                        fingertips.append(start_point)
        
        # Remove duplicate fingertips that are too close
        filtered_fingertips = []
        min_distance = 20
        
        for tip in fingertips:
            is_duplicate = False
            for existing_tip in filtered_fingertips:
                distance = math.sqrt((tip[0] - existing_tip[0]) ** 2 + (tip[1] - existing_tip[1]) ** 2)
                if distance < min_distance:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_fingertips.append(tip)
        
        return filtered_fingertips
    
    def classify_gesture(self, properties: Dict[str, Any], fingertips: List[Tuple[int, int]]) -> str:
        """
        Basic gesture classification based on contour properties and fingertips.
        
        Args:
            properties: Contour properties
            fingertips: Detected fingertip positions
            
        Returns:
            Gesture name
        """
        num_fingers = len(fingertips)
        area = properties['area']
        solidity = properties['solidity']
        aspect_ratio = properties['aspect_ratio']
        
        # Simple gesture rules
        if num_fingers == 0:
            if solidity > 0.9:
                return "fist"
            else:
                return "partial_hand"
        elif num_fingers == 1:
            return "pointing"
        elif num_fingers == 2:
            return "peace_sign"
        elif num_fingers == 3:
            return "three_fingers"
        elif num_fingers == 4:
            return "four_fingers"
        elif num_fingers >= 5:
            return "open_hand"
        else:
            return "unknown"
    
    def update(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Update contour tracker with new frame.
        
        Args:
            image: Input image
            
        Returns:
            List of tracking results
        """
        # Preprocess image
        binary = self.preprocess_image(image)
        
        # Detect contours
        contours = self.detect_contours(binary)
        
        # Match and track contours
        tracked_contours = self.match_contours(contours)
        
        # Analyze each tracked contour
        results = []
        for track_id, contour, properties in tracked_contours:
            # Detect fingertips for hand contours
            fingertips = self.detect_fingers(contour, properties)
            
            # Classify gesture
            gesture = self.classify_gesture(properties, fingertips)
            
            result = {
                'track_id': track_id,
                'contour': contour,
                'properties': properties,
                'fingertips': fingertips,
                'gesture': gesture,
                'num_fingers': len(fingertips)
            }
            
            results.append(result)
        
        return results
    
    def draw_results(self, image: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw tracking results on image.
        
        Args:
            image: Input image
            results: Tracking results from update()
            
        Returns:
            Image with drawn results
        """
        output = image.copy()
        
        for result in results:
            contour = result['contour']
            properties = result['properties']
            fingertips = result['fingertips']
            gesture = result['gesture']
            track_id = result['track_id']
            
            # Draw contour
            cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
            
            # Draw bounding rectangle
            x, y, w, h = properties['bounding_rect']
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Draw centroid
            centroid = properties['centroid']
            cv2.circle(output, centroid, 5, (0, 0, 255), -1)
            
            # Draw fingertips
            for fingertip in fingertips:
                cv2.circle(output, fingertip, 8, (255, 0, 255), -1)
            
            # Draw gesture label
            label = f"ID:{track_id} {gesture}"
            cv2.putText(output, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return output
    
    def reset(self):
        """Reset tracker state."""
        self.previous_contours = []
        self.contour_history = {}
        self.next_id = 0


if __name__ == "__main__":
    # Example usage
    tracker = ContourTracker(min_area=5000, max_area=50000)
    
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update tracker
            results = tracker.update(frame)
            
            # Draw results
            output = tracker.draw_results(frame, results)
            
            cv2.imshow('Contour Tracking', output)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("No webcam available for testing")
