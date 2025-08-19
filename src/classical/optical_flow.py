"""
Optical flow tracking module for hand and face tracking.

This module implements Lucas-Kanade optical flow and other tracking
algorithms to maintain temporal consistency in video sequences.
The tracker can handle multiple objects and provides robust tracking
even with partial occlusions.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class OpticalFlowTracker:
    """
    A robust optical flow tracker that can track multiple points
    and regions across video frames using Lucas-Kanade method.
    """
    
    def __init__(self, max_points: int = 100, quality_level: float = 0.01,
                 min_distance: float = 10, detection_interval: int = 10):
        """
        Initialize the optical flow tracker.
        
        Args:
            max_points: Maximum number of points to track
            quality_level: Quality threshold for corner detection
            min_distance: Minimum distance between tracked points
            detection_interval: Frames between feature re-detection
        """
        self.max_points = max_points
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.detection_interval = detection_interval
        
        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Good features to track parameters
        self.feature_params = dict(
            maxCorners=max_points,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=7
        )
        
        # Tracking state
        self.previous_frame = None
        self.current_points = None
        self.track_ids = []
        self.frame_count = 0
        self.next_id = 0
        
        # Track history for smoothing
        self.point_history = {}
        self.max_history_length = 10
        
    def detect_features(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Detect good features to track in the current frame.
        
        Args:
            frame: Grayscale input frame
            mask: Optional mask to restrict feature detection
            
        Returns:
            Array of detected corner points
        """
        corners = cv2.goodFeaturesToTrack(
            frame, mask=mask, **self.feature_params)
        
        if corners is not None:
            return corners.reshape(-1, 1, 2).astype(np.float32)
        else:
            return np.array([]).reshape(-1, 1, 2).astype(np.float32)
    
    def track_points(self, current_frame: np.ndarray) -> Tuple[np.ndarray, List[int], np.ndarray]:
        """
        Track points from previous frame to current frame.
        
        Args:
            current_frame: Current grayscale frame
            
        Returns:
            Tuple of (tracked_points, valid_track_ids, tracking_status)
        """
        if self.previous_frame is None or self.current_points is None:
            return np.array([]).reshape(-1, 1, 2), [], np.array([])
        
        # Calculate optical flow
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.previous_frame, current_frame, self.current_points, None, **self.lk_params)
        
        # Select good points based on status and error
        good_new = next_points[status == 1]
        good_old = self.current_points[status == 1]
        good_ids = [self.track_ids[i] for i in range(len(status)) if status[i] == 1]
        
        # Additional filtering based on tracking error
        if len(error) > 0:
            error_threshold = 30.0
            valid_indices = error[status == 1].flatten() < error_threshold
            good_new = good_new[valid_indices]
            good_old = good_old[valid_indices]
            good_ids = [good_ids[i] for i in range(len(valid_indices)) if valid_indices[i]]
        
        # Update point history for smoothing
        for i, track_id in enumerate(good_ids):
            if track_id not in self.point_history:
                self.point_history[track_id] = []
            
            self.point_history[track_id].append(good_new[i])
            
            # Keep only recent history
            if len(self.point_history[track_id]) > self.max_history_length:
                self.point_history[track_id].pop(0)
        
        return good_new.reshape(-1, 1, 2), good_ids, status
    
    def smooth_trajectories(self, points: np.ndarray, track_ids: List[int]) -> np.ndarray:
        """
        Apply temporal smoothing to point trajectories.
        
        Args:
            points: Current tracked points
            track_ids: Corresponding track IDs
            
        Returns:
            Smoothed point positions
        """
        smoothed_points = points.copy()
        
        for i, track_id in enumerate(track_ids):
            if track_id in self.point_history and len(self.point_history[track_id]) > 3:
                # Apply simple moving average smoothing
                history = np.array(self.point_history[track_id])
                smoothed_position = np.mean(history[-3:], axis=0)
                smoothed_points[i] = smoothed_position
        
        return smoothed_points
    
    def update(self, frame: np.ndarray, detection_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Update tracker with new frame.
        
        Args:
            frame: Input frame (BGR or grayscale)
            detection_mask: Optional mask for feature detection
            
        Returns:
            Dictionary with tracking results
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame.copy()
        
        self.frame_count += 1
        
        # Track existing points
        tracked_points, valid_ids, status = self.track_points(gray_frame)
        
        # Update current state
        self.current_points = tracked_points
        self.track_ids = valid_ids
        
        # Detect new features periodically or when few points remain
        need_detection = (
            self.frame_count % self.detection_interval == 0 or
            len(self.current_points) < self.max_points // 2
        )
        
        if need_detection:
            # Create mask to avoid detecting near existing points
            detection_mask_combined = detection_mask
            if len(self.current_points) > 0:
                temp_mask = np.ones_like(gray_frame)
                for point in self.current_points:
                    x, y = point.ravel().astype(int)
                    cv2.circle(temp_mask, (x, y), int(self.min_distance), 0, -1)
                
                if detection_mask is not None:
                    detection_mask_combined = cv2.bitwise_and(detection_mask, temp_mask)
                else:
                    detection_mask_combined = temp_mask
            
            # Detect new features
            new_points = self.detect_features(gray_frame, detection_mask_combined)
            
            # Add new points to tracking
            if len(new_points) > 0:
                new_ids = list(range(self.next_id, self.next_id + len(new_points)))
                self.next_id += len(new_points)
                
                if len(self.current_points) > 0:
                    self.current_points = np.vstack([self.current_points, new_points])
                else:
                    self.current_points = new_points
                
                self.track_ids.extend(new_ids)
        
        # Apply smoothing
        if len(self.current_points) > 0:
            self.current_points = self.smooth_trajectories(self.current_points, self.track_ids)
        
        # Update previous frame
        self.previous_frame = gray_frame.copy()
        
        # Prepare output
        result = {
            'points': self.current_points,
            'track_ids': self.track_ids,
            'num_tracks': len(self.track_ids),
            'frame_count': self.frame_count
        }
        
        return result
    
    def get_track_trajectories(self, min_length: int = 5) -> Dict[int, np.ndarray]:
        """
        Get trajectories for tracks with sufficient history.
        
        Args:
            min_length: Minimum trajectory length to return
            
        Returns:
            Dictionary mapping track IDs to trajectory arrays
        """
        trajectories = {}
        
        for track_id, history in self.point_history.items():
            if len(history) >= min_length:
                trajectories[track_id] = np.array(history)
        
        return trajectories
    
    def reset(self):
        """Reset tracker state."""
        self.previous_frame = None
        self.current_points = None
        self.track_ids = []
        self.frame_count = 0
        self.next_id = 0
        self.point_history = {}


class RegionTracker:
    """
    Track regions (bounding boxes) across frames using various methods.
    """
    
    def __init__(self, tracker_type: str = 'kcf'):
        """
        Initialize region tracker.
        
        Args:
            tracker_type: Type of tracker ('kcf', 'csrt', 'medianflow')
        """
        self.tracker_type = tracker_type
        self.trackers = {}
        self.next_id = 0
        
    def create_tracker(self) -> cv2.Tracker:
        """Create a new tracker instance."""
        if self.tracker_type.lower() == 'kcf':
            return cv2.TrackerKCF_create()
        elif self.tracker_type.lower() == 'csrt':
            return cv2.TrackerCSRT_create()
        elif self.tracker_type.lower() == 'medianflow':
            return cv2.TrackerMedianFlow_create()
        else:
            logger.warning(f"Unknown tracker type {self.tracker_type}, using KCF")
            return cv2.TrackerKCF_create()
    
    def add_region(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> int:
        """
        Add a new region to track.
        
        Args:
            frame: Input frame
            bbox: Bounding box (x, y, width, height)
            
        Returns:
            Tracker ID
        """
        tracker = self.create_tracker()
        success = tracker.init(frame, bbox)
        
        if success:
            track_id = self.next_id
            self.trackers[track_id] = tracker
            self.next_id += 1
            return track_id
        else:
            logger.error("Failed to initialize tracker")
            return -1
    
    def update_all(self, frame: np.ndarray) -> Dict[int, Tuple[bool, Tuple[int, int, int, int]]]:
        """
        Update all active trackers.
        
        Args:
            frame: Current frame
            
        Returns:
            Dictionary mapping track IDs to (success, bbox) tuples
        """
        results = {}
        failed_trackers = []
        
        for track_id, tracker in self.trackers.items():
            success, bbox = tracker.update(frame)
            results[track_id] = (success, bbox)
            
            if not success:
                failed_trackers.append(track_id)
        
        # Remove failed trackers
        for track_id in failed_trackers:
            del self.trackers[track_id]
            
        return results
    
    def remove_tracker(self, track_id: int):
        """Remove a specific tracker."""
        if track_id in self.trackers:
            del self.trackers[track_id]
    
    def reset(self):
        """Reset all trackers."""
        self.trackers = {}
        self.next_id = 0


def estimate_motion_vectors(prev_frame: np.ndarray, curr_frame: np.ndarray,
                          grid_size: int = 16) -> np.ndarray:
    """
    Estimate dense motion vectors using block matching.
    
    Args:
        prev_frame: Previous grayscale frame
        curr_frame: Current grayscale frame
        grid_size: Size of blocks for motion estimation
        
    Returns:
        Motion vector field (height/grid_size, width/grid_size, 2)
    """
    h, w = prev_frame.shape
    motion_vectors = np.zeros((h // grid_size, w // grid_size, 2))
    
    for i in range(0, h - grid_size, grid_size):
        for j in range(0, w - grid_size, grid_size):
            # Extract block from previous frame
            block = prev_frame[i:i+grid_size, j:j+grid_size]
            
            # Search window in current frame
            search_size = grid_size + 8
            start_i = max(0, i - 4)
            end_i = min(h - grid_size, i + search_size - grid_size)
            start_j = max(0, j - 4)
            end_j = min(w - grid_size, j + search_size - grid_size)
            
            search_region = curr_frame[start_i:end_i, start_j:end_j]
            
            # Template matching
            result = cv2.matchTemplate(search_region, block, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)
            
            # Calculate motion vector
            mv_x = max_loc[0] + start_j - j
            mv_y = max_loc[1] + start_i - i
            
            motion_vectors[i // grid_size, j // grid_size] = [mv_x, mv_y]
    
    return motion_vectors


if __name__ == "__main__":
    # Example usage
    tracker = OpticalFlowTracker()
    region_tracker = RegionTracker('kcf')
    
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update optical flow tracker
            result = tracker.update(frame)
            
            # Draw tracked points
            if len(result['points']) > 0:
                for point in result['points']:
                    x, y = point.ravel().astype(int)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            cv2.imshow('Optical Flow Tracking', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("No webcam available for testing")
