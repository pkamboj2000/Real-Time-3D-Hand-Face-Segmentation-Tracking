"""
Classical computer vision module initialization.

This module provides classical computer vision algorithms for hand and face
detection and tracking, including skin detection, optical flow, and contour
analysis methods.
"""

from .skin_detection import SkinDetector, adaptive_skin_detection
from .optical_flow import OpticalFlowTracker, RegionTracker, estimate_motion_vectors
from .contour_tracking import ContourTracker

__all__ = [
    'SkinDetector',
    'adaptive_skin_detection',
    'OpticalFlowTracker', 
    'RegionTracker',
    'estimate_motion_vectors',
    'ContourTracker'
]
