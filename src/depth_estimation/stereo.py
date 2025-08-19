"""
Stereo vision depth estimation module.

This module implements stereo vision algorithms for depth estimation
using calibrated stereo camera pairs. It includes stereo matching,
disparity computation, and 3D reconstruction capabilities.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional, Any, List
import logging

logger = logging.getLogger(__name__)


class StereoCalibrator:
    """
    Camera calibration for stereo vision systems.
    
    Handles calibration of stereo camera pairs for accurate
    depth estimation from disparity maps.
    """
    
    def __init__(self, chessboard_size: Tuple[int, int] = (9, 6),
                 square_size: float = 25.0):
        """
        Initialize stereo calibrator.
        
        Args:
            chessboard_size: Size of calibration chessboard (width, height)
            square_size: Size of chessboard squares in mm
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        # Prepare object points
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Storage for calibration data
        self.objpoints = []  # 3D points in real world space
        self.imgpoints_left = []  # 2D points in left image plane
        self.imgpoints_right = []  # 2D points in right image plane
        
        # Calibration results
        self.camera_matrix_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_left = None
        self.dist_coeffs_right = None
        self.R = None  # Rotation matrix
        self.T = None  # Translation vector
        self.E = None  # Essential matrix
        self.F = None  # Fundamental matrix
        
        # Rectification parameters
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None
        self.map1_left = None
        self.map2_left = None
        self.map1_right = None
        self.map2_right = None
    
    def add_calibration_images(self, img_left: np.ndarray, 
                              img_right: np.ndarray) -> bool:
        """
        Add stereo image pair for calibration.
        
        Args:
            img_left: Left camera image
            img_right: Right camera image
            
        Returns:
            True if chessboard found in both images
        """
        # Convert to grayscale
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY) if len(img_left.shape) == 3 else img_left
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY) if len(img_right.shape) == 3 else img_right
        
        # Find chessboard corners
        ret_left, corners_left = cv2.findChessboardCorners(
            gray_left, self.chessboard_size, None)
        ret_right, corners_right = cv2.findChessboardCorners(
            gray_right, self.chessboard_size, None)
        
        if ret_left and ret_right:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
            
            # Add to calibration data
            self.objpoints.append(self.objp)
            self.imgpoints_left.append(corners_left)
            self.imgpoints_right.append(corners_right)
            
            return True
        
        return False
    
    def calibrate_stereo(self, image_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        Perform stereo calibration.
        
        Args:
            image_size: Size of images (width, height)
            
        Returns:
            Dictionary with calibration results
        """
        if len(self.objpoints) < 10:
            raise ValueError("Need at least 10 calibration image pairs")
        
        # Individual camera calibration
        ret_left, self.camera_matrix_left, self.dist_coeffs_left, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_left, image_size, None, None)
        
        ret_right, self.camera_matrix_right, self.dist_coeffs_right, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_right, image_size, None, None)
        
        # Stereo calibration
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        
        ret_stereo, _, _, _, _, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_left, self.imgpoints_right,
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            image_size, criteria=criteria, flags=flags)
        
        # Stereo rectification
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            image_size, self.R, self.T, alpha=0)
        
        # Generate rectification maps
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            self.camera_matrix_left, self.dist_coeffs_left, self.R1, self.P1, image_size, cv2.CV_16SC2)
        
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            self.camera_matrix_right, self.dist_coeffs_right, self.R2, self.P2, image_size, cv2.CV_16SC2)
        
        return {
            'reprojection_error': ret_stereo,
            'camera_matrix_left': self.camera_matrix_left,
            'camera_matrix_right': self.camera_matrix_right,
            'dist_coeffs_left': self.dist_coeffs_left,
            'dist_coeffs_right': self.dist_coeffs_right,
            'rotation_matrix': self.R,
            'translation_vector': self.T,
            'essential_matrix': self.E,
            'fundamental_matrix': self.F,
            'Q_matrix': self.Q
        }
    
    def rectify_images(self, img_left: np.ndarray, 
                      img_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify stereo image pair.
        
        Args:
            img_left: Left camera image
            img_right: Right camera image
            
        Returns:
            Tuple of rectified (left, right) images
        """
        if self.map1_left is None:
            raise ValueError("Stereo calibration not performed")
        
        rectified_left = cv2.remap(img_left, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(img_right, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
        
        return rectified_left, rectified_right


class StereoMatcher:
    """
    Stereo matching for disparity computation.
    
    Implements various stereo matching algorithms for computing
    disparity maps from rectified stereo image pairs.
    """
    
    def __init__(self, matcher_type: str = 'sgbm', **kwargs):
        """
        Initialize stereo matcher.
        
        Args:
            matcher_type: Type of matcher ('bm', 'sgbm', 'sgbm3way')
            **kwargs: Additional parameters for the matcher
        """
        self.matcher_type = matcher_type
        self.matcher = self._create_matcher(**kwargs)
        
        # Post-processing filter
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.matcher)
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.matcher)
    
    def _create_matcher(self, **kwargs) -> cv2.StereoMatcher:
        """Create stereo matcher based on type."""
        if self.matcher_type == 'bm':
            # Block Matching
            matcher = cv2.StereoBM_create()
            
            # Set parameters
            matcher.setNumDisparities(kwargs.get('num_disparities', 16))
            matcher.setBlockSize(kwargs.get('block_size', 15))
            matcher.setPreFilterCap(kwargs.get('pre_filter_cap', 31))
            matcher.setMinDisparity(kwargs.get('min_disparity', 0))
            matcher.setTextureThreshold(kwargs.get('texture_threshold', 10))
            matcher.setUniquenessRatio(kwargs.get('uniqueness_ratio', 15))
            matcher.setSpeckleWindowSize(kwargs.get('speckle_window_size', 100))
            matcher.setSpeckleRange(kwargs.get('speckle_range', 32))
            
        elif self.matcher_type == 'sgbm':
            # Semi-Global Block Matching
            matcher = cv2.StereoSGBM_create()
            
            # Set parameters
            matcher.setNumDisparities(kwargs.get('num_disparities', 96))
            matcher.setBlockSize(kwargs.get('block_size', 5))
            matcher.setMinDisparity(kwargs.get('min_disparity', 0))
            matcher.setP1(kwargs.get('P1', 8 * 3 * kwargs.get('block_size', 5) ** 2))
            matcher.setP2(kwargs.get('P2', 32 * 3 * kwargs.get('block_size', 5) ** 2))
            matcher.setDisp12MaxDiff(kwargs.get('disp12_max_diff', 1))
            matcher.setPreFilterCap(kwargs.get('pre_filter_cap', 63))
            matcher.setUniquenessRatio(kwargs.get('uniqueness_ratio', 10))
            matcher.setSpeckleWindowSize(kwargs.get('speckle_window_size', 100))
            matcher.setSpeckleRange(kwargs.get('speckle_range', 32))
            matcher.setMode(cv2.STEREO_SGBM_MODE_SGBM_3WAY)
            
        else:
            raise ValueError(f"Unknown matcher type: {self.matcher_type}")
        
        return matcher
    
    def compute_disparity(self, img_left: np.ndarray, 
                         img_right: np.ndarray,
                         use_filter: bool = True) -> np.ndarray:
        """
        Compute disparity map from stereo pair.
        
        Args:
            img_left: Left rectified image
            img_right: Right rectified image
            use_filter: Whether to apply WLS filtering
            
        Returns:
            Disparity map
        """
        # Convert to grayscale if needed
        if len(img_left.shape) == 3:
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = img_left
            gray_right = img_right
        
        # Compute disparity
        disparity_left = self.matcher.compute(gray_left, gray_right)
        
        if use_filter:
            # Apply WLS filtering for better results
            disparity_right = self.right_matcher.compute(gray_right, gray_left)
            filtered_disparity = self.wls_filter.filter(
                disparity_left, gray_left, None, disparity_right)
            
            return filtered_disparity.astype(np.float32) / 16.0
        else:
            return disparity_left.astype(np.float32) / 16.0
    
    def set_parameter(self, param_name: str, value: Any):
        """Set matcher parameter."""
        if hasattr(self.matcher, f'set{param_name}'):
            getattr(self.matcher, f'set{param_name}')(value)


class DepthFromStereo:
    """
    3D reconstruction from stereo disparity maps.
    
    Converts disparity maps to depth maps and point clouds
    using camera calibration parameters.
    """
    
    def __init__(self, Q_matrix: np.ndarray, baseline: float, focal_length: float):
        """
        Initialize depth reconstruction.
        
        Args:
            Q_matrix: Disparity-to-depth mapping matrix from stereo calibration
            baseline: Distance between stereo cameras in mm
            focal_length: Focal length in pixels
        """
        self.Q = Q_matrix
        self.baseline = baseline
        self.focal_length = focal_length
    
    def disparity_to_depth(self, disparity: np.ndarray, 
                          min_disparity: float = 1.0) -> np.ndarray:
        """
        Convert disparity map to depth map.
        
        Args:
            disparity: Disparity map
            min_disparity: Minimum disparity value to avoid division by zero
            
        Returns:
            Depth map in same units as baseline
        """
        # Avoid division by zero
        disparity_safe = np.where(disparity > min_disparity, disparity, np.nan)
        
        # Convert disparity to depth
        depth = (self.baseline * self.focal_length) / disparity_safe
        
        return depth
    
    def disparity_to_pointcloud(self, disparity: np.ndarray,
                               color_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert disparity map to 3D point cloud.
        
        Args:
            disparity: Disparity map
            color_image: Optional color image for textured point cloud
            
        Returns:
            Point cloud array (N, 3) or (N, 6) with colors
        """
        # Reproject to 3D
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
        
        # Remove invalid points (infinite or too far)
        mask = (disparity > 0) & (points_3d[:, :, 2] > 0) & (points_3d[:, :, 2] < 10000)
        
        # Extract valid points
        valid_points = points_3d[mask]
        
        if color_image is not None:
            # Add color information
            valid_colors = color_image[mask]
            return np.column_stack([valid_points, valid_colors])
        else:
            return valid_points
    
    def filter_depth_range(self, depth: np.ndarray, 
                          min_depth: float = 100.0, 
                          max_depth: float = 2000.0) -> np.ndarray:
        """
        Filter depth map by valid depth range.
        
        Args:
            depth: Input depth map
            min_depth: Minimum valid depth
            max_depth: Maximum valid depth
            
        Returns:
            Filtered depth map
        """
        filtered_depth = depth.copy()
        filtered_depth[(depth < min_depth) | (depth > max_depth)] = 0
        
        return filtered_depth


class StereoVisionSystem:
    """
    Complete stereo vision system for real-time depth estimation.
    
    Combines calibration, rectification, matching, and 3D reconstruction
    for complete stereo vision pipeline.
    """
    
    def __init__(self, calibration_data: Optional[Dict[str, Any]] = None):
        """
        Initialize stereo vision system.
        
        Args:
            calibration_data: Pre-computed calibration parameters
        """
        self.calibrator = StereoCalibrator()
        self.matcher = StereoMatcher('sgbm')
        self.depth_processor = None
        
        if calibration_data:
            self.load_calibration(calibration_data)
    
    def load_calibration(self, calibration_data: Dict[str, Any]):
        """Load calibration parameters."""
        self.calibrator.camera_matrix_left = calibration_data['camera_matrix_left']
        self.calibrator.camera_matrix_right = calibration_data['camera_matrix_right']
        self.calibrator.dist_coeffs_left = calibration_data['dist_coeffs_left']
        self.calibrator.dist_coeffs_right = calibration_data['dist_coeffs_right']
        self.calibrator.R = calibration_data['rotation_matrix']
        self.calibrator.T = calibration_data['translation_vector']
        self.calibrator.Q = calibration_data['Q_matrix']
        
        # Calculate baseline and focal length
        baseline = np.linalg.norm(self.calibrator.T)
        focal_length = self.calibrator.camera_matrix_left[0, 0]
        
        self.depth_processor = DepthFromStereo(
            self.calibrator.Q, baseline, focal_length)
    
    def process_stereo_pair(self, img_left: np.ndarray, 
                           img_right: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process stereo image pair to generate depth information.
        
        Args:
            img_left: Left camera image
            img_right: Right camera image
            
        Returns:
            Dictionary with processing results
        """
        if self.depth_processor is None:
            raise ValueError("System not calibrated")
        
        # Rectify images
        rect_left, rect_right = self.calibrator.rectify_images(img_left, img_right)
        
        # Compute disparity
        disparity = self.matcher.compute_disparity(rect_left, rect_right)
        
        # Convert to depth
        depth = self.depth_processor.disparity_to_depth(disparity)
        
        # Filter depth
        depth_filtered = self.depth_processor.filter_depth_range(depth)
        
        # Create visualizations
        disparity_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_vis = cv2.normalize(depth_filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        
        return {
            'rectified_left': rect_left,
            'rectified_right': rect_right,
            'disparity': disparity,
            'disparity_vis': disparity_vis,
            'depth': depth_filtered,
            'depth_vis': depth_vis,
            'depth_colored': depth_colored
        }


def create_default_stereo_system() -> StereoVisionSystem:
    """
    Create stereo vision system with default parameters.
    
    Returns:
        Configured stereo vision system
    """
    # Create default calibration data (for simulation)
    default_calibration = {
        'camera_matrix_left': np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32),
        'camera_matrix_right': np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32),
        'dist_coeffs_left': np.zeros((4, 1)),
        'dist_coeffs_right': np.zeros((4, 1)),
        'rotation_matrix': np.eye(3),
        'translation_vector': np.array([-60, 0, 0]),  # 60mm baseline
        'Q_matrix': np.array([
            [1, 0, 0, -320],
            [0, 1, 0, -240],
            [0, 0, 0, 800],
            [0, 0, 1/60, 0]
        ], dtype=np.float32)
    }
    
    system = StereoVisionSystem(default_calibration)
    return system


if __name__ == "__main__":
    # Test stereo vision system
    stereo_system = create_default_stereo_system()
    
    # Test with synthetic stereo pair
    img_left = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img_right = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    try:
        results = stereo_system.process_stereo_pair(img_left, img_right)
        
        print("Stereo processing results:")
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                print(f"{key}: {value.shape}, dtype: {value.dtype}")
            else:
                print(f"{key}: {value}")
        
        print("Stereo vision system test completed successfully")
        
    except Exception as e:
        print(f"Error in stereo processing: {e}")
        print("Note: This is expected with random synthetic data")
