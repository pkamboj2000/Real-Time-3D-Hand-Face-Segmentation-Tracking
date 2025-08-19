"""
Core segmentation models for real-time hand/face tracking.
Contains U-Net and Mask R-CNN implementations for real-time inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import cv2
import numpy as np


class UNet(nn.Module):
    """
    U-Net model for semantic segmentation.
    Built for real-time hand/face segmentation.
    """
    
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        
    def conv_block(self, in_channels, out_channels):
        """Conv block with BatchNorm and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final(dec1)


class MaskRCNN:
    """
    Mask R-CNN wrapper for instance segmentation.
    Set up for hand and face detection.
    """
    
    def __init__(self, num_classes=3):  # background, hand, face
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained Mask R-CNN
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        
        # Replace classifier head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Replace mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, image):
        """
        Get masks for input image.
        
        Args:
            image: Input image (H, W, 3) numpy array
            
        Returns:
            Dictionary with predictions
        """
        # Preprocess
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        return predictions[0]


class ClassicalBaseline:
    """
    Classical computer vision baseline with skin detection and optical flow.
    """
    
    def __init__(self):
        self.prev_frame = None
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Harris corner detection parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
    
    def detect_skin(self, image):
        """
        Find skin regions using HSV and YCrCb color spaces.
        
        Args:
            image: Input BGR image
            
        Returns:
            Binary mask of skin regions
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # HSV skin detection
        lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
        upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # YCrCb skin detection  
        lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
        upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        # Combine masks
        skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        
        # Clean up the mask a bit
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        return skin_mask
    
    def track_features(self, image):
        """
        Track features using Lucas-Kanade optical flow.
        
        Args:
            image: Current frame
            
        Returns:
            Tracked points and status
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            # First frame - just set up tracking
            self.prev_frame = gray
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            return [], []
        
        if self.prev_points is not None and len(self.prev_points) > 0:
            # Track the points we found before
            next_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_frame, gray, self.prev_points, None, **self.lk_params
            )
            
            # Keep only the good ones
            good_new = next_points[status == 1]
            good_old = self.prev_points[status == 1]
            
            self.prev_points = good_new.reshape(-1, 1, 2)
            self.prev_frame = gray
            
            return good_new, good_old
        
        self.prev_frame = gray
        return [], []
    
    def segment_hands_faces(self, image):
        """
        Find hands and faces using classical methods.
        
        Args:
            image: Input image
            
        Returns:
            Segmentation mask with hand/face regions
        """
        # Get skin mask
        skin_mask = self.detect_skin(image)
        
        # Find contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Make output mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 1000:  # Skip small blobs
                # Simple guess: bigger regions are hands, smaller ones faces
                if area > 5000:
                    cv2.fillPoly(mask, [contour], 1)  # Hand class
                else:
                    cv2.fillPoly(mask, [contour], 2)  # Face class
        
        return mask


class DepthEstimator:
    """
    Depth estimation with MiDaS for 3D stuff.
    """
    
    def __init__(self):
        try:
            # Try to load MiDaS model
            self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
            self.transform = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            self.available = True
        except Exception as e:
            print(f"MiDaS not available: {e}")
            self.available = False
    
    def estimate_depth(self, image):
        """
        Get depth from single image.
        
        Args:
            image: Input BGR image
            
        Returns:
            Depth map
        """
        if not self.available:
            # Return dummy depth map
            return np.ones(image.shape[:2], dtype=np.float32)
        
        # Prep the image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb).to(self.device)
        
        with torch.no_grad():
            depth = self.model(input_tensor)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        return depth.cpu().numpy()


def create_models():
    """
    Make all the models we need.
    
    Returns:
        Dictionary with model instances
    """
    models = {
        'unet': UNet(in_channels=3, out_channels=3),
        'maskrcnn': MaskRCNN(num_classes=3),
        'classical': ClassicalBaseline(),
        'depth': DepthEstimator()
    }
    
    return models


if __name__ == "__main__":
    # Quick test
    models = create_models()
    print("Models created successfully!")
    
    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test classical method
    classical_mask = models['classical'].segment_hands_faces(dummy_image)
    print(f"Classical segmentation shape: {classical_mask.shape}")
    
    # Test depth estimation
    depth_map = models['depth'].estimate_depth(dummy_image)
    print(f"Depth map shape: {depth_map.shape}")
