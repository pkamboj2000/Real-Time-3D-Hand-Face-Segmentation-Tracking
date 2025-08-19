"""
Evaluation script for hand and face segmentation models.

This script includes detailed evaluation capabilities including
segmentation metrics, tracking performance, and 3D alignment accuracy.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import time
import argparse
import logging
from pathlib import Path
import sys
import json
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.classical import SkinDetector, OpticalFlowTracker, ContourTracker
from src.deep_learning import create_unet_model, create_mask_rcnn, IoUMetric, DiceMetric
from src.depth_estimation import RealTimeDepthEstimator
from src.utils import setup_logging, load_config, save_config, calculate_iou
from config.config import MODEL_CONFIGS, EVALUATION_METRICS

logger = setup_logging()


class SegmentationEvaluator:
    """
    Evaluator for segmentation models with comprehensive metrics.
    """
    
    def __init__(self, num_classes: int = 3):
        """
        Initialize evaluator.
        
        Args:
            num_classes: Number of segmentation classes
        """
        self.num_classes = num_classes
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset all evaluation metrics."""
        self.iou_metric = IoUMetric(self.num_classes)
        self.dice_metric = DiceMetric(self.num_classes)
        
        self.pixel_accuracy_correct = 0
        self.pixel_accuracy_total = 0
        
        self.class_predictions = defaultdict(list)
        self.class_targets = defaultdict(list)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with new predictions.
        
        Args:
            predictions: Model predictions (B, C, H, W) or (B, H, W)
            targets: Ground truth targets (B, H, W)
        """
        # Convert predictions to class labels if needed
        if len(predictions.shape) == 4:
            pred_labels = torch.argmax(predictions, dim=1)
        else:
            pred_labels = predictions
        
        # Update IoU and Dice metrics
        self.iou_metric.update(pred_labels, targets)
        self.dice_metric.update(pred_labels, targets)
        
        # Update pixel accuracy
        correct = (pred_labels == targets).sum().item()
        total = targets.numel()
        
        self.pixel_accuracy_correct += correct
        self.pixel_accuracy_total += total
        
        # Store per-class predictions for detailed analysis
        for class_id in range(self.num_classes):
            pred_mask = (pred_labels == class_id)
            target_mask = (targets == class_id)
            
            self.class_predictions[class_id].append(pred_mask.cpu().numpy())
            self.class_targets[class_id].append(target_mask.cpu().numpy())
    
    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute all evaluation metrics.
        
        Returns:
            Dictionary with computed metrics
        """
        # IoU metrics
        iou_results = self.iou_metric.compute()
        
        # Dice metrics
        dice_results = self.dice_metric.compute()
        
        # Pixel accuracy
        pixel_accuracy = self.pixel_accuracy_correct / max(self.pixel_accuracy_total, 1)
        
        # Per-class analysis
        class_metrics = {}
        for class_id in range(self.num_classes):
            if class_id in self.class_predictions:
                class_name = self._get_class_name(class_id)
                
                # Combine all predictions and targets for this class
                all_preds = np.concatenate(self.class_predictions[class_id])
                all_targets = np.concatenate(self.class_targets[class_id])
                
                # Calculate precision, recall, F1
                tp = np.sum(all_preds & all_targets)
                fp = np.sum(all_preds & ~all_targets)
                fn = np.sum(~all_preds & all_targets)
                
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                
                class_metrics[class_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
        
        return {
            'mIoU': iou_results['mIoU'],
            'IoU_per_class': iou_results['IoU_per_class'],
            'mDice': dice_results['mDice'],
            'Dice_per_class': dice_results['Dice_per_class'],
            'pixel_accuracy': pixel_accuracy,
            'class_metrics': class_metrics
        }
    
    def _get_class_name(self, class_id: int) -> str:
        """Get human-readable class name."""
        class_names = ['background', 'hand', 'face']
        return class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'


class TrackingEvaluator:
    """
    Evaluator for tracking performance metrics.
    """
    
    def __init__(self):
        """Initialize tracking evaluator."""
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset tracking metrics."""
        self.frame_times = []
        self.detection_counts = []
        self.track_continuity = defaultdict(list)
        self.track_stability = []
    
    def update_timing(self, frame_time: float):
        """Update timing metrics."""
        self.frame_times.append(frame_time)
    
    def update_detections(self, num_detections: int):
        """Update detection count metrics."""
        self.detection_counts.append(num_detections)
    
    def update_tracking(self, track_results: List[Dict[str, Any]]):
        """Update tracking metrics."""
        for result in track_results:
            track_id = result.get('track_id', -1)
            if track_id >= 0:
                self.track_continuity[track_id].append(True)
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute tracking performance metrics."""
        # FPS and latency
        if self.frame_times:
            avg_frame_time = np.mean(self.frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            latency_ms = avg_frame_time * 1000
        else:
            fps = 0
            latency_ms = 0
        
        # Detection stability
        avg_detections = np.mean(self.detection_counts) if self.detection_counts else 0
        detection_variance = np.var(self.detection_counts) if self.detection_counts else 0
        
        # Track continuity
        track_lengths = [len(track) for track in self.track_continuity.values()]
        avg_track_length = np.mean(track_lengths) if track_lengths else 0
        
        # Stability score (inverse of detection variance)
        stability_score = 1.0 / (1.0 + detection_variance)
        
        return {
            'fps': fps,
            'latency_ms': latency_ms,
            'avg_detections': avg_detections,
            'detection_variance': detection_variance,
            'avg_track_length': avg_track_length,
            'stability_score': stability_score,
            'num_tracks': len(self.track_continuity)
        }


class ModelEvaluator:
    """
    Comprehensive model evaluator for the entire pipeline.
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize model evaluator.
        
        Args:
            device: Device for evaluation ('cpu', 'cuda', 'auto')
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize evaluators
        self.seg_evaluator = SegmentationEvaluator()
        self.tracking_evaluator = TrackingEvaluator()
        
        # Initialize methods
        self.classical_methods = {
            'skin_detector': SkinDetector('combined'),
            'optical_flow': OpticalFlowTracker(),
            'contour_tracker': ContourTracker()
        }
        
        self.deep_learning_models = {}
        self.depth_estimator = None
    
    def load_models(self, model_configs: Dict[str, Any]):
        """
        Load deep learning models for evaluation.
        
        Args:
            model_configs: Configuration for models to load
        """
        try:
            # Load U-Net
            if 'unet' in model_configs:
                config = model_configs['unet']
                self.deep_learning_models['unet'] = create_unet_model(
                    'standard',
                    in_channels=3,
                    out_channels=config.get('num_classes', 3)
                )
                self.deep_learning_models['unet'] = self.deep_learning_models['unet'].to(self.device)
                self.deep_learning_models['unet'].eval()
                logger.info("U-Net model loaded")
            
            # Load Mask R-CNN
            if 'maskrcnn' in model_configs:
                config = model_configs['maskrcnn']
                self.deep_learning_models['maskrcnn'] = create_mask_rcnn(
                    num_classes=config.get('num_classes', 3)
                )
                self.deep_learning_models['maskrcnn'] = self.deep_learning_models['maskrcnn'].to(self.device)
                self.deep_learning_models['maskrcnn'].eval()
                logger.info("Mask R-CNN model loaded")
            
        except Exception as e:
            logger.warning(f"Failed to load some models: {e}")
    
    def load_depth_estimator(self):
        """Load depth estimation model."""
        try:
            self.depth_estimator = RealTimeDepthEstimator()
            logger.info("Depth estimator loaded")
        except Exception as e:
            logger.warning(f"Failed to load depth estimator: {e}")
    
    def evaluate_classical_methods(self, images: List[np.ndarray], 
                                 ground_truth_masks: List[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate classical computer vision methods.
        
        Args:
            images: List of input images
            ground_truth_masks: Optional ground truth masks for evaluation
            
        Returns:
            Evaluation results for classical methods
        """
        logger.info("Evaluating classical methods")
        
        results = {}
        total_time = 0
        
        for i, image in enumerate(images):
            start_time = time.time()
            
            # Skin detection
            skin_mask = self.classical_methods['skin_detector'].detect_skin(image)
            
            # Optical flow tracking
            flow_results = self.classical_methods['optical_flow'].update(image)
            
            # Contour tracking
            contour_results = self.classical_methods['contour_tracker'].update(image)
            
            frame_time = time.time() - start_time
            total_time += frame_time
            
            # Update tracking metrics
            self.tracking_evaluator.update_timing(frame_time)
            self.tracking_evaluator.update_detections(len(contour_results))
            self.tracking_evaluator.update_tracking(contour_results)
            
            # Compare with ground truth if available
            if ground_truth_masks and i < len(ground_truth_masks):
                gt_mask = ground_truth_masks[i]
                
                # Convert skin mask to multi-class format (simplified)
                pred_mask = np.zeros_like(gt_mask)
                pred_mask[skin_mask > 0] = 1  # Assume skin regions are hands
                
                # Update segmentation metrics
                pred_tensor = torch.from_numpy(pred_mask).long()
                gt_tensor = torch.from_numpy(gt_mask).long()
                
                self.seg_evaluator.update(pred_tensor.unsqueeze(0), gt_tensor.unsqueeze(0))
        
        # Compute metrics
        tracking_metrics = self.tracking_evaluator.compute_metrics()
        
        if ground_truth_masks:
            segmentation_metrics = self.seg_evaluator.compute_metrics()
            results['segmentation'] = segmentation_metrics
        
        results['tracking'] = tracking_metrics
        results['avg_processing_time'] = total_time / len(images)
        
        return results
    
    def evaluate_deep_learning_models(self, images: List[np.ndarray],
                                    ground_truth_masks: List[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate deep learning models.
        
        Args:
            images: List of input images
            ground_truth_masks: Optional ground truth masks
            
        Returns:
            Evaluation results for deep learning models
        """
        logger.info("Evaluating deep learning models")
        
        results = {}
        
        # Evaluate U-Net
        if 'unet' in self.deep_learning_models:
            results['unet'] = self._evaluate_unet(images, ground_truth_masks)
        
        # Evaluate Mask R-CNN
        if 'maskrcnn' in self.deep_learning_models:
            results['maskrcnn'] = self._evaluate_maskrcnn(images, ground_truth_masks)
        
        return results
    
    def _evaluate_unet(self, images: List[np.ndarray], 
                      ground_truth_masks: List[np.ndarray] = None) -> Dict[str, Any]:
        """Evaluate U-Net model."""
        model = self.deep_learning_models['unet']
        
        # Reset evaluators
        unet_seg_evaluator = SegmentationEvaluator()
        unet_tracking_evaluator = TrackingEvaluator()
        
        total_time = 0
        
        with torch.no_grad():
            for i, image in enumerate(images):
                start_time = time.time()
                
                # Preprocess image
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
                
                # Forward pass
                output = model(image_tensor)
                pred_mask = torch.argmax(output, dim=1)
                
                frame_time = time.time() - start_time
                total_time += frame_time
                
                # Update tracking metrics
                unet_tracking_evaluator.update_timing(frame_time)
                
                # Count detections (non-background pixels)
                num_detections = torch.sum(pred_mask > 0).item()
                unet_tracking_evaluator.update_detections(num_detections)
                
                # Compare with ground truth
                if ground_truth_masks and i < len(ground_truth_masks):
                    gt_mask = torch.from_numpy(ground_truth_masks[i]).long().to(self.device)
                    unet_seg_evaluator.update(pred_mask, gt_mask.unsqueeze(0))
        
        # Compute metrics
        results = {
            'tracking': unet_tracking_evaluator.compute_metrics(),
            'avg_processing_time': total_time / len(images)
        }
        
        if ground_truth_masks:
            results['segmentation'] = unet_seg_evaluator.compute_metrics()
        
        return results
    
    def _evaluate_maskrcnn(self, images: List[np.ndarray],
                          ground_truth_masks: List[np.ndarray] = None) -> Dict[str, Any]:
        """Evaluate Mask R-CNN model."""
        model = self.deep_learning_models['maskrcnn']
        
        tracking_evaluator = TrackingEvaluator()
        total_time = 0
        
        with torch.no_grad():
            for i, image in enumerate(images):
                start_time = time.time()
                
                # Preprocess image
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
                
                # Forward pass
                try:
                    outputs = model(image_tensor)
                    
                    # Extract detections (simplified)
                    if isinstance(outputs, dict):
                        num_detections = len(outputs.get('boxes', []))
                    else:
                        num_detections = 0
                    
                except Exception as e:
                    logger.warning(f"Mask R-CNN inference failed: {e}")
                    num_detections = 0
                
                frame_time = time.time() - start_time
                total_time += frame_time
                
                # Update tracking metrics
                tracking_evaluator.update_timing(frame_time)
                tracking_evaluator.update_detections(num_detections)
        
        return {
            'tracking': tracking_evaluator.compute_metrics(),
            'avg_processing_time': total_time / len(images)
        }
    
    def evaluate_depth_estimation(self, images: List[np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate depth estimation performance.
        
        Args:
            images: List of input images
            
        Returns:
            Depth estimation evaluation results
        """
        if self.depth_estimator is None:
            return {}
        
        logger.info("Evaluating depth estimation")
        
        depth_tracking_evaluator = TrackingEvaluator()
        depth_qualities = []
        total_time = 0
        
        for image in images:
            start_time = time.time()
            
            # Process frame
            results = self.depth_estimator.process_frame(image)
            
            frame_time = time.time() - start_time
            total_time += frame_time
            
            # Update tracking metrics
            depth_tracking_evaluator.update_timing(frame_time)
            
            # Assess depth quality (simplified metric)
            depth_map = results['depth_raw']
            depth_variance = np.var(depth_map[depth_map > 0])
            depth_qualities.append(depth_variance)
        
        return {
            'tracking': depth_tracking_evaluator.compute_metrics(),
            'avg_processing_time': total_time / len(images),
            'avg_depth_variance': np.mean(depth_qualities) if depth_qualities else 0,
            'depth_quality_std': np.std(depth_qualities) if depth_qualities else 0
        }
    
    def run_comprehensive_evaluation(self, test_images: List[np.ndarray],
                                   ground_truth_masks: List[np.ndarray] = None,
                                   output_path: Path = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation of all methods.
        
        Args:
            test_images: List of test images
            ground_truth_masks: Optional ground truth masks
            output_path: Optional path to save results
            
        Returns:
            Complete evaluation results
        """
        logger.info(f"Starting comprehensive evaluation with {len(test_images)} images")
        
        results = {
            'dataset_info': {
                'num_images': len(test_images),
                'has_ground_truth': ground_truth_masks is not None,
                'image_size': test_images[0].shape if test_images else None
            }
        }
        
        # Evaluate classical methods
        try:
            classical_results = self.evaluate_classical_methods(test_images, ground_truth_masks)
            results['classical_methods'] = classical_results
        except Exception as e:
            logger.error(f"Classical evaluation failed: {e}")
            results['classical_methods'] = {'error': str(e)}
        
        # Evaluate deep learning models
        try:
            dl_results = self.evaluate_deep_learning_models(test_images, ground_truth_masks)
            results['deep_learning'] = dl_results
        except Exception as e:
            logger.error(f"Deep learning evaluation failed: {e}")
            results['deep_learning'] = {'error': str(e)}
        
        # Evaluate depth estimation
        try:
            depth_results = self.evaluate_depth_estimation(test_images)
            results['depth_estimation'] = depth_results
        except Exception as e:
            logger.error(f"Depth estimation evaluation failed: {e}")
            results['depth_estimation'] = {'error': str(e)}
        
        # Save results if requested
        if output_path:
            save_config(results, output_path)
            logger.info(f"Evaluation results saved to {output_path}")
        
        return results


def load_test_images(data_dir: Path, max_images: int = 100) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load test images and masks.
    
    Args:
        data_dir: Path to test data directory
        max_images: Maximum number of images to load
        
    Returns:
        Tuple of (images, masks)
    """
    images = []
    masks = []
    
    test_images_dir = data_dir / "images" / "test"
    test_masks_dir = data_dir / "masks" / "test"
    
    if not test_images_dir.exists():
        logger.warning("Test images directory not found, creating dummy data")
        # Create dummy test data
        for i in range(min(max_images, 10)):
            dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            dummy_mask = np.random.randint(0, 3, (512, 512), dtype=np.uint8)
            images.append(dummy_image)
            masks.append(dummy_mask)
        return images, masks
    
    # Load real test data
    image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    image_files = sorted(image_files)[:max_images]
    
    for img_file in image_files:
        # Load image
        image = cv2.imread(str(img_file))
        if image is not None:
            images.append(image)
            
            # Look for corresponding mask
            mask_file = test_masks_dir / (img_file.stem + ".png")
            if mask_file.exists():
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                masks.append(mask)
            else:
                masks.append(None)
    
    # Filter out None masks if we want only valid pairs
    valid_pairs = [(img, mask) for img, mask in zip(images, masks) if mask is not None]
    if valid_pairs:
        images, masks = zip(*valid_pairs)
        images, masks = list(images), list(masks)
    
    logger.info(f"Loaded {len(images)} test images")
    if masks and masks[0] is not None:
        logger.info(f"Loaded {len(masks)} corresponding masks")
    
    return images, masks


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Hand/Face Segmentation Models')
    
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to test dataset directory')
    parser.add_argument('--output-path', type=str,
                       help='Path to save evaluation results')
    parser.add_argument('--max-images', type=int, default=100,
                       help='Maximum number of test images to evaluate')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device for evaluation')
    parser.add_argument('--models-config', type=str,
                       help='Path to models configuration file')
    parser.add_argument('--skip-classical', action='store_true',
                       help='Skip classical methods evaluation')
    parser.add_argument('--skip-deep-learning', action='store_true',
                       help='Skip deep learning models evaluation')
    parser.add_argument('--skip-depth', action='store_true',
                       help='Skip depth estimation evaluation')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load test data
    data_dir = Path(args.data_dir)
    test_images, test_masks = load_test_images(data_dir, args.max_images)
    
    if not test_images:
        logger.error("No test images found")
        return
    
    # Create evaluator
    evaluator = ModelEvaluator(device=args.device)
    
    # Load models if requested
    if not args.skip_deep_learning:
        if args.models_config:
            models_config = load_config(args.models_config)
        else:
            models_config = MODEL_CONFIGS
        
        evaluator.load_models(models_config)
    
    # Load depth estimator if requested
    if not args.skip_depth:
        evaluator.load_depth_estimator()
    
    # Set up output path
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = Path("evaluation_results.json")
    
    # Run evaluation
    logger.info("Starting comprehensive evaluation")
    
    # Filter out None masks for ground truth
    valid_masks = [mask for mask in test_masks if mask is not None] if test_masks else None
    
    results = evaluator.run_comprehensive_evaluation(
        test_images, valid_masks, output_path)
    
    # Print summary
    logger.info("Evaluation completed. Summary:")
    
    for method_name, method_results in results.items():
        if method_name == 'dataset_info':
            continue
            
        if isinstance(method_results, dict) and 'error' not in method_results:
            logger.info(f"\n{method_name.upper()}:")
            
            if 'tracking' in method_results:
                tracking = method_results['tracking']
                logger.info(f"  FPS: {tracking.get('fps', 0):.2f}")
                logger.info(f"  Latency: {tracking.get('latency_ms', 0):.2f}ms")
                logger.info(f"  Stability: {tracking.get('stability_score', 0):.3f}")
            
            if 'segmentation' in method_results:
                segmentation = method_results['segmentation']
                logger.info(f"  mIoU: {segmentation.get('mIoU', 0):.3f}")
                logger.info(f"  mDice: {segmentation.get('mDice', 0):.3f}")
                logger.info(f"  Pixel Accuracy: {segmentation.get('pixel_accuracy', 0):.3f}")
        
        elif isinstance(method_results, dict) and 'error' in method_results:
            logger.warning(f"{method_name}: {method_results['error']}")
    
    logger.info(f"Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
