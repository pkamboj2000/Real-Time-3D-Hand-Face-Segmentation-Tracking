"""
Evaluation pipeline for segmentation models.

This module provides comprehensive evaluation capabilities including
metrics computation, benchmarking, and result reporting for all
supported segmentation methods.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from core_models import create_models, InferenceResult
from train_model import SegmentationDataset

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    pixel_accuracy: float = 0.0
    mean_iou: float = 0.0
    mean_dice: float = 0.0
    per_class_iou: list[float] = field(default_factory=list)
    per_class_dice: list[float] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pixel_accuracy": self.pixel_accuracy,
            "mean_iou": self.mean_iou,
            "mean_dice": self.mean_dice,
            "per_class_iou": self.per_class_iou,
            "per_class_dice": self.per_class_dice,
        }


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    method_name: str
    num_samples: int
    avg_processing_time_ms: float
    fps: float
    metrics: EvaluationMetrics
    errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "method": self.method_name,
            "num_samples": self.num_samples,
            "avg_processing_time_ms": self.avg_processing_time_ms,
            "fps": self.fps,
            "metrics": self.metrics.to_dict(),
            "errors": self.errors,
        }


class MetricsCalculator:
    """
    Calculator for segmentation metrics.
    
    Computes pixel accuracy, IoU, and Dice scores for multi-class
    segmentation predictions.
    """
    
    CLASS_NAMES = ["Background", "Hand", "Face"]
    
    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
    
    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> EvaluationMetrics:
        """
        Compute all segmentation metrics.
        
        Args:
            predictions: Predicted masks (B, H, W) or logits (B, C, H, W)
            targets: Ground truth masks (B, H, W)
            
        Returns:
            EvaluationMetrics instance with all computed metrics
        """
        # Handle logits vs class predictions
        if predictions.dim() == 4:
            pred_labels = torch.argmax(predictions, dim=1)
        else:
            pred_labels = predictions
        
        # Pixel accuracy
        correct = (pred_labels == targets).sum().float()
        total = targets.numel()
        pixel_accuracy = (correct / total).item()
        
        # Per-class metrics
        ious = []
        dice_scores = []
        
        for class_id in range(self.num_classes):
            pred_mask = (pred_labels == class_id)
            target_mask = (targets == class_id)
            
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            pred_sum = pred_mask.sum().float()
            target_sum = target_mask.sum().float()
            
            # IoU with edge case handling
            if union > 0:
                iou = (intersection / union).item()
            else:
                iou = 1.0 if intersection == 0 else 0.0
            ious.append(iou)
            
            # Dice score
            denominator = pred_sum + target_sum
            if denominator > 0:
                dice = (2 * intersection / denominator).item()
            else:
                dice = 1.0 if intersection == 0 else 0.0
            dice_scores.append(dice)
        
        return EvaluationMetrics(
            pixel_accuracy=pixel_accuracy,
            mean_iou=float(np.mean(ious)),
            mean_dice=float(np.mean(dice_scores)),
            per_class_iou=ious,
            per_class_dice=dice_scores,
        )


class ModelEvaluator:
    """
    Comprehensive evaluator for segmentation models.
    
    Supports evaluation of multiple methods with benchmarking
    and detailed metrics reporting.
    
    Args:
        models: Dictionary of model instances
        device: Evaluation device
        num_classes: Number of segmentation classes
    """
    
    def __init__(
        self,
        models: dict[str, Any],
        device: str = "auto",
        num_classes: int = 3
    ):
        self.device = self._resolve_device(device)
        self.models = models
        self.num_classes = num_classes
        self.metrics_calculator = MetricsCalculator(num_classes)
        
        # Move models to device and set eval mode
        self._prepare_models()
        
        logger.info(f"Evaluator initialized on {self.device}")
    
    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve compute device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)
    
    def _prepare_models(self) -> None:
        """Prepare models for evaluation."""
        if "unet" in self.models:
            self.models["unet"] = self.models["unet"].to(self.device)
            self.models["unet"].eval()
    
    def _predict_classical(self, image_np: np.ndarray) -> torch.Tensor:
        """Run classical method prediction."""
        result = self.models["classical"].predict(image_np)
        if isinstance(result, InferenceResult):
            return torch.from_numpy(result.mask)
        return torch.from_numpy(result)
    
    def _predict_unet(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Run U-Net prediction."""
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.models["unet"](image_tensor)
            return torch.argmax(output, dim=1).squeeze().cpu()
    
    def _predict_maskrcnn(self, image_np: np.ndarray) -> torch.Tensor:
        """Run Mask R-CNN prediction."""
        result = self.models["maskrcnn"].predict(image_np)
        if isinstance(result, InferenceResult):
            return torch.from_numpy(result.mask).long()
        
        # Legacy dict format handling
        h, w = image_np.shape[:2]
        pred_mask = torch.zeros((h, w), dtype=torch.long)
        
        if "masks" in result and len(result["masks"]) > 0:
            masks = result["masks"].cpu().numpy()
            labels = result["labels"].cpu().numpy()
            scores = result["scores"].cpu().numpy()
            
            for mask, label, score in zip(masks, labels, scores):
                if score > 0.5:
                    pred_mask[mask[0] > 0.5] = label
        
        return pred_mask
    
    def evaluate_single(
        self,
        method_name: str,
        image: torch.Tensor,
        image_np: np.ndarray,
        target_mask: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """
        Evaluate a single sample with one method.
        
        Args:
            method_name: Name of the method to use
            image: Image tensor (C, H, W)
            image_np: Image as numpy array (H, W, 3) BGR
            target_mask: Ground truth mask
            
        Returns:
            Tuple of (predicted_mask, processing_time_seconds)
        """
        start_time = time.perf_counter()
        
        if method_name == "classical":
            pred_mask = self._predict_classical(image_np)
        elif method_name == "unet":
            pred_mask = self._predict_unet(image)
        elif method_name == "maskrcnn":
            pred_mask = self._predict_maskrcnn(image_np)
        else:
            raise ValueError(f"Unknown method: {method_name}")
        
        processing_time = time.perf_counter() - start_time
        
        # Resize if needed
        if pred_mask.shape != target_mask.shape:
            pred_mask = torch.nn.functional.interpolate(
                pred_mask.unsqueeze(0).unsqueeze(0).float(),
                size=target_mask.shape,
                mode="nearest"
            ).squeeze().long()
        
        return pred_mask, processing_time
    
    def evaluate_method(
        self,
        method_name: str,
        dataloader: DataLoader,
        max_samples: Optional[int] = None
    ) -> BenchmarkResult:
        """
        Evaluate a specific method on the dataset.
        
        Args:
            method_name: Name of method to evaluate
            dataloader: Test data loader
            max_samples: Maximum samples to evaluate
            
        Returns:
            BenchmarkResult with metrics and timing
        """
        logger.info(f"Evaluating {method_name}...")
        
        all_metrics: dict[str, list] = defaultdict(list)
        processing_times: list[float] = []
        errors: list[str] = []
        sample_count = 0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            if max_samples and sample_count >= max_samples:
                break
            
            batch_size = images.shape[0]
            
            for i in range(batch_size):
                if max_samples and sample_count >= max_samples:
                    break
                
                image = images[i]
                mask = masks[i]
                
                # Convert to numpy for classical/maskrcnn
                image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                try:
                    pred_mask, proc_time = self.evaluate_single(
                        method_name, image, image_np, mask
                    )
                    processing_times.append(proc_time)
                    
                    # Compute metrics
                    metrics = self.metrics_calculator.compute(
                        pred_mask.unsqueeze(0),
                        mask.unsqueeze(0)
                    )
                    
                    all_metrics["pixel_accuracy"].append(metrics.pixel_accuracy)
                    all_metrics["mean_iou"].append(metrics.mean_iou)
                    all_metrics["mean_dice"].append(metrics.mean_dice)
                    all_metrics["per_class_iou"].append(metrics.per_class_iou)
                    all_metrics["per_class_dice"].append(metrics.per_class_dice)
                    
                except Exception as e:
                    error_msg = f"Sample {sample_count}: {str(e)}"
                    errors.append(error_msg)
                    logger.warning(error_msg)
                
                sample_count += 1
                
                if sample_count % 20 == 0:
                    logger.info(f"  Processed {sample_count} samples...")
        
        # Aggregate metrics
        avg_time = np.mean(processing_times) if processing_times else 0
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        aggregated_metrics = EvaluationMetrics(
            pixel_accuracy=float(np.mean(all_metrics["pixel_accuracy"])) if all_metrics["pixel_accuracy"] else 0,
            mean_iou=float(np.mean(all_metrics["mean_iou"])) if all_metrics["mean_iou"] else 0,
            mean_dice=float(np.mean(all_metrics["mean_dice"])) if all_metrics["mean_dice"] else 0,
            per_class_iou=np.mean(all_metrics["per_class_iou"], axis=0).tolist() if all_metrics["per_class_iou"] else [],
            per_class_dice=np.mean(all_metrics["per_class_dice"], axis=0).tolist() if all_metrics["per_class_dice"] else [],
        )
        
        return BenchmarkResult(
            method_name=method_name,
            num_samples=sample_count,
            avg_processing_time_ms=avg_time * 1000,
            fps=fps,
            metrics=aggregated_metrics,
            errors=errors,
        )
    
    def run_evaluation(
        self,
        dataloader: DataLoader,
        methods: Optional[list[str]] = None,
        max_samples: Optional[int] = None
    ) -> dict[str, Any]:
        """
        Run evaluation on multiple methods.
        
        Args:
            dataloader: Test data loader
            methods: List of methods to evaluate (None for all)
            max_samples: Maximum samples per method
            
        Returns:
            Dictionary with complete evaluation results
        """
        if methods is None:
            methods = [m for m in self.models.keys() if m != "depth"]
        
        results = {
            "dataset_info": {
                "num_samples": len(dataloader.dataset),
                "batch_size": dataloader.batch_size,
            },
            "methods": {},
        }
        
        for method in methods:
            if method not in self.models:
                logger.warning(f"Method {method} not available, skipping")
                continue
            
            try:
                benchmark = self.evaluate_method(method, dataloader, max_samples)
                results["methods"][method] = benchmark.to_dict()
            except Exception as e:
                logger.error(f"Failed to evaluate {method}: {e}")
                results["methods"][method] = {"error": str(e)}
        
        return results


class ResultsReporter:
    """Formats and displays evaluation results."""
    
    CLASS_NAMES = ["Background", "Hand", "Face"]
    
    @classmethod
    def print_results(cls, results: dict[str, Any]) -> None:
        """Print formatted evaluation results."""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        # Dataset info
        dataset_info = results.get("dataset_info", {})
        print(f"\nDataset:")
        print(f"  Total Samples: {dataset_info.get('num_samples', 'N/A')}")
        print(f"  Batch Size: {dataset_info.get('batch_size', 'N/A')}")
        
        # Per-method results
        for method_name, method_results in results.get("methods", {}).items():
            print(f"\n{method_name.upper()}")
            print("-" * 50)
            
            if "error" in method_results:
                print(f"  Error: {method_results['error']}")
                continue
            
            print(f"  Samples: {method_results.get('num_samples', 'N/A')}")
            print(f"  FPS: {method_results.get('fps', 0):.2f}")
            print(f"  Latency: {method_results.get('avg_processing_time_ms', 0):.2f} ms")
            
            metrics = method_results.get("metrics", {})
            print(f"\n  Metrics:")
            print(f"    Pixel Accuracy: {metrics.get('pixel_accuracy', 0):.4f}")
            print(f"    Mean IoU: {metrics.get('mean_iou', 0):.4f}")
            print(f"    Mean Dice: {metrics.get('mean_dice', 0):.4f}")
            
            per_class_iou = metrics.get("per_class_iou", [])
            if per_class_iou:
                print(f"\n  Per-class IoU:")
                for i, iou in enumerate(per_class_iou):
                    class_name = cls.CLASS_NAMES[i] if i < len(cls.CLASS_NAMES) else f"Class_{i}"
                    print(f"    {class_name}: {iou:.4f}")
            
            errors = method_results.get("errors", [])
            if errors:
                print(f"\n  Errors ({len(errors)}):")
                for err in errors[:3]:
                    print(f"    - {err}")
                if len(errors) > 3:
                    print(f"    ... and {len(errors) - 3} more")
        
        print("\n" + "=" * 70)


def create_test_loader(
    data_dir: str | Path,
    batch_size: int = 1,
    image_size: int = 512,
    num_workers: int = 4
) -> DataLoader:
    """Create test data loader."""
    dataset = SegmentationDataset(
        data_dir,
        split="test",
        target_size=(image_size, image_size),
        create_synthetic=True,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--methods", nargs="+",
        choices=["classical", "unet", "maskrcnn"],
        help="Methods to evaluate (default: all)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=100,
        help="Maximum samples per method"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--image-size", type=int, default=512,
        help="Input image size"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Evaluation device"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to U-Net checkpoint"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main evaluation entry point."""
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logger.info("Starting evaluation pipeline")
    
    # Create models
    models = create_models()
    
    # Load checkpoint if specified
    if args.checkpoint and "unet" in models:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                models["unet"].load_state_dict(checkpoint["model_state_dict"])
            else:
                models["unet"].load_state_dict(checkpoint)
    
    # Create data loader
    test_loader = create_test_loader(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )
    logger.info(f"Test dataset: {len(test_loader.dataset)} samples")
    
    # Run evaluation
    evaluator = ModelEvaluator(models, args.device)
    results = evaluator.run_evaluation(
        test_loader,
        methods=args.methods,
        max_samples=args.max_samples,
    )
    
    # Print results
    ResultsReporter.print_results(results)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")
    
    logger.info("Evaluation completed")


# Backwards compatibility
SegmentationEvaluator = ModelEvaluator


if __name__ == "__main__":
    main()
