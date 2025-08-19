"""
Evaluation script for hand/face segmentation models.
Provides comprehensive metrics including IoU, Dice, FPS, and tracking performance.
"""

import torch
import cv2
import numpy as np
import time
import argparse
from pathlib import Path
import json
from collections import defaultdict
from core_models import create_models
from train_model import HandFaceDataset


class SegmentationEvaluator:
	"""
	Comprehensive evaluator for segmentation models.
	"""
	def __init__(self, models, device='auto'):
		"""
		Initialize evaluator.
		Args:
			models: Dictionary of models to evaluate
			device: Evaluation device
		"""
		if device == 'auto':
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		else:
			self.device = torch.device(device)
		self.models = models
		if 'unet' in self.models:
			self.models['unet'] = self.models['unet'].to(self.device)
			self.models['unet'].eval()
		if 'maskrcnn' in self.models:
			self.models['maskrcnn'] = self.models['maskrcnn'].to(self.device)
			self.models['maskrcnn'].eval()

	def calculate_metrics(self, predictions, targets, num_classes=3):
		"""
		Calculate segmentation metrics.
		Args:
			predictions: Model predictions (B, H, W) or (B, C, H, W)
			targets: Ground truth (B, H, W)
			num_classes: Number of classes
		Returns:
			Dictionary with metrics
		"""
		if len(predictions.shape) == 4:
			pred_labels = torch.argmax(predictions, dim=1)
		else:
			pred_labels = predictions
		correct = (pred_labels == targets).sum().float()
		total = targets.numel()
		pixel_accuracy = (correct / total).item()
		ious = []
		dice_scores = []
		for class_id in range(num_classes):
			pred_mask = (pred_labels == class_id)
			target_mask = (targets == class_id)
			intersection = (pred_mask & target_mask).sum().float()
			union = (pred_mask | target_mask).sum().float()
			if union > 0:
				iou = (intersection / union).item()
				ious.append(iou)
				dice = (2 * intersection / (pred_mask.sum() + target_mask.sum())).item()
				dice_scores.append(dice)
			else:
				ious.append(1.0 if intersection == 0 else 0.0)
				dice_scores.append(1.0 if intersection == 0 else 0.0)
		return {
			'pixel_accuracy': pixel_accuracy,
			'mean_iou': np.mean(ious),
			'per_class_iou': ious,
			'mean_dice': np.mean(dice_scores),
			'per_class_dice': dice_scores
		}

	def evaluate_method(self, method_name, test_loader, max_samples=None):
		"""
		Evaluate a specific method.
		Args:
			method_name: Name of method to evaluate
			test_loader: Test data loader
			max_samples: Maximum number of samples to evaluate
		Returns:
			Evaluation results
		"""
		print(f"Evaluating {method_name}...")
		all_metrics = defaultdict(list)
		processing_times = []
		sample_count = 0
		for batch_idx, (images, masks) in enumerate(test_loader):
			if max_samples and sample_count >= max_samples:
				break
			batch_size = images.shape[0]
			for i in range(batch_size):
				if max_samples and sample_count >= max_samples:
					break
				image = images[i]
				mask = masks[i]
				if isinstance(image, torch.Tensor):
					image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
					image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
				else:
					image_np = image
				start_time = time.time()
				try:
					if method_name == 'classical':
						pred_mask = self.models['classical'].segment_hands_faces(image_np)
						pred_mask = torch.from_numpy(pred_mask)
					elif method_name == 'unet':
						image_tensor = image.unsqueeze(0).to(self.device)
						with torch.no_grad():
							output = self.models['unet'](image_tensor)
							pred_mask = torch.argmax(output, dim=1).squeeze().cpu()
					elif method_name == 'maskrcnn':
						predictions = self.models['maskrcnn'].predict(image_np)
						h, w = image_np.shape[:2]
						pred_mask = torch.zeros((h, w), dtype=torch.long)
						if 'masks' in predictions and len(predictions['masks']) > 0:
							masks_np = predictions['masks'].cpu().numpy()
							labels = predictions['labels'].cpu().numpy()
							scores = predictions['scores'].cpu().numpy()
							for mask_np, label, score in zip(masks_np, labels, scores):
								if score > 0.5:
									pred_mask[mask_np[0] > 0.5] = label
					else:
						raise ValueError(f"Unknown method: {method_name}")
				except Exception as e:
					print(f"Error processing sample {sample_count}: {e}")
					pred_mask = torch.zeros_like(mask)
				processing_time = time.time() - start_time
				processing_times.append(processing_time)
				if pred_mask.shape != mask.shape:
					pred_mask = torch.nn.functional.interpolate(
						pred_mask.unsqueeze(0).unsqueeze(0).float(),
						size=mask.shape,
						mode='nearest'
					).squeeze().long()
				metrics = self.calculate_metrics(
					pred_mask.unsqueeze(0), 
					mask.unsqueeze(0)
				)
				for key, value in metrics.items():
					all_metrics[key].append(value)
				sample_count += 1
				if sample_count % 10 == 0:
					print(f"  Processed {sample_count} samples...")
		results = {
			'num_samples': sample_count,
			'avg_processing_time': np.mean(processing_times),
			'fps': 1.0 / np.mean(processing_times),
			'latency_ms': np.mean(processing_times) * 1000,
		}
		for key, values in all_metrics.items():
			if key.startswith('per_class'):
				per_class_avg = np.mean(values, axis=0)
				results[key] = per_class_avg.tolist()
			else:
				results[key] = np.mean(values)
		return results

	def run_evaluation(self, test_loader, methods=None, max_samples=None):
		"""
		Run evaluation on multiple methods.
		Args:
			test_loader: Test data loader
			methods: List of methods to evaluate (None for all)
			max_samples: Maximum samples per method
		Returns:
			Complete evaluation results
		"""
		if methods is None:
			methods = list(self.models.keys())
		results = {
			'dataset_info': {
				'num_samples': len(test_loader.dataset),
				'batch_size': test_loader.batch_size
			},
			'methods': {}
		}
		for method in methods:
			if method in self.models:
				try:
					method_results = self.evaluate_method(method, test_loader, max_samples)
					results['methods'][method] = method_results
				except Exception as e:
					print(f"Error evaluating {method}: {e}")
					results['methods'][method] = {'error': str(e)}
		return results

def create_test_loader(data_dir, batch_size=1, image_size=512):
	"""
	Create test data loader.
	"""
	dataset = HandFaceDataset(data_dir, image_size=image_size, mode='test')
	loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
	return loader

def print_results(results):
	"""
	Print evaluation results in a formatted way.
	Args:
		results: Evaluation results dictionary
	"""
	print("\n" + "="*60)
	print("EVALUATION RESULTS")
	print("="*60)
	print(f"\nDataset Info:")
	dataset_info = results.get('dataset_info', {})
	print(f"  Samples: {dataset_info.get('num_samples', 'N/A')}")
	print(f"  Batch Size: {dataset_info.get('batch_size', 'N/A')}")
	for method_name, method_results in results.get('methods', {}).items():
		print(f"\n{method_name.upper()} Results:")
		print("-" * 40)
		if 'error' in method_results:
			print(f"  Error: {method_results['error']}")
			continue
		print(f"  Samples Evaluated: {method_results.get('num_samples', 'N/A')}")
		print(f"  FPS: {method_results.get('fps', 0):.2f}")
		print(f"  Latency: {method_results.get('latency_ms', 0):.2f} ms")
		print(f"  Mean IoU: {method_results.get('mean_iou', 0):.4f}")
		print(f"  Mean Dice: {method_results.get('mean_dice', 0):.4f}")
		print(f"  Pixel Accuracy: {method_results.get('pixel_accuracy', 0):.4f}")
		per_class_iou = method_results.get('per_class_iou', [])
		if per_class_iou:
			class_names = ['Background', 'Hand', 'Face']
			print("  Per-class IoU:")
			for i, iou in enumerate(per_class_iou):
				class_name = class_names[i] if i < len(class_names) else f'Class_{i}'
				print(f"    {class_name}: {iou:.4f}")

def main():
	"""Main evaluation function."""
	parser = argparse.ArgumentParser(description='Evaluate Hand/Face Segmentation Models')
	parser.add_argument('--data-dir', type=str, default='data',
					   help='Path to dataset directory')
	parser.add_argument('--methods', nargs='+', 
					   choices=['classical', 'unet', 'maskrcnn'],
					   help='Methods to evaluate (default: all available)')
	parser.add_argument('--max-samples', type=int, default=100,
					   help='Maximum samples to evaluate per method')
	parser.add_argument('--batch-size', type=int, default=1,
					   help='Evaluation batch size')
	parser.add_argument('--device', type=str, default='auto',
					   choices=['auto', 'cpu', 'cuda'],
					   help='Evaluation device')
	parser.add_argument('--image-size', type=int, default=512,
					   help='Input image size')
	parser.add_argument('--output', type=str,
					   help='Output file to save results (JSON)')
	parser.add_argument('--checkpoint', type=str,
					   help='Path to model checkpoint for U-Net')
	args = parser.parse_args()
	models = create_models()
	if args.checkpoint and 'unet' in models:
		print(f"Loading checkpoint: {args.checkpoint}")
		checkpoint = torch.load(args.checkpoint, map_location='cpu')
		if 'model_state_dict' in checkpoint:
			models['unet'].load_state_dict(checkpoint['model_state_dict'])
		else:
			models['unet'].load_state_dict(checkpoint)
	test_loader = create_test_loader(
		args.data_dir, 
		args.batch_size, 
		args.image_size
	)
	if len(test_loader.dataset) == 0:
		print("No test data found. Creating dummy test data...")
		test_loader = create_test_loader(
			args.data_dir, 
			args.batch_size, 
			args.image_size
		)
	print(f"Found {len(test_loader.dataset)} test samples")
	evaluator = SegmentationEvaluator(models, args.device)
	results = evaluator.run_evaluation(
		test_loader, 
		args.methods, 
		args.max_samples
	)
	print_results(results)
	if args.output:
		with open(args.output, 'w') as f:
			json.dump(results, f, indent=2)
		print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
	main()
