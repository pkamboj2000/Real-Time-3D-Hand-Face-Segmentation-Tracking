"""
Model optimization utilities for real-time performance.

This module provides tools for model optimization including ONNX export,
quantization, pruning, and hardware acceleration setup.
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple, Union
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """
    Comprehensive model optimization for real-time inference.
    """
    
    def __init__(self, target_device: str = 'cpu'):
        """
        Initialize model optimizer.
        
        Args:
            target_device: Target device for optimization ('cpu', 'cuda', 'tensorrt')
        """
        self.target_device = target_device
        self.optimization_history = []
        
    def export_to_onnx(self, model: nn.Module, input_shape: Tuple[int, ...], 
                      output_path: Path, dynamic_axes: Optional[Dict[str, Any]] = None) -> bool:
        """
        Export PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model to export
            input_shape: Input tensor shape
            output_path: Path to save ONNX model
            dynamic_axes: Dynamic axes for variable input sizes
            
        Returns:
            Success status
        """
        try:
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(input_shape)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes
            )
            
            # Verify exported model
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            
            logger.info(f"Successfully exported model to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False
    
    def quantize_model(self, model: nn.Module, calibration_data: torch.Tensor,
                      quantization_type: str = 'dynamic') -> nn.Module:
        """
        Quantize model for faster inference.
        
        Args:
            model: Model to quantize
            calibration_data: Data for calibration
            quantization_type: Type of quantization ('dynamic', 'static')
            
        Returns:
            Quantized model
        """
        try:
            if quantization_type == 'dynamic':
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                )
            else:
                # Static quantization
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                model_prepared = torch.quantization.prepare(model)
                
                # Calibration
                model_prepared.eval()
                with torch.no_grad():
                    for data in calibration_data:
                        model_prepared(data)
                
                quantized_model = torch.quantization.convert(model_prepared)
            
            logger.info(f"Model quantized using {quantization_type} quantization")
            return quantized_model
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return model
    
    def prune_model(self, model: nn.Module, pruning_ratio: float = 0.3) -> nn.Module:
        """
        Prune model to reduce size and computation.
        
        Args:
            model: Model to prune
            pruning_ratio: Fraction of weights to prune
            
        Returns:
            Pruned model
        """
        try:
            import torch.nn.utils.prune as prune
            
            # Identify layers to prune
            parameters_to_prune = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    parameters_to_prune.append((module, 'weight'))
            
            # Apply global magnitude-based pruning
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_ratio,
            )
            
            # Make pruning permanent
            for module, param_name in parameters_to_prune:
                prune.remove(module, param_name)
            
            logger.info(f"Model pruned with ratio {pruning_ratio}")
            return model
            
        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            return model
    
    def optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """
        Apply various optimizations for inference.
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimized model
        """
        try:
            # Set to evaluation mode
            model.eval()
            
            # Freeze batch norm layers
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None
            
            # Fuse operations if possible
            try:
                model = torch.jit.script(model)
                model = torch.jit.optimize_for_inference(model)
                logger.info("Applied JIT optimizations")
            except Exception as e:
                logger.warning(f"JIT optimization failed: {e}")
            
            return model
            
        except Exception as e:
            logger.error(f"Inference optimization failed: {e}")
            return model
    
    def benchmark_model(self, model: nn.Module, input_shape: Tuple[int, ...],
                       num_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark model performance.
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape
            num_iterations: Number of iterations for benchmarking
            
        Returns:
            Performance metrics
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Warmup
        dummy_input = torch.randn(input_shape).to(device)
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.time()
                _ = model(dummy_input)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
        
        return {
            'avg_inference_time': np.mean(times),
            'std_inference_time': np.std(times),
            'fps': 1.0 / np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times)
        }


class ONNXRuntimeOptimizer:
    """
    ONNX Runtime optimization utilities.
    """
    
    def __init__(self, providers: Optional[list] = None):
        """
        Initialize ONNX Runtime optimizer.
        
        Args:
            providers: List of execution providers
        """
        if providers is None:
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
        
        self.providers = providers
        
    def create_optimized_session(self, model_path: Path) -> ort.InferenceSession:
        """
        Create optimized ONNX Runtime session.
        
        Args:
            model_path: Path to ONNX model
            
        Returns:
            Optimized inference session
        """
        try:
            # Session options for optimization
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 4
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            # Create session
            session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=self.providers
            )
            
            logger.info(f"Created optimized ONNX session with providers: {self.providers}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to create ONNX session: {e}")
            raise
    
    def benchmark_onnx_model(self, session: ort.InferenceSession,
                           input_shape: Tuple[int, ...], num_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark ONNX model performance.
        
        Args:
            session: ONNX Runtime session
            input_shape: Input tensor shape
            num_iterations: Number of iterations
            
        Returns:
            Performance metrics
        """
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Warmup
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        for _ in range(10):
            _ = session.run(None, {input_name: dummy_input})
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start_time = time.time()
            _ = session.run(None, {input_name: dummy_input})
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            'avg_inference_time': np.mean(times),
            'std_inference_time': np.std(times),
            'fps': 1.0 / np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times)
        }


class RealTimeOptimizer:
    """
    Real-time processing optimizations.
    """
    
    def __init__(self):
        """Initialize real-time optimizer."""
        self.frame_buffer = []
        self.processing_times = []
        
    def setup_threading(self, num_threads: int = 4):
        """
        Setup threading for parallel processing.
        
        Args:
            num_threads: Number of threads to use
        """
        import threading
        
        torch.set_num_threads(num_threads)
        
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.8)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        logger.info(f"Threading setup complete with {num_threads} threads")
    
    def optimize_memory_usage(self):
        """Optimize memory usage for real-time processing."""
        import gc
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Set memory growth
        if torch.cuda.is_available():
            torch.cuda.memory.set_per_process_memory_fraction(0.8)
        
        logger.info("Memory optimization applied")
    
    def adaptive_resolution(self, frame: np.ndarray, target_fps: float,
                          current_fps: float, min_resolution: int = 256,
                          max_resolution: int = 1024) -> np.ndarray:
        """
        Adaptively adjust frame resolution based on performance.
        
        Args:
            frame: Input frame
            target_fps: Target FPS
            current_fps: Current FPS
            min_resolution: Minimum resolution
            max_resolution: Maximum resolution
            
        Returns:
            Resized frame
        """
        height, width = frame.shape[:2]
        current_resolution = min(height, width)
        
        # Calculate adjustment factor
        fps_ratio = current_fps / target_fps
        
        if fps_ratio < 0.8:  # Too slow, reduce resolution
            new_resolution = max(int(current_resolution * 0.9), min_resolution)
        elif fps_ratio > 1.2:  # Too fast, can increase resolution
            new_resolution = min(int(current_resolution * 1.1), max_resolution)
        else:
            new_resolution = current_resolution
        
        # Resize if needed
        if new_resolution != current_resolution:
            scale_factor = new_resolution / current_resolution
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            
            frame = cv2.resize(frame, (new_width, new_height))
            logger.debug(f"Adaptive resolution: {current_resolution} -> {new_resolution}")
        
        return frame
    
    def frame_skipping(self, target_fps: float, current_fps: float) -> bool:
        """
        Determine if frame should be skipped for performance.
        
        Args:
            target_fps: Target FPS
            current_fps: Current FPS
            
        Returns:
            Whether to skip the frame
        """
        if current_fps < target_fps * 0.7:
            # Skip frames when performance is poor
            skip_probability = 1.0 - (current_fps / target_fps)
            return np.random.random() < skip_probability
        
        return False


def optimize_model_pipeline(model: nn.Module, input_shape: Tuple[int, ...],
                          output_dir: Path, optimization_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete model optimization pipeline.
    
    Args:
        model: Model to optimize
        input_shape: Input tensor shape
        output_dir: Directory to save optimized models
        optimization_config: Optimization configuration
        
    Returns:
        Optimization results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    optimizer = ModelOptimizer()
    results = {}
    
    # Original model benchmark
    original_metrics = optimizer.benchmark_model(model, input_shape)
    results['original'] = original_metrics
    
    # Quantization
    if optimization_config.get('quantize', False):
        try:
            dummy_data = [torch.randn(input_shape) for _ in range(10)]
            quantized_model = optimizer.quantize_model(model, dummy_data)
            quantized_metrics = optimizer.benchmark_model(quantized_model, input_shape)
            results['quantized'] = quantized_metrics
            
            # Save quantized model
            torch.save(quantized_model.state_dict(), output_dir / 'quantized_model.pth')
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
    
    # Pruning
    if optimization_config.get('prune', False):
        try:
            pruned_model = optimizer.prune_model(model.copy() if hasattr(model, 'copy') else model)
            pruned_metrics = optimizer.benchmark_model(pruned_model, input_shape)
            results['pruned'] = pruned_metrics
            
            # Save pruned model
            torch.save(pruned_model.state_dict(), output_dir / 'pruned_model.pth')
            
        except Exception as e:
            logger.error(f"Pruning failed: {e}")
    
    # ONNX export
    if optimization_config.get('export_onnx', False):
        try:
            onnx_path = output_dir / 'model.onnx'
            success = optimizer.export_to_onnx(model, input_shape, onnx_path)
            
            if success:
                # Benchmark ONNX model
                onnx_optimizer = ONNXRuntimeOptimizer()
                session = onnx_optimizer.create_optimized_session(onnx_path)
                onnx_metrics = onnx_optimizer.benchmark_onnx_model(session, input_shape)
                results['onnx'] = onnx_metrics
                
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
    
    # Inference optimization
    try:
        optimized_model = optimizer.optimize_for_inference(model)
        optimized_metrics = optimizer.benchmark_model(optimized_model, input_shape)
        results['inference_optimized'] = optimized_metrics
        
        # Save optimized model
        torch.save(optimized_model.state_dict(), output_dir / 'inference_optimized_model.pth')
        
    except Exception as e:
        logger.error(f"Inference optimization failed: {e}")
    
    # Summary
    logger.info("Optimization Pipeline Results:")
    for optimization_type, metrics in results.items():
        fps = metrics.get('fps', 0)
        avg_time = metrics.get('avg_inference_time', 0) * 1000  # Convert to ms
        logger.info(f"  {optimization_type}: {fps:.1f} FPS ({avg_time:.2f}ms)")
    
    return results
