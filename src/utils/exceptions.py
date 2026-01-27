"""
Custom exception hierarchy for the segmentation pipeline.

Provides specialized exceptions for different failure modes with
contextual information to aid in debugging and error handling.
"""

from __future__ import annotations

from typing import Any, Optional


class SegmentationBaseError(Exception):
    """
    Base exception for all segmentation pipeline errors.
    
    Provides consistent error message formatting and optional
    context information for debugging.
    """
    
    def __init__(
        self,
        message: str,
        *,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize the exception with message and optional context.
        
        Args:
            message: Human-readable error description
            context: Additional context information
            cause: Original exception that caused this error
        """
        self.context = context or {}
        self.cause = cause
        
        full_message = message
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            full_message = f"{message} [{context_str}]"
        
        if cause:
            full_message = f"{full_message} (caused by: {cause})"
        
        super().__init__(full_message)


class ModelError(SegmentationBaseError):
    """Errors related to model operations."""
    pass


class ModelLoadError(ModelError):
    """Failed to load model weights or configuration."""
    
    def __init__(
        self,
        model_name: str,
        path: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Failed to load model '{model_name}'",
            context={"model": model_name, "path": path},
            cause=cause
        )


class ModelInferenceError(ModelError):
    """Error during model inference/prediction."""
    
    def __init__(
        self,
        model_name: str,
        input_shape: Optional[tuple] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Inference failed for model '{model_name}'",
            context={"model": model_name, "input_shape": input_shape},
            cause=cause
        )


class DataError(SegmentationBaseError):
    """Errors related to data loading and processing."""
    pass


class DataLoadError(DataError):
    """Failed to load data from disk."""
    
    def __init__(
        self,
        file_path: str,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Failed to load data from '{file_path}'",
            context={"path": file_path},
            cause=cause
        )


class InvalidDataError(DataError):
    """Data validation failed."""
    
    def __init__(
        self,
        message: str,
        expected: Optional[Any] = None,
        actual: Optional[Any] = None
    ):
        super().__init__(
            message,
            context={"expected": expected, "actual": actual}
        )


class ImageProcessingError(SegmentationBaseError):
    """Errors during image processing operations."""
    
    def __init__(
        self,
        operation: str,
        image_shape: Optional[tuple] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Image processing failed during '{operation}'",
            context={"operation": operation, "shape": image_shape},
            cause=cause
        )


class DeviceError(SegmentationBaseError):
    """Errors related to hardware devices (GPU, camera, etc.)."""
    pass


class CameraError(DeviceError):
    """Camera-related errors."""
    
    def __init__(
        self,
        camera_id: int,
        operation: str = "access",
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Camera error: failed to {operation} camera {camera_id}",
            context={"camera_id": camera_id, "operation": operation},
            cause=cause
        )


class GPUError(DeviceError):
    """GPU-related errors."""
    
    def __init__(
        self,
        message: str,
        device_id: Optional[int] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message,
            context={"device_id": device_id},
            cause=cause
        )


class ConfigurationError(SegmentationBaseError):
    """Configuration-related errors."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message,
            context={"config_key": config_key},
            cause=cause
        )


class TrainingError(SegmentationBaseError):
    """Errors during model training."""
    
    def __init__(
        self,
        message: str,
        epoch: Optional[int] = None,
        batch: Optional[int] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message,
            context={"epoch": epoch, "batch": batch},
            cause=cause
        )


class CheckpointError(SegmentationBaseError):
    """Errors related to checkpoint save/load operations."""
    
    def __init__(
        self,
        operation: str,
        path: str,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Checkpoint {operation} failed at '{path}'",
            context={"operation": operation, "path": path},
            cause=cause
        )
