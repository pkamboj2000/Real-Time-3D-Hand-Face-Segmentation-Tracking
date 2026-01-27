"""
Configuration module for Real-Time 3D Hand/Face Segmentation & Tracking.

This module provides typed configuration classes using dataclasses for
type safety, validation, and IDE support. Configurations can be loaded
from YAML files or environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import json


class DeviceType(Enum):
    """Supported compute device types."""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class SegmentationMethod(Enum):
    """Available segmentation methods."""
    CLASSICAL = "classical"
    UNET = "unet"
    MASK_RCNN = "maskrcnn"


class QuantizationMethod(Enum):
    """Model quantization methods."""
    DYNAMIC = "dynamic"
    STATIC = "static"


@dataclass(frozen=True)
class PathConfig:
    """Immutable path configuration."""
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"
    
    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"
    
    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"
    
    @property
    def cache_dir(self) -> Path:
        return self.project_root / ".cache"
    
    @property
    def checkpoints_dir(self) -> Path:
        return self.project_root / "checkpoints"
    
    def ensure_directories(self) -> None:
        """Create all required directories."""
        for dir_path in [
            self.data_dir,
            self.models_dir,
            self.logs_dir,
            self.cache_dir,
            self.checkpoints_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetConfig:
    """Configuration for a dataset source."""
    url: str
    local_path: Path
    annotation_format: str
    classes: list[str]


@dataclass
class ModelConfig:
    """Base configuration for models."""
    input_size: tuple[int, int] = (512, 512)
    batch_size: int = 4
    learning_rate: float = 1e-4
    epochs: int = 50
    num_workers: int = 4


@dataclass
class UNetConfig(ModelConfig):
    """U-Net specific configuration."""
    encoder_name: str = "resnet50"
    encoder_weights: str = "imagenet"
    num_classes: int = 3
    in_channels: int = 3
    use_attention: bool = False
    dropout_rate: float = 0.1
    
    def __post_init__(self):
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")


@dataclass
class MaskRCNNConfig(ModelConfig):
    """Mask R-CNN specific configuration."""
    backbone: str = "resnet50"
    num_classes: int = 3
    min_size: int = 512
    max_size: int = 1024
    score_threshold: float = 0.5
    nms_threshold: float = 0.5


@dataclass
class DepthEstimationConfig:
    """Depth estimation configuration."""
    model_name: str = "MiDaS_small"
    input_size: tuple[int, int] = (384, 384)
    enable_stereo: bool = False
    baseline_meters: float = 0.065


@dataclass
class ClassicalConfig:
    """Classical CV pipeline configuration."""
    # HSV skin detection thresholds
    hsv_lower: tuple[int, int, int] = (0, 20, 70)
    hsv_upper: tuple[int, int, int] = (20, 255, 255)
    
    # YCrCb skin detection thresholds
    ycrcb_lower: tuple[int, int, int] = (0, 133, 77)
    ycrcb_upper: tuple[int, int, int] = (255, 173, 127)
    
    # Optical flow parameters
    max_corners: int = 100
    quality_level: float = 0.3
    min_distance: int = 7
    block_size: int = 7
    
    # Contour filtering
    min_contour_area: int = 1000
    hand_area_threshold: int = 5000


@dataclass
class CameraConfig:
    """Camera capture configuration."""
    device_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 1
    flip_horizontal: bool = True


@dataclass
class OptimizationConfig:
    """Model optimization configuration."""
    quantization_enabled: bool = True
    quantization_method: QuantizationMethod = QuantizationMethod.DYNAMIC
    onnx_export_enabled: bool = True
    onnx_opset_version: int = 11
    tensorrt_enabled: bool = False
    mixed_precision: bool = True


@dataclass
class VisualizationConfig:
    """Visualization settings."""
    hand_color: tuple[int, int, int] = (0, 255, 0)
    face_color: tuple[int, int, int] = (255, 0, 0)
    background_color: tuple[int, int, int] = (0, 0, 0)
    mask_alpha: float = 0.5
    bbox_thickness: int = 2
    show_fps: bool = True
    show_latency: bool = True


@dataclass
class TrainingConfig:
    """Training-specific configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    save_every_n_epochs: int = 5
    early_stopping_patience: int = 10
    gradient_clip_value: Optional[float] = 1.0
    use_amp: bool = True
    log_every_n_steps: int = 10
    validate_every_n_epochs: int = 1


@dataclass
class AppConfig:
    """Main application configuration."""
    paths: PathConfig = field(default_factory=PathConfig)
    unet: UNetConfig = field(default_factory=UNetConfig)
    maskrcnn: MaskRCNNConfig = field(default_factory=MaskRCNNConfig)
    depth: DepthEstimationConfig = field(default_factory=DepthEstimationConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: DeviceType = DeviceType.AUTO
    debug: bool = False
    
    def __post_init__(self):
        self.paths.ensure_directories()
    
    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)
    
    @classmethod
    def from_json(cls, path: Path) -> "AppConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        """Create config from dictionary."""
        # This would need proper recursive instantiation in production
        return cls()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)


# Global configuration instance (lazy loaded)
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def set_config(config: AppConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


# Legacy compatibility - path constants
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / ".cache"

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, CACHE_DIR]:
    dir_path.mkdir(exist_ok=True)

# Legacy dataset configurations (kept for backwards compatibility)
DATASETS = {
    "hands": {
        "egohands": {
            "url": "http://vision.soic.indiana.edu/projects/egohands/",
            "local_path": DATA_DIR / "raw" / "egohands",
            "annotation_format": "mat",
            "classes": ["hand"]
        },
        "freihand": {
            "url": "https://lmb.informatik.uni-freiburg.de/projects/freihand/",
            "local_path": DATA_DIR / "raw" / "freihand",
            "annotation_format": "json",
            "classes": ["hand"]
        }
    },
    "faces": {
        "celebamask_hq": {
            "url": "https://github.com/switchablenorms/CelebAMask-HQ",
            "local_path": DATA_DIR / "raw" / "celebamask_hq",
            "annotation_format": "png",
            "classes": ["face", "skin", "nose", "eye_g", "l_eye", "r_eye", 
                       "l_brow", "r_brow", "l_ear", "r_ear", "mouth", 
                       "u_lip", "l_lip", "hair", "hat", "ear_r", "neck_l"]
        },
        "wflw": {
            "url": "https://wywu.github.io/projects/LAB/WFLW.html",
            "local_path": DATA_DIR / "raw" / "wflw",
            "annotation_format": "txt",
            "classes": ["face"]
        }
    }
}

# Model configurations
MODEL_CONFIGS = {
    "unet": {
        "encoder_name": "resnet50",
        "encoder_weights": "imagenet",
        "classes": 2,  # background + hand/face
        "activation": "sigmoid",
        "input_size": (512, 512),
        "batch_size": 8,
        "learning_rate": 1e-4,
        "epochs": 100
    },
    "mask_rcnn": {
        "backbone": "resnet50",
        "num_classes": 3,  # background + hand + face
        "min_size": 512,
        "max_size": 1024,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "epochs": 50
    },
    "depth_estimation": {
        "midas_model": "MiDaS_small",
        "input_size": (384, 384),
        "enable_stereo": True,
        "baseline": 0.065  # meters, typical stereo baseline
    }
}

# Classical CV configurations
CLASSICAL_CONFIG = {
    "skin_detection": {
        "color_spaces": ["HSV", "YCrCb", "LAB"],
        "hsv_lower": [0, 48, 80],
        "hsv_upper": [20, 255, 255],
        "ycrcb_lower": [0, 133, 77],
        "ycrcb_upper": [255, 173, 127]
    },
    "optical_flow": {
        "method": "lucas_kanade",
        "max_corners": 100,
        "quality_level": 0.3,
        "min_distance": 7,
        "block_size": 7
    },
    "contour_tracking": {
        "min_contour_area": 1000,
        "approximation_epsilon": 0.02,
        "smoothing_factor": 0.7
    }
}

# Optimization configurations
OPTIMIZATION_CONFIG = {
    "quantization": {
        "enabled": True,
        "method": "dynamic",  # "dynamic" or "static"
        "calibration_samples": 100
    },
    "onnx": {
        "export_enabled": True,
        "opset_version": 11,
        "optimize": True,
        "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]
    },
    "coreml": {
        "export_enabled": True,
        "compute_units": "ALL",  # "ALL", "CPU_ONLY", "CPU_AND_GPU"
        "minimum_deployment_target": "iOS15"
    },
    "tensorrt": {
        "enabled": False,  # Enable if TensorRT is available
        "precision": "fp16",
        "max_batch_size": 8
    }
}

# Real-time processing configurations
REALTIME_CONFIG = {
    "target_fps": 30,
    "max_latency_ms": 33,  # 1000ms / 30fps
    "camera": {
        "device_id": 0,
        "width": 1280,
        "height": 720,
        "fps": 30
    },
    "buffer_size": 3,
    "threading": {
        "use_multiprocessing": True,
        "num_workers": 4
    }
}

# Evaluation metrics
EVALUATION_METRICS = {
    "segmentation": ["iou", "dice", "pixel_accuracy", "mean_iou"],
    "tracking": ["fps", "latency", "stability_score"],
    "3d_alignment": ["mesh_error", "keypoint_error", "pose_accuracy"]
}

# Visualization settings
VISUALIZATION_CONFIG = {
    "colors": {
        "hand": (0, 255, 0),     # Green
        "face": (255, 0, 0),     # Red
        "background": (0, 0, 0)   # Black
    },
    "3d_viewer": {
        "mesh_alpha": 0.7,
        "show_wireframe": True,
        "lighting": "three_lights",
        "background_color": "white"
    },
    "overlay": {
        "mask_alpha": 0.5,
        "bbox_thickness": 2,
        "keypoint_radius": 3
    }
}

# Hardware configurations
HARDWARE_CONFIG = {
    "device": "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu",
    "gpu_memory_fraction": 0.8,
    "mixed_precision": True,
    "compile_model": True  # PyTorch 2.0 compilation
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_dir": LOGS_DIR,
    "wandb": {
        "enabled": True,
        "project": "hand-face-segmentation-3d",
        "entity": "your-wandb-username"
    }
}

# Export all configurations
__all__ = [
    "PROJECT_ROOT", "DATA_DIR", "MODELS_DIR", "LOGS_DIR", "CACHE_DIR",
    "DATASETS", "MODEL_CONFIGS", "CLASSICAL_CONFIG", "OPTIMIZATION_CONFIG",
    "REALTIME_CONFIG", "EVALUATION_METRICS", "VISUALIZATION_CONFIG",
    "HARDWARE_CONFIG", "LOGGING_CONFIG"
]
