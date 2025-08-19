"""
Configuration file for Real-Time 3D Hand/Face Segmentation & Tracking
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / ".cache"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, CACHE_DIR]:
    dir_path.mkdir(exist_ok=True)

# Dataset configurations
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
