# Real-Time 3D Hand/Face Segmentation & Tracking

Real-time segmentation and tracking system for hands and faces in AR/VR applications.

## Features

- Classical computer vision methods (skin detection, optical flow)
- Deep learning models (U-Net, Mask R-CNN)
- 3D depth estimation with MiDaS
- Real-time processing and optimization
- Live webcam demo with performance monitoring

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run live demo
python live_demo.py --method classical

# Train models
python train_model.py --data-dir data --epochs 20

# Evaluate performance
python evaluate_model.py --data-dir data
```

## Requirements

- Python 3.8+
- OpenCV, PyTorch, NumPy
- CUDA support recommended for training

## Architecture

The system implements both classical and deep learning approaches:

### Classical Methods
- Multi-color space skin detection (HSV, YCrCb, RGB)
- Lucas-Kanade optical flow tracking
- Contour analysis and gesture recognition

### Deep Learning
- U-Net with attention mechanisms for segmentation
- Mask R-CNN for instance segmentation
- MiDaS for monocular depth estimation

### Real-time Optimization
- ONNX model export for deployment
- Quantization support for edge devices
- Multi-threading for parallel processing

## Evaluation Metrics

- **Segmentation**: IoU, Dice coefficient, pixel accuracy
- **Performance**: FPS, latency, memory usage
- **Tracking**: Stability, continuity, accuracy

## Skills Demonstrated

- Computer vision and image processing
- Deep learning model development
- Real-time system optimization
- AR/VR pipeline integration
- Performance benchmarking and evaluation

