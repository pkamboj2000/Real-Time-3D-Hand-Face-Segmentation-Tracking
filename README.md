
# Real-Time 3D Hand/Face Segmentation & Tracking

This project provides a real-time segmentation and tracking system for hands and faces in AR/VR, enabling natural interaction in mixed reality environments.

## Datasets Supported
- Hands: EgoHands, FreiHAND
- Faces: CelebAMask-HQ, WFLW
- Webcam/live capture for demo

## Methods
- **Classical Baseline:** Skin detection, optical flow, contour tracking
- **Deep Learning:** U-Net and Mask R-CNN for segmentation
- **Depth Estimation:** MiDaS (monocular), with placeholder for stereo
- **Real-time Inference:** ONNX export and quantization scripts included

## Evaluation
- Segmentation metrics: IoU, Dice coefficient
- Tracking metrics: FPS, latency, stability
- 3D alignment: Depth map overlay, 3D mesh/landmark visualization (see `scripts/visualize_3d_mesh.py`)

## Demo/Tools
- Live webcam demo with overlays and bounding boxes
- Switch between classical, U-Net, Mask R-CNN in real time
- FPS and IoU displayed in demo
- Depth map and 3D mesh visualization

## Quick Start

1. **Install Dependencies**
  ```bash
  pip install -r requirements.txt
  ```
2. **Run Live Demo**
  ```bash
  python live_demo.py --method classical
  python live_demo.py --method unet --show-depth
  python live_demo.py --method maskrcnn
  ```
3. **Export Models to ONNX/Quantized**
  ```bash
  python scripts/export_onnx_quantized.py --model unet --quantize
  python scripts/export_onnx_quantized.py --model maskrcnn
  ```
4. **Visualize 3D Mesh Overlay**
  ```bash
  python scripts/visualize_3d_mesh.py
  ```
5. **Train Models**
  ```bash
  python train_model.py --model unet --data-dir /path/to/data --epochs 50
  ```
6. **Evaluate Models**
  ```bash
  python evaluate_model.py --data-dir /path/to/data
  ```

---

For more details, see the code and comments in each script. All code and documentation are written in a clear, human style.

## Goal

Build a real-time system that can:
- Segment hands and faces in live video streams
- Track movements with low latency
- Provide 3D depth information for AR/VR alignment
- Run efficiently on various hardware configurations

## Features

### Core Methods
- **Classical Baseline**: Skin detection + optical flow + contour tracking
- **Deep Learning**: U-Net and Mask R-CNN for segmentation  
- **Depth Estimation**: MiDaS integration for 3D alignment
- **Real-time Optimization**: ONNX export and quantization support

### Performance
- Real-time inference (30+ FPS)
- Low latency processing
- Hardware-aware deployment options
- Full evaluation metrics

## Quick Start

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Run Live Demo**
```bash
python live_demo.py --method classical
```

3. **Train Models**
```bash
python train_model.py --data-dir data --epochs 20
```

4. **Evaluate Performance**
```bash
python evaluate_model.py --data-dir data --max-samples 100
```

## Usage

### Live Demo Options
```bash
# Classical computer vision approach
python live_demo.py --method classical

# Deep learning approach  
python live_demo.py --method unet

# With depth estimation
python live_demo.py --method unet --enable-depth

# Save output video
python live_demo.py --method classical --save-video output.mp4
```

### Training Custom Models
```bash
# Train U-Net on custom data
python train_model.py --model unet --data-dir /path/to/data --epochs 50

# Train with specific image size
python train_model.py --model unet --image-size 512 --batch-size 8

# Resume training from checkpoint
python train_model.py --model unet --resume checkpoints/best_model.pth
```

### Evaluation and Benchmarking
```bash
# Comprehensive evaluation
python evaluate_model.py --data-dir test_data --max-samples 200

# Speed benchmarking
python evaluate_model.py --benchmark-only --device cuda

# Generate detailed report
python evaluate_model.py --data-dir test_data --output-report results.json
```

## Architecture

### Classical Pipeline
1. **Skin Detection**: Multi-color space thresholding (HSV, YCrCb)
2. **Optical Flow**: Lucas-Kanade tracking for motion estimation
3. **Contour Analysis**: Hand/face shape detection and tracking

### Deep Learning Pipeline
1. **U-Net Segmentation**: Encoder-decoder with skip connections
2. **Mask R-CNN**: Instance segmentation with bounding boxes
3. **Depth Integration**: MiDaS monocular depth estimation

### 3D Alignment
- Real-time depth map generation
- Hand/face pose estimation
- AR/VR coordinate system alignment

## Datasets Supported

- **EgoHands**: First-person hand segmentation
- **FreiHAND**: 3D hand pose dataset
- **CelebAMask-HQ**: High-quality face segmentation
- **WFLW**: Facial landmark detection
- **Custom datasets**: Your own annotated data

## Performance Metrics

### Segmentation Quality
- **IoU (Intersection over Union)**: Pixel-level accuracy
- **Dice Coefficient**: Overlap similarity measure
- **Pixel Accuracy**: Overall correctness

### Real-time Performance
- **FPS**: Frames per second processing rate
- **Latency**: End-to-end processing time
- **Stability**: Temporal consistency tracking

### 3D Alignment
- **Depth Accuracy**: Distance estimation quality
- **Pose Consistency**: 3D tracking stability
- **AR Registration**: Mixed reality alignment precision

## Hardware Requirements

### Minimum
- CPU: Intel i5 or AMD Ryzen 5
- RAM: 8GB
- GPU: Optional (CPU inference supported)

### Recommended
- CPU: Intel i7 or AMD Ryzen 7
- RAM: 16GB
- GPU: NVIDIA GTX 1060 or better
- Storage: SSD for faster data loading

### Optimization Features
- **ONNX Export**: Cross-platform deployment
- **Model Quantization**: Reduced memory usage
- **Multi-threading**: Parallel processing support
- **GPU Acceleration**: CUDA and OpenCL support

## Technical Implementation

### Core Models (`core_models.py`)
- U-Net architecture with attention mechanisms
- Mask R-CNN with Feature Pyramid Networks
- Classical computer vision algorithms
- Depth estimation integration

### Training Framework (`train_model.py`)
- PyTorch-based training loops
- Data augmentation and preprocessing
- Checkpoint management and resuming
- Distributed training support

### Live Demo (`live_demo.py`)
- Real-time webcam processing
- Multiple algorithm switching
- Performance monitoring overlay
- Video recording capabilities

### Evaluation Suite (`evaluate_model.py`)
- Detailed metric calculation
- Benchmark testing framework
- Performance profiling tools
- Report generation utilities

## Skills Demonstrated

### Computer Vision
- Classical image processing techniques
- Deep learning for segmentation
- Real-time video analysis
- 3D computer vision and depth estimation

### Software Engineering
- Modular, maintainable code architecture
- Performance optimization and profiling
- Hardware-aware deployment strategies
- Comprehensive testing and evaluation

### AR/VR Development
- Real-time processing pipelines
- Low-latency system design
- 3D coordinate system alignment
- Mixed reality integration principles

## Future Enhancements

- Multi-hand tracking and gesture recognition
- Facial expression analysis and emotion detection
- Hand-object interaction modeling
- Advanced 3D mesh reconstruction
- Mobile deployment (iOS/Android)
- WebRTC integration for remote processing


