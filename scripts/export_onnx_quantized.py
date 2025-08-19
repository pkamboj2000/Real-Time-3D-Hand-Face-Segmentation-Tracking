"""
Export U-Net and Mask R-CNN models to ONNX and quantized ONNX for real-time inference.
This script is written in a clear, human style.
"""

import torch
import torch.quantization
import argparse
from core_models import UNet, MaskRCNN
import os


def export_onnx(model, dummy_input, out_path, quantize=False):
    if quantize:
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Exported {'quantized ' if quantize else ''}model to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX/quantized ONNX.")
    parser.add_argument('--model', type=str, choices=['unet', 'maskrcnn'], required=True)
    parser.add_argument('--quantize', action='store_true', help='Export quantized model')
    parser.add_argument('--output', type=str, default=None, help='Output ONNX file path')
    args = parser.parse_args()

    if args.model == 'unet':
        model = UNet(in_channels=3, out_channels=3)
        dummy_input = torch.randn(1, 3, 256, 256)
        out_path = args.output or ('unet_quantized.onnx' if args.quantize else 'unet.onnx')
    else:
        model = MaskRCNN(num_classes=3).model
        dummy_input = torch.randn(1, 3, 256, 256)
        out_path = args.output or ('maskrcnn_quantized.onnx' if args.quantize else 'maskrcnn.onnx')

    export_onnx(model, dummy_input, out_path, quantize=args.quantize)

if __name__ == "__main__":
    main()
