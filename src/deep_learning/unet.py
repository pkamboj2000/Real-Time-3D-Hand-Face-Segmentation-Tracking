"""
U-Net implementation for hand and face segmentation.

This module provides a flexible U-Net architecture with various backbone
encoders and decoder configurations optimized for real-time segmentation
of hands and faces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """
    Double convolution block used in U-Net architecture.
    
    Consists of two 3x3 convolutions, each followed by batch normalization
    and ReLU activation. This is the basic building block of U-Net.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 use_batch_norm: bool = True, dropout_rate: float = 0.0):
        """
        Initialize double convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate (0.0 to disable)
        """
        super(DoubleConv, self).__init__()
        
        layers = []
        
        # First convolution
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        
        # Second convolution
        layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        self.double_conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock(nn.Module):
    """
    Downsampling block for U-Net encoder.
    
    Applies max pooling followed by double convolution to reduce
    spatial dimensions while increasing feature depth.
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 use_batch_norm: bool = True, dropout_rate: float = 0.0):
        """
        Initialize downsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate
        """
        super(DownBlock, self).__init__()
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_batch_norm, dropout_rate)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """
    Upsampling block for U-Net decoder.
    
    Performs upsampling via transposed convolution or interpolation,
    concatenates with skip connection, and applies double convolution.
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 use_transpose: bool = True, use_batch_norm: bool = True,
                 dropout_rate: float = 0.0):
        """
        Initialize upsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            use_transpose: Whether to use transpose convolution for upsampling
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate
        """
        super(UpBlock, self).__init__()
        
        if use_transpose:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 
                                       kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', 
                                align_corners=True)
        
        self.conv = DoubleConv(in_channels, out_channels, 
                              use_batch_norm, dropout_rate)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connection.
        
        Args:
            x1: Feature map from decoder path
            x2: Feature map from encoder path (skip connection)
            
        Returns:
            Upsampled and concatenated feature map
        """
        x1 = self.up(x1)
        
        # Handle size mismatches due to padding
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention gate module for U-Net.
    
    Implements attention mechanism to focus on relevant features
    and suppress irrelevant ones during skip connections.
    """
    
    def __init__(self, gate_channels: int, skip_channels: int, 
                 inter_channels: int):
        """
        Initialize attention gate.
        
        Args:
            gate_channels: Number of channels in gating signal
            skip_channels: Number of channels in skip connection
            inter_channels: Number of intermediate channels
        """
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of attention gate.
        
        Args:
            g: Gating signal from decoder
            x: Skip connection from encoder
            
        Returns:
            Attention-weighted skip connection
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Resize gating signal to match skip connection
        g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', 
                          align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation.
    
    Implements the standard U-Net with optional attention gates,
    batch normalization, and dropout for robust segmentation.
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 features: List[int] = [64, 128, 256, 512],
                 use_attention: bool = False, use_batch_norm: bool = True,
                 dropout_rate: float = 0.1, use_transpose: bool = True):
        """
        Initialize U-Net model.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output channels (classes)
            features: List of feature dimensions for each level
            use_attention: Whether to use attention gates
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate for regularization
            use_transpose: Whether to use transpose convolution for upsampling
        """
        super(UNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        
        # Encoder (downsampling path)
        self.initial_conv = DoubleConv(in_channels, features[0], 
                                     use_batch_norm, dropout_rate)
        
        self.down_blocks = nn.ModuleList()
        for i in range(len(features) - 1):
            self.down_blocks.append(
                DownBlock(features[i], features[i + 1], 
                         use_batch_norm, dropout_rate)
            )
        
        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        self.attention_gates = nn.ModuleList() if use_attention else None
        
        for i in range(len(features) - 1, 0, -1):
            self.up_blocks.append(
                UpBlock(features[i] + features[i - 1], features[i - 1],
                       use_transpose, use_batch_norm, dropout_rate)
            )
            
            if use_attention:
                self.attention_gates.append(
                    AttentionGate(features[i], features[i - 1], 
                                features[i - 1] // 2)
                )
        
        # Final classification layer
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Segmentation output (B, out_channels, H, W)
        """
        # Store skip connections
        skip_connections = []
        
        # Initial convolution
        x = self.initial_conv(x)
        skip_connections.append(x)
        
        # Encoder path
        for down_block in self.down_blocks:
            x = down_block(x)
            skip_connections.append(x)
        
        # Remove the last skip connection (it's the bottleneck)
        skip_connections = skip_connections[:-1]
        
        # Decoder path
        for i, up_block in enumerate(self.up_blocks):
            skip_connection = skip_connections[-(i + 1)]
            
            if self.use_attention:
                skip_connection = self.attention_gates[i](x, skip_connection)
            
            x = up_block(x, skip_connection)
        
        # Final classification
        x = self.final_conv(x)
        
        return x


class UNetWithBackbone(nn.Module):
    """
    U-Net with pre-trained backbone encoder.
    
    Uses pre-trained models like ResNet or EfficientNet as encoder
    for better feature extraction and faster convergence.
    """
    
    def __init__(self, backbone_name: str = 'resnet34', 
                 in_channels: int = 3, out_channels: int = 1,
                 pretrained: bool = True, use_attention: bool = False):
        """
        Initialize U-Net with backbone.
        
        Args:
            backbone_name: Name of backbone ('resnet34', 'resnet50', etc.)
            in_channels: Number of input channels
            out_channels: Number of output channels
            pretrained: Whether to use pre-trained weights
            use_attention: Whether to use attention gates
        """
        super(UNetWithBackbone, self).__init__()
        
        self.backbone_name = backbone_name
        self.out_channels = out_channels
        self.use_attention = use_attention
        
        # Load backbone encoder
        self.encoder = self._create_encoder(backbone_name, in_channels, pretrained)
        
        # Get encoder feature dimensions
        encoder_features = self._get_encoder_features()
        
        # Create decoder
        self.decoder = self._create_decoder(encoder_features)
        
        # Final classification layer
        self.final_conv = nn.Conv2d(encoder_features[0], out_channels, 1)
    
    def _create_encoder(self, backbone_name: str, in_channels: int, 
                       pretrained: bool) -> nn.Module:
        """Create encoder from backbone model."""
        try:
            import torchvision.models as models
            
            if backbone_name == 'resnet34':
                backbone = models.resnet34(pretrained=pretrained)
                encoder_features = [64, 64, 128, 256, 512]
            elif backbone_name == 'resnet50':
                backbone = models.resnet50(pretrained=pretrained)
                encoder_features = [64, 256, 512, 1024, 2048]
            else:
                raise ValueError(f"Unsupported backbone: {backbone_name}")
            
            # Modify first layer if input channels != 3
            if in_channels != 3:
                backbone.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, 
                                         padding=3, bias=False)
            
            # Extract encoder layers
            encoder_layers = [
                nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu),
                nn.Sequential(backbone.maxpool, backbone.layer1),
                backbone.layer2,
                backbone.layer3,
                backbone.layer4
            ]
            
            self.encoder_features = encoder_features
            return nn.ModuleList(encoder_layers)
            
        except ImportError:
            logger.warning("torchvision not available, using simple encoder")
            return self._create_simple_encoder(in_channels)
    
    def _create_simple_encoder(self, in_channels: int) -> nn.Module:
        """Create simple encoder if torchvision is not available."""
        features = [64, 128, 256, 512, 1024]
        self.encoder_features = features
        
        layers = []
        current_channels = in_channels
        
        for feature_dim in features:
            if len(layers) == 0:
                layers.append(DoubleConv(current_channels, feature_dim))
            else:
                layers.append(DownBlock(current_channels, feature_dim))
            current_channels = feature_dim
        
        return nn.ModuleList(layers)
    
    def _get_encoder_features(self) -> List[int]:
        """Get encoder feature dimensions."""
        return self.encoder_features
    
    def _create_decoder(self, encoder_features: List[int]) -> nn.Module:
        """Create decoder layers."""
        decoder_layers = []
        attention_gates = [] if self.use_attention else None
        
        for i in range(len(encoder_features) - 1, 0, -1):
            decoder_layers.append(
                UpBlock(encoder_features[i] + encoder_features[i - 1], 
                       encoder_features[i - 1])
            )
            
            if self.use_attention:
                attention_gates.append(
                    AttentionGate(encoder_features[i], encoder_features[i - 1],
                                encoder_features[i - 1] // 2)
                )
        
        self.attention_gates = nn.ModuleList(attention_gates) if attention_gates else None
        return nn.ModuleList(decoder_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone U-Net."""
        # Encoder path
        skip_connections = []
        
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            skip_connections.append(x)
        
        # Remove bottleneck from skip connections
        skip_connections = skip_connections[:-1]
        
        # Decoder path
        for i, decoder_layer in enumerate(self.decoder):
            skip_connection = skip_connections[-(i + 1)]
            
            if self.use_attention:
                skip_connection = self.attention_gates[i](x, skip_connection)
            
            x = decoder_layer(x, skip_connection)
        
        # Final classification
        x = self.final_conv(x)
        
        return x


def create_unet_model(model_type: str = 'standard', **kwargs) -> nn.Module:
    """
    Factory function to create U-Net models.
    
    Args:
        model_type: Type of model ('standard', 'attention', 'backbone')
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Initialized U-Net model
    """
    if model_type == 'standard':
        return UNet(**kwargs)
    elif model_type == 'attention':
        kwargs['use_attention'] = True
        return UNet(**kwargs)
    elif model_type == 'backbone':
        return UNetWithBackbone(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation and forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test standard U-Net
    model = create_unet_model('standard', in_channels=3, out_channels=2)
    model = model.to(device)
    
    # Test input
    x = torch.randn(2, 3, 256, 256).to(device)
    
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test attention U-Net
    attention_model = create_unet_model('attention', in_channels=3, out_channels=2)
    attention_model = attention_model.to(device)
    
    with torch.no_grad():
        attention_output = attention_model(x)
        print(f"Attention U-Net output shape: {attention_output.shape}")
        print(f"Attention model parameters: {sum(p.numel() for p in attention_model.parameters())}")
