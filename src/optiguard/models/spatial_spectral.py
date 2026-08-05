"""
Spatial-Spectral Multi-Scale Neural Restoration Model for Hyperspectral Raman Maps.
Step 8: 2D Spatial Convolutions + Spectral Channel Preservation with Max 2 Downsampling Stages.
"""
from typing import Tuple, Dict, Any, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class ConvBlock2D(nn.Module):
        """2D Spatial Convolution Block with GroupNorm and GELU."""
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
            super().__init__()
            padding = kernel_size // 2
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
            self.act = nn.GELU()
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
            self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.act(self.norm1(self.conv1(x)))
            h = self.act(self.norm2(self.conv2(h)))
            return h

    class SpatialSpectralUNet(nn.Module):
        """
        Spatial-Spectral Multi-Scale Restoration U-Net.
        
        Processes 3D hyperspectral map cubes formatted as (B, C, H, W) where:
          - C: Cropped spectral channels (e.g. 128)
          - H, W: Spatial map grid dimensions (e.g. 64x64)
          
        Uses 2 downsampling stages maximum to preserve fine spatial defect boundaries
        while aggregating spatial SNR across smooth bulk material.
        """
        def __init__(
            self,
            in_channels: int = 128,
            out_channels: int = 128,
            base_channels: int = 64,
            depth: int = 2,
            dropout: float = 0.05
        ):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            
            # Initial spectral-spatial feature projection
            self.in_conv = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
                nn.GroupNorm(8, base_channels),
                nn.GELU()
            )
            
            # Encoder Stage 1 (Full resolution: 64x64)
            self.enc1 = ConvBlock2D(base_channels, base_channels)
            self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=2, stride=2) # 64 -> 32
            
            # Encoder Stage 2 (Resolution: 32x32)
            self.enc2 = ConvBlock2D(base_channels * 2, base_channels * 2)
            self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=2, stride=2) # 32 -> 16
            
            # Bottleneck (Resolution: 16x16, Max 2 downsamplings)
            self.bottleneck = nn.Sequential(
                ConvBlock2D(base_channels * 4, base_channels * 4),
                nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
            )
            
            # Decoder Stage 2 (16 -> 32)
            self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
            self.dec2 = ConvBlock2D(base_channels * 4, base_channels * 2) # Concat skip
            
            # Decoder Stage 1 (32 -> 64)
            self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
            self.dec1 = ConvBlock2D(base_channels * 2, base_channels)     # Concat skip
            
            # Output projection to spectral channels with Softplus (enforce non-negative counts)
            self.out_conv = nn.Sequential(
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(base_channels, out_channels, kernel_size=1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (B, C, H, W) noisy input counts
            Returns:
                (B, C, H, W) restored expected photon counts
            """
            # Residual base connection
            feat0 = self.in_conv(x)
            
            e1 = self.enc1(feat0)
            d1 = self.down1(e1)
            
            e2 = self.enc2(d1)
            d2 = self.down2(e2)
            
            b = self.bottleneck(d2)
            
            u2 = self.up2(b)
            dec2_in = torch.cat([u2, e2], dim=1)
            dec2_out = self.dec2(dec2_in)
            
            u1 = self.up1(dec2_out)
            dec1_in = torch.cat([u1, e1], dim=1)
            dec1_out = self.dec1(dec1_in)
            
            res = self.out_conv(dec1_out)
            out = F.softplus(x + res)
            return out


    class SpatialPoissonLoss(nn.Module):
        """
        Loss for Spatial-Spectral Restoration.
        Combines supervised L1/MSE on expected ground-truth rate with Poisson NLL.
        """
        def __init__(self, alpha_nll: float = 0.05, eps: float = 1e-6):
            super().__init__()
            self.alpha_nll = alpha_nll
            self.eps = eps

        def forward(self, pred: torch.Tensor, target_clean: torch.Tensor, raw_noisy: torch.Tensor) -> torch.Tensor:
            # 1. Supervised reconstruction error on clean expected rate
            l1_loss = F.l1_loss(pred, target_clean)
            
            # 2. Unsupervised Poisson deviance against raw counts
            # Poisson NLL = mu - y * log(mu)
            mu = torch.clamp(pred, min=self.eps)
            poisson_nll = torch.mean(mu - raw_noisy * torch.log(mu))
            
            return l1_loss + self.alpha_nll * poisson_nll
else:
    class SpatialSpectralUNet:
        pass
    class SpatialPoissonLoss:
        pass
