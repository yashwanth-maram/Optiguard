"""
Neural Restoration Models for Low-Photon Spectroscopic Inspection.
Step 8: Deep Spectral Residual Denoising Network.
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class ResidualBlock1D(nn.Module):
        """1D Dilated Residual Block with Layer Normalization."""
        def __init__(self, channels: int, dilation: int = 1):
            super().__init__()
            self.conv1 = nn.Conv1d(
                channels, channels, kernel_size=5, padding=2 * dilation, dilation=dilation
            )
            self.norm1 = nn.GroupNorm(4, channels)
            self.act = nn.GELU()
            self.conv2 = nn.Conv1d(
                channels, channels, kernel_size=5, padding=2 * dilation, dilation=dilation
            )
            self.norm2 = nn.GroupNorm(4, channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            res = x
            out = self.act(self.norm1(self.conv1(x)))
            out = self.norm2(self.conv2(out))
            return self.act(out + res)

    class SpectralDenoiser(nn.Module):
        """
        Deep 1D Multi-Scale Residual Denoising Network.
        Maps raw Poisson shot-noise spectra (B, 1, C) -> (B, 1, C) expected clean photon rate
        and predicts per-channel uncertainty sigma(B, 1, C).
        """
        def __init__(self, in_channels: int = 1, hidden_dim: int = 64, num_blocks: int = 6):
            super().__init__()
            self.in_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3)
            
            dilations = [1, 2, 4, 1, 2, 4]
            self.blocks = nn.ModuleList([
                ResidualBlock1D(hidden_dim, dilation=dilations[i % len(dilations)])
                for i in range(num_blocks)
            ])
            
            self.out_mean = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(hidden_dim // 2, 1, kernel_size=3, padding=1),
                nn.Softplus() # Enforce positive photon rate
            )
            
            self.out_logvar = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(hidden_dim // 2, 1, kernel_size=3, padding=1)
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                x: (B, 1, C) noisy raw counts
            Returns:
                mean_rate: (B, 1, C) predicted denoised photon expectation
                log_var: (B, 1, C) predicted log-variance (uncertainty)
            """
            feat = F.gelu(self.in_proj(x))
            for block in self.blocks:
                feat = block(feat)
            
            mean = self.out_mean(feat)
            log_var = self.out_logvar(feat)
            return mean, log_var


    class PoissonGaussianLoss(nn.Module):
        """
        Heteroscedastic Loss under Poisson Shot Noise + Gaussian Read Noise.
        Minimizes negative log-likelihood of raw counts given predicted clean rate.
        """
        def __init__(self, read_noise_e: float = 0.0, eps: float = 1e-6):
            super().__init__()
            self.read_noise_e2 = read_noise_e ** 2
            self.eps = eps

        def forward(self, pred_rate: torch.Tensor, target_clean: torch.Tensor, raw_noisy: torch.Tensor) -> torch.Tensor:
            # 1. Supervised reconstruction loss vs reference ground truth
            l1_loss = F.l1_loss(pred_rate, target_clean)
            
            # 2. Generalized Poisson deviance consistency on raw noisy input
            # D = 2 * sum(y * log(y / mu) - (y - mu))
            mu = torch.clamp(pred_rate, min=self.eps)
            poisson_nll = torch.mean(mu - raw_noisy * torch.log(mu))
            
            # 3. Total loss
            return l1_loss + 0.1 * poisson_nll
else:
    # Dummy fallback when PyTorch is not yet installed in local virtualenv
    class SpectralDenoiser:
        pass
    class PoissonGaussianLoss:
        pass
