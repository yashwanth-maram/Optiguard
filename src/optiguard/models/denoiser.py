"""
Neural Restoration Models for Low-Photon Spectroscopic Inspection.
Step 8: Spatial-Spectral Multi-Scale Restoration U-Net.
"""
from optiguard.models.spatial_spectral import (
    SpatialSpectralUNet,
    SpatialPoissonLoss,
    ConvBlock2D,
    TORCH_AVAILABLE
)

# Alias for backwards compatibility
RestorationModel = SpatialSpectralUNet
