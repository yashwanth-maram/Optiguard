"""
Model Wrapper and Interface for Evaluation Harness.
Provides `load_restoration_method(path)` which returns an evaluatable callable.
"""
from typing import Callable, Dict, Any, Optional
from pathlib import Path
import numpy as np
import yaml

from optiguard.models.spatial_spectral import SpatialSpectralUNet, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch


def load_restoration_method(checkpoint_path: str, config_path: Optional[str] = None) -> Callable[[np.ndarray, Dict[str, Any], np.ndarray], np.ndarray]:
    """
    Load a trained restoration model and wrap it into a callable accepted by `evaluate()`.

    Args:
        checkpoint_path: Path to PyTorch .pt model checkpoint
        config_path: Optional path to restoration YAML config

    Returns:
        Callable with signature `method(short_counts, meta, axis) -> restored_counts`
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to run neural restoration inference.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load configuration
    cfg = {}
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f).get("model", {})
            
    in_channels = cfg.get("in_channels", 128)
    base_channels = cfg.get("base_channels", 64)
    depth = cfg.get("depth", 2)
    
    # Initialize model
    model = SpatialSpectralUNet(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=base_channels,
        depth=depth
    ).to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    def restoration_method(sample, exposure, **kwargs) -> np.ndarray:
        """
        Inference function for a full sample matching the harness baseline signature.
        """
        H, W, C = short_counts.shape
        peak_nominal = _nominal(sample)
        
        peak_idx = int(np.argmin(np.abs(axis - peak_nominal)))
        w_start = max(0, peak_idx - in_channels // 2)
        w_end = min(C, peak_idx + in_channels // 2)
        
        # Prepare input tensor: (1, C_crop, H, W)
        cropped_in = short_counts[:, :, w_start:w_end] # (H, W, C_crop)
        actual_channels = cropped_in.shape[2]
        
        # If boundary crop size differs from in_channels, pad
        if actual_channels != in_channels:
            padded = np.zeros((H, W, in_channels), dtype=np.float32)
            padded[:, :, :actual_channels] = cropped_in
            inp_tensor = torch.from_numpy(padded.transpose(2, 0, 1)).unsqueeze(0).to(device)
        else:
            inp_tensor = torch.from_numpy(cropped_in.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0).to(device)
            
        with torch.no_grad():
            out_tensor = model(inp_tensor)
            out_np = out_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) # (H, W, C_crop)
            
        # Splice back into full array
        restored = np.copy(short_counts).astype(np.float64)
        restored[:, :, w_start:w_end] = out_np[:, :, :actual_channels]
        
        return fit_map(axis, restored, meta.get("read_noise_e", 0.0), peak_nominal)

    return restoration_method
