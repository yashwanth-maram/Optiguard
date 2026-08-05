"""
Export trained PyTorch OptiGuard Restoration Model to ONNX format.
Usage:
    python scripts/export_onnx.py --checkpoint runs/restoration_v1/best.pt --out runs/restoration_v1/model.onnx
"""
import sys
import argparse
from pathlib import Path
import yaml

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from optiguard.models.spatial_spectral import SpatialSpectralUNet, TORCH_AVAILABLE


def export_onnx(checkpoint_path: str, out_path: str, config_path: str = "configs/restoration_v1.yaml"):
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required to export ONNX models.")
        sys.exit(1)

    import torch

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    cfg = {}
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f).get("model", {})
            
    in_channels = cfg.get("in_channels", 128)
    base_channels = cfg.get("base_channels", 64)
    depth = cfg.get("depth", 2)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    model = SpatialSpectralUNet(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=base_channels,
        depth=depth
    )
    
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Dummy input: (1, 128, 64, 64)
    dummy_input = torch.randn(1, in_channels, 64, 64, dtype=torch.float32)
    
    print(f"Exporting to ONNX -> {out_file}")
    torch.onnx.export(
        model,
        dummy_input,
        str(out_file),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["short_counts"],
        output_names=["restored_counts"],
        dynamic_axes={
            "short_counts": {0: "batch_size", 2: "height", 3: "width"},
            "restored_counts": {0: "batch_size", 2: "height", 3: "width"}
        }
    )
    
    print(f"ONNX export successful: {out_file} (Size: {out_file.stat().st_size / (1024 * 1024):.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--out", type=str, required=True, help="Output .onnx file path")
    parser.add_argument("--config", type=str, default="configs/restoration_v1.yaml", help="Path to config YAML")
    args = parser.parse_args()

    export_onnx(checkpoint_path=args.checkpoint, out_path=args.out, config_path=args.config)


if __name__ == "__main__":
    main()
