import sys; sys.path.insert(0, 'src')
import numpy as np, torch
import argparse
from optiguard.data.simulator import MapSimulator
from optiguard.models.spatial_spectral import SpatialSpectralUNet

def compute_neff(checkpoint_path, data_config='configs/simulator.yaml', index=6, exposure=0.1):
    sim = MapSimulator.from_yaml(data_config)
    s = sim.generate(index=index)
    
    # Load model
    model = SpatialSpectralUNet(128, 128, 64, 2)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    model.load_state_dict(ckpt)
    model.eval()

    # Prepare input
    short_counts = s.short_counts[exposure]
    peak_nominal = 520.7
    peak_idx = int(np.argmin(np.abs(s.axis - peak_nominal)))
    w_start = max(0, peak_idx - 64)
    w_end = min(len(s.axis), peak_idx + 64)
    cropped = short_counts[:, :, w_start:w_end]

    inp_t = torch.from_numpy(cropped.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)
    inp_t.requires_grad_(True)

    # Forward pass
    out_t = model(inp_t) # (1, 128, H, W)
    center_y, center_x = out_t.shape[2] // 2, out_t.shape[3] // 2

    # We compute the Jacobian of the sum of the spectrum at the center pixel
    # with respect to the input spatial pixels.
    val_sum = out_t[0, :, center_y, center_x].sum()
    val_sum.backward()
    
    # Sum the gradients over the spectral channels to get the spatial footprint
    spatial_grad = inp_t.grad[0].sum(dim=0).abs()
    
    sum_w = spatial_grad.sum().item()
    sum_sq_w = (spatial_grad ** 2).sum().item()
    
    N_eff = (sum_w ** 2) / sum_sq_w if sum_sq_w > 0 else 0
    
    print(f"Jacobian Probe Results for {checkpoint_path}")
    print(f"--------------------------------------------------")
    print(f"Effective pooled pixels (N_eff): {N_eff:.2f}")
    
    # Print the top 5 contributing pixels
    flat_indices = torch.argsort(spatial_grad.flatten(), descending=True)
    print(f"Top spatial contributors:")
    for i in range(5):
        idx = flat_indices[i].item()
        y, x = idx // 64, idx % 64
        weight_frac = spatial_grad[y, x].item() / sum_w
        dy, dx = y - center_y, x - center_x
        print(f"  Rank {i+1}: offset (dy={dy:2d}, dx={dx:2d}) -> {weight_frac:.2%} of gradient")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute effective spatial pooling (N_eff)')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    args = parser.parse_args()
    
    compute_neff(args.checkpoint)
