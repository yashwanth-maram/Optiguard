import sys; sys.path.insert(0, 'src')
import argparse
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from optiguard.data.simulator import MapSimulator
from optiguard.models.spatial_spectral import SpatialSpectralUNet
import time

# IMPORTANT: N_eff is activation-dependent for nonlinear networks.
# Random input activates all spatial positions uniformly -> inflated N_eff.
# Real spectroscopic data has a dominant central peak -> the network's
# activations collapse and N_eff reflects the true operational footprint.
# Always probe with real data from the actual data distribution.

def get_spatial_gauss_neff_map(sigma=2.0, H=64, W=64):
    """Compute N_eff map for a spatial gaussian filter by building the Jacobian."""
    print(f"Running known-answer test on spatial_gauss (sigma={sigma})")
    
    # We can just probe the convolution by passing a delta function at each pixel.
    # For a linear filter, the impulse response IS the weight footprint!
    neff_map = np.zeros((H, W))
    t0 = time.time()
    
    # To do it across the map, we place a 1 at (y,x) and apply the filter
    for y in range(H):
        for x in range(W):
            impulse = np.zeros((H, W))
            impulse[y, x] = 1.0
            
            # The weights used to compute pixel (y,x) are given by the filter applied to the impulse,
            # because Gaussian filter is symmetric.
            # (Wait, actually for any linear shift invariant filter, the response to a delta at (y,x)
            # evaluated everywhere gives the weights, but we want the weights that go INTO (y,x).
            # By symmetry it's the same.)
            weights = gaussian_filter(impulse, sigma=sigma, mode='reflect')
            
            sum_w = weights.sum()
            sum_sq = (weights**2).sum()
            neff_map[y, x] = (sum_w**2) / sum_sq
            
    t1 = time.time()
    print(f"Gaussian Neff Map computed in {t1-t0:.2f}s")
    
    center_neff = neff_map[H//2, W//2]
    theoretical = 4 * np.pi * (sigma**2)
    print(f"Center N_eff: {center_neff:.2f}")
    print(f"Theoretical : {theoretical:.2f}")
    print(f"Matches theoretical: {abs(center_neff - theoretical) < 1.0}")
    
    return neff_map

def get_network_neff_map(checkpoint_path, data_config='configs/simulator.yaml',
                         index=6, exposure=0.1):
    """Compute per-pixel N_eff using REAL spectroscopic data.

    N_eff is activation-dependent for nonlinear networks. Random input
    inflates N_eff because it activates all spatial positions uniformly.
    Real data with a dominant central peak produces the correct, much lower value.
    """
    sim = MapSimulator.from_yaml(data_config)
    s = sim.generate(index=index)
    short_counts = s.short_counts[exposure]
    peak_idx = int(np.argmin(np.abs(s.axis - 520.7)))
    w_start = max(0, peak_idx - 64)
    w_end = min(len(s.axis), peak_idx + 64)
    cropped = short_counts[:, :, w_start:w_end]  # (H, W, 128)
    H, W = cropped.shape[0], cropped.shape[1]

    print(f"Computing per-pixel N_eff map for network ({checkpoint_path})...")
    print(f"Input: real sample {index}, exposure={exposure}, shape={cropped.shape}")
    print(f"This will take a few minutes because it runs {H*W} backward passes.")

    model = SpatialSpectralUNet(128, 128, 64, 2)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    model.load_state_dict(ckpt)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Use real data as input — NOT random noise
    inp = torch.from_numpy(cropped.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)
    inp.requires_grad_(True)
    out = model(inp)

    neff_map = np.zeros((H, W))
    t0 = time.time()

    for y in range(H):
        for x in range(W):
            val = out[0, :, y, x].sum()
            is_last = (y == H - 1 and x == W - 1)
            grad = torch.autograd.grad(val, inp, retain_graph=not is_last)[0]
            spatial_grad = grad[0].sum(dim=0).abs()
            sum_w = spatial_grad.sum().item()
            sum_sq = (spatial_grad ** 2).sum().item()
            neff_map[y, x] = (sum_w ** 2) / sum_sq if sum_sq > 0 else 1.0

        if y % 8 == 0:
            print(f"Row {y}/{H} done. Elapsed: {time.time()-t0:.1f}s")

    t1 = time.time()
    print(f"Network Neff Map computed in {t1-t0:.2f}s")
    print(f"Median N_eff: {np.median(neff_map):.2f}")
    print(f"Mean N_eff:   {np.mean(neff_map):.2f}")

    return neff_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, help='Path to checkpoint')
    parser.add_argument('--gauss-only', action='store_true', help='Only run the gaussian test')
    parser.add_argument('--data-config', default='configs/simulator.yaml')
    parser.add_argument('--index', type=int, default=6)
    parser.add_argument('--exposure', type=float, default=0.1)
    args = parser.parse_args()

    gauss_map = get_spatial_gauss_neff_map(sigma=2.0)

    if not args.gauss_only and args.network:
        net_map = get_network_neff_map(
            args.network, args.data_config, args.index, args.exposure
        )
