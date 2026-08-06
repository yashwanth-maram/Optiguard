import sys; sys.path.insert(0, 'src')
import argparse
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from optiguard.data.simulator import MapSimulator
from optiguard.models.spatial_spectral import SpatialSpectralUNet
import time

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

def get_network_neff_map(checkpoint_path, H=64, W=64):
    print(f"Computing per-pixel N_eff map for network ({checkpoint_path})...")
    print("This will take a few minutes because it runs 4096 backward passes.")
    
    model = SpatialSpectralUNet(128, 128, 64, 2)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    model.load_state_dict(ckpt)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
        
    # We use a dummy input of the correct size
    inp = torch.randn(1, 128, H, W, requires_grad=True)
    out = model(inp)
    
    neff_map = np.zeros((H, W))
    
    t0 = time.time()
    # To avoid 4096 separate passes, we can compute them row by row or just one by one.
    # One by one is foolproof.
    for y in range(H):
        for x in range(W):
            # Sum over spectral channels to get spatial footprint
            val = out[0, :, y, x].sum()
            
            is_last = (y == H-1 and x == W-1)
            grad = torch.autograd.grad(val, inp, retain_graph=not is_last)[0]
            
            spatial_grad = grad[0].sum(dim=0).abs()
            sum_w = spatial_grad.sum().item()
            sum_sq = (spatial_grad**2).sum().item()
            
            neff_map[y, x] = (sum_w**2) / sum_sq if sum_sq > 0 else 1.0
            
            # clear grad for the next iteration (not strictly necessary with autograd.grad, 
            # but good practice if using .backward())
            
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
    args = parser.parse_args()
    
    gauss_map = get_spatial_gauss_neff_map(sigma=2.0)
    
    if not args.gauss_only and args.network:
        net_map = get_network_neff_map(args.network)
