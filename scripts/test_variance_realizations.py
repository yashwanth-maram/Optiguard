import sys; sys.path.insert(0, 'src')
import numpy as np
import torch
from optiguard.data.simulator import MapSimulator
from optiguard.models.spatial_spectral import SpatialSpectralUNet
from optiguard.estimation.fit import fit_lorentzian
import time

sim = MapSimulator.from_yaml("configs/simulator.yaml")
s = sim.generate(index=6)
exposure = 0.1

# Pick a non-defect pixel
y, x = 32, 32
while s.defect_mask[y, x]:
    x += 1

print(f"Testing pixel (y={y}, x={x})")

# compute rate for this pixel
C = len(s.axis)
center = s.theta_true["center"][y, x]
fwhm = s.theta_true["fwhm"][y, x]
amp = s.theta_true["amplitude"][y, x]
bg = s.theta_true["background"][y, x]

gamma = fwhm / 2.0
L = (gamma**2) / ((s.axis - center)**2 + gamma**2)
rate = amp * L + bg

dark = sim.config["detector"]["dark_rate_e_per_s"]
read_noise_e = sim.config["detector"]["read_noise_e"]

N = 500
rng = np.random.default_rng(42)

# Generate 500 realizations for this one pixel
signal = rng.poisson(rate * exposure, size=(N, C))
dark_counts = rng.poisson(dark * exposure, size=(N, C))
read_counts = rng.normal(0, read_noise_e, size=(N, C))
raw_spectra = (signal + dark_counts + read_counts).astype(np.float64)

# Network setup
model = SpatialSpectralUNet(128, 128, 64, 2)
ckpt = torch.load('src/optiguard/models/step8_checkpoint.pt', map_location='cpu', weights_only=True)
if 'model_state_dict' in ckpt:
    ckpt = ckpt['model_state_dict']
model.load_state_dict(ckpt)
model.eval()
model.to('cpu')

# Window parameters
peak_nominal = 520.7
peak_idx = int(np.argmin(np.abs(s.axis - peak_nominal)))
w_start = max(0, peak_idx - 64)
w_end = min(C, peak_idx + 64)

base_map = s.short_counts[exposure].astype(np.float64) # (H, W, C)

# To process quickly, we can process batches of 50
nn_spectra = []
print("Running network...")
for i in range(0, N, 50):
    batch_size = min(50, N - i)
    # create batch of maps
    batch_maps = np.tile(base_map, (batch_size, 1, 1, 1)) # (B, H, W, C)
    batch_maps[:, y, x, :] = raw_spectra[i:i+batch_size]
    
    # crop window
    cropped = batch_maps[:, :, :, w_start:w_end]
    # network expects (B, Cw, H, W)
    inp = torch.from_numpy(cropped.transpose(0, 3, 1, 2).astype(np.float32))
    with torch.no_grad():
        out = model(inp)
    
    out_np = out.numpy().transpose(0, 2, 3, 1) # (B, H, W, Cw)
    
    # extract the pixel's output
    nn_spectra.append(out_np[:, y, x, :])

nn_spectra_w = np.concatenate(nn_spectra, axis=0) # (500, 128)
raw_spectra_w = raw_spectra[:, w_start:w_end]

# 1. Variance of the output vs input across realizations
var_raw = np.var(raw_spectra_w, axis=0)
var_nn = np.var(nn_spectra_w, axis=0)

print(f"Mean variance across realizations (raw window): {np.mean(var_raw):.3f}")
print(f"Mean variance across realizations (NN window):  {np.mean(var_nn):.3f}")
print(f"Ratio NN_var / Raw_var: {np.mean(var_nn)/np.mean(var_raw):.3f}")

# 2. Fit each realization
print("Fitting realizations...")
raw_centers = []
nn_centers = []

for i in range(N):
    try:
        r_fit = fit_lorentzian(s.axis[w_start:w_end], raw_spectra_w[i], read_noise_e=read_noise_e)
        raw_centers.append(r_fit["center"])
        if i == 0: print("First raw fit:", r_fit, flush=True)
    except Exception as e:
        if i == 0: print("First raw fit EXCEPTION:", e, flush=True)
        pass
        
    try:
        n_fit = fit_lorentzian(s.axis[w_start:w_end], nn_spectra_w[i], read_noise_e=read_noise_e)
        nn_centers.append(n_fit["center"])
        if i == 0: print("First nn fit:", n_fit, flush=True)
    except Exception as e:
        if i == 0: print("First nn fit EXCEPTION:", e, flush=True)
        pass

print(f"Scatter (std) of fitted centers (RAW): {np.nanstd(raw_centers):.5f}")
print(f"Scatter (std) of fitted centers (NN):  {np.nanstd(nn_centers):.5f}")
print(f"Gain (Raw_std / NN_std): {np.nanstd(raw_centers)/np.nanstd(nn_centers):.3f}")
