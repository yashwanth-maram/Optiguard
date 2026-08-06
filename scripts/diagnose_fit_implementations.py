import sys; sys.path.insert(0, 'src')
import numpy as np
from optiguard.data.simulator import MapSimulator
from optiguard.estimation.fit import fit_lorentzian, fit_lorentzian_map

sim = MapSimulator.from_yaml('configs/simulator.yaml')
s = sim.generate(index=6)

axis = s.axis
short_counts = s.short_counts[0.1]
read_noise_e = s.meta.get('read_noise_e', 0.0)

peak_nominal = 520.7
peak_idx = int(np.argmin(np.abs(axis - peak_nominal)))
w_start = max(0, peak_idx - 64)
w_end = min(len(axis), peak_idx + 64)
axis_w = axis[w_start:w_end]
counts_w = short_counts[:, :, w_start:w_end]

# Run map fit
print("Running fit_lorentzian_map on sample 6...")
map_fit = fit_lorentzian_map(axis, short_counts, read_noise_e=read_noise_e, nominal_center_cm1=peak_nominal)
sigma_map = map_fit["sigma_center"]

# Pick a few pixels and run single fit
pixels = [(32, 32), (10, 10), (60, 60), (120, 120)]
print("\nComparing single pixel fit to map fit:")
for y, x in pixels:
    spec_w = counts_w[y, x]
    single_fit = fit_lorentzian(axis_w, spec_w, read_noise_e=read_noise_e)
    
    s_single = single_fit["sigma_center"]
    s_map = sigma_map[y, x]
    
    print(f"Pixel ({y:3d}, {x:3d}):")
    print(f"  Single fit sigma_center: {s_single:.6f}")
    print(f"  Map fit sigma_center:    {s_map:.6f}")
    print(f"  Ratio (Map / Single):    {s_map / s_single:.4f}")
    print(f"  Center (Single vs Map):  {single_fit['center']:.4f} vs {map_fit['center'][y,x]:.4f}")
    print(f"  FWHM (Single vs Map):    {single_fit['fwhm']:.4f} vs {map_fit['fwhm'][y,x]:.4f}")
    print()
