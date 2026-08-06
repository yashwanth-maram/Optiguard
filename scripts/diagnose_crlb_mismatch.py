import sys; sys.path.insert(0, 'src')
import numpy as np
from optiguard.data.simulator import MapSimulator
from optiguard.physics.crlb import crlb_peak_position, crlb_plugin_map
from optiguard.estimation.fit import fit_lorentzian, fit_lorentzian_map

sim = MapSimulator.from_yaml('configs/simulator.yaml')
s = sim.generate(index=6)

axis = s.axis
short_counts = s.short_counts[0.1]
read_noise_e = s.meta.get('read_noise_e', 0.0)

y, x = 32, 32
spec = short_counts[y, x]

peak_nominal = 520.7
peak_idx = int(np.argmin(np.abs(axis - peak_nominal)))
w_start = max(0, peak_idx - 64)
w_end = min(len(axis), peak_idx + 64)
axis_w = axis[w_start:w_end]
spec_w = spec[w_start:w_end]

# Path 1: single-pixel fit
fit = fit_lorentzian(axis_w, spec_w, read_noise_e=read_noise_e)
print('=== Single-pixel fit (fit_lorentzian) ===')
print('  center:    ', round(fit['center'], 4), 'cm-1')
print('  fwhm:      ', round(fit['fwhm'], 4), 'cm-1')
print('  amplitude: ', round(fit['amplitude'], 2), 'counts (PEAK HEIGHT)')
print('  background:', round(fit['background'], 2))
print('  sigma_center from covariance:', round(fit['sigma_center'], 6), 'cm-1')

# Path 2: crlb_peak_position with the identical fitted params and window
crlb_direct = crlb_peak_position(
    axis=axis_w, center=fit['center'], fwhm=fit['fwhm'],
    amplitude=fit['amplitude'], background=fit['background'],
    read_noise_e=read_noise_e
)
print()
print('=== crlb_peak_position (same window + fitted params) ===')
print('  CRLB:      ', round(crlb_direct, 6), 'cm-1')
print('  sigma/CRLB:', round(fit['sigma_center'] / crlb_direct, 3))

# Path 3: crlb_plugin_map on the same pixel window
cube_w = short_counts[:, :, w_start:w_end]
crlb_map_out = crlb_plugin_map(
    axis=axis_w,
    counts_window=cube_w,
    fitted_center=np.full((64, 64), fit['center']),
    read_noise_e=read_noise_e
)
print()
print('=== crlb_plugin_map (same pixel [32,32]) ===')
plugin_val = crlb_map_out[32, 32]
print('  CRLB:         ', round(plugin_val, 6), 'cm-1')
print('  sigma/plugin: ', round(fit['sigma_center'] / plugin_val, 3))
print('  direct/plugin:', round(crlb_direct / plugin_val, 3), ' <- should be 1.0 if same')

# Path 4: what amplitude does crlb_plugin_map actually inject?
shoulder = 16
bg_est = (np.mean(spec_w[:shoulder]) + np.mean(spec_w[-shoulder:])) / 2.0
total_signal = np.sum(spec_w) - bg_est * len(spec_w)
disp = float(axis_w[1] - axis_w[0])
print()
print('=== crlb_plugin_map amplitude diagnosis ===')
print('  bg from shoulders:        ', round(bg_est, 2))
print('  total_signal (INTEGRAL):  ', round(total_signal, 2))
print('  fit amplitude (PEAK):     ', round(fit['amplitude'], 2))
print('  ratio integral/peak:      ', round(total_signal / fit['amplitude'], 2))
print('  expected pi*fwhm/(2*disp):', round(np.pi * fit['fwhm'] / (2 * disp), 2))
print()
print('  sqrt(integral/peak):      ', round((total_signal / fit['amplitude']) ** 0.5, 3),
      ' <- predicted CRLB underestimate factor')
