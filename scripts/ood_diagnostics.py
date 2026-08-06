import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8')
from optiguard.data.simulator import MapSimulator

sim = MapSimulator.from_yaml('configs/simulator.yaml')

# 1. Check whether axis differs by material
print("=== Axis range per material ===")
for mat in ['silicon', 'gan', 'amorphous_silicon', 'sic_4h', 'si_doublet']:
    s = sim.generate(index=0, material=mat)
    peak = s.meta['peak_cm1']
    print(f"{mat}: axis [{s.axis.min():.1f}, {s.axis.max():.1f}], anchor={peak:.1f}")

# 2. Total integrated photons (Lorentzian integral = A * pi * gamma/2)
print("\n=== Total integrated photons/s per material ===")
base_amp = sim.config['map']['peak_photons_per_s']
for mat, data in sim.config['materials'].items():
    total_photons = 0.0
    for p in data['peaks']:
        amp = base_amp * p['rel_amplitude']
        fwhm = p['fwhm']
        total_photons += amp * np.pi * (fwhm / 2)
    print(f"{mat}: total_photons/s={total_photons:.0f}")

# 3. Does parameter_plausibility use the same nominal 520.7 regardless of material?
print("\n=== Plausibility score (nominal_center=520.7 for all) ===")
from optiguard.estimation.fit import fit_lorentzian_map
from optiguard.assurance.ood import score_parameter_plausibility
for mat in ['silicon', 'gan']:
    s = sim.generate(index=0, material=mat)
    fitted = fit_lorentzian_map(s.axis, s.long_counts.astype(np.float64))
    sc = score_parameter_plausibility(fitted)  # default nominal_center=520.7
    print(f"{mat}: fit_center_mean={fitted['center'].mean():.2f}  fit_fwhm_mean={fitted['fwhm'].mean():.2f}  score_mean={sc.mean():.3f}")
