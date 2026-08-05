"""
Investigate T2 chi2 and z-score distributions on clean vs defect pixels.
"""
import numpy as np
from optiguard.data.simulator import MapSimulator
from optiguard.assurance.gate import test_pooling_legitimacy_map
from scipy.ndimage import gaussian_filter

sim = MapSimulator.from_yaml("configs/simulator.yaml")
s = sim.generate(index=6) # test sample
counts = s.short_counts[0.1]
rn = s.meta.get("read_noise_e", 0.0)

# Full spectrum vs 128-channel window
H, W, C = counts.shape
peak_idx = int(np.argmin(np.abs(s.axis - float(s.meta["peak_cm1"]))))
window_size = 128
w_start = max(0, peak_idx - window_size // 2)
w_end = min(C, w_start + window_size)

counts_crop = counts[:, :, w_start:w_end]

res_full = test_pooling_legitimacy_map(counts, sigma=2.0, read_noise_e=rn, alpha=0.01)
res_crop = test_pooling_legitimacy_map(counts_crop, sigma=2.0, read_noise_e=rn, alpha=0.01)

d_mask = s.defect_mask
c_mask = ~s.defect_mask

print("Full spectrum (1024 ch):")
print(f"  Clean pixels z-score:  mean={res_full['z_score'][c_mask].mean():.2f}, median={res_full['z_score'][c_mask].median() if hasattr(res_full['z_score'][c_mask], 'median') else np.median(res_full['z_score'][c_mask]):.2f}, max={res_full['z_score'][c_mask].max():.2f}")
print(f"  Defect pixels z-score: mean={res_full['z_score'][d_mask].mean():.2f}, median={np.median(res_full['z_score'][d_mask]):.2f}, min={res_full['z_score'][d_mask].min():.2f}")

print("\nCropped window (128 ch):")
print(f"  Clean pixels z-score:  mean={res_crop['z_score'][c_mask].mean():.2f}, median={np.median(res_crop['z_score'][c_mask]):.2f}, max={res_crop['z_score'][c_mask].max():.2f}")
print(f"  Defect pixels z-score: mean={res_crop['z_score'][d_mask].mean():.2f}, median={np.median(res_crop['z_score'][d_mask]):.2f}, min={res_crop['z_score'][d_mask].min():.2f}")
