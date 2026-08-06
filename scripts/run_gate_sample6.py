import sys; sys.path.insert(0, 'src')
import numpy as np, torch
from optiguard.data.simulator import MapSimulator
from optiguard.models.spatial_spectral import SpatialSpectralUNet
from optiguard.assurance.gate import evaluate_gate

sim = MapSimulator.from_yaml('configs/simulator.yaml')
s = sim.generate(index=6)

model = SpatialSpectralUNet(128, 128, 64, 2)
ckpt = torch.load('src/optiguard/models/step8_checkpoint.pt', map_location='cpu', weights_only=True)
if 'model_state_dict' in ckpt:
    ckpt = ckpt['model_state_dict']
model.load_state_dict(ckpt)
model.eval()

short_counts = s.short_counts[0.1]
axis = s.axis
peak_nominal = 520.7
peak_idx = int(np.argmin(np.abs(axis - peak_nominal)))
w_start = max(0, peak_idx - 64)
w_end = min(len(axis), peak_idx + 64)
cropped = short_counts[:, :, w_start:w_end]

inp = torch.from_numpy(cropped.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)
with torch.no_grad():
    out = model(inp)

out_np = out.squeeze(0).numpy().transpose(1, 2, 0)  # (H, W, 128)

restored = np.copy(short_counts).astype(np.float64)
restored[:, :, w_start:w_end] = out_np

H, W = short_counts.shape[:2]
neff_map = np.full((H, W), 1.02)
read_noise_e = s.meta.get('read_noise_e', 0.0)

print('Running evaluate_gate on sample 6 with step8_checkpoint...')
result = evaluate_gate(s, restored, neff_map, read_noise_e=read_noise_e)
summary = result['summary']

print()
print('=== Gate Summary ===')
n_pass = summary['n_pass']
n_t1 = summary['n_fail_t1']
n_t2 = summary['n_fail_t2']
n_t3 = summary['n_fail_t3']
total = summary['total']
pass_rate = summary['pass_rate']
print(f'  PASS:    {n_pass:4d} / {total} ({pass_rate*100:.1f}%)')
print(f'  FAIL_T1: {n_t1:4d} (precision beats N_eff-adjusted floor)')
print(f'  FAIL_T2: {n_t2:4d} (neighbourhood heterogeneous)')
print(f'  FAIL_T3: {n_t3:4d} (restored spectrum inconsistent with raw counts)')

print()
ratio = result['precision_ratio']
claimed = result['floor_map']  # used as proxy for sigma relationship check
crlb_map = result['crlb_map']
print('Precision ratio stats (sigma_center from fit / floor):')
print(f'  Mean:   {np.nanmean(ratio):.3f}')
print(f'  Median: {np.nanmedian(ratio):.3f}')
print(f'  < 1.0 (T1 would flag): {(ratio < 1.0).sum()} pixels ({(ratio < 1.0).mean()*100:.1f}%)')

print()
print('CRLB / floor stats:')
print(f'  CRLB mean:  {crlb_map.mean():.4f} cm-1')
print(f'  Floor mean: {result["floor_map"].mean():.4f} cm-1')

# --- Difficulty stratification ---
# Difficulty = CRLB, thresholds from the Step 8 eval
# "1.5 CRLB band" means CRLB > some threshold -- in the training,
# difficulty is relative to the single pixel CRLB
raw_sigma_center = s.meta.get('crlb', None)
print()
print('=== T1 failures by CRLB difficulty tier ===')
# Bin pixels by crlb_map value in CRLB-relative difficulty
crlb_vals = crlb_map.flatten()
fail_t1_flat = result['fail_t1'].flatten()
# Quantile bins of CRLB: easiest (low CRLB) to hardest (high CRLB)
qs = np.quantile(crlb_vals[np.isfinite(crlb_vals)], [0.0, 0.25, 0.5, 0.75, 1.0])
labels = ['easy (Q1)', 'Q2', 'Q3', 'hard (Q4)']
for i, lbl in enumerate(labels):
    mask = (crlb_vals >= qs[i]) & (crlb_vals < qs[i+1])
    n_in_bin = mask.sum()
    n_fail = fail_t1_flat[mask].sum()
    print(f'  {lbl}: CRLB [{qs[i]:.4f}, {qs[i+1]:.4f}] -> T1 fail {n_fail}/{n_in_bin} ({n_fail/max(n_in_bin,1)*100:.1f}%)')

