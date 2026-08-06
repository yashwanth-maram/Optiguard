"""Gate runner: compares RAW vs spatial_gauss (sigma=2, N_eff=1.0).

Checks:
1. RAW counts: chi2_nu should be ~1.0 (known-answer calibration).
2. spatial_gauss: T1a should fire (chi2_nu << 1, variance removed).
                  T1b should fire (sigma_scaled < CRLB, fabricated precision).

This closes Step 10: the gate must fire on spatial_gauss and stay quiet on raw.
"""
import sys; sys.path.insert(0, 'src')
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from optiguard.data.simulator import MapSimulator
from optiguard.models.spatial_spectral import SpatialSpectralUNet
from optiguard.assurance.gate import evaluate_gate

sim = MapSimulator.from_yaml('configs/simulator.yaml')
s = sim.generate(index=6)

# Load the neural network from Step 8
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

H, W = short_counts.shape[:2]
neff_map_1 = np.full((H, W), 1.0)
read_noise_e = s.meta.get('read_noise_e', 0.0)


def report(gate_result, label, chi2nu_threshold=0.5, t1b_threshold=None):
    summary = gate_result['summary']
    chi2_nu = gate_result['chi2_nu_map']
    ratio = gate_result['precision_ratio']
    crlb_map = gate_result['crlb_map']
    floor_map = gate_result['floor_map']
    fail_t1a_raw = gate_result['fail_t1a']   # actual bool array, not priority-masked
    fail_t1b_raw = gate_result['fail_t1b']   # actual bool array, not priority-masked
    fail_t2_raw = gate_result['fail_t2']     # actual bool array
    fail_t3_raw = gate_result['fail_t3']     # actual bool array
    fail_t4_raw = gate_result['fail_t4']     # actual bool array (T4 feature erasure)

    t1b_thr_str = f"{t1b_threshold:.3f}" if t1b_threshold is not None else "<floor"
    print(f'=== {label} ===')
    print(f"  fused risk summary:")
    print(f"    PASS:               {summary['n_pass']:5d} / {summary['total']}")
    print(f"    EXPLOIT:            {summary['n_exploit']:5d}  (T1a & T1b)")
    print(f"    HALLUCINATION:      {summary['n_hallucination']:5d}  (T3)")
    print(f"    FEATURE_ERASURE:    {summary['n_feature_erasure']:5d}  (T4)")
    print(f"    BOUNDARY_VIOLATION: {summary['n_boundary']:5d}  (T2 spatial mask)")
    defect_mask = s.defect_mask
    t4_on_defect = (fail_t4_raw & defect_mask).sum()
    t4_on_bg = (fail_t4_raw & ~defect_mask).sum()

    print(f"  scores:")
    print(f"    confidence score (map mean): {summary['mean_confidence']:.4f}  (honesty={summary['mean_honesty']:.4f}, feature={summary['mean_feature']:.4f})")
    print(f"    feature score on supported:  {summary['feature_score_on_supported']:.4f}")
    print(f"  feature erasure rate (on supported): {summary['n_erased_supported']} / {summary['n_supported_features']} ({summary['feature_erasure_rate']*100:.1f}%)")
    print(f"  T4 defect overlap: {t4_on_defect} / {defect_mask.sum()} defect pixels caught ({t4_on_bg} on background)")
    print(f"  raw gate counts (un-masked):")
    print(f"    T1a fires: {fail_t1a_raw.sum():5d} / {fail_t1a_raw.size} ({fail_t1a_raw.mean()*100:.1f}%)  chi2_nu < {chi2nu_threshold}")
    thr_label = f"< {t1b_thr_str}" if t1b_threshold else "< floor"
    t1b_calibrated = (ratio < t1b_threshold).sum() if t1b_threshold else fail_t1b_raw.sum()
    t1b_pct = (ratio < t1b_threshold).mean()*100 if t1b_threshold else fail_t1b_raw.mean()*100
    print(f"    T1b fires: {t1b_calibrated:5d} / {ratio.size} ({t1b_pct:.1f}%)  ratio {thr_label}")
    print(f"    T2  fires: {fail_t2_raw.sum():5d} / {fail_t2_raw.size} ({fail_t2_raw.mean()*100:.1f}%)")
    print(f"    T3  fires: {fail_t3_raw.sum():5d} / {fail_t3_raw.size} ({fail_t3_raw.mean()*100:.1f}%)")
    print(f"    T4  fires: {fail_t4_raw.sum():5d} / {fail_t4_raw.size} ({fail_t4_raw.mean()*100:.1f}%)")
    print()
    print(f"  chi2_nu: mean={chi2_nu.mean():.3f}  median={np.median(chi2_nu):.3f}  "
          f"min={chi2_nu.min():.3f}  max={chi2_nu.max():.3f}")
    print(f"  sigma_scaled/floor (T1b ratio): mean={np.nanmean(ratio):.3f}  median={np.nanmedian(ratio):.3f}")
    print(f"  CRLB mean: {crlb_map.mean():.4f} cm-1   floor mean: {floor_map.mean():.4f} cm-1")
    print()


# ── 1. RAW: known-answer check. chi2_nu should be ≈ 1.0 ────────────────────
print('--- 1. RAW counts (known-answer: chi2_nu ~ 1.0) ---')
raw_gate = evaluate_gate(
    s, short_counts, neff_map_1,
    read_noise_e=read_noise_e,
    spectral_window=(w_start, w_end)
)
# Empirical T1b threshold: 5th percentile of raw ratio distribution
raw_ratios = raw_gate['precision_ratio']
valid_raw = raw_ratios[np.isfinite(raw_ratios)]
t1b_threshold = float(np.percentile(valid_raw, 5.0))
print(f'  T1b threshold (raw 5th pct): {t1b_threshold:.4f}')
report(raw_gate, "RAW DATA", t1b_threshold=t1b_threshold)

# ── 2. spatial_gauss sigma=2.0, N_eff=1.0 (dishonest: no pooling credit) ───
print('--- 2. spatial_gauss sigma=2.0, N_eff=1.0 (dishonest) ---')
gauss_restored = gaussian_filter(short_counts.astype(np.float64), sigma=(2.0, 2.0, 0))
gauss_gate = evaluate_gate(
    s, gauss_restored, neff_map_1,
    read_noise_e=read_noise_e,
    t1b_threshold_ratio=t1b_threshold if t1b_threshold is not None else 1.0,
    spectral_window=(w_start, w_end)
)
report(gauss_gate, "spatial_gauss (sigma=2.0, N_eff=1.0 claimed)", t1b_threshold=t1b_threshold)

# ── 3. spatial_gauss with correct N_eff = 4*pi*sigma^2 ─────────────────────
# For a continuous 2D Gaussian filter, the variance of the output is
# 1/(4*pi*sigma^2) of the input variance, so N_eff = 4*pi*sigma^2.
# At sigma=2: N_eff = 4*pi*4 = 50.3.
neff_gauss_correct = 4 * np.pi * 2.0**2   # 50.27
neff_map_gauss_correct = np.full((H, W), neff_gauss_correct)
print(f'--- 3. spatial_gauss sigma=2.0, N_eff=4*pi*4={neff_gauss_correct:.1f} (correct) ---')
gauss_gate_correct = evaluate_gate(
    s, gauss_restored, neff_map_gauss_correct,
    read_noise_e=read_noise_e,
    t1b_threshold_ratio=t1b_threshold if t1b_threshold is not None else 1.0,
    spectral_window=(w_start, w_end)
)
report(gauss_gate_correct, f"spatial_gauss (sigma=2.0, N_eff={neff_gauss_correct:.1f} correct)", t1b_threshold=t1b_threshold)

# ── 4. Neural Network (Step 8 checkpoint, N_eff=1.02) ──────────────────────
print('--- 4. Neural Network (step8_checkpoint.pt, N_eff=1.02) ---')
cropped = short_counts[:, :, w_start:w_end]
inp = torch.from_numpy(cropped.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)
with torch.no_grad():
    out = model(inp)
out_np = out.squeeze(0).numpy().transpose(1, 2, 0)

nn_restored = np.copy(short_counts).astype(np.float64)
nn_restored[:, :, w_start:w_end] = out_np

neff_map_nn = np.full((H, W), 1.02)
nn_gate = evaluate_gate(
    s, nn_restored, neff_map_nn,
    read_noise_e=read_noise_e,
    t1b_threshold_ratio=t1b_threshold if t1b_threshold is not None else 1.0,
    spectral_window=(w_start, w_end)
)
report(nn_gate, "Neural Network (step8_checkpoint, N_eff=1.02 claimed)", t1b_threshold=t1b_threshold)
