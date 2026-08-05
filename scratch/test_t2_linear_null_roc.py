"""
Test upgraded T2 Gate with Locally Linear Spatial Null Model and compute ROC / AUROC.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy import stats
from sklearn.metrics import roc_curve, auc
import json
from pathlib import Path

from optiguard.data.simulator import MapSimulator
from optiguard.assurance.pooling import gaussian_pooling_weights_2d, effective_pooling_multiplier

def compute_t2_linear_null(counts, sigma=2.0, read_noise_e=0.0, crop_window=None):
    """Compute T2 statistic with locally linear spatial null model."""
    if crop_window is not None:
        counts = counts[:, :, crop_window[0]:crop_window[1]]
        
    H, W, C = counts.shape
    kernel_weights = gaussian_pooling_weights_2d(sigma)
    M_eff = effective_pooling_multiplier(kernel_weights)
    
    counts_f = counts.astype(np.float64)
    
    # 1. Local mean and second moment
    mu_hat = gaussian_filter(counts_f, [sigma, sigma, 0], mode="reflect")
    sq_smooth = gaussian_filter(counts_f ** 2, [sigma, sigma, 0], mode="reflect")
    var_local_raw = np.maximum(sq_smooth - (mu_hat ** 2), 0.0)
    
    # 2. Local spatial gradients: d/dx and d/dy
    grad_x = gaussian_filter(counts_f, [sigma, sigma, 0], order=[1, 0, 0], mode="reflect")
    grad_y = gaussian_filter(counts_f, [sigma, sigma, 0], order=[0, 1, 0], mode="reflect")
    
    # Gradient variance contribution across the Gaussian window: sigma^2 * (grad_x^2 + grad_y^2)
    var_grad = (sigma ** 2) * (grad_x ** 2 + grad_y ** 2)
    
    # 3. Pure excess variance after subtracting smooth spatial gradient
    var_excess = np.maximum(var_local_raw - var_grad, 0.0)
    
    # 4. Expected Poisson + read noise variance per channel
    var_expected = np.maximum(mu_hat + (read_noise_e ** 2), 1e-6)
    
    # 5. Chi-squared test statistic scaled by (M_eff - 1)
    chi2_map = (M_eff - 1.0) * np.sum(var_excess / var_expected, axis=-1)
    
    df = (M_eff - 1.0) * C
    z_map = (chi2_map - df) / np.sqrt(2.0 * df)
    
    return z_map

# Run evaluation on shift-only test samples
sim = MapSimulator.from_yaml("configs/simulator.yaml")
samples = [sim.generate(index=i) for i in range(6, 12)]
EXPOSURE = 0.1
SIGMA = 2.0

all_z_scores = []
all_true_labels = []
all_difficulties = []

for s in samples:
    counts = s.short_counts[EXPOSURE]
    rn = s.meta.get("read_noise_e", 0.0)
    diff_map = s.difficulty(EXPOSURE)
    
    peak_idx = int(np.argmin(np.abs(s.axis - float(s.meta["peak_cm1"]))))
    w_start = max(0, peak_idx - 64)
    w_end = min(counts.shape[2], peak_idx + 64)
    
    z = compute_t2_linear_null(counts, sigma=SIGMA, read_noise_e=rn, crop_window=(w_start, w_end))
    
    all_z_scores.append(z.ravel())
    all_true_labels.append(s.defect_mask.ravel())
    all_difficulties.append(diff_map.ravel())

z_all = np.concatenate(all_z_scores)
y_all = np.concatenate(all_true_labels)
diff_all = np.concatenate(all_difficulties)

# Calculate ROC and AUROC
fpr, tpr, thresholds = roc_curve(y_all, z_all)
roc_auc = auc(fpr, tpr)

print(f"=== UPGRADED T2 GATE WITH LOCALLY LINEAR NULL MODEL ===")
print(f"Overall AUROC: {roc_auc:.4f}")

# Check performance at specific FPR operating points
target_fprs = [0.01, 0.02, 0.05, 0.10]
print("\nTrue Positive Rate (TPR) at target False Positive Rates (FPR):")
for tf in target_fprs:
    idx = np.argmin(np.abs(fpr - tf))
    actual_fpr = fpr[idx]
    actual_tpr = tpr[idx]
    thresh = thresholds[idx]
    
    # Recall per difficulty at this threshold
    flagged = z_all > thresh
    rec_by_diff = {}
    for d_bin in [0.5, 1.5, 2.5, 4.0, 10.0]:
        if d_bin == 0.5:
            mask = (diff_all >= 0.0) & (diff_all < 1.0) & y_all
        elif d_bin == 1.5:
            mask = (diff_all >= 1.0) & (diff_all < 2.0) & y_all
        elif d_bin == 2.5:
            mask = (diff_all >= 2.0) & (diff_all < 3.0) & y_all
        elif d_bin == 4.0:
            mask = (diff_all >= 3.0) & (diff_all < 5.0) & y_all
        else:
            mask = (diff_all >= 5.0) & y_all
        rec = flagged[mask].sum() / mask.sum() if mask.sum() > 0 else 0.0
        rec_by_diff[d_bin] = rec
        
    print(f"  FPR = {actual_fpr*100:4.2f}% (Threshold z = {thresh:5.2f}) -> Overall Defect TPR = {actual_tpr*100:5.2f}%")
    print(f"    Recalls: 0.5: {rec_by_diff[0.5]*100:4.1f}%, 1.5: {rec_by_diff[1.5]*100:4.1f}%, 2.5: {rec_by_diff[2.5]*100:4.1f}%, 4.0: {rec_by_diff[4.0]*100:4.1f}%, 10.0: {rec_by_diff[10.0]*100:4.1f}%")

# Plot ROC curve
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"T2 Linear Null (AUROC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Chance (0.500)")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate (Clean Pixels Flagged)", fontsize=11)
ax.set_ylabel("True Positive Rate (Defects Flagged)", fontsize=11)
ax.set_title("T2 Assurance Gate ROC Curve (Shift-Only Defects)", fontsize=12, fontweight="bold")
ax.legend(loc="lower right", frameon=True)
ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
fig.savefig("evidence/t2_gate_roc_curve.png", dpi=300)
print("\nWrote evidence/t2_gate_roc_curve.png")
