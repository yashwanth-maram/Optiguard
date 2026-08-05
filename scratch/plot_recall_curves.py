"""
Generate publication-quality figure: Defect Recall vs Difficulty
Demonstrating the core tension: 16.8x spatial gain vs catastrophic defect erasure.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Load baseline results
evidence_path = Path("evidence/baselines.json")
with open(evidence_path) as f:
    data = json.load(f)

# Extract curves
diff_bins = [0.5, 1.5, 2.5, 4.0, 10.0]
raw_rec = [data["raw"]["recall_by_difficulty"][str(b)] for b in diff_bins]
spatial_rec = [data["spatial_gauss"]["recall_by_difficulty"][str(b)] for b in diff_bins]
binning_rec = [data["binning"]["recall_by_difficulty"][str(b)] for b in diff_bins]
ref_rec = [data["reference"]["recall_by_difficulty"][str(b)] for b in diff_bins]

# Set up clean, modern figure style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

# Plot curves
ax.plot(diff_bins, ref_rec, "o--", color="#2ca02c", linewidth=2, markersize=7, label="Reference (5.0 s, 49.3× gain)", alpha=0.7)
ax.plot(diff_bins, raw_rec, "s-", color="#1f77b4", linewidth=2.5, markersize=8, label="Raw Exposure (0.1 s, 1.0× gain, honest bound)")
ax.plot(diff_bins, spatial_rec, "D-", color="#d62728", linewidth=3, markersize=8, label="Spatial Gaussian (σ=2.0, 16.8× gain)")
ax.plot(diff_bins, binning_rec, "^:", color="#ff7f0e", linewidth=2, markersize=7, label="7×7 Binning (14.9× gain)", alpha=0.8)

# Highlight the defect erasure zone (0.3 to 3.0 CRLB)
ax.axvspan(0.3, 3.0, color="#d62728", alpha=0.08, label="Critical Feature Zone (0.3–3.0 CRLB)")

# Annotations
ax.annotate(
    "16.8× Exposure Gain\nBUT 0% Recall\nat 1.5 CRLB (Erased)",
    xy=(1.5, spatial_rec[1]),
    xytext=(1.8, 0.22),
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.8),
    fontsize=11,
    fontweight="bold",
    color="#d62728",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff0f0", edgecolor="#d62728", alpha=0.95)
)

ax.annotate(
    "Raw detects 72% of\n1.5 CRLB defects",
    xy=(1.5, raw_rec[1]),
    xytext=(0.7, 0.82),
    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.8),
    fontsize=10,
    fontweight="semibold",
    color="#1f77b4",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f7ff", edgecolor="#1f77b4", alpha=0.95)
)

ax.set_xscale("log")
ax.set_xticks(diff_bins)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlim(0.4, 11.0)
ax.set_ylim(-0.05, 1.05)

ax.set_xlabel("Defect Physical Difficulty (Shift / Local CRLB at 0.1 s)", fontsize=12, fontweight="semibold")
ax.set_ylabel("Defect Detection Recall", fontsize=12, fontweight="semibold")
ax.set_title("The Restoration Dilemma: Spatial Pooling Buys 16.8× SNR but Erases Defects", fontsize=13, fontweight="bold", pad=15)

ax.legend(loc="lower right", frameon=True, framealpha=0.95, facecolor="white", edgecolor="#cccccc", fontsize=10)
plt.tight_layout()

out_path = Path("evidence/recall_vs_difficulty.png")
fig.savefig(out_path, dpi=300)
print(f"Saved figure to {out_path.resolve()}")
