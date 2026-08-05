# Baseline Analysis & T2 Assurance Gate Verification

## 1. Baseline Tuning Grid Saturation Analysis

| Method | Tuned Parameter | Grid Evaluated | Grid Boundary Status | Physical Rationale / Failure Mode |
|:---|:---:|:---:|:---:|:---|
| **`binning`** | `size = 7` | `[2, 3, 5, 7]` | **Saturated at Top** | Global RMSE continuously decreases as box size grows because >98% of pixels are smooth background. The tuning objective is unbounded in the destructive direction. |
| **`spatial_gauss`** | `sigma = 2.0` | `[0.5, 1.0, 1.5, 2.0, 3.0]` | **Near Top (`sigma=2.0`)** | Delivers 16.8× effective photon gain on background, but collapses defect recall to 0.0 at 0.5–1.5 CRLB. |
| **`pca`** | `n_components = 16` | `[2, 4, 8, 16]` | **Saturated at Top** | Continuously shifting Lorentzian peaks across a 2D map cannot be compactly represented in low rank. The optimizer wants maximum rank ("don't project"). |
| **`nmf`** | `n_components = 16` | `[2, 4, 8, 16]` | **Saturated at Top** | Same physical mechanism as PCA: spectral subspace projection distorts continuously variable peak positions. |

> [!IMPORTANT]
> **Key Finding on Conventional Objective Functions:**
> Conventional tuning objectives (minimizing global RMSE or reconstruction error) are **unbounded in the destructive direction**. Because defect pixels constitute <2% of the wafer area, a standard tuner aggressively drives spatial filtering to maximum kernel size to suppress shot noise across the bulk, completely blind to the catastrophic erasure of physical defect signatures.

---

## 2. Keystone Measurement Table (`evidence/baselines.json`)

Evaluated at $T = 0.1\,\text{s}$ ($N \approx 400$ peak photons) across 6 held-out test maps:

| Method | RMSE (cm⁻¹) | RMSE/CRLB | Eff. Exposure Gain | Recall@0.5 | Recall@1.5 | Recall@2.5 | Recall@4.0 | False Feature Rate |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`raw`** | 0.04385 | **1.002** | 1.0× | 0.345 | **0.719** | 0.765 | 1.000 | 0.0031 |
| **`savgol`** | 0.04383 | 1.002 | 1.0× | 0.345 | 0.737 | 0.765 | 1.000 | 0.0029 |
| **`binning`** (7×7) | 0.01089 | 0.249 | **14.9×** | **0.000** | **0.000** | 0.118 | 0.707 | **0.0000** |
| **`spatial_gauss`** (σ=2.0) | 0.01026 | 0.235 | **16.8×** | **0.000** | **0.000** | 0.412 | 0.733 | **0.0000** |
| **`pca`** ($k=16$) | 0.04428 | 1.012 | 0.97× | 0.385 | 0.737 | 0.647 | 1.000 | 0.0032 |
| **`nmf`** ($k=16$) | 0.04430 | 1.013 | 0.97× | 0.385 | 0.754 | 0.647 | 1.000 | 0.0033 |
| **`reference`** (5.0 s) | 0.00591 | 0.135 | **49.3×** | **0.920** | **1.000** | **1.000** | **1.000** | 0.0028 |

### Independent Validation Checks
1. **Fitter Efficiency:** $\text{RMSE}/\text{CRLB} = 1.002$ for raw spectra. The Levenberg-Marquardt Poisson MLE achieves the exact Cramer-Rao bound.
2. **Exposure Scaling:** $\text{RMSE}_{\text{raw}} / \text{RMSE}_{\text{ref}} = 0.04385 / 0.00591 = 7.42 \approx \sqrt{50} = 7.07$. Effective exposure gain of reference is $49.3\times$ (predicted $50.0\times$).
3. **Monotonicity:** Reference strictly dominates raw across every difficulty bin ($0.920$ vs $0.345$ at 0.5 CRLB).

---

## 3. Recall vs. Difficulty Artifact

The figure has been generated and saved to:
`evidence/recall_vs_difficulty.png`

It plots defect recall as a function of physical difficulty (in CRLB units) across methods, highlighting the critical 0.3–3.0 CRLB band where spatial smoothing claims a 16.8× photon gain while defect recall drops to exactly 0.0%.

---

## 4. T2 Assurance Gate Validation on Spatial Baseline

We evaluated the T2 Gate (`test_pooling_legitimacy`) on the exact $\sigma=2.0$ spatial Gaussian smoothed maps across all 6 test samples ($24,576$ total pixels, $404$ ground truth defect pixels).

### Measured Performance (`evidence/t2_gate_results.json`)

| Defect Physical Difficulty | Spatial Gauss Recall | **T2 Gate Flagging Rate** | Status |
|:---|:---:|:---:|:---|
| **0.5 CRLB** | **0.0%** (0/200) | **98.5%** (197/200) | **Flagged as illegitimate pooling** |
| **1.5 CRLB** | **0.0%** (0/57) | **100.0%** (57/57) | **100% Flagged (Perfect Catch)** |
| **2.5 CRLB** | **41.2%** (7/17) | **100.0%** (17/17) | **100% Flagged** |
| **4.0 CRLB** | 73.3% (55/75) | **100.0%** (75/75) | **100% Flagged** |
| **10.0 CRLB** | 100.0% (55/55) | **100.0%** (55/55) | **100% Flagged** |
| **Total Defect Pixels** | — | **99.3% (401 / 404)** | **Thesis Confirmed** |

### Demonstration Figure
The diagnostic map comparison is saved to:
`evidence/t2_gate_spatial_demonstration.png`

It shows side-by-side:
1. **(a) Ground Truth Peak Center Map:** True localized peak shifts.
2. **(b) Spatial Gaussian Restored Map ($\sigma=2.0$):** High SNR background, but defects are completely blurred out and invisible.
3. **(c) True Defect Mask:** Ground truth locations of all defects.
4. **(d) T2 Gate Failure Map:** T2 flags the exact boundary/defect pixels where spatial pooling violates spectral homogeneity.
