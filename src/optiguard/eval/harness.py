import json
from dataclasses import dataclass
from typing import Dict, Any, List, Callable, Optional
import numpy as np
from pathlib import Path
from optiguard.physics.crlb import crlb_peak_position, crlb_peak_position_map

@dataclass
class Result:
    mae_center: float            # over NON-defect pixels, cm^-1
    rmse_center: float           # over NON-defect pixels — use this against CRLB (same units)
    median_center: float         # catch outlier-driven MAE
    mae_fwhm: float
    mean_crlb: float             # mean CRLB over non-defect pixels at this exposure
    rmse_over_crlb: float        # rmse_center / mean_crlb — efficiency ratio; >1.3 => fitter lossy
    convergence_rate: float      # fraction of pixels with valid sigma_center from fitter
    recall_by_difficulty: Dict[float, float]     # the key curve
    false_feature_rate: float    # non-defect pixels where |error| > 3*sigma_pred
    per_pixel: Dict[str, np.ndarray]
    n_pixels_scored: int


def evaluate(method: Callable, samples: List, exposure: float = 0.1) -> Result:
    errors_center = []
    sq_errors_center = []
    errors_fwhm = []
    crlb_vals = []
    false_features = []
    total_non_defect = 0
    valid_pixels = 0
    total_pixels = 0

    diff_bins = [0.0, 1.0, 2.0, 3.0, 5.0, np.inf]
    diff_bin_centers = [0.5, 1.5, 2.5, 4.0, 10.0]
    recalls = {c: {"hits": 0, "total": 0} for c in diff_bin_centers}

    # Sort samples deterministically
    sorted_samples = sorted(samples, key=lambda s: float(np.mean(s.theta_true["center"])))

    for s in sorted_samples:
        theta_pred = method(s, exposure)

        mask = ~s.defect_mask
        c_err = np.abs(theta_pred["center"][mask] - s.theta_true["center"][mask])
        f_err = np.abs(theta_pred["fwhm"][mask] - s.theta_true["fwhm"][mask])

        errors_center.extend(c_err.tolist())
        sq_errors_center.extend((c_err ** 2).tolist())
        errors_fwhm.extend(f_err.tolist())
        total_non_defect += int(mask.sum())

        # sigma_center from the fitter — this is the per-pixel uncertainty
        sigma_pred = theta_pred.get(
            "sigma_center", np.full_like(theta_pred["center"], np.inf)
        )
        valid_pixels += int(np.sum(np.isfinite(sigma_pred) & (sigma_pred > 0) & (sigma_pred < 100)))
        total_pixels += sigma_pred.size

        rn = s.meta.get("read_noise_e", 0.0)
        # difficulty at 'exposure' — fixed coordinate frame regardless of what the method uses
        diff_map = s.difficulty(exposure)

        # CRLB at evaluation exposure — used for false-feature threshold on non-defect pixels
        crlb_map = crlb_peak_position_map(
            axis=s.axis,
            center=s.theta_true["center"],
            fwhm=s.theta_true["fwhm"],
            amplitude=s.theta_true["amplitude"] * exposure,
            background=s.theta_true["background"] * exposure,
            read_noise_e=rn
        )

        # False features: non-defect pixels where error exceeds 3 * sigma_pred.
        # Use sigma_pred (method's own uncertainty) not crlb_map: this scales with
        # the method's actual exposure, so reference doesn't get a widened gate.
        sigma_non_defect = sigma_pred[mask]
        c_err_masked = np.abs(theta_pred["center"] - s.theta_true["center"])[mask]
        # Fall back to crlb_map if sigma_pred is infinite (e.g. non-fitter baselines)
        threshold_ff = np.where(
            np.isfinite(sigma_non_defect) & (sigma_non_defect > 0),
            3.0 * sigma_non_defect,
            3.0 * crlb_map[mask]
        )
        crlb_vals.extend(crlb_map[mask].tolist())
        false_features.extend((c_err_masked > threshold_ff).astype(int).tolist())

        # Recall on defect pixels.
        # Detection criterion: |pred_center - bg_center| > k * sigma_pred.
        # Using sigma_pred (the method's noise) makes the threshold scale correctly:
        # - raw at 0.1 s: sigma_pred ~ CRLB(0.1s), threshold = 1 * CRLB(0.1s)
        # - reference at 5.0 s: sigma_pred ~ CRLB(5.0s), threshold = 1 * CRLB(5.0s) = CRLB(0.1s)/sqrt(50)
        # A 0.5-CRLB(0.1s) defect is 0.5*sqrt(50) ~ 3.5 sigma for reference => easily found.
        # Without this: reference's tight predictions are compared to CRLB(0.1s) threshold
        # and sub-CRLB(0.1s) defects are never flagged even though they are detected.
        defect_mask = s.defect_mask
        bg_center = s.theta_true["center"][defect_mask] - s.defect_shift[defect_mask]
        sigma_defect = sigma_pred[defect_mask]
        crlb_defect = crlb_map[defect_mask]
        threshold_recall = np.where(
            np.isfinite(sigma_defect) & (sigma_defect > 0),
            1.0 * sigma_defect,
            1.0 * crlb_defect
        )
        found = np.abs(theta_pred["center"][defect_mask] - bg_center) > threshold_recall

        d_vals = diff_map[defect_mask]
        for d, f in zip(d_vals.tolist(), found.tolist()):
            for k in range(len(diff_bins) - 1):
                if diff_bins[k] <= d < diff_bins[k + 1]:
                    b = diff_bin_centers[k]
                    recalls[b]["total"] += 1
                    if f:
                        recalls[b]["hits"] += 1
                    break

    recall_curve = {
        b: counts["hits"] / counts["total"]
        for b, counts in recalls.items()
        if counts["total"] > 0
    }

    mean_crlb = float(np.mean(crlb_vals)) if crlb_vals else float("nan")
    mae_c = float(np.mean(errors_center)) if errors_center else 0.0
    rmse_c = float(np.sqrt(np.mean(sq_errors_center))) if sq_errors_center else 0.0

    return Result(
        mae_center=mae_c,
        rmse_center=rmse_c,
        median_center=float(np.median(errors_center)) if errors_center else 0.0,
        mae_fwhm=float(np.mean(errors_fwhm)) if errors_fwhm else 0.0,
        mean_crlb=mean_crlb,
        rmse_over_crlb=rmse_c / mean_crlb if mean_crlb > 0 else float("nan"),
        convergence_rate=float(valid_pixels / total_pixels) if total_pixels > 0 else 0.0,
        recall_by_difficulty=recall_curve,
        false_feature_rate=float(np.mean(false_features)) if false_features else 0.0,
        per_pixel={},
        n_pixels_scored=total_non_defect,
    )


def sweep_exposures(method: Callable, samples: List, exposures: List[float]) -> Dict[float, float]:
    return {t: evaluate(method, samples, t).rmse_center for t in exposures}


def effective_exposure_gain(
    method: Callable, samples: List, exposure: float = 0.1
) -> Optional[float]:
    """Effective exposure gain relative to raw at 'exposure'.

    Sweeps raw RMSE across all available short_counts exposures, then appends
    the reference (long_counts at reference_integration_s) as the top anchor.
    This gives the log-log fit a wide lever arm and allows the reference row
    to report its ~50x gain correctly.

    Returns None when extrapolation falls outside a credible range.
    Never returns a silent 1.0 — that was the broken behaviour.
    """
    from optiguard.eval.baselines import BASELINES

    rmse_m = evaluate(method, samples, exposure).rmse_center

    # Use only exposures that exist in short_counts — avoids KeyError
    available_short = sorted(samples[0].short_counts.keys())
    ref_t = float(samples[0].meta["reference_integration_s"])

    raw_curve: Dict[float, float] = {}
    for t in available_short:
        raw_curve[t] = evaluate(BASELINES["raw"], samples, t).rmse_center
    # Top anchor: reference method at reference exposure gives raw RMSE at ref_t
    raw_curve[ref_t] = evaluate(BASELINES["reference"], samples, exposure).rmse_center

    ts = np.array(sorted(raw_curve.keys()))
    rmses = np.array([raw_curve[t] for t in ts])

    # log-log fit: RMSE ~ 1/sqrt(t) => log RMSE = -0.5 log t + c
    p = np.polyfit(np.log(ts), np.log(rmses), 1)

    if rmse_m <= 0:
        return None
    t_raw = np.exp((np.log(rmse_m) - p[1]) / p[0])
    gain = float(t_raw / exposure)

    if gain < 0.1 or gain > 1000:
        return None

    return gain


def write_baseline_table(samples: List, exposure: float, out_dir: str):
    from optiguard.eval.baselines import BASELINES, tune_baseline
    table = {}

    n = len(samples)
    val_samples = samples[: n // 2]
    test_samples = samples[n // 2 :]

    for name in ["raw", "savgol", "binning", "spatial_gauss", "pca", "nmf", "reference"]:
        method, params = tune_baseline(name, val_samples, exposure)
        res = evaluate(method, test_samples, exposure)
        gain = effective_exposure_gain(method, test_samples, exposure)

        table[name] = {
            "mae_center": res.mae_center,
            "rmse_center": res.rmse_center,
            "median_center": res.median_center,
            "mean_crlb": res.mean_crlb,
            "rmse_over_crlb": res.rmse_over_crlb,
            "convergence_rate": res.convergence_rate,
            "effective_exposure_gain": gain,
            "tuned_params": params,
        }

    out_path = Path(out_dir) / "baseline_table.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, indent=2))
    return out_path
