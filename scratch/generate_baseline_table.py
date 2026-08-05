"""
Step 7 deliverable: generate evidence/baselines.json
Runs all baselines with tuning (val split) and evaluation (test split).
exposure=0.1s. Spatial baselines included.
"""
import sys, json
sys.path.insert(0, "src")

from optiguard.data.simulator import MapSimulator
from optiguard.eval.harness import evaluate, effective_exposure_gain
from optiguard.eval.baselines import BASELINES, tune_baseline

print("Generating 12 samples...", flush=True)
sim = MapSimulator.from_yaml("configs/simulator.yaml")
all_samples = [sim.generate(index=i) for i in range(12)]
val_samples  = all_samples[:6]
test_samples = all_samples[6:]
print(f"Done. Val={len(val_samples)} Test={len(test_samples)}", flush=True)

EXPOSURE = 0.1

table = {}
for name in ["raw", "savgol", "binning", "spatial_gauss", "pca", "nmf", "reference"]:
    print(f"\n=== {name} ===", flush=True)
    method, params = tune_baseline(name, val_samples, EXPOSURE)
    print(f"  Tuned params: {params}", flush=True)
    res = evaluate(method, test_samples, EXPOSURE)
    print(f"  RMSE center: {res.rmse_center:.5f}", flush=True)
    print(f"  MAE center:  {res.mae_center:.5f}", flush=True)
    print(f"  Mean CRLB:   {res.mean_crlb:.5f}", flush=True)
    print(f"  RMSE/CRLB:   {res.rmse_over_crlb:.3f}  (>1.3 => fitter lossy)", flush=True)
    print(f"  Convergence: {res.convergence_rate:.4f}", flush=True)
    print(f"  Recall by difficulty: {res.recall_by_difficulty}", flush=True)
    print(f"  False feature rate: {res.false_feature_rate:.5f}", flush=True)

    gain = effective_exposure_gain(method, test_samples, EXPOSURE)
    print(f"  Effective exposure gain: {gain}", flush=True)

    table[name] = {
        "rmse_center_cm1": res.rmse_center,
        "mae_center_cm1": res.mae_center,
        "median_center_cm1": res.median_center,
        "mae_fwhm_cm1": res.mae_fwhm,
        "mean_crlb_cm1": res.mean_crlb,
        "rmse_over_crlb": res.rmse_over_crlb,
        "convergence_rate": res.convergence_rate,
        "recall_by_difficulty": {str(k): v for k, v in res.recall_by_difficulty.items()},
        "false_feature_rate": res.false_feature_rate,
        "effective_exposure_gain": gain,
        "tuned_params": params
    }

table["_meta"] = {
    "test_indices": [6, 7, 8, 9, 10, 11],
    "train_indices": [0, 1, 2, 3, 4, 5],
    "exposure_s": EXPOSURE,
    "reference_exposure_s": 5.0
}

import pathlib
out = pathlib.Path("evidence/baselines.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(table, indent=2))
print(f"\nWrote {out}")
print(json.dumps(table, indent=2))
