import numpy as np
import sys
import json
sys.stdout.reconfigure(encoding='utf-8')

from optiguard.data.simulator import MapSimulator
from optiguard.assurance.ood import auroc_against_silicon, calibrate_threshold, score_map
from sklearn.metrics import roc_auc_score

print("=== OOD Difficulty Sweep ===")
sim = MapSimulator.from_yaml('configs/simulator.yaml')

# Calibrate null distribution
print("Generating null distribution (silicon)...")
maps_si = [sim.generate(index=i, material="silicon") for i in range(6)]
maps_si_test = [sim.generate(index=i, material="silicon") for i in range(6, 9)]

thr = calibrate_threshold(material="silicon", indices=range(0, 6), target_fpr=0.05)
null_scores = np.concatenate([score_map(m.axis, m.long_counts).flatten() for m in maps_si_test])
null_fpr = (null_scores > thr).mean()
print(f"Null FPR at threshold {thr:.3f}: {null_fpr:.3f}")

# Sweep FWHM for a single broad mode, centered exactly at 520.7
fwhm_sweep = np.linspace(3.5, 6.0, 11)
results = {}

for fwhm in fwhm_sweep:
    # Patch the config dynamically for this run
    sim.config["materials"]["si_broad"]["peaks"][0]["fwhm"] = float(fwhm)
    
    print(f"Testing FWHM {fwhm:.2f}... ", end="", flush=True)
    maps_broad = [sim.generate(index=i, material="si_broad") for i in range(6, 9)]
    
    auroc = float(auroc_against_silicon(maps_broad, maps_si_test, method="physics"))
    
    # Calculate True Positive Rate at the calibrated threshold
    scores = np.concatenate([score_map(m.axis, m.long_counts).flatten() for m in maps_broad])
    tpr = float((scores > thr).mean())
    
    results[float(fwhm)] = {"auroc": auroc, "tpr": tpr}
    print(f"AUROC: {auroc:.3f} | TPR: {tpr:.3f}")

# Re-run full test suite internally to get new numbers for the real OOD axes
print("\n=== Re-evaluating standard OOD materials ===")
auroc_standard = {}
for mat in ["amorphous_silicon", "sic_4h", "gan", "si_doublet"]:
    maps_mat = [sim.generate(index=i, material=mat) for i in range(6, 9)]
    a = float(auroc_against_silicon(maps_mat, maps_si_test, method="physics"))
    t = float(np.mean([ (score_map(m.axis, m.long_counts) > thr).mean() for m in maps_mat ]))
    auroc_standard[mat] = {"auroc": a, "tpr": t}
    print(f"{mat}: AUROC {a:.3f} | TPR {t:.3f}")

out = {
    "fwhm_sweep": results,
    "standard_materials": auroc_standard
}

with open("evidence/ood_sweep_results.json", "w") as f:
    json.dump(out, f, indent=2)
    
print("\nResults saved to evidence/ood_sweep_results.json")
