import numpy as np
import pytest

@pytest.fixture(scope="module")
def maps():
    from optiguard.data.simulator import MapSimulator
    sim = MapSimulator.from_yaml("configs/simulator.yaml")
    return {m: [sim.generate(index=i, material=m) for i in range(6, 9)]
            for m in ["silicon", "amorphous_silicon", "sic_4h", "gan", "si_doublet"]}

def test_in_distribution_null_is_calibrated(maps):
    """Held-out silicon must sit below the operating threshold at the designed rate.
    Calibrate on separate silicon maps; never on the maps being scored."""
    from optiguard.assurance.ood import score_map, calibrate_threshold
    thr = calibrate_threshold(material="silicon", indices=range(0, 6), target_fpr=0.05)
    rate = np.mean([ (score_map(m.axis, m.long_counts) > thr).mean() for m in maps["silicon"] ])
    assert 0.02 <= rate <= 0.10, f"null firing rate {rate:.3f}, expected ~0.05"

@pytest.mark.parametrize("material", ["amorphous_silicon", "sic_4h", "gan"])
def test_gross_material_shift_is_detected(maps, material):
    from optiguard.assurance.ood import score_map, calibrate_threshold
    thr = calibrate_threshold(material="silicon", indices=range(0, 6), target_fpr=0.05)
    rate = np.mean([ (score_map(m.axis, m.long_counts) > thr).mean() for m in maps[material] ])
    assert rate > 0.80, f"{material} detected on only {rate:.1%} of pixels"

def test_the_hard_case_overlapping_doublet(maps):
    """Two modes 2 cm^-1 apart fit as one broad peak. The fit converges and reports a
    tight sigma, so T1a/T1b/T4 all pass. This is the case that must not be missed."""
    from optiguard.assurance.ood import score_map, calibrate_threshold
    thr = calibrate_threshold(material="silicon", indices=range(0, 6), target_fpr=0.05)
    rate = np.mean([ (score_map(m.axis, m.long_counts) > thr).mean() for m in maps["si_doublet"] ])
    assert rate > 0.50, f"overlapping doublet detected on only {rate:.1%} of pixels"

def test_existing_gate_does_NOT_catch_the_doublet(maps):
    """Establishes why this step exists. If the gate already caught it, say so and
    simplify. Run T1a/T1b/T4 on the doublet and record the pass rate."""
    from optiguard.assurance.gate import evaluate_gate
    sample = maps["si_doublet"][0]
    res = evaluate_gate(sample, restored_counts=None, neff_map=None)
    print("doublet gate pass rate:", res["summary"]["pass_rate"])
    
def test_auroc_per_material(maps):
    """Report AUROC per material, never as one aggregate. A single number hides
    exactly the case a customer cares about."""
    from optiguard.assurance.ood import auroc_against_silicon
    for m in ["amorphous_silicon", "sic_4h", "gan", "si_doublet"]:
        a = auroc_against_silicon(maps[m], maps["silicon"])
        print(f"{m}: AUROC {a:.3f}")
        assert a >= 0.85, f"{m} AUROC {a:.3f}"

def test_physics_detector_is_compared_against_the_ml_baseline(maps):
    """Report both. The hypothesis is that residual structure beats embedding
    distance, because the failure IS model mismatch. If the embedding wins, that is
    also a finding — but it must be measured, not assumed."""
    from optiguard.assurance.ood import auroc_by_method
    table = auroc_by_method(maps)
    for method in ("residual_structure", "parameter_plausibility", "embedding_distance"):
        assert method in table
    print(table)

def test_score_never_touches_ground_truth(maps):
    """Production path runs on real measurements where theta_true does not exist."""
    import inspect
    from optiguard.assurance import ood
    src = inspect.getsource(ood)
    assert "theta_true" not in src, "OOD scoring must not reference ground truth"

def test_ood_suppresses_confidence_not_just_annotates(maps):
    """A confident number from a wrong model is worse than no number."""
    from optiguard.assurance.gate import evaluate_gate
    si_sample = maps["silicon"][0]
    ood_sample = maps["sic_4h"][0]
    si = evaluate_gate(si_sample, restored_counts=None, neff_map=None)
    ood_res = evaluate_gate(ood_sample, restored_counts=None, neff_map=None)
    assert ood_res["summary"]["mean_confidence"] < 0.5 * si["summary"]["mean_confidence"]

def test_rationale_names_the_failed_assumption(maps):
    from optiguard.assurance.ood import explain
    r = explain(maps["amorphous_silicon"][0].axis, maps["amorphous_silicon"][0].long_counts)
    assert "fwhm" in r.lower() or "linewidth" in r.lower()
    assert any(ch.isdigit() for ch in r), "rationale must carry the measured value"
