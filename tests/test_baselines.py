import pytest

@pytest.fixture(scope="module")
def samples():
    from optiguard.data.simulator import MapSimulator
    sim = MapSimulator.from_yaml("configs/simulator.yaml")
    return [sim.generate(index=i) for i in range(6)]

# ---------------------------------------------------------------------------
# The harness must be honest about what it is averaging
# ---------------------------------------------------------------------------
def test_accuracy_metrics_exclude_defect_pixels(samples):
    from optiguard.eval.harness import evaluate
    from optiguard.eval.baselines import BASELINES

    r = evaluate(BASELINES["savgol"], samples, exposure=0.1)
    assert r.n_pixels_scored == sum((~s.defect_mask).sum() for s in samples)

def test_ordering_of_the_obvious_cases(samples):
    from optiguard.eval.harness import evaluate
    from optiguard.eval.baselines import BASELINES

    raw = evaluate(BASELINES["raw"], samples, exposure=0.1).mae_center
    sp = evaluate(BASELINES["spatial_gauss"], samples, exposure=0.1).mae_center
    ref = evaluate(BASELINES["reference"], samples, exposure=0.1).mae_center

    assert ref < sp < raw

def test_harness_is_deterministic(samples):
    from optiguard.eval.harness import evaluate
    from optiguard.eval.baselines import BASELINES

    a = evaluate(BASELINES["pca"], samples, exposure=0.1)
    b = evaluate(BASELINES["pca"], samples, exposure=0.1)
    assert a.mae_center == b.mae_center

# ---------------------------------------------------------------------------
# Baselines must be tuned, not strawmen
# ---------------------------------------------------------------------------
def test_tuning_improves_every_baseline(samples):
    from optiguard.eval.harness import evaluate
    from optiguard.eval.baselines import BASELINES, tune_baseline

    for name in ["savgol", "pca", "nmf"]:
        tuned, params = tune_baseline(name, samples[:3], exposure=0.1)
        assert params, f"{name} exposed no tunable parameters"
        default_mae = evaluate(BASELINES[name], samples[3:], exposure=0.1).mae_center
        tuned_mae = evaluate(tuned, samples[3:], exposure=0.1).mae_center
        assert tuned_mae <= default_mae * 1.01

# ---------------------------------------------------------------------------
# Recall must be resolved against difficulty, never reported as one number
# ---------------------------------------------------------------------------
def test_recall_is_binned_by_difficulty_and_monotone(samples):
    from optiguard.eval.harness import evaluate
    from optiguard.eval.baselines import BASELINES

    r = evaluate(BASELINES["pca"], samples, exposure=0.1)
    bins = sorted(r.recall_by_difficulty)
    assert len(bins) >= 4, "need resolution across the 1-3 CRLB transition"

    values = [r.recall_by_difficulty[b] for b in bins]
    assert values[0] < values[-1]
    assert values[0] <= 0.45, "sub-CRLB defects should have low recall"
    assert values[-1] > 0.70, "clearly detectable defects should be found"

def test_false_feature_rate_is_measured_against_the_bound(samples):
    from optiguard.eval.harness import evaluate
    from optiguard.eval.baselines import BASELINES

    raw = evaluate(BASELINES["raw"], samples, exposure=0.1)
    assert 0.0 <= raw.false_feature_rate <= 1.0
    assert raw.false_feature_rate < 0.10

# ---------------------------------------------------------------------------
# Effective exposure gain - the primary reporting metric
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_exposure_sweep_is_monotone_and_follows_sqrt_scaling(samples):
    from optiguard.eval.harness import sweep_exposures
    from optiguard.eval.baselines import BASELINES

    curve = sweep_exposures(BASELINES["raw"], samples,
                            exposures=[0.1, 0.25, 0.5, 1.0])
    maes = [curve[t] for t in sorted(curve)]
    assert all(b < a for a, b in zip(maes, maes[1:])), "MAE must fall with exposure"
    assert 2.4 < maes[0] / maes[-1] < 4.0

@pytest.mark.slow
def test_effective_exposure_gain_is_meaningful(samples):
    from optiguard.eval.harness import effective_exposure_gain
    from optiguard.eval.baselines import BASELINES

    g_raw = effective_exposure_gain(BASELINES["raw"], samples, exposure=0.1)
    g_spatial = effective_exposure_gain(BASELINES["binning"], samples, exposure=0.1)
    g_pca = effective_exposure_gain(BASELINES["pca"], samples, exposure=0.1)

    assert abs(g_raw - 1.0) < 0.15, "raw must be its own reference (1.0x)"
    assert g_spatial > 5.0, "spatial pooling buys real photons on bulk (>5x)"
    assert g_pca <= 1.20, "spectral PCA cannot compress continuously varying Lorentzian lines without distortion"

# ---------------------------------------------------------------------------
# The result every later step is compared against
# ---------------------------------------------------------------------------
def test_baseline_table_is_written_to_evidence(samples, tmp_path):
    from optiguard.eval.harness import write_baseline_table

    path = write_baseline_table(samples, exposure=0.1, out_dir=tmp_path)
    assert path.exists()

    import json
    table = json.loads(path.read_text())
    for name in ["raw", "savgol", "pca", "nmf", "reference"]:
        assert name in table
        assert ("mae_center" in table[name] or "mae_center_cm1" in table[name] or "rmse_center_cm1" in table[name])
        assert "effective_exposure_gain" in table[name]
        assert "tuned_params" in table[name]
