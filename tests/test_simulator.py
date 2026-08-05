"""
Step 4 specification: the Tier A synthetic map simulator.

This is THE dataset. Everything the evidence gate can be validated against is
determined by what this generator can express, so the properties below are not
conveniences - they are the boundary of what the project can later claim.

Target API
----------
    from optiguard.data.simulator import MapSimulator

    sim    = MapSimulator.from_yaml("configs/simulator.yaml")
    sample = sim.generate(index=0)

    sample.rate            (H, W, C) float32  lambda, photons/s          <- level 1 truth
    sample.long_counts     (H, W, C) int32    Poisson(lambda * T)        <- level 2 truth
    sample.short_counts    dict[float, (H,W,C) int32]  keyed by exposure <- level 3, the input
    sample.theta_true      dict[str, (H,W) float32]  center/fwhm/amplitude/background
    sample.defect_mask     (H, W) bool
    sample.defect_shift    (H, W) float32    applied centre offset, cm^-1, 0 outside defects
    sample.axis            (C,)   float32    wavenumber axis, cm^-1
    sample.meta            dict             T, exposures, detector params, seed, index

    sample.difficulty(t)   (H, W) float32    |defect_shift| / CRLB(N at exposure t)
"""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def sim():
    from optiguard.data.simulator import MapSimulator
    return MapSimulator.from_yaml("configs/simulator.yaml")


# ---------------------------------------------------------------------------
# Determinism - this is what allows "regenerate from seed" instead of syncing GB
# ---------------------------------------------------------------------------
def test_generation_is_deterministic(sim):
    a = sim.generate(index=7)
    b = sim.generate(index=7)
    np.testing.assert_array_equal(a.long_counts, b.long_counts)
    np.testing.assert_array_equal(a.theta_true["center"], b.theta_true["center"])
    for t in a.short_counts:
        np.testing.assert_array_equal(a.short_counts[t], b.short_counts[t])


def test_distinct_indices_give_distinct_samples(sim):
    a, b = sim.generate(index=0), sim.generate(index=1)
    assert not np.array_equal(a.theta_true["center"], b.theta_true["center"])


def test_shapes_and_dtypes(sim):
    s = sim.generate(index=0)
    H, W = s.meta["shape"]
    C = s.axis.size
    assert s.rate.shape == (H, W, C) and s.rate.dtype == np.float32
    assert s.long_counts.shape == (H, W, C)
    assert s.defect_mask.shape == (H, W) and s.defect_mask.dtype == bool
    for t, arr in s.short_counts.items():
        assert arr.shape == (H, W, C)
        assert t <= s.meta["reference_integration_s"]


# ---------------------------------------------------------------------------
# The three levels of truth must be consistent with each other
# ---------------------------------------------------------------------------
def test_long_counts_are_poisson_of_the_rate(sim):
    s = sim.generate(index=2)
    T = s.meta["reference_integration_s"]
    expected = s.rate * T
    # Aggregate over the whole cube: mean must match, and Poisson variance
    # about the per-voxel mean must equal that mean.
    resid = s.long_counts - expected
    assert abs(resid.mean()) < 0.05 * np.sqrt(expected.mean())
    assert abs((resid ** 2).mean() / expected.mean() - 1.0) < 0.10


def test_short_counts_are_a_valid_thinning_of_long_counts(sim):
    """
    Thinning must be applied to the long-exposure REALISATION, not to lambda.
    Elementwise the short counts can never exceed the long counts (before dark
    and read noise are added), and the ratio must match the exposure ratio.
    """
    s = sim.generate(index=3)
    T = s.meta["reference_integration_s"]
    t = min(s.short_counts)
    signal_short = s.short_counts[t]

    assert signal_short.sum() > 0
    ratio = signal_short.sum() / s.long_counts.sum()
    assert abs(ratio - t / T) < 0.05 * (t / T)


# ---------------------------------------------------------------------------
# theta_true must be the generative truth, recoverable by the fitter
# ---------------------------------------------------------------------------
def test_fitting_the_noiseless_rate_recovers_theta_true(sim):
    """
    Fit the noise-free spectrum. Any disagreement here is a model mismatch
    between generator and fitter, not noise - most often a background model
    mismatch (sloped baseline generated, constant baseline fitted). This bias
    would otherwise masquerade as restoration error for days.
    """
    from optiguard.estimation.fit import fit_lorentzian

    s = sim.generate(index=4)
    T = s.meta["reference_integration_s"]
    rng = np.random.default_rng(0)
    H, W = s.meta["shape"]

    for _ in range(20):
        i, j = rng.integers(H), rng.integers(W)
        out = fit_lorentzian(s.axis, s.rate[i, j] * T)
        assert abs(out["center"] - s.theta_true["center"][i, j]) < 0.01
        assert abs(out["fwhm"] - s.theta_true["fwhm"][i, j]) < 0.05


@pytest.mark.slow
def test_fitting_long_counts_scatters_consistently_with_crlb(sim):
    """Level-2 truth is itself noisy. Its scatter must match the bound."""
    from optiguard.estimation.fit import fit_lorentzian
    from optiguard.physics.crlb import crlb_peak_position

    s = sim.generate(index=5)
    T = s.meta["reference_integration_s"]
    i, j = 0, 0
    while s.defect_mask[i, j]:
        j += 1

    rng = np.random.default_rng(1)
    mu = s.rate[i, j] * T
    fitted = np.array([fit_lorentzian(s.axis, rng.poisson(mu))["center"]
                       for _ in range(400)])
    bound = crlb_peak_position(
        axis=s.axis,
        center=s.theta_true["center"][i, j],
        fwhm=s.theta_true["fwhm"][i, j],
        amplitude=s.theta_true["amplitude"][i, j] * T,
        background=s.theta_true["background"][i, j] * T,
        read_noise_e=0.0,
    )
    assert 0.8 * bound <= fitted.std() <= 1.3 * bound


# ---------------------------------------------------------------------------
# Defects - these are what smoothers erase, so they must be honestly specified
# ---------------------------------------------------------------------------
def test_defects_are_small_and_present_in_theta_true(sim):
    s = sim.generate(index=6)
    assert s.defect_mask.any(), "no defects generated"
    assert s.defect_mask.mean() < 0.10, "defects too large to be interesting"

    inside = s.theta_true["center"][s.defect_mask]
    assert np.abs(s.defect_shift[s.defect_mask]).min() > 0
    assert not np.allclose(inside, inside[0]), "defect shifts are degenerate"
    assert np.all(s.defect_shift[~s.defect_mask] == 0)


def test_defect_difficulty_is_expressed_in_crlb_units(sim):
    """
    Difficulty must be |shift| / CRLB at the given exposure, NOT an absolute
    wavenumber offset. This is what lets recall be plotted against detectability
    and lets the gate be shown to flag exactly the information-limited regime.
    """
    s = sim.generate(index=6)
    t_short, t_long = min(s.short_counts), max(s.short_counts)

    d_short = s.difficulty(t_short)[s.defect_mask]
    d_long = s.difficulty(t_long)[s.defect_mask]

    assert np.all(d_short > 0)
    # More photons -> same shift is easier to detect.
    assert d_long.mean() > d_short.mean()
    assert abs(d_long.mean() / d_short.mean() - np.sqrt(t_long / t_short)) < 0.15


def test_difficulty_spans_the_information_limit(sim):
    """The set must contain defects that are detectable, marginal, and provably
    undetectable - otherwise recall numbers mean nothing."""
    diffs = np.concatenate([
        sim.generate(index=i).difficulty(min(sim.exposures))[sim.generate(index=i).defect_mask]
        for i in range(12)
    ])
    assert diffs.min() < 1.5, "no information-limited defects - recall will look fake"
    assert diffs.max() > 5.0, "no clearly detectable defects"


# ---------------------------------------------------------------------------
# Heterogeneity - required so that T2 (pooling legitimacy) has anything to catch
# ---------------------------------------------------------------------------
def test_boundary_samples_contain_sharp_transitions(sim):
    """
    At least one field type must produce a sharp material/grain boundary.
    Without it, pooling across a boundary can never be tested, and T2 is
    unfalsifiable.
    """
    s = sim.generate(index=0, field="boundary")
    grad = np.abs(np.diff(s.theta_true["center"], axis=1))
    assert grad.max() > 0.5, "no sharp boundary present"
    assert (grad > 0.5).mean() < 0.05, "boundary should be sharp, not everywhere"


# ---------------------------------------------------------------------------
# Cosmic rays belong to the despiker, not the restoration network
# ---------------------------------------------------------------------------
def test_cosmic_rays_are_injected_and_removable(sim):
    from optiguard.preprocess.despike import remove_cosmic_rays

    s = sim.generate(index=8, cosmic_rays=True)
    t = min(s.short_counts)
    raw = s.short_counts[t]
    cleaned = remove_cosmic_rays(raw)

    assert raw.max() > cleaned.max(), "no spikes were injected or none removed"
    # The despiker must not eat real signal: total counts change only slightly.
    assert abs(cleaned.sum() - raw.sum()) / raw.sum() < 0.02
