"""
Keystone physics tests. These ARE the specification for Steps 1-3, 6, 9 and 10.

Write implementations until these pass. Do not weaken an assertion to make it
pass - if a test fails, the physics is wrong, and everything downstream
(the evidence gate, the planner, the whole product claim) is unsound.
"""
import numpy as np
import pytest
from scipy import stats


# ---------------------------------------------------------------------------
# STEP 3 - photon thinning must be exact, not approximate
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_thinning_matches_direct_simulation():
    """
    A short-integration acquisition is a binomially thinned long-integration
    acquisition. This is a theorem about Poisson processes, not a modelling
    choice - and it is the reason the v1.0 circularity objection goes away.

    Thinned samples must be statistically indistinguishable from samples drawn
    directly at the short exposure.
    """
    from optiguard.physics.thinning import thin_counts

    rng = np.random.default_rng(0)
    rate, T, t, n = 300.0, 5.0, 0.5, 20000

    long_counts = rng.poisson(rate * T, size=n)
    thinned = thin_counts(long_counts, t_target=t, t_source=T, rng=rng)
    direct = rng.poisson(rate * t, size=n)

    assert stats.ks_2samp(thinned, direct).pvalue > 0.01
    assert abs(thinned.mean() - direct.mean()) < 0.05 * direct.mean()
    # Poisson: variance equals mean. A wrong implementation usually breaks this.
    assert abs(thinned.var() / thinned.mean() - 1.0) < 0.05


def test_read_noise_and_dark_do_not_thin():
    """
    Only the signal thins. Read noise is per-readout and dark current scales
    with time independently. Conflating them is the single most likely bug in
    the whole physics module.
    """
    from optiguard.physics.detector import simulate_acquisition

    rng = np.random.default_rng(1)
    kw = dict(signal_rate=np.zeros(256), read_noise_e=4.0,
              dark_rate_e_per_s=0.02, rng=rng)
    short = simulate_acquisition(integration_s=0.1, **kw)
    long_ = simulate_acquisition(integration_s=5.0, **kw)

    # With zero signal, variance is dominated by read noise, which does NOT
    # grow with integration time; only the dark term does.
    assert long_.var() > short.var()
    assert long_.var() < 5.0 * short.var()


# ---------------------------------------------------------------------------
# STEP 1/2 - detector calibration and lineshapes
# ---------------------------------------------------------------------------
def test_gain_recovered_from_photon_transfer_curve():
    """Var(ADU) = mean(ADU)/gain + const  ->  slope gives gain."""
    from optiguard.physics.detector import estimate_gain_from_ptc

    rng = np.random.default_rng(2)
    true_gain = 2.4
    levels = [500, 2000, 8000, 20000, 50000]
    means, variances = [], []
    for e in levels:
        adu = rng.poisson(e, size=4000) / true_gain
        means.append(adu.mean())
        variances.append(adu.var())

    assert abs(estimate_gain_from_ptc(np.array(means), np.array(variances))
               - true_gain) / true_gain < 0.05


def test_lorentzian_analytic_properties():
    from optiguard.physics.lineshapes import lorentzian

    x = np.linspace(490, 550, 20001)
    y = lorentzian(x, center=520.7, fwhm=3.5, amplitude=1.0)

    assert abs(x[np.argmax(y)] - 520.7) < 0.01
    half = y.max() / 2
    above = x[y >= half]
    assert abs((above[-1] - above[0]) - 3.5) < 0.02


# ---------------------------------------------------------------------------
# STEP 6 - THE KEYSTONE. If this fails, nothing downstream is trustworthy.
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_empirical_scatter_matches_crlb():
    """
    Fit many independent noisy realisations of the same spectrum. The scatter
    of the fitted peak positions must match the Cramer-Rao lower bound computed
    from the photon budget.

    NOTE: the fit must be a Poisson MLE or a correctly weighted least squares.
    Unweighted least squares is inefficient at low counts and will legitimately
    exceed the bound - that is a fitting bug, not a CRLB bug.
    """
    from optiguard.physics.lineshapes import lorentzian
    from optiguard.physics.crlb import crlb_peak_position
    from optiguard.estimation.fit import fit_lorentzian

    rng = np.random.default_rng(3)
    axis = np.arange(500.0, 541.0, 0.55)
    center, fwhm, peak_photons, background = 520.7, 3.5, 400.0, 5.0

    mu = lorentzian(axis, center, fwhm, peak_photons) + background
    fitted = np.array([
        fit_lorentzian(axis, rng.poisson(mu))["center"] for _ in range(1500)
    ])

    empirical = fitted.std()
    bound = crlb_peak_position(axis=axis, center=center, fwhm=fwhm,
                               amplitude=peak_photons, background=background,
                               read_noise_e=0.0)

    assert abs(fitted.mean() - center) < 0.2 * empirical, "fit is biased"
    assert empirical >= 0.85 * bound, "scatter below the bound - CRLB is wrong"
    assert empirical <= 1.25 * bound, "fit is inefficient - check the weighting"


def test_crlb_scales_as_inverse_sqrt_photons():
    """sigma proportional to 1/sqrt(N). Quadrupling photons halves the bound."""
    from optiguard.physics.crlb import crlb_peak_position

    axis = np.arange(500.0, 541.0, 0.55)
    kw = dict(axis=axis, center=520.7, fwhm=3.5, background=0.0, read_noise_e=0.0)
    assert abs(crlb_peak_position(amplitude=100.0, **kw)
               / crlb_peak_position(amplitude=400.0, **kw) - 2.0) < 0.05


# ---------------------------------------------------------------------------
# STEP 9 - effective pooled photon budget
# ---------------------------------------------------------------------------
def test_effective_photon_count_uniform_pooling():
    """Uniform pooling over M pixels of N photons must return exactly M*N."""
    from optiguard.assurance.pooling import effective_photon_count

    M, N = 9, 250.0
    w = np.full(M, 1.0 / M)
    counts = np.full(M, N)
    assert abs(effective_photon_count(w, counts) - M * N) < 1e-6

    # A single dominant weight collapses to a single pixel.
    w2 = np.array([1.0] + [0.0] * (M - 1))
    assert abs(effective_photon_count(w2, counts) - N) < 1e-6


# ---------------------------------------------------------------------------
# STEP 10 - each gate test must fire on its own failure mode
# ---------------------------------------------------------------------------
def test_T1_flags_fabricated_precision():
    """A claimed sigma below the N_eff-adjusted CRLB is physically impossible.

    With neff=1 (default), the floor is the raw single-pixel CRLB.
    With neff=4 (2x2 pool), the floor tightens to crlb/sqrt(4) = crlb/2.
    A claim below either floor must be flagged regardless.
    """
    from optiguard.assurance.gate import test_precision_floor

    # Single-pixel case (neff=1.0 default) - preserved original assertions
    assert test_precision_floor(claimed_sigma=0.010, crlb=0.050)["failed"]
    assert not test_precision_floor(claimed_sigma=0.060, crlb=0.050)["failed"]

    # N_eff-pooled case: floor = 0.050 / sqrt(4) = 0.025
    assert test_precision_floor(claimed_sigma=0.020, crlb=0.050, neff=4.0)["failed"]
    assert not test_precision_floor(claimed_sigma=0.030, crlb=0.050, neff=4.0)["failed"]

    # Verify floor is exactly crlb / sqrt(neff)
    result = test_precision_floor(claimed_sigma=0.030, crlb=0.050, neff=4.0)
    assert abs(result["floor"] - 0.025) < 1e-9


def test_T1_no_double_counting_of_neff():
    """The crlb argument must be the raw single-pixel bound; neff is applied once.

    If a caller pre-divides crlb by sqrt(N_eff) before passing it in AND also
    passes neff>1, the gate divides by sqrt(N_eff) a second time, producing a
    floor that is too tight and will incorrectly reject valid restorations.

    Guard: floor = single_pixel_crlb / sqrt(neff), not further divided.
    Concretely: neff=9 -> floor = crlb / 3 (one division).
    """
    from optiguard.assurance.gate import test_precision_floor

    single_pixel_crlb = 0.090
    neff = 9.0
    expected_floor = single_pixel_crlb / 3.0  # 0.030

    r = test_precision_floor(claimed_sigma=0.040, crlb=single_pixel_crlb, neff=neff)
    assert abs(r["floor"] - expected_floor) < 1e-9, (
        f"Floor should be {expected_floor:.4f} (crlb/sqrt(9)), got {r['floor']:.4f}"
    )
    assert not r["failed"], "0.040 > floor=0.030, should pass"

    # Double-counting trap: caller pre-pools crlb AND passes neff -- do NOT do this.
    # The gate computes floor = (crlb/3) / 3 = 0.010, which is too tight.
    pre_pooled_crlb = single_pixel_crlb / 3.0  # already divided once
    r_double = test_precision_floor(claimed_sigma=0.040, crlb=pre_pooled_crlb, neff=neff)
    assert r_double["floor"] < 0.020, (
        "Double-counting produces a spuriously tight floor -- caller must pass raw CRLB"
    )

    # Correct usage when caller has only pre-pooled crlb: pass neff=1.0
    r_correct = test_precision_floor(claimed_sigma=0.040, crlb=pre_pooled_crlb, neff=1.0)
    assert abs(r_correct["floor"] - expected_floor) < 1e-9, (
        "Passing pre-pooled crlb with neff=1 gives the same floor as single-pixel crlb with neff=9"
    )




def test_T2_flags_pooling_across_a_boundary():
    """
    Pooling is only legitimate over a homogeneous neighbourhood. Across a
    material boundary it contaminates the estimate - this is the mechanism by
    which small defects get erased and false shifts appear.
    """
    from optiguard.physics.lineshapes import lorentzian
    from optiguard.assurance.gate import test_pooling_legitimacy

    rng = np.random.default_rng(4)
    axis = np.arange(500.0, 541.0, 0.55)
    a = lorentzian(axis, 520.7, 3.5, 400.0) + 5.0
    b = lorentzian(axis, 522.1, 3.5, 400.0) + 5.0   # different material

    homogeneous = rng.poisson(np.tile(a, (9, 1)))
    mixed = rng.poisson(np.vstack([np.tile(a, (5, 1)), np.tile(b, (4, 1))]))

    assert not test_pooling_legitimacy(homogeneous, read_noise_e=0.0)["failed"]
    assert test_pooling_legitimacy(mixed, read_noise_e=0.0)["failed"]


def test_T3_flags_an_injected_peak_shift():
    """A restoration that shifts a peak must fail to explain the raw counts."""
    from optiguard.physics.lineshapes import lorentzian
    from optiguard.assurance.gate import test_photon_consistency

    rng = np.random.default_rng(5)
    axis = np.arange(500.0, 541.0, 0.55)
    truth = lorentzian(axis, 520.7, 3.5, 400.0) + 5.0
    observed = rng.poisson(truth)

    honest = truth
    fabricated = lorentzian(axis, 521.6, 3.5, 400.0) + 5.0

    assert not test_photon_consistency(observed, honest, read_noise_e=0.0)["failed"]
    assert test_photon_consistency(observed, fabricated, read_noise_e=0.0)["failed"]
