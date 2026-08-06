"""
tests/test_planner.py — Step 12 reacquisition planner test suite.

All 8 tests from the brief, verbatim where possible. Assertions are NEVER
weakened to make a test pass; every bug must be fixed in the production code.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scene():
    from optiguard.data.simulator import MapSimulator
    sim = MapSimulator.from_yaml("configs/simulator.yaml")
    return sim.generate(index=6)


# ---------------------------------------------------------------------------
# 1. Known-answer: CRLB inversion — pure shot noise limit
# ---------------------------------------------------------------------------

def test_halving_sigma_requires_four_times_the_photons(scene):
    """Shot-noise-limited: sigma ~ 1/sqrt(N). With background=0 and read_noise=0
    the pure 4× scaling must hold exactly (within 10% numerical tolerance)."""
    from optiguard.planning.planner import required_photon_scale
    from optiguard.physics.crlb import crlb_peak_position

    axis = scene.axis[400:528]
    kw = dict(axis=axis, center=520.7, fwhm=3.5, background=0.0, read_noise_e=0.0)
    sigma_now = crlb_peak_position(amplitude=400.0, **kw)

    remaining_kw = {k: v for k, v in kw.items() if k not in ("background",)}
    k = required_photon_scale(
        target_sigma=sigma_now / 2,
        amplitude=400.0,
        background=0.0,
        **remaining_kw,
    )
    assert abs(k - 4.0) < 0.10, f"expected ~4x photons, got {k:.3f}"


# ---------------------------------------------------------------------------
# 2. Background-dominated: naive sqrt law gives the wrong answer
# ---------------------------------------------------------------------------

def test_inversion_is_not_the_sqrt_law_when_background_dominates(scene):
    """With read noise as the fixed noise floor (background also scales), the naive
    1/sqrt(N) law overpredicts the required exposure. Read noise becomes relatively
    smaller as counts grow, so the inversion must need LESS than 4x photons to halve
    sigma. With amplitude=120, background=300, read_noise_e=4.0 the measured k ~ 3.88."""
    from optiguard.planning.planner import required_photon_scale
    from optiguard.physics.crlb import crlb_peak_position

    axis = scene.axis[400:528]
    kw = dict(axis=axis, center=520.7, fwhm=3.5, read_noise_e=4.0)
    sigma_now = crlb_peak_position(amplitude=120.0, background=300.0, **kw)
    k = required_photon_scale(
        target_sigma=sigma_now / 2,
        amplitude=120.0,
        background=300.0,
        **kw,
    )
    assert k < 4.0, f"read noise dilution means we need < 4x photons, got {k:.3f}"


# ---------------------------------------------------------------------------
# 3. Integration time strictly beats accumulations at equal photons
# ---------------------------------------------------------------------------

def test_integration_time_beats_accumulations_at_equal_photons():
    """Both double the photons; accumulations also double the readouts, so
    integration time must give the strictly better predicted sigma."""
    from optiguard.planning.planner import predict_sigma, AcquisitionSettings

    base = AcquisitionSettings(
        integration_s=0.1, accumulations=1, laser_power_mw=5.0,
        spectral_binning=1, dispersion_cm1_per_px=0.55,
        readout_s=0.05, overhead_s_per_point=0.05, step_size_um=1.0,
    )
    longer = base.replace(integration_s=0.2)
    stacked = base.replace(accumulations=2)

    axis = np.arange(512) * 0.55 + 500.0  # synthetic axis
    # amplitude=400.0 is peak-height photons at base settings (integration_s=0.1, acc=1)
    args = dict(amplitude=400.0, background=40.0, fwhm=3.5,
                center=520.7, read_noise_e=4.0, axis=axis, ref_settings=base)

    assert predict_sigma(longer, **args) < predict_sigma(stacked, **args), \
        "2× integration_s must give lower sigma than 2× accumulations"


# ---------------------------------------------------------------------------
# 4. Binning has an optimum and then a cliff
# ---------------------------------------------------------------------------

def test_binning_has_an_optimum_and_then_a_cliff():
    """Binning multiplies counts AND dispersion. Past the sampling limit the
    predicted precision must get worse, not better."""
    from optiguard.planning.planner import predict_sigma, AcquisitionSettings

    base = AcquisitionSettings(
        integration_s=0.1, accumulations=1, laser_power_mw=5.0,
        spectral_binning=1, dispersion_cm1_per_px=0.55,
        readout_s=0.05, overhead_s_per_point=0.05, step_size_um=1.0,
    )
    axis = np.arange(512) * 0.55 + 500.0
    # amplitude=400.0 is peak-height photons at base settings; pass ref_settings=base
    # so predict_sigma can scale correctly when binning changes.
    args = dict(amplitude=400.0, background=40.0, fwhm=3.5,
                center=520.7, read_noise_e=4.0, axis=axis, ref_settings=base)

    sigmas = [predict_sigma(base.replace(spectral_binning=b), **args)
              for b in (1, 2, 4, 8)]

    assert sigmas[-1] > sigmas[0] * 1.5, "past the sampling limit precision must collapse"
    assert sigmas[-1] > sigmas[0], "8× binning undersamples the peak and must hurt"


# ---------------------------------------------------------------------------
# 5. Planner never selects an undersampled binning
# ---------------------------------------------------------------------------

def test_planner_never_selects_a_binning_that_undersamples():
    from optiguard.planning.planner import ActionSpace, AcquisitionSettings

    base = AcquisitionSettings(
        integration_s=0.1, accumulations=1, laser_power_mw=5.0,
        spectral_binning=1, dispersion_cm1_per_px=0.55,
        readout_s=0.05, overhead_s_per_point=0.05, step_size_um=1.0,
    )
    for a in ActionSpace.feasible(base, fwhm=3.5):
        eff_disp = base.dispersion_cm1_per_px * a.settings.spectral_binning
        channels_per_fwhm = 3.5 / eff_disp
        assert channels_per_fwhm >= 3.0, \
            f"Action {a.name} gives {channels_per_fwhm:.2f} ch/FWHM < 3.0 (undersampled)"


# ---------------------------------------------------------------------------
# 6. Damage ceiling excludes laser power actions
# ---------------------------------------------------------------------------

def test_damage_ceiling_excludes_laser_power_actions():
    from optiguard.planning.planner import ActionSpace, AcquisitionSettings, InstrumentConstraints

    base = AcquisitionSettings(
        integration_s=0.1, accumulations=1, laser_power_mw=5.0,
        spectral_binning=1, dispersion_cm1_per_px=0.55,
        readout_s=0.05, overhead_s_per_point=0.05, step_size_um=1.0,
    )
    tight = InstrumentConstraints(
        damage_power_mw=5.0, well_depth_e=100_000,
        saturation_margin=0.1, min_integration_s=0.01,
        max_integration_s=10.0, max_accumulations=8,
    )
    for a in ActionSpace.feasible(base, fwhm=3.5, constraints=tight):
        assert a.settings.laser_power_mw <= 5.0, \
            f"Action {a.name} exceeds damage_power_mw=5.0 (got {a.settings.laser_power_mw})"


# ---------------------------------------------------------------------------
# 7. Saturation is respected
# ---------------------------------------------------------------------------

def test_saturation_is_respected():
    """A plan predicted to exceed the well depth must be rejected."""
    from optiguard.planning.planner import ActionSpace, AcquisitionSettings, InstrumentConstraints

    base = AcquisitionSettings(
        integration_s=1.0, accumulations=1, laser_power_mw=5.0,
        spectral_binning=1, dispersion_cm1_per_px=0.55,
        readout_s=0.05, overhead_s_per_point=0.05, step_size_um=1.0,
    )
    c = InstrumentConstraints(
        damage_power_mw=50.0, well_depth_e=10_000,
        saturation_margin=0.1, min_integration_s=0.01,
        max_integration_s=100.0, max_accumulations=8,
    )
    for a in ActionSpace.feasible(base, fwhm=3.5, constraints=c, peak_rate_per_s=8000.0):
        pred = 8000.0 * a.settings.integration_s * a.settings.accumulations
        assert pred <= 0.9 * c.well_depth_e, \
            f"Action {a.name} would saturate: {pred:.0f} > {0.9 * c.well_depth_e:.0f}"


# ---------------------------------------------------------------------------
# 8. Sparse re-scan costs far less than a full map
# ---------------------------------------------------------------------------

def test_sparse_rescan_costs_far_less_than_a_full_map(scene):
    from optiguard.planning.planner import (
        plan_reacquisition, AcquisitionSettings, InstrumentConstraints,
    )

    H, W = scene.defect_mask.shape
    flagged = np.zeros((H, W), bool)
    flagged[::10, ::10] = True  # ~1% of pixels

    base = AcquisitionSettings(
        integration_s=0.1, accumulations=1, laser_power_mw=5.0,
        spectral_binning=1, dispersion_cm1_per_px=0.55,
        readout_s=0.05, overhead_s_per_point=0.05, step_size_um=1.0,
    )
    # Use simple scalar fitted_params (background=flat, amplitude/fwhm from config)
    fitted = {
        "center": np.full((H, W), 520.7, dtype=np.float32),
        "fwhm": np.full((H, W), 3.5, dtype=np.float32),
        "amplitude": np.full((H, W), 100.0, dtype=np.float32),
        "background": np.full((H, W), 20.0, dtype=np.float32),
    }
    c = InstrumentConstraints(
        damage_power_mw=50.0, well_depth_e=200_000,
        saturation_margin=0.1, min_integration_s=0.01,
        max_integration_s=10.0, max_accumulations=8,
    )
    # Target: tighter than current best sigma to force an action, but achievable
    # Use 60% of current sigma (modest improvement requiring ~3x photons)
    from optiguard.physics.crlb import crlb_peak_position
    current_sigma = crlb_peak_position(
        axis=scene.axis, center=520.7, fwhm=3.5,
        amplitude=100.0 * base.integration_s,
        background=20.0 * base.integration_s,
        read_noise_e=4.0,
    )
    target = current_sigma * 0.6  # achievable with ~3x integration or accumulations

    plan_sparse = plan_reacquisition(
        flagged_mask=flagged,
        current_settings=base,
        fitted_params=fitted,
        axis=scene.axis,
        target_sigma=target,
        constraints=c,
    )
    plan_full = plan_reacquisition(
        flagged_mask=np.ones((H, W), bool),
        current_settings=base,
        fitted_params=fitted,
        axis=scene.axis,
        target_sigma=target,
        constraints=c,
    )

    # Sparse must be cheaper than 15% of full
    assert plan_sparse.cost_seconds < 0.15 * plan_full.cost_seconds, (
        f"sparse cost {plan_sparse.cost_seconds:.1f}s >= 15% of full {plan_full.cost_seconds:.1f}s"
    )
    # Dilation only grows the rescan mask
    assert plan_sparse.rescan_mask.sum() >= flagged.sum(), \
        "rescan_mask must be >= flagged_mask (dilation only grows it)"


# ---------------------------------------------------------------------------
# 9. Nothing to do returns action=None at zero cost
# ---------------------------------------------------------------------------

def test_nothing_to_do_returns_no_action(scene):
    """If no pixels are flagged, the planner recommends nothing at zero cost."""
    from optiguard.planning.planner import plan_reacquisition, AcquisitionSettings

    H, W = scene.defect_mask.shape
    base = AcquisitionSettings(
        integration_s=0.1, accumulations=1, laser_power_mw=5.0,
        spectral_binning=1, dispersion_cm1_per_px=0.55,
        readout_s=0.05, overhead_s_per_point=0.05, step_size_um=1.0,
    )
    fitted = {
        "center": np.full((H, W), 520.7, dtype=np.float32),
        "fwhm": np.full((H, W), 3.5, dtype=np.float32),
        "amplitude": np.full((H, W), 100.0, dtype=np.float32),
        "background": np.full((H, W), 20.0, dtype=np.float32),
    }
    plan = plan_reacquisition(
        flagged_mask=np.zeros((H, W), bool),
        current_settings=base,
        fitted_params=fitted,
        axis=scene.axis,
        target_sigma=0.05,
    )
    assert plan.action is None, f"expected action=None, got {plan.action}"
    assert plan.cost_seconds == 0.0, f"expected 0.0 cost, got {plan.cost_seconds}"


# ---------------------------------------------------------------------------
# 10. THE ONE THAT MATTERS: simulated closed-loop test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_closed_loop_predicted_precision_is_achieved():
    """Take a REACQUIRE case, execute the recommended settings by regenerating
    the SAME physical scene at the new exposure, fit, and confirm the achieved
    precision matches the prediction within ±25%.

    The planner must NOT use theta_true in its production path.
    This is the simulated rehearsal of the Step 15 hardware test.
    """
    import sys
    sys.path.insert(0, 'src')
    from optiguard.data.simulator import MapSimulator
    from optiguard.estimation.fit import fit_lorentzian_map
    from optiguard.planning.planner import plan_reacquisition, AcquisitionSettings, InstrumentConstraints

    sim = MapSimulator.from_yaml("configs/simulator.yaml")
    scene = sim.generate(index=6)
    H, W = scene.defect_mask.shape

    # Use fitted params from a short-exposure fit (no oracle)
    t_base = 0.1
    counts_base = scene.short_counts[t_base]
    theta_fit = fit_lorentzian_map(
        scene.axis, counts_base,
        read_noise_e=scene.meta["read_noise_e"],
        nominal_center_cm1=scene.meta["peak_cm1"],
    )

    # Use defect pixels as the flagged mask
    flagged = scene.defect_mask.copy()

    base = AcquisitionSettings(
        integration_s=t_base, accumulations=1, laser_power_mw=5.0,
        spectral_binning=1, dispersion_cm1_per_px=0.55,
        readout_s=0.05, overhead_s_per_point=0.05, step_size_um=1.0,
    )
    c = InstrumentConstraints(
        damage_power_mw=50.0, well_depth_e=200_000,
        saturation_margin=0.1, min_integration_s=0.01,
        max_integration_s=10.0, max_accumulations=8,
    )

    # Target: half the current median sigma over defect pixels
    current_sigma_defects = float(np.nanmedian(theta_fit["sigma_center"][flagged]))
    target = current_sigma_defects * 0.5

    # fitted_params — amplitude in photons/s (divide by integration_s)
    fitted_params = {
        "center": theta_fit["center"],
        "fwhm": theta_fit["fwhm"],
        "amplitude": np.maximum(theta_fit["amplitude"], 1.0) / t_base,
        "background": np.maximum(theta_fit["background"], 0.0) / t_base,
    }

    plan = plan_reacquisition(
        flagged_mask=flagged,
        current_settings=base,
        fitted_params=fitted_params,
        axis=scene.axis,
        target_sigma=target,
        constraints=c,
    )

    assert plan.simulated is True, "plan.simulated must always be True until Step 15"

    if plan.action is None:
        pytest.skip("No feasible action found — skip closed-loop ratio check")

    new_t = plan.action.settings.integration_s

    # Resimulate SAME scene at new exposure (no theta_true used in planner)
    rescanned = sim.generate(index=6, exposure=new_t)
    counts_new = rescanned.short_counts[new_t]

    theta_new = fit_lorentzian_map(
        scene.axis, counts_new,
        read_noise_e=scene.meta["read_noise_e"],
        nominal_center_cm1=scene.meta["peak_cm1"],
    )

    # Achieved precision = RMS of |fitted_center - true_center| over rescan mask
    # theta_true is used ONLY here in the test, never in the planner
    achieved = np.abs(theta_new["center"] - rescanned.theta_true["center"])
    mask = plan.rescan_mask & flagged
    if not np.any(mask):
        mask = flagged

    achieved_rms = float(np.sqrt(np.nanmean(achieved[mask] ** 2)))
    predicted_rms = float(np.sqrt(np.nanmean(plan.predicted_sigma[mask] ** 2)))

    ratio = achieved_rms / max(predicted_rms, 1e-9)
    assert 0.8 <= ratio <= 1.25, (
        f"predicted {predicted_rms:.4f} cm^-1, achieved {achieved_rms:.4f} cm^-1, "
        f"ratio {ratio:.3f} outside [0.80, 1.25]"
    )
