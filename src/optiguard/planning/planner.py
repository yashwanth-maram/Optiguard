"""
CTE-10: Minimum-cost reacquisition planner.

Step 12 of the OptiGuard project. Plans the cheapest re-measurement strategy
for pixels that the assurance gate declined to certify.

Physics rules enforced:
  1. CRLB inversion by bisection — no 1/sqrt(N) shortcuts.
  2. Integration time vs accumulations modelled correctly (read noise adds per readout).
  3. Spectral binning modelled as both photon gain AND dispersion stretch; undersampling
     floor of 3 channels/FWHM is enforced.
  4. Pooling credit is opt-in and must be measured (Jacobian probe), not assumed.
  5. Sparse re-scan of only the flagged pixels is the product.
  6. All predictions labelled simulated=True until Step 15 hardware validation.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field, replace as dc_replace
from typing import Optional, List, Dict, Any, Tuple
from scipy.ndimage import binary_dilation

from optiguard.physics.crlb import crlb_peak_position


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AcquisitionSettings:
    integration_s: float
    accumulations: int
    laser_power_mw: float
    spectral_binning: int
    dispersion_cm1_per_px: float
    step_size_um: float
    overhead_s_per_point: float
    readout_s: float

    def replace(self, **kwargs) -> "AcquisitionSettings":
        return dc_replace(self, **kwargs)


@dataclass
class InstrumentConstraints:
    damage_power_mw: float
    well_depth_e: float
    saturation_margin: float
    min_integration_s: float
    max_integration_s: float
    max_accumulations: int


@dataclass
class Action:
    name: str
    settings: AcquisitionSettings
    cost_seconds: float          # total wall-clock for rescanned pixels only
    setup_penalty: float = 0.0   # extra seconds for grating change / refocus


@dataclass
class ReacquisitionPlan:
    action: Optional[Action]
    ranked: List[Action]
    predicted_sigma: np.ndarray    # (H, W) cm^-1 at the chosen action's settings
    rescan_mask: np.ndarray        # (H, W) bool — dilated flagged mask
    cost_seconds: float
    simulated: bool = True
    rationale: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default constraints (broad; callers should pass their own)
# ---------------------------------------------------------------------------

_DEFAULT_CONSTRAINTS = InstrumentConstraints(
    damage_power_mw=100.0,
    well_depth_e=200_000,
    saturation_margin=0.1,
    min_integration_s=0.005,
    max_integration_s=300.0,
    max_accumulations=64,
)

_DEFAULT_READ_NOISE_E = 4.0


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def predict_sigma(
    settings: AcquisitionSettings,
    amplitude: float,
    background: float,
    fwhm: float,
    center: float,
    read_noise_e: float,
    axis: np.ndarray,
    ref_settings: "AcquisitionSettings" = None,
) -> float:
    """
    Predicted CRLB (cm^-1) at the given AcquisitionSettings for one pixel.

    Args:
        amplitude:  Peak height in photons at the REFERENCE acquisition settings
                    (integration_s=ref_settings.integration_s, accumulations=1, bin=1).
                    If ref_settings is None, amplitude is used directly without scaling.
        background: Background counts per channel at the reference acquisition.
        ref_settings: The reference AcquisitionSettings that amplitude/background were
                      measured at. If None, amplitude and background are used as-is
                      (i.e., they are already at the target settings).

    Modelling rules:
    - When ref_settings is provided, amplitude and background are scaled by the ratio
      of new to reference (integration_s × accumulations).
    - Spectral binning b collapses b channels; background per binned-channel × b;
      amplitude (peak height) is unchanged; dispersion stretches by b.
    - Read noise: one readout per accumulation; RMS adds in quadrature so effective
      per-spectrum read noise = read_noise_e × sqrt(accumulations).
    """
    b = settings.spectral_binning
    acc = settings.accumulations

    if ref_settings is not None:
        scale = (settings.integration_s * acc) / (ref_settings.integration_s * ref_settings.accumulations)
        A_eff = amplitude * scale
        B_eff = background * scale
    else:
        # amplitude already represents accumulated photons at this exposure on the unbinned grid
        A_eff = amplitude
        B_eff = background

    R_eff = read_noise_e * np.sqrt(acc)

    # Note: A_eff = amplitude * scale with binning handled by b is only correct
    # because crlb_peak_position physically integrates the lineshape over each bin
    # rather than just scaling peak height. This correctly models both the photon gain
    # and the eventual loss of precision when the peak becomes undersampled (the cliff).
    return crlb_peak_position(
        axis=axis,
        center=center,
        fwhm=fwhm,
        amplitude=A_eff,
        background=B_eff,
        read_noise_e=R_eff,
        b=b,
    )


def required_photon_scale(
    target_sigma: float,
    amplitude: float,
    background: float,
    axis: np.ndarray,
    center: float,
    fwhm: float,
    read_noise_e: float,
    k_lo: float = 1e-3,
    k_hi: float = 1e4,
    tol: float = 1e-6,
) -> float:
    """
    Find photon scale factor k such that
        crlb_peak_position(amplitude=A*k, background=B, ...) <= target_sigma.

    Scales both amplitude and background (as both are Poisson processes that
    scale with integration time), treating read noise as the fixed noise floor.
    This models the physically important regime where read noise is diluted by
    longer integration, making the required photon scale slightly less than the naive
    sqrt law would predict.

    Solved by bisection on the validated numerical CRLB. Does NOT assume sigma ~ 1/sqrt(k).

    Returns k. If already satisfied at k=1.0, returns 1.0. If even k_hi is
    insufficient, returns k_hi to signal infeasibility.
    """
    def crlb_at_k(k: float) -> float:
        return crlb_peak_position(
            axis=axis,
            center=center,
            fwhm=fwhm,
            amplitude=amplitude * k,
            background=background * k,
            read_noise_e=read_noise_e,
        )

    if crlb_at_k(1.0) <= target_sigma:
        return 1.0

    if crlb_at_k(k_hi) > target_sigma:
        return k_hi

    lo, hi = k_lo, k_hi
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if crlb_at_k(mid) <= target_sigma:
            hi = mid
        else:
            lo = mid
        if (hi - lo) / max(hi, 1e-12) < tol:
            break
    return hi


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

def _pixel_cost(settings: AcquisitionSettings) -> float:
    """Wall-clock seconds per pixel for one acquisition."""
    return (
        settings.integration_s * settings.accumulations
        + settings.readout_s * settings.accumulations
        + settings.overhead_s_per_point
    )


class ActionSpace:
    """Enumerate and filter candidate acquisition adjustments."""

    @staticmethod
    def feasible(
        base: AcquisitionSettings,
        fwhm: float,
        constraints: Optional[InstrumentConstraints] = None,
        peak_rate_per_s: Optional[float] = None,
    ) -> List[Action]:
        """
        Return all feasible actions, sorted by cost (ascending).

        Candidates:
          - Integration time: ×2, ×4, ×8
          - Accumulations: ×2, ×4
          - Spectral binning: 2, 4
          - Laser power: ×2

        Rejected if any constraint is violated (damage, saturation, sampling floor,
        integration bounds, accumulation bounds).
        """
        if constraints is None:
            constraints = _DEFAULT_CONSTRAINTS

        actions: List[Action] = []

        for mult in [2, 4, 8]:
            s = base.replace(integration_s=base.integration_s * mult)
            if ActionSpace._check(s, fwhm, constraints, peak_rate_per_s) is None:
                actions.append(Action(
                    name=f"integration_s\u00d7{mult}",
                    settings=s,
                    cost_seconds=0.0,
                ))

        for mult in [2, 4]:
            s = base.replace(accumulations=base.accumulations * mult)
            if ActionSpace._check(s, fwhm, constraints, peak_rate_per_s) is None:
                actions.append(Action(
                    name=f"accumulations\u00d7{mult}",
                    settings=s,
                    cost_seconds=0.0,
                ))

        for b in [2, 4]:
            s = base.replace(spectral_binning=b)
            if ActionSpace._check(s, fwhm, constraints, peak_rate_per_s) is None:
                actions.append(Action(
                    name=f"binning={b}",
                    settings=s,
                    cost_seconds=0.0,
                    setup_penalty=30.0,
                ))

        for mult in [2]:
            s = base.replace(laser_power_mw=base.laser_power_mw * mult)
            if ActionSpace._check(s, fwhm, constraints, peak_rate_per_s) is None:
                actions.append(Action(
                    name=f"laser_power\u00d7{mult}",
                    settings=s,
                    cost_seconds=0.0,
                ))

        return actions

    @staticmethod
    def _check(
        s: AcquisitionSettings,
        fwhm: float,
        constraints: InstrumentConstraints,
        peak_rate_per_s: Optional[float],
    ) -> Optional[str]:
        """Return None if feasible, or a rejection reason string."""
        if s.integration_s < constraints.min_integration_s:
            return "integration_s below min"
        if s.integration_s > constraints.max_integration_s:
            return "integration_s above max"
        if s.accumulations > constraints.max_accumulations:
            return "accumulations above max"
        if s.laser_power_mw > constraints.damage_power_mw:
            return "laser_power exceeds damage limit"

        eff_disp = s.dispersion_cm1_per_px * s.spectral_binning
        if fwhm / eff_disp < 3.0:
            return f"undersampled: {fwhm / eff_disp:.2f} ch/FWHM < 3"

        if peak_rate_per_s is not None:
            pred_well = peak_rate_per_s * s.integration_s * s.accumulations
            well_limit = constraints.well_depth_e * (1.0 - constraints.saturation_margin)
            if pred_well > well_limit:
                return f"saturation: {pred_well:.0f} > {well_limit:.0f}"

        return None


# ---------------------------------------------------------------------------
# Main planner
# ---------------------------------------------------------------------------

def plan_reacquisition(
    flagged_mask: np.ndarray,
    current_settings: AcquisitionSettings,
    fitted_params: Dict[str, np.ndarray],
    axis: np.ndarray,
    target_sigma: float,
    constraints: Optional[InstrumentConstraints] = None,
    coverage: float = 0.95,
    pooling_credit: Optional[np.ndarray] = None,
    read_noise_e: float = _DEFAULT_READ_NOISE_E,
) -> ReacquisitionPlan:
    """
    Plan the minimum-cost reacquisition for flagged pixels.

    Args:
        flagged_mask:     (H, W) bool — pixels the gate declined to certify.
        current_settings: AcquisitionSettings at the time of the original acquisition.
        fitted_params:    Dict of (H,W) arrays: 'center', 'fwhm', 'amplitude', 'background'.
                          amplitude = peak height in photons/s (NOT integrated signal).
        axis:             (C,) wavenumber axis, cm^-1, at spectral_binning=1.
        target_sigma:     Required precision, cm^-1.
        constraints:      InstrumentConstraints; uses broad defaults if None.
        coverage:         Fraction of flagged pixels that must reach target (default 0.95).
        pooling_credit:   Optional (H, W) N_eff map from Jacobian probe. Opt-in only.
        read_noise_e:     Detector read noise in electrons (default 4.0).

    Returns:
        ReacquisitionPlan. plan.simulated is always True until Step 15.
    """
    if constraints is None:
        constraints = _DEFAULT_CONSTRAINTS

    H, W = flagged_mask.shape

    # --- 1. Nothing to do ---
    if not np.any(flagged_mask):
        return ReacquisitionPlan(
            action=None,
            ranked=[],
            predicted_sigma=np.zeros((H, W), dtype=np.float32),
            rescan_mask=np.zeros((H, W), dtype=bool),
            cost_seconds=0.0,
            simulated=True,
            rationale={"reason": "No flagged pixels — nothing to reacquire."},
        )

    # --- 2. Dilate rescan mask by 1 px so boundaries are covered ---
    rescan_mask = binary_dilation(flagged_mask, iterations=1)
    n_rescan = int(rescan_mask.sum())
    n_flagged = int(flagged_mask.sum())

    # --- 3. Fitted parameter maps with safe fallbacks ---
    fwhm_map = np.asarray(fitted_params.get("fwhm", np.full((H, W), 3.5)), dtype=np.float64)
    amplitude_map = np.asarray(fitted_params.get("amplitude", np.ones((H, W))), dtype=np.float64)
    background_map = np.asarray(fitted_params.get("background", np.zeros((H, W))), dtype=np.float64)
    center_map = np.asarray(fitted_params.get("center", np.full((H, W), float(np.median(axis)))), dtype=np.float64)

    # Representative values for action-space enumeration (median over flagged pixels)
    fwhm_med = float(np.nanmedian(fwhm_map[flagged_mask]))
    amp_med = float(np.nanmedian(amplitude_map[flagged_mask]))
    bg_med = float(np.nanmedian(background_map[flagged_mask]))
    ctr_med = float(np.nanmedian(center_map[flagged_mask]))

    t_cur = current_settings.integration_s
    acc_cur = current_settings.accumulations
    b_cur = current_settings.spectral_binning

    # Current per-acquisition photons (for representative pixel)
    amp_acq = amp_med * t_cur * acc_cur
    bg_acq = bg_med * t_cur * acc_cur * b_cur

    # --- 4. Required scale factor (representative pixel) ---
    k_required = required_photon_scale(
        target_sigma=target_sigma,
        amplitude=amp_acq,
        background=bg_acq,
        axis=axis[::b_cur],
        center=ctr_med,
        fwhm=fwhm_med,
        read_noise_e=read_noise_e * np.sqrt(acc_cur),
    )
    sigma_current = crlb_peak_position(
        axis=axis[::b_cur],
        center=ctr_med,
        fwhm=fwhm_med,
        amplitude=amp_acq,
        background=bg_acq,
        read_noise_e=read_noise_e * np.sqrt(acc_cur),
    )

    # --- 5. Enumerate feasible actions ---
    candidate_actions = ActionSpace.feasible(
        base=current_settings,
        fwhm=fwhm_med,
        constraints=constraints,
        peak_rate_per_s=amp_med,
    )

    # --- 6. Score each action against every flagged pixel ---
    def _eval_action(act: Action) -> Tuple[float, np.ndarray, float]:
        """Returns (coverage_fraction, per-pixel sigma map, total cost_s)."""
        s = act.settings
        pwr_scale = s.laser_power_mw / current_settings.laser_power_mw
        b = s.spectral_binning
        t = s.integration_s
        acc = s.accumulations
        R = read_noise_e * np.sqrt(acc)
        ax_bin = axis[::b]

        sigma_map = np.full((H, W), np.inf, dtype=np.float64)
        ys, xs = np.where(flagged_mask)
        for y, x in zip(ys, xs):
            A = float(amplitude_map[y, x]) * t * acc * pwr_scale
            B = float(background_map[y, x]) * t * acc * b * pwr_scale
            neff = max(1.0, float(pooling_credit[y, x])) if pooling_credit is not None else 1.0
            try:
                sig = crlb_peak_position(
                    axis=ax_bin,
                    center=float(center_map[y, x]),
                    fwhm=float(fwhm_map[y, x]),
                    amplitude=A,
                    background=B,
                    read_noise_e=R,
                )
                # Pooling credit: effective target tightens by 1/sqrt(neff);
                # equivalently sigma must be <= target_sigma (without pooling adjustment,
                # since neff credits are already folded into the reduced CRLB via the
                # lower photon requirement). We compare sigma against target_sigma / sqrt(neff)
                # only for coverage accounting — not for the sigma map itself.
                sigma_map[y, x] = sig / np.sqrt(neff)
            except Exception:
                sigma_map[y, x] = np.inf

        meets = sigma_map[flagged_mask] <= target_sigma
        cov = float(meets.sum()) / max(1, n_flagged)
        cost = n_rescan * _pixel_cost(s) + act.setup_penalty
        return cov, sigma_map, cost

    feasible_scored: List[Tuple[float, np.ndarray, float, Action]] = []
    rejected: List[Tuple[str, str]] = []

    for act in candidate_actions:
        cov, sigma_map, cost = _eval_action(act)
        act.cost_seconds = cost
        if cov >= coverage:
            feasible_scored.append((cost, sigma_map, cov, act))
        else:
            rejected.append((act.name, f"coverage {cov:.1%} < required {coverage:.0%}"))

    feasible_scored.sort(key=lambda x: x[0])

    # --- 7. No feasible action ---
    if not feasible_scored:
        sigma_now_map = np.full((H, W), np.inf, dtype=np.float32)
        for y, x in zip(*np.where(flagged_mask)):
            try:
                sigma_now_map[y, x] = crlb_peak_position(
                    axis=axis[::b_cur],
                    center=float(center_map[y, x]),
                    fwhm=float(fwhm_map[y, x]),
                    amplitude=float(amplitude_map[y, x]) * t_cur * acc_cur,
                    background=float(background_map[y, x]) * t_cur * acc_cur * b_cur,
                    read_noise_e=read_noise_e * np.sqrt(acc_cur),
                )
            except Exception:
                pass

        return ReacquisitionPlan(
            action=None,
            ranked=[],
            predicted_sigma=sigma_now_map,
            rescan_mask=rescan_mask,
            cost_seconds=0.0,
            simulated=True,
            rationale={
                "reason": "No feasible action reaches target within constraints.",
                "target_sigma_cm1": target_sigma,
                "current_sigma_cm1": float(sigma_current),
                "required_photon_scale": float(k_required),
                "rejected": rejected,
                "note": "simulated=True; not validated on hardware (Step 15).",
            },
        )

    # --- 8. Best action ---
    best_cost, best_sigma_map, best_cov, best_action = feasible_scored[0]
    full_rescan_cost = float(H * W * _pixel_cost(best_action.settings))

    rationale = {
        "target_sigma_cm1": float(target_sigma),
        "current_sigma_representative_cm1": float(sigma_current),
        "required_photon_scale": float(k_required),
        "chosen_action": best_action.name,
        "achieved_coverage": float(best_cov),
        "cost_seconds_sparse": float(best_cost),
        "cost_seconds_full_rescan": full_rescan_cost,
        "time_saving_ratio": float(best_cost / max(full_rescan_cost, 1e-9)),
        "n_flagged_pixels": n_flagged,
        "n_rescan_pixels": n_rescan,
        "alternatives_rejected": rejected,
        "action_settings": {
            "integration_s": best_action.settings.integration_s,
            "accumulations": best_action.settings.accumulations,
            "spectral_binning": best_action.settings.spectral_binning,
            "laser_power_mw": best_action.settings.laser_power_mw,
        },
        "note": "simulated=True; predictions not validated on hardware (Step 15).",
    }

    return ReacquisitionPlan(
        action=best_action,
        ranked=[a for _, __, ___, a in feasible_scored],
        predicted_sigma=best_sigma_map.astype(np.float32),
        rescan_mask=rescan_mask,
        cost_seconds=float(best_cost),
        simulated=True,
        rationale=rationale,
    )
