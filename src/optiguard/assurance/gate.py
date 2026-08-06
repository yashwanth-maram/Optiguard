"""
Assurance Gate Tests: T1 (Precision Floor), T2 (Pooling Legitimacy), T3 (Photon Consistency).
Step 10 specification.
"""
from typing import Dict, Any, Optional
import numpy as np
from scipy import stats


def test_precision_floor(
    claimed_sigma: float,
    crlb: float,
    neff: float = 1.0,
    tol: float = 1e-9
) -> Dict[str, Any]:
    """T1 Gate: Check if claimed precision violates the N_eff-adjusted Cramer-Rao bound.

    Physics rationale:
    No unbiased estimator can achieve a variance below the CRLB for the available
    effective photon budget. Any restoration reporting an uncertainty tighter
    than the information-theoretic floor is physically impossible (hallucination).

    When the method legitimately pools N_eff independent pixels, the floor
    tightens by 1/sqrt(N_eff) -- from the single-pixel CRLB to CRLB/sqrt(N_eff).
    Passing neff > 1 grants that credit. Passing neff derived from a probe on
    random noise instead of real data will artificially inflate N_eff and weaken
    the gate -- probing must use real, in-distribution data.

    Args:
        claimed_sigma: Standard error claimed by the estimator / network, cm^-1
        crlb: Single-pixel information-theoretic lower bound, cm^-1
        neff: Effective number of pooled pixels (Kish effective sample size, >= 1).
              Default 1.0 = no pooling credit granted.
        tol: Small tolerance for numerical edge cases

    Returns:
        Dict with "failed": bool, "claimed_sigma", "crlb", "neff", "floor", "ratio", "slack"
    """
    claimed = float(claimed_sigma)
    bound = float(crlb)
    neff = max(float(neff), 1.0)

    # N_eff-adjusted floor: pooling N_eff pixels reduces uncertainty by 1/sqrt(N_eff)
    floor = bound / (neff ** 0.5)
    failed = bool(claimed < (floor - tol))

    return {
        "gate": "T1",
        "name": "precision_floor",
        "failed": failed,
        "claimed_sigma": claimed,
        "crlb": bound,
        "neff": neff,
        "floor": floor,
        "ratio": claimed / floor if floor > 0 else float("inf"),
        "slack": claimed - floor
    }


def test_pooling_legitimacy(
    neighbourhood_counts: np.ndarray,
    read_noise_e: float = 0.0,
    weights: Optional[np.ndarray] = None,
    alpha: float = 0.01
) -> Dict[str, Any]:
    """T2 Gate: Test whether neighbourhood spectra are statistically homogeneous.

    Physics rationale:
    Spatial pooling is valid ONLY when neighbouring pixels sample the same underlying
    ground-truth material state. Across a material boundary or over a localized defect,
    spatial pooling contaminates the spectrum and erases small physical features.

    Statistical test:
    H0: All M spectra Y_1, ..., Y_M in the neighbourhood are Poisson realisations
        of a common mean spectrum mu in R^C.
    Pooled estimate: mu_hat = sum_m (w_m * Y_m) (or mean Y_bar if weights=None).
    Variance under Poisson + read noise: sigma_c^2 = mu_hat_c + read_noise_e^2.
    Pearson test statistic:
        chi2 = sum_m sum_c (Y_mc - mu_hat_c)^2 / sigma_c^2
    Degrees of freedom: df = (M - 1) * C.
    p-value = P(chi2 >= statistic | H0).

    If p < alpha, H0 is rejected: the neighbourhood is heterogeneous, and pooling
    is illegitimate at this pixel.

    Args:
        neighbourhood_counts: (M, C) array of counts for M pixels and C spectral channels
        read_noise_e: Detector read noise standard deviation in electrons
        weights: Optional (M,) array of pooling weights (must sum to 1)
        alpha: Significance level for rejection (default: 0.01)

    Returns:
        Dict with "failed": bool, "p_value", "statistic", "df", "critical_value"
    """
    counts = np.asarray(neighbourhood_counts, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(f"neighbourhood_counts must be 2D (M, C), got shape {counts.shape}")

    M, C = counts.shape
    if M < 2:
        # Trivial single-pixel neighbourhood is always homogeneous
        return {
            "gate": "T2",
            "name": "pooling_legitimacy",
            "failed": False,
            "p_value": 1.0,
            "statistic": 0.0,
            "df": 0
        }

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).ravel()
        w = w / np.sum(w)
        mu_hat = np.sum(counts * w[:, None], axis=0)
    else:
        mu_hat = np.mean(counts, axis=0)

    var = np.maximum(mu_hat + (read_noise_e ** 2), 1e-6)  # shape (C,)

    # Pearson chi-squared statistic
    residuals_sq = (counts - mu_hat[None, :]) ** 2  # (M, C)
    statistic = float(np.sum(residuals_sq / var[None, :]))

    df = int((M - 1) * C)
    p_val = float(stats.chi2.sf(statistic, df))
    crit_val = float(stats.chi2.ppf(1.0 - alpha, df))

    failed = bool(statistic > crit_val or p_val < alpha)

    return {
        "gate": "T2",
        "name": "pooling_legitimacy",
        "failed": failed,
        "p_value": p_val,
        "statistic": statistic,
        "df": df,
        "critical_value": crit_val
    }


def test_pooling_legitimacy_map(
    counts: np.ndarray,
    sigma: float = 2.0,
    read_noise_e: float = 0.0,
    alpha: float = 0.01,
    spectral_window: Optional[tuple] = None
) -> Dict[str, np.ndarray]:
    """Vectorized T2 gate across a full hyperspectral map for Gaussian pooling.

    Evaluates pooling legitimacy at every pixel simultaneously via spatial convolution
    under a **locally linear spatial null model**. Subtracts the expected variance
    contributed by smooth background strain gradients across the pooling kernel so that
    unperturbed wafer bulk does not trigger spurious boundary rejections.

    Args:
        counts: (H, W, C) hyperspectral count map
        sigma: Spatial Gaussian standard deviation in pixels
        read_noise_e: Detector read noise standard deviation
        alpha: Significance level for rejection (default: 0.01)
        spectral_window: Optional (start_idx, end_idx) channel crop

    Returns:
        Dict with "failed": (H, W) bool, "z_score": (H, W) float, "p_value": (H, W) float
    """
    from scipy.ndimage import gaussian_filter
    from optiguard.assurance.pooling import gaussian_pooling_weights_2d, effective_pooling_multiplier

    if spectral_window is not None:
        counts_active = counts[:, :, spectral_window[0]:spectral_window[1]]
    else:
        counts_active = counts

    H, W, C = counts_active.shape
    kernel_weights = gaussian_pooling_weights_2d(sigma)
    M_eff = effective_pooling_multiplier(kernel_weights)

    counts_f = counts_active.astype(np.float64)

    # 1. Smoothed mean spectrum and second moment
    mu_hat = gaussian_filter(counts_f, [sigma, sigma, 0], mode="reflect")
    sq_smooth = gaussian_filter(counts_f ** 2, [sigma, sigma, 0], mode="reflect")
    var_local_raw = np.maximum(sq_smooth - (mu_hat ** 2), 0.0)

    # 2. Local spatial gradient estimation: dI/dx and dI/dy
    grad_x = gaussian_filter(counts_f, [sigma, sigma, 0], order=[1, 0, 0], mode="reflect")
    grad_y = gaussian_filter(counts_f, [sigma, sigma, 0], order=[0, 1, 0], mode="reflect")
    var_grad = (sigma ** 2) * (grad_x ** 2 + grad_y ** 2)

    # 3. Excess variance beyond smooth spatial linear gradient
    var_excess = np.maximum(var_local_raw - var_grad, 0.0)

    # 4. Expected Poisson + read noise variance per channel under H0: (H, W, C)
    var_expected = np.maximum(mu_hat + (read_noise_e ** 2), 1e-6)

    # 5. Chi-squared statistic scaled by (M_eff - 1)
    chi2_map = (M_eff - 1.0) * np.sum(var_excess / var_expected, axis=-1)  # (H, W)

    df = (M_eff - 1.0) * C
    z_map = (chi2_map - df) / np.sqrt(2.0 * df)
    z_crit = stats.norm.ppf(1.0 - alpha)

    failed_map = z_map > z_crit
    p_val_map = stats.norm.sf(z_map)

    return {
        "failed": failed_map,
        "z_score": z_map,
        "p_value": p_val_map,
        "chi2": chi2_map,
        "df": df,
        "z_critical": z_crit
    }


def test_photon_consistency(
    observed_counts: np.ndarray,
    model_spectrum: np.ndarray,
    read_noise_e: float = 0.0,
    alpha: float = 0.01
) -> Dict[str, Any]:
    """T3 Gate: Test whether reconstructed spectrum is consistent with raw photon counts.

    Physics rationale:
    A restored spectrum must be a plausible physical source for the observed noisy
    shot counts. If a restoration invents or shifts a peak, the raw counts will
    systematically reject the model hypothesis under Poisson deviance.

    Statistical test:
    H0: observed_counts Y ~ Poisson(model_spectrum) + N(0, read_noise_e^2)
    Statistic: chi2 = sum_c (Y_c - model_c)^2 / (model_c + read_noise_e^2)
    df = C
    p-value = P(chi2 >= statistic | H0)

    If p < alpha, model is rejected as physically inconsistent with raw counts.

    Args:
        observed_counts: (C,) raw observed spectrum counts
        model_spectrum: (C,) proposed model / reconstruction
        read_noise_e: Detector read noise standard deviation
        alpha: Significance level (default: 0.01)

    Returns:
        Dict with "failed": bool, "p_value", "statistic", "df", "critical_value"
    """
    obs = np.asarray(observed_counts, dtype=np.float64)
    model = np.asarray(model_spectrum, dtype=np.float64)

    if obs.shape != model.shape:
        raise ValueError(f"Shape mismatch: obs {obs.shape} vs model {model.shape}")

    C = obs.size
    var = np.maximum(model + (read_noise_e ** 2), 1e-6)

    statistic = float(np.sum(((obs - model) ** 2) / var))
    df = int(C)
    p_val = float(stats.chi2.sf(statistic, df))
    crit_val = float(stats.chi2.ppf(1.0 - alpha, df))

    failed = bool(statistic > crit_val or p_val < alpha)

    return {
        "gate": "T3",
        "name": "photon_consistency",
        "failed": failed,
        "p_value": p_val,
        "statistic": statistic,
        "df": df,
        "critical_value": crit_val
    }
