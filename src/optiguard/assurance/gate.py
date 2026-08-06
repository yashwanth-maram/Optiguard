"""
Assurance Gate Tests.

T1a (Variance Deficit): chi2_nu self-consistency test — catches any processing
    that removed noise the photons should have produced. Fires on spatial smoothing.
    This is a self-consistency test, NOT an information-limit test.

T1b (Information Limit): residual-scaled sigma vs CRLB/sqrt(N_eff) — catches
    precision fabrication even after variance has been accounted for by N_eff.
    This is the patentable mechanism: §5.5 of the project spec.

T2 (Pooling Legitimacy): chi2 test for neighbourhood homogeneity.
T3 (Photon Consistency): Poisson deviance between restored and raw counts.

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


def test_variance_deficit(
    observed_counts: np.ndarray,
    fitted_center: float,
    fitted_fwhm: float,
    fitted_amplitude: float,
    fitted_background: float,
    axis: np.ndarray,
    read_noise_e: float = 0.0,
    chi2nu_threshold: float = 0.5
) -> Dict[str, Any]:
    """T1a Gate: Test whether the fit residuals are consistent with the Poisson noise model.

    Self-consistency test: measures chi2_nu = chi2 / df.
    For honest (un-smoothed) data, chi2_nu ~ 1.0.
    For spatially-smoothed data, variance is removed so chi2_nu << 1.
    Any processing that suppresses shot noise below the Poisson floor will be detected.

    IMPORTANT: This is a self-consistency test, NOT an information-limit test.
    A good denoiser that legitimately pools N_eff photons SHOULD also show chi2_nu < 1,
    because it has legitimately reduced variance. The combination of chi2_nu and N_eff
    (via T1b) distinguishes legitimate pooling from variance fabrication.

    Args:
        observed_counts: (C,) raw or restored spectral counts in fit window
        fitted_center, fitted_fwhm, fitted_amplitude, fitted_background: Lorentzian params
        axis: (C,) spectral axis values corresponding to observed_counts
        read_noise_e: Detector read noise standard deviation in electrons
        chi2nu_threshold: Flag if chi2_nu < this value (default 0.5 = deficit > 2-sigma)

    Returns:
        Dict with 'failed': bool, 'chi2_nu', 'chi2', 'df', 'threshold'
    """
    from optiguard.physics.lineshapes import lorentzian

    obs = np.asarray(observed_counts, dtype=np.float64)
    ax = np.asarray(axis, dtype=np.float64)
    C = obs.size

    # Poisson model prediction
    mu = lorentzian(ax, fitted_center, fitted_fwhm, fitted_amplitude) + fitted_background
    var = np.maximum(mu + read_noise_e ** 2, 1e-9)

    # Pearson chi2 statistic
    chi2 = float(np.sum((obs - mu) ** 2 / var))
    df = C - 4  # 4 free parameters: center, fwhm, amplitude, background
    chi2_nu = chi2 / max(df, 1)

    # Flag when variance is significantly suppressed below the Poisson expectation
    failed = bool(chi2_nu < chi2nu_threshold)

    return {
        "gate": "T1a",
        "name": "variance_deficit",
        "failed": failed,
        "chi2_nu": chi2_nu,
        "chi2": chi2,
        "df": df,
        "threshold": chi2nu_threshold
    }


def test_information_limit(
    sigma_center_fit: float,
    chi2_nu: float,
    crlb: float,
    neff: float = 1.0,
    tol: float = 1e-9
) -> Dict[str, Any]:
    """T1b Gate: Test whether residual-scaled precision beats the N_eff-adjusted CRLB.

    Information limit test: the patentable mechanism from §5.5 of the spec.

    The fitter derives sigma_center from the Fisher Information of the Poisson
    model mean, which is correct but blind to upstream smoothing. Multiplying by
    sqrt(chi2_nu) corrects for the actual residual variance:
        sigma_scaled = sigma_center * sqrt(chi2_nu)

    If the processing is honest and pools N_eff pixels, then:
        sigma_scaled >= CRLB / sqrt(N_eff)    [passes T1b]

    If the processing fabricates precision by removing noise without pooling
    real photons, then sigma_scaled will beat the floor despite low chi2_nu:
        sigma_scaled < CRLB / sqrt(N_eff)     [fails T1b]

    The combination of T1a (chi2_nu << 1) AND T1b (beats floor) is the signature
    of the exploit: variance has been removed AND that removal cannot be explained
    by legitimate spatial pooling of independent photons.

    Args:
        sigma_center_fit: sigma_center from Jacobian-based fit covariance (cm^-1)
        chi2_nu: chi2 per degree of freedom from the fit residuals
        crlb: Single-pixel information-theoretic lower bound (cm^-1)
        neff: Effective pooled pixels (from Jacobian probe on real data, >= 1)
        tol: Numerical tolerance

    Returns:
        Dict with 'failed': bool, 'sigma_scaled', 'floor', 'ratio', 'chi2_nu'
    """
    sigma_fit = float(sigma_center_fit)
    chi2_nu_val = float(chi2_nu)
    neff_val = max(float(neff), 1.0)

    # Residual-scaled sigma: corrects for variance removal by upstream processing
    sigma_scaled = sigma_fit * (chi2_nu_val ** 0.5)

    floor = float(crlb) / (neff_val ** 0.5)
    failed = bool(sigma_scaled < (floor - tol))

    return {
        "gate": "T1b",
        "name": "information_limit",
        "failed": failed,
        "sigma_scaled": sigma_scaled,
        "sigma_fit": sigma_fit,
        "chi2_nu": chi2_nu_val,
        "floor": floor,
        "ratio": sigma_scaled / floor if floor > 0 else float("inf"),
        "neff": neff_val
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


def evaluate_gate(
    sample: object,
    restored_counts: np.ndarray,
    neff_map: np.ndarray,
    read_noise_e: float = 0.0,
    chi2nu_threshold: float = 0.5,
    t1b_threshold_ratio: float = 1.0,
    t4_threshold_ratio: float = 1.0,
    alpha_t2: float = 0.01,
    alpha_t3: float = 0.01,
    t2_sigma: float = 2.0,
    spectral_window: Optional[tuple] = None
) -> Dict[str, Any]:
    """Wire T1a/T1b/T2/T3 into a single per-pixel support classification.

    For each pixel, evaluates:
    - T1a (variance_deficit): Are fit residuals consistent with Poisson noise?
          chi2_nu << 1 means variance was removed (smoothing, denoising).
          Self-consistency test — fires on ANY denoiser, including good ones.
    - T1b (information_limit): Does residual-scaled precision beat CRLB/sqrt(N_eff)?
          The patentable mechanism: sigma_fit*sqrt(chi2_nu) vs CRLB/sqrt(neff).
          Fires only if variance removal cannot be explained by legitimate pooling.
    - T2 (pooling_legitimacy): Is the neighbourhood spectrally homogeneous?
    - T3 (photon_consistency): Is the restored spectrum consistent with raw counts?

    Support class priority: T1b > T1a > T2 > T3.
    T1b is the most fundamental violation (fabricated information).
    T1a alone means variance is suppressed but pooling credit may explain it.

    Args:
        sample: MapSample object with .axis, .short_counts, .theta_true, .meta
        restored_counts: (H, W, C) array of restored spectral counts
        neff_map: (H, W) per-pixel N_eff from Jacobian probe on real data
        read_noise_e: Detector read noise in electrons
        alpha_t2: Significance level for T2 boundary test
        alpha_t3: Significance level for T3 photon consistency test
        t2_sigma: Gaussian sigma (pixels) for T2 neighbourhood
        spectral_window: Optional (start, end) channel indices for T2 and T3
        chi2nu_threshold: T1a threshold — flag if chi2_nu < this (default 0.5)

    Returns:
        Dict with:
          'support_class':  (H, W) str array ('PASS'/'FAIL_T1a'/'FAIL_T1b'/'FAIL_T2'/'FAIL_T3')
          'fail_t1a':       (H, W) bool — variance deficit
          'fail_t1b':       (H, W) bool — information limit
          'fail_t2':        (H, W) bool — neighbourhood heterogeneous
          'fail_t3':        (H, W) bool — photon inconsistency
          'chi2_nu_map':    (H, W) float — per-pixel chi2/df from fit residuals
          'sigma_scaled_map': (H, W) float — sigma_fit * sqrt(chi2_nu)
          'crlb_map':       (H, W) single-pixel CRLB
          'floor_map':      (H, W) N_eff-adjusted CRLB floor
          'precision_ratio': (H, W) sigma_scaled / floor (T1b quantity)
          'summary':        dict of counts per class
    """
    from optiguard.physics.crlb import crlb_plugin_map
    from optiguard.estimation.fit import fit_lorentzian_map
    from optiguard.physics.lineshapes import lorentzian

    axis = sample.axis
    raw_counts = sample.short_counts[list(sample.short_counts.keys())[0]]
    H, W, C = raw_counts.shape
    peak_nominal = float(sample.meta.get("peak_cm1", 520.7))

    # Allow calling without a restoration network: fall back to raw counts
    if restored_counts is None:
        restored_counts = raw_counts.astype(np.float64)

    # Allow calling without a probe-derived neff: no pooling credit
    if neff_map is None:
        neff_map = np.ones((H, W), dtype=np.float64)

    # --- Fit the restored map to get per-pixel parameters and sigma_center ---
    theta_restored = fit_lorentzian_map(axis, restored_counts,
                                        read_noise_e=read_noise_e,
                                        nominal_center_cm1=peak_nominal)
    sigma_center_map = theta_restored["sigma_center"]  # (H, W)

    # Spectral window used by the fitter (128 channels around peak)
    peak_idx = int(np.argmin(np.abs(axis - peak_nominal)))
    fw_start = max(0, peak_idx - 64)
    fw_end = min(C, peak_idx + 64)
    axis_w = axis[fw_start:fw_end]
    restored_w = restored_counts[:, :, fw_start:fw_end]   # (H, W, Cw)

    # --- T1a: per-pixel variance deficit (chi2_nu) ---
    Cw = fw_end - fw_start
    df = max(Cw - 4, 1)  # 4 free parameters

    centers = theta_restored["center"]           # (H, W)
    fwhms = theta_restored["fwhm"]               # (H, W)
    amplitudes = theta_restored["amplitude"]     # (H, W)
    backgrounds = theta_restored["background"]   # (H, W)

    # Vectorised chi2 computation over the fit window
    # axis_w: (Cw,), model params: (H, W)
    dx = axis_w[None, None, :] - centers[:, :, None]          # (H, W, Cw)
    gamma = fwhms[:, :, None] / 2.0
    L = amplitudes[:, :, None] * gamma**2 / (dx**2 + gamma**2)
    mu_w = L + backgrounds[:, :, None]                        # (H, W, Cw)
    var_w = np.maximum(mu_w + read_noise_e**2, 1e-9)

    chi2_map = np.sum((restored_w - mu_w)**2 / var_w, axis=-1)  # (H, W)
    chi2_nu_map = chi2_map / df                                   # (H, W)
    fail_t1a = chi2_nu_map < chi2nu_threshold

    # --- CRLB plugin map ---
    crlb_map_vals = crlb_plugin_map(
        axis=axis_w,
        counts_window=restored_w,
        fitted_center=centers,
        read_noise_e=read_noise_e
    )
    floor_map = crlb_map_vals / np.maximum(neff_map ** 0.5, 1.0)

    # --- T1b: residual-scaled sigma vs N_eff-adjusted CRLB ---
    sigma_scaled_map = sigma_center_map * np.sqrt(np.maximum(chi2_nu_map, 0.0))
    precision_ratio = np.where(floor_map > 0, sigma_scaled_map / floor_map, np.inf)
    fail_t1b = precision_ratio < t1b_threshold_ratio

    # --- T2: pooling legitimacy across the map ---
    t2_result = test_pooling_legitimacy_map(
        raw_counts,
        sigma=t2_sigma,
        read_noise_e=read_noise_e,
        alpha=alpha_t2,
        spectral_window=spectral_window
    )
    fail_t2 = t2_result["failed"]

    # --- T3: per-pixel photon consistency (vectorised) ---
    if spectral_window is not None:
        t3_start, t3_end = spectral_window
    else:
        t3_start, t3_end = 0, C

    raw_t3  = raw_counts[:, :, t3_start:t3_end].astype(np.float64)      # (H, W, Ct)
    rest_t3 = restored_counts[:, :, t3_start:t3_end].astype(np.float64)  # (H, W, Ct)
    Ct = t3_end - t3_start
    var_t3   = np.maximum(rest_t3 + read_noise_e ** 2, 1e-6)
    chi2_t3  = np.sum((raw_t3 - rest_t3) ** 2 / var_t3, axis=-1)         # (H, W)
    crit_t3  = float(stats.chi2.ppf(1.0 - alpha_t3, Ct))
    fail_t3  = chi2_t3 > crit_t3

    # --- T4: Feature Retention / Structural Preservation Test ---
    from scipy.ndimage import median_filter
    theta_raw = fit_lorentzian_map(axis, raw_counts.astype(np.float64),
                                  read_noise_e=read_noise_e,
                                  nominal_center_cm1=peak_nominal)
    c_raw = theta_raw["center"]
    crlb_raw = theta_raw["sigma_center"]

    # Spatial background expectation (5x5 median filter on raw centers)
    c_spatial_bg = median_filter(c_raw, size=5)

    # Raw peak shift from spatial background in CRLB units
    dev_raw = np.abs(c_raw - c_spatial_bg)
    dev_rest = np.abs(theta_restored["center"] - c_spatial_bg)

    d_raw = np.where((crlb_raw > 0) & np.isfinite(crlb_raw), dev_raw / crlb_raw, 0.0)
    raw_supports_feature = d_raw >= 1.0

    # Feature Erasure Discrepancy (D_T4): Amount of peak shift from spatial background erased by restoration
    # D_T4 = max(0, dev_raw - dev_rest) / crlb_raw
    c_restored = theta_restored["center"]
    d_t4_map = np.zeros((H, W), dtype=np.float64)
    valid_fits = np.isfinite(c_raw) & np.isfinite(c_restored) & np.isfinite(crlb_raw) & (crlb_raw > 0)
    active_feat = valid_fits & raw_supports_feature
    d_t4_map[active_feat] = np.maximum(0.0, dev_raw[active_feat] - dev_rest[active_feat]) / crlb_raw[active_feat]

    fail_t4 = d_t4_map > t4_threshold_ratio

    # --- Continuous Physical Confidence Scoring ---
    # 1. Variance deficit penalty (only applies when chi2_nu < chi2nu_threshold)
    d_t1a = np.where(chi2_nu_map < chi2nu_threshold,
                     (chi2nu_threshold - chi2_nu_map) / chi2nu_threshold, 0.0)
    d_t1a = np.maximum(0.0, d_t1a)

    # 2. Precision ratio penalty (only applies when precision_ratio < t1b_threshold_ratio)
    d_t1b = np.where(precision_ratio < t1b_threshold_ratio,
                     (t1b_threshold_ratio - precision_ratio) / t1b_threshold_ratio, 0.0)
    d_t1b = np.maximum(0.0, d_t1b)

    # 3. Feature erasure penalty (only applies when D_T4 > t4_threshold_ratio)
    d_t4_pen = np.where(d_t4_map > t4_threshold_ratio,
                        (d_t4_map - t4_threshold_ratio) / t4_threshold_ratio, 0.0)
    d_t4_pen = np.maximum(0.0, d_t4_pen)

    from optiguard.assurance.ood import score_map, explain

    # --- OOD Detection ---
    ood_score = score_map(axis, raw_counts.astype(np.float64))
    ood_penalty = np.exp(-ood_score)
    
    honesty_score = np.exp(- (2.0 * d_t1a**2 + 2.0 * d_t1b**2))
    feature_score = np.exp(- (1.0 * d_t4_pen**2))
    confidence_score = np.clip(honesty_score * feature_score * ood_penalty, 0.0, 1.0)

    # --- Categorical Risk Fusion ---
    risk_class = np.full((H, W), "PASS", dtype=object)
    risk_class[fail_t4] = "FEATURE_ERASURE"
    risk_class[fail_t3] = "HALLUCINATION"
    exploit_mask = fail_t1a & fail_t1b
    risk_class[exploit_mask] = "EXPLOIT"
    
    # We can flag OOD if the penalty drops confidence heavily
    ood_mask = ood_penalty < 0.5
    risk_class[ood_mask] = "REVIEW"

    n_supported = int(np.sum(raw_supports_feature))
    n_erased_supp = int(np.sum(fail_t4 & raw_supports_feature))
    feature_erasure_rate = float(n_erased_supp / max(1, n_supported))
    feat_score_supp = float(feature_score[raw_supports_feature].mean()) if n_supported > 0 else 1.0

    # Attach explanation if any pixel is flagged for OOD
    ood_rationale = None
    if np.any(ood_mask):
        # We need a dummy object or just pass axis and counts to explain
        ood_rationale = explain(axis, raw_counts.astype(np.float64), ood_score)

    summary = {
        "n_pass":                   int(np.sum(risk_class == "PASS")),
        "n_feature_erasure":        int(np.sum(risk_class == "FEATURE_ERASURE")),
        "n_hallucination":          int(np.sum(risk_class == "HALLUCINATION")),
        "n_exploit":                int(np.sum(risk_class == "EXPLOIT")),
        "n_ood":                    int(np.sum(risk_class == "REVIEW")),   # OOD veto pixels
        "n_boundary":               int(np.sum(fail_t2)),
        "n_supported_features":     n_supported,
        "n_erased_supported":       n_erased_supp,
        "feature_erasure_rate":     feature_erasure_rate,
        "feature_score_on_supported": feat_score_supp,
        "total":                    int(H * W),
        "mean_confidence":          float(confidence_score.mean()),
        "mean_honesty":             float(honesty_score.mean()),
        "mean_feature":             float(feature_score.mean()),
        "mean_ood_score":           float(ood_score.mean()),
        "ood_threshold_used":       0.5,
    }

    if ood_rationale:
        summary["ood_rationale"] = ood_rationale
    summary["pass_rate"] = summary["n_pass"] / summary["total"]

    return {
        "risk_class":           risk_class,
        "confidence_score":     confidence_score,
        "honesty_score":        honesty_score,
        "feature_score":        feature_score,
        "raw_supports_feature": raw_supports_feature,
        "fail_t1a":             fail_t1a,
        "fail_t1b":             fail_t1b,
        "fail_t2":              fail_t2,
        "fail_t3":              fail_t3,
        "fail_t4":              fail_t4,
        "d_t4_map":             d_t4_map,
        "chi2_nu_map":          chi2_nu_map,
        "sigma_scaled_map":     sigma_scaled_map,
        "crlb_map":             crlb_map_vals,
        "floor_map":            floor_map,
        "precision_ratio":      precision_ratio,
        "ood_score_map":        ood_score,
        "ood_penalty_map":      ood_penalty,
        "summary":              summary,
        "center_map":           theta_restored["center"],
        "sigma_center_map":     theta_restored["sigma_center"],
    }


