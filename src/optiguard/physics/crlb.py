import numpy as np
from optiguard.physics.lineshapes import lorentzian

def crlb_peak_position(axis, center, fwhm, amplitude, background, read_noise_e):
    # Match the 128-channel window of the estimator
    window_size = 128
    peak_idx = np.argmin(np.abs(axis - center))
    start_idx = max(0, peak_idx - window_size // 2)
    end_idx = min(len(axis), start_idx + window_size)
    if end_idx - start_idx < window_size and start_idx > 0:
        start_idx = max(0, end_idx - window_size)
        
    axis = axis[start_idx:end_idx]
    
    theta = np.array([center, fwhm, amplitude, background])
    
    def model(t):
        c, f, a, b = t
        return lorentzian(axis, c, f, a) + b
    
    # Central difference
    FIM = np.zeros((4, 4))
    n = len(axis)
    grad = np.zeros((4, n))
    
    for i in range(4):
        eps = 1e-6 * max(1.0, abs(theta[i]))
        t_plus = theta.copy()
        t_plus[i] += eps
        t_minus = theta.copy()
        t_minus[i] -= eps
        
        mu_plus = model(t_plus)
        mu_minus = model(t_minus)
        grad[i] = (mu_plus - mu_minus) / (2 * eps)
        
    mu = model(theta)
    var = mu + read_noise_e**2
    
    for i in range(4):
        for j in range(4):
            FIM[i, j] = np.sum((grad[i] * grad[j]) / var)
            
    cov = np.linalg.inv(FIM)
    return np.sqrt(cov[0, 0])

def crlb_plugin_map(axis, counts_window, fitted_center, read_noise_e,
                    fwhm_pooled=None):
    """Plug-in CRLB for the deployed path — no oracle (theta_true) required.

    Design rationale (per project spec):
    - N (signal photons) comes from integrated raw counts in the window minus
      background estimated from window shoulders. This is well-determined even
      at 0.1 s because it's a sum over 128 channels.
    - Gamma (linewidth) is spatially smooth across a map so a robust pooled
      estimate — map median of the fitted FWHM or an explicit prior — is much
      more stable than the per-pixel fit which is poorly determined at 0.1 s.
    - Background is estimated from the window shoulder counts.

    Args:
        axis: full spectral axis (C,) cm^-1
        counts_window: (H, W, Cw) counts in the 128-channel fit window
        fitted_center: (H, W) fitted peak centres, cm^-1
        read_noise_e: detector read noise
        fwhm_pooled: scalar FWHM estimate to use for all pixels. If None,
            uses median of counts-weighted FWHM inferred from window width,
            which is a crude but stable default.

    Returns:
        (H, W) array of plug-in CRLB values, cm^-1
    """
    from optiguard.physics.lineshapes import lorentzian

    H, W, Cw = counts_window.shape
    disp = float(axis[1] - axis[0]) if len(axis) > 1 else 0.55

    # Background: mean of 16 channels on each shoulder
    shoulder = 16
    bg_map = (np.mean(counts_window[..., :shoulder], axis=-1) +
              np.mean(counts_window[..., -shoulder:], axis=-1)) / 2.0
    bg_map = np.maximum(bg_map, 0.0)

    # Pooled FWHM: map median is robust to a few badly-fit pixels.
    # Default of 3.5 cm-1 from instrument prior; caller should pass the
    # map-median fitted FWHM when available.
    if fwhm_pooled is None:
        fwhm_pooled = 3.5  # instrument prior (cm-1)
    fwhm_map = np.full((H, W), float(fwhm_pooled), dtype=np.float32)

    # Signal integral: total counts minus background contribution.
    # For a Lorentzian with peak height A and FWHM Gamma over a wide window:
    #   integral = A * (pi * Gamma / 2) / disp   (in discrete-channel units)
    # crlb_peak_position_map expects A (peak height), not the integral.
    # Bug: previous code passed the integral directly, underestimating CRLB
    # by sqrt(integral/A) ~ sqrt(pi*fwhm/(2*disp)) ≈ 3×.
    total_signal_integral = np.maximum(
        np.sum(counts_window, axis=-1) - bg_map * Cw, 1.0
    )
    # Convert integral -> peak height: A = integral / (pi * fwhm / (2 * disp))
    lorentzian_integral_factor = np.pi * fwhm_map / (2.0 * disp)
    amplitude_map = np.maximum(
        total_signal_integral / lorentzian_integral_factor, 1.0
    )

    return crlb_peak_position_map(
        axis=axis,
        center=fitted_center,
        fwhm=fwhm_map,
        amplitude=amplitude_map,
        background=bg_map,
        read_noise_e=read_noise_e
    )

def crlb_peak_position_map(axis, center, fwhm, amplitude, background, read_noise_e):
    """Vectorized CRLB across a map."""
    # Crop axis to the mean center to match the estimator's 128 window
    mean_c = np.mean(center)
    window_size = 128
    peak_idx = np.argmin(np.abs(axis - mean_c))
    start_idx = max(0, peak_idx - window_size // 2)
    end_idx = min(len(axis), start_idx + window_size)
    if end_idx - start_idx < window_size and start_idx > 0:
        start_idx = max(0, end_idx - window_size)
        
    axis = axis[start_idx:end_idx]
    
    H, W = center.shape
    C = len(axis)
    
    theta = np.stack([center, fwhm, amplitude, background], axis=-1) # (H, W, 4)
    
    def model(t):
        c = t[..., 0:1]
        f = t[..., 1:2]
        a = t[..., 2:3]
        b = t[..., 3:4]
        return lorentzian(axis, c, f, a) + b
        
    FIM = np.zeros((H, W, 4, 4), dtype=np.float32)
    grad = np.zeros((4, H, W, C), dtype=np.float32)
    
    for i in range(4):
        eps = 1e-6 * np.maximum(1.0, np.abs(theta[..., i]))
        
        t_plus = theta.copy()
        t_plus[..., i] += eps
        t_minus = theta.copy()
        t_minus[..., i] -= eps
        
        mu_plus = model(t_plus)
        mu_minus = model(t_minus)
        grad[i] = (mu_plus - mu_minus) / (2 * eps[..., None])
        
    mu = model(theta)
    var = mu + read_noise_e**2
    
    for i in range(4):
        for j in range(4):
            FIM[..., i, j] = np.sum((grad[i] * grad[j]) / var, axis=-1)
            
    cov = np.linalg.inv(FIM)
    return np.sqrt(cov[..., 0, 0])
