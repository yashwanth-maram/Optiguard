import numpy as np
from scipy.optimize import least_squares
from optiguard.physics.lineshapes import lorentzian

def fit_lorentzian(x, y, read_noise_e=0.0):
    # Seed center from weighted centroid
    center_0 = np.sum(x * y) / np.sum(y) if np.sum(y) > 0 else np.mean(x)
    
    # Seed background from the edge channels
    background_0 = np.mean([y[0], y[-1]])
    
    # Seed amplitude and fwhm
    amplitude_0 = np.max(y) - background_0
    if amplitude_0 <= 0:
        amplitude_0 = 1.0
        
    fwhm_0 = (np.max(x) - np.min(x)) / 10.0 # simple fallback

    theta_0 = np.array([center_0, fwhm_0, amplitude_0, background_0])
    
    def residual(theta):
        center, fwhm, amplitude, background = theta
        mu = lorentzian(x, center, fwhm, amplitude) + background
        var = np.maximum(mu + read_noise_e**2, 1e-9)
        return (y - mu) / np.sqrt(var)
        
    res = least_squares(
        residual, 
        theta_0,
        bounds=([-np.inf, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf])
    )
    
    center, fwhm, amplitude, background = res.x
    
    # Compute covariance from Jacobian
    J = res.jac
    try:
        cov = np.linalg.inv(J.T @ J)
        sigma_center = np.sqrt(cov[0, 0])
    except np.linalg.LinAlgError:
        cov = np.full((4, 4), np.inf)
        sigma_center = np.inf
        
    return {
        "center": center,
        "fwhm": fwhm,
        "amplitude": amplitude,
        "background": background,
        "sigma_center": sigma_center,
        "sigma_center": sigma_center,
        "cov": cov
    }

def fit_lorentzian_map(axis, counts_cube, read_noise_e=0.0, max_iter=20, tol=1e-5,
                       nominal_center_cm1=None):
    """Batched Levenberg-Marquardt fitter for an entire (H, W, C) datacube.

    Args:
        axis: spectral axis array (C,), cm^-1
        counts_cube: (H, W, C) integer or float counts
        read_noise_e: detector read noise in electrons
        max_iter: maximum LM iterations
        tol: convergence tolerance on parameter step
        nominal_center_cm1: the expected peak position in cm^-1.
            If provided, the 128-channel window is anchored here — appropriate
            when the band position is known from config or instrument profile.
            If None, the map-median spectrum argmax is used: median across
            thousands of pixels is high-SNR even at 0.1 s and does not depend
            on how the axis was constructed, making it safe for real data.
            **Do not use axis[len//2] as a default — that only works when the
            simulator builds a symmetric axis, which real spectrographs don't.**
    """
    H, W, C = counts_cube.shape
    y_batch_full = counts_cube.reshape(H * W, C).astype(np.float32)
    N = H * W

    # 1. Global Window Crop — anchor on explicit nominal centre or map-median argmax.
    # median is taken across pixels first so cosmic rays / hot pixels don't pull argmax.
    if nominal_center_cm1 is not None:
        peak_idx = int(np.argmin(np.abs(axis - nominal_center_cm1)))
    else:
        median_spec = np.median(y_batch_full, axis=0)
        peak_idx = int(np.argmax(median_spec))

    window_size = 128
    start_idx = max(0, peak_idx - window_size // 2)
    end_idx = min(C, start_idx + window_size)
    if end_idx - start_idx < window_size and start_idx > 0:
        start_idx = max(0, end_idx - window_size)

    y_batch = y_batch_full[:, start_idx:end_idx]
    axis_crop = axis[start_idx:end_idx].astype(np.float32)

    
    # Initialization
    background_0 = np.mean(y_batch[:, [0, -1]], axis=1)
    amplitude_0 = np.max(y_batch, axis=1) - background_0
    amplitude_0 = np.maximum(amplitude_0, 1.0)
    
    y_sum = np.sum(y_batch, axis=1)
    y_sum = np.where(y_sum == 0, 1e-9, y_sum)
    center_0 = np.sum(y_batch * axis_crop, axis=1) / y_sum
    fwhm_0 = np.full(N, (axis_crop[-1] - axis_crop[0]) / 10.0, dtype=np.float32)
    
    theta = np.stack([center_0, fwhm_0, amplitude_0, background_0], axis=1)
    lambda_damp = np.full(N, 1e-3, dtype=np.float32)
    active_mask = np.ones(N, dtype=bool) # True means not yet converged
    
    def calc_residual_and_jac(t_active, y_active):
        c = t_active[:, 0:1]
        f = t_active[:, 1:2]
        a = t_active[:, 2:3]
        b = t_active[:, 3:4]
        
        dx = axis_crop - c
        gamma = f / 2.0
        gamma_sq = gamma**2
        denom = dx**2 + gamma_sq
        
        L = a * gamma_sq / denom
        mu = L + b
        
        var = np.maximum(mu + read_noise_e**2, 1e-9)
        std = np.sqrt(var)
        
        res = (y_active - mu) / std
        
        # Jacobian elements
        dL_da = gamma_sq / denom
        J_c = 2.0 * dx * L / denom
        J_f = L / gamma - gamma * L / denom
        
        J = -np.stack([J_c, J_f, dL_da, np.ones_like(L)], axis=-1) / std[..., None]
        cost = np.sum(res**2, axis=1)
        return res, J, cost

    I = np.eye(4, dtype=np.float32)
    
    for _ in range(max_iter):
        if not np.any(active_mask):
            break
            
        t_active = theta[active_mask]
        y_active = y_batch[active_mask]
        l_active = lambda_damp[active_mask]
        
        res, J, cost = calc_residual_and_jac(t_active, y_active)
        
        JT = np.swapaxes(J, 1, 2)
        JTJ = np.matmul(JT, J)
        JTr = np.matmul(JT, res[:, :, None])[:, :, 0]
        diag_JTJ = np.diagonal(JTJ, axis1=1, axis2=2)
        
        A = JTJ + l_active[:, None, None] * diag_JTJ[:, :, None] * I
        
        # Solve
        delta = np.zeros_like(t_active)
        for i in range(len(t_active)):
            try:
                delta[i] = np.linalg.solve(A[i], -JTr[i])
            except np.linalg.LinAlgError:
                l_active[i] *= 10.0
                
        t_new = t_active + delta
        # Bounds
        t_new[:, 1] = np.maximum(t_new[:, 1], 1e-3)
        t_new[:, 2] = np.maximum(t_new[:, 2], 1e-3)
        
        res_new, J_new, cost_new = calc_residual_and_jac(t_new, y_active)
        
        improved = cost_new < cost
        
        t_active = np.where(improved[:, None], t_new, t_active)
        l_active = np.where(improved, l_active / 10.0, l_active * 10.0)
        l_active = np.clip(l_active, 1e-7, 1e7)
        
        theta[active_mask] = t_active
        lambda_damp[active_mask] = l_active
        
        # Check convergence
        converged = np.max(np.abs(delta), axis=1) < tol
        active_indices = np.where(active_mask)[0]
        active_mask[active_indices[converged]] = False
            
    # Final evaluation for covariance
    res, J, _ = calc_residual_and_jac(theta, y_batch)
    JT = np.swapaxes(J, 1, 2)
    JTJ = np.matmul(JT, J)
    cov = np.zeros_like(JTJ)
    sigma_center = np.full(N, np.inf, dtype=np.float32)
    
    for n in range(N):
        try:
            cov[n] = np.linalg.inv(JTJ[n])
            sigma_center[n] = np.sqrt(cov[n, 0, 0])
        except np.linalg.LinAlgError:
            cov[n] = np.inf
            sigma_center[n] = np.inf
            
    return {
        "center": theta[:, 0].reshape(H, W),
        "fwhm": theta[:, 1].reshape(H, W),
        "amplitude": theta[:, 2].reshape(H, W),
        "background": theta[:, 3].reshape(H, W),
        "cov": cov.reshape(H, W, 4, 4),
        "sigma_center": sigma_center.reshape(H, W)
    }
