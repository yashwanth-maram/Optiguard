import numpy as np
from typing import Dict, Any, Tuple
from optiguard.physics.lineshapes import lorentzian
from optiguard.estimation.fit import fit_lorentzian_map

def score_residual_structure(axis: np.ndarray, counts: np.ndarray, fitted: Dict[str, np.ndarray], read_noise_e: float = 0.0) -> np.ndarray:
    """Detect model mismatch via structured residuals (e.g. from overlapping doublets).

    Fully vectorised: no Python pixel loop.  ~100× faster than the loop version on 64×64.
    """
    H, W, C = counts.shape

    # Broadcast fitted params: (H, W, C)
    c_map  = fitted["center"][:, :, None]      # (H, W, 1)
    f_map  = fitted["fwhm"][:, :, None]
    a_map  = fitted["amplitude"][:, :, None]
    bg_map = fitted["background"][:, :, None]

    gamma = f_map / 2.0
    dx    = axis[None, None, :] - c_map        # (H, W, C)
    mu    = a_map * gamma**2 / (dx**2 + gamma**2) + bg_map   # (H, W, C)
    mu    = np.where(np.isfinite(mu), mu, 0.0)
    var   = np.maximum(mu + read_noise_e**2, 1e-9)

    # Standardised residuals  (H, W, C)
    res   = (counts.astype(np.float64) - mu) / np.sqrt(var)
    res   = np.where(np.isfinite(res), res, 0.0)

    # Chi2_nu
    df      = max(1, C - 4)
    chi2_nu = np.sum(res**2, axis=-1) / df      # (H, W)

    # Lag-1 autocorrelation (vectorised over the spectral axis)
    if C > 1:
        res_mean = np.mean(res, axis=-1, keepdims=True)   # (H, W, 1)
        r        = res - res_mean
        num      = np.sum(r[..., :-1] * r[..., 1:], axis=-1)   # (H, W)
        den      = np.sum(r**2, axis=-1)                        # (H, W)
        lag1     = np.where(den > 0, num / den, 0.0)
    else:
        lag1 = np.zeros((H, W), dtype=np.float32)

    score = np.maximum(0.0, chi2_nu - 1.0) * np.maximum(0.0, lag1)
    return score.astype(np.float32)

def score_parameter_plausibility(fitted: Dict[str, np.ndarray], prior_fwhm: float = 3.5, nominal_center: float = 520.7) -> np.ndarray:
    """Detect gross material shifts via implausible fitted parameters."""
    H, W = fitted["center"].shape
    score = np.zeros((H, W), dtype=np.float32)
    
    # Extreme FWHM differences (e.g. amorphous silicon)
    fwhm = fitted["fwhm"]
    fwhm_dev = np.abs(fwhm - prior_fwhm) / prior_fwhm
    
    # Extreme center differences (e.g. GaN or SiC)
    center = fitted["center"]
    center_dev = np.abs(center - nominal_center) / 10.0 # scale by roughly the window size or reasonable expected shift
    
    # Just max deviation
    score = np.maximum(fwhm_dev, center_dev)
    
    # Penalize negative amplitudes
    score[fitted["amplitude"] <= 0] = 100.0
    
    return score

def score_embedding_distance(counts: np.ndarray, reference_pca) -> np.ndarray:
    """ML Baseline: Mahalanobis distance in PCA space."""
    H, W, C = counts.shape
    X = counts.reshape(H * W, C)
    
    # Project and reconstruct
    X_mean = X - reference_pca['mean']
    proj = X_mean @ reference_pca['components'].T
    
    # Mahalanobis distance
    dists = np.sum((proj / reference_pca['singular_values'])**2, axis=1)
    
    return dists.reshape(H, W)

def explain(axis: np.ndarray, counts: np.ndarray, score_map=None, prior_fwhm=3.5, nominal_center=520.7) -> str:
    """Return a rationale for the highest OOD scoring pixel in the map."""
    # Fit the map to get parameters
    fitted = fit_lorentzian_map(axis, counts.astype(np.float64), nominal_center_cm1=nominal_center)
    
    if score_map is None:
        score_map = score_residual_structure(axis, counts, fitted) + score_parameter_plausibility(fitted, prior_fwhm, nominal_center)
        
    y, x = np.unravel_index(np.argmax(score_map), score_map.shape)
    
    fwhm = fitted["fwhm"][y, x]
    center = fitted["center"][y, x]
    
    if np.abs(fwhm - prior_fwhm) / prior_fwhm > 0.5:
        return f"fitted linewidth {fwhm:.1f} cm⁻¹ against instrument prior {prior_fwhm}"
    if np.abs(center - nominal_center) > 10.0:
        return f"fitted center {center:.1f} cm⁻¹ far from nominal {nominal_center}"
        
    return f"structured residuals indicating overlapping modes or wrong lineshape"

def score_map(axis: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Primary physics-first OOD scoring function."""
    fitted = fit_lorentzian_map(axis, counts.astype(np.float64))
    
    s_res = score_residual_structure(axis, counts, fitted)
    s_param = score_parameter_plausibility(fitted)
    
    # The doublet will be caught by s_res. The gross shifts by s_param.
    return s_res + s_param

def calibrate_threshold(material: str, indices, target_fpr: float = 0.05) -> float:
    from optiguard.data.simulator import MapSimulator
    sim = MapSimulator.from_yaml("configs/simulator.yaml")
    
    scores = []
    for i in indices:
        sample = sim.generate(index=i, material=material)
        scores.append(score_map(sample.axis, sample.long_counts))
        
    all_scores = np.concatenate([s.flatten() for s in scores])
    return float(np.quantile(all_scores, 1.0 - target_fpr))

def auroc_against_silicon(ood_maps, si_maps, method="physics") -> float:
    from sklearn.metrics import roc_auc_score
    y_true = []
    y_score = []
    
    for m in si_maps:
        y_true.append(np.zeros(m.long_counts.shape[:2]))
        y_score.append(score_map(m.axis, m.long_counts) if method == "physics" else _score_ml(m, si_maps))
        
    for m in ood_maps:
        y_true.append(np.ones(m.long_counts.shape[:2]))
        y_score.append(score_map(m.axis, m.long_counts) if method == "physics" else _score_ml(m, si_maps))
        
    return roc_auc_score(np.concatenate([y.flatten() for y in y_true]), 
                         np.concatenate([y.flatten() for y in y_score]))

def _score_ml(sample, si_maps):
    # Fit PCA on si_maps once if not done
    if not hasattr(_score_ml, 'pca'):
        X_train = np.concatenate([m.long_counts.reshape(-1, m.long_counts.shape[-1]) for m in si_maps])
        mean = np.mean(X_train, axis=0)
        U, S, Vt = np.linalg.svd(X_train - mean, full_matrices=False)
        k = min(5, len(S))
        _score_ml.pca = {'mean': mean, 'components': Vt[:k], 'singular_values': S[:k]}
        
    return score_embedding_distance(sample.long_counts, _score_ml.pca)

def auroc_by_method(maps_dict):
    si = maps_dict["silicon"]
    methods = ["residual_structure", "parameter_plausibility", "embedding_distance"]
    results = {}
    
    for method in methods:
        results[method] = {}
        for mat in ["amorphous_silicon", "sic_4h", "gan", "si_doublet"]:
            if method == "residual_structure":
                def score_fn(m):
                    f = fit_lorentzian_map(m.axis, m.long_counts.astype(np.float64))
                    return score_residual_structure(m.axis, m.long_counts, f)
            elif method == "parameter_plausibility":
                def score_fn(m):
                    f = fit_lorentzian_map(m.axis, m.long_counts.astype(np.float64))
                    return score_parameter_plausibility(f)
            else:
                def score_fn(m): return _score_ml(m, si)
                
            from sklearn.metrics import roc_auc_score
            y_true = np.concatenate([np.zeros(np.prod(m.long_counts.shape[:2])) for m in si] + 
                                    [np.ones(np.prod(m.long_counts.shape[:2])) for m in maps_dict[mat]])
            y_score = np.concatenate([score_fn(m).flatten() for m in si] + 
                                     [score_fn(m).flatten() for m in maps_dict[mat]])
            results[method][mat] = float(roc_auc_score(y_true, y_score))
            
    return results
