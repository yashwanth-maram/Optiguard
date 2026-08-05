import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter, uniform_filter
from sklearn.decomposition import PCA, NMF
from optiguard.estimation.fit import fit_lorentzian_map

def fit_map(axis, counts_cube, read_noise_e, nominal_center_cm1=None):
    return fit_lorentzian_map(axis, counts_cube, read_noise_e=read_noise_e,
                              nominal_center_cm1=nominal_center_cm1)

def _nominal(sample):
    """Return the nominal band centre from the config, stored in sample.meta.
    This is the physics-grounded anchor: it comes from the config's peak_cm1,
    not from axis geometry. axis.mean() == axis.median() == peak_cm1 only when
    the simulator happens to centre the axis, which real spectrographs don't."""
    return float(sample.meta["peak_cm1"])

def baseline_raw(sample, exposure, **kwargs):
    return fit_map(sample.axis, sample.short_counts[exposure],
                   sample.meta.get("read_noise_e", 0.0), _nominal(sample))

def baseline_reference(sample, exposure, **kwargs):
    return fit_map(sample.axis, sample.long_counts,
                   sample.meta.get("read_noise_e", 0.0), _nominal(sample))

def baseline_savgol(sample, exposure, window_length=5, polyorder=2, **kwargs):
    counts = sample.short_counts[exposure]
    filtered = savgol_filter(counts, window_length, polyorder, axis=-1)
    return fit_map(sample.axis, filtered, sample.meta.get("read_noise_e", 0.0), _nominal(sample))

def baseline_binning(sample, exposure, size=3, **kwargs):
    counts = sample.short_counts[exposure]
    filtered = uniform_filter(counts, size=(size, size, 1))
    return fit_map(sample.axis, filtered, sample.meta.get("read_noise_e", 0.0), _nominal(sample))

def baseline_spatial_gauss(sample, exposure, sigma=1.0, **kwargs):
    counts = sample.short_counts[exposure]
    filtered = gaussian_filter(counts, sigma=(sigma, sigma, 0))
    return fit_map(sample.axis, filtered, sample.meta.get("read_noise_e", 0.0), _nominal(sample))

def baseline_pca(sample, exposure, n_components=5, **kwargs):
    counts = sample.short_counts[exposure]
    H, W, C = counts.shape
    flat = counts.reshape(-1, C)
    pca = PCA(n_components=n_components, svd_solver='full')
    reconstructed = pca.inverse_transform(pca.fit_transform(flat)).reshape(H, W, C)
    return fit_map(sample.axis, reconstructed, sample.meta.get("read_noise_e", 0.0), _nominal(sample))

def baseline_nmf(sample, exposure, n_components=5, **kwargs):
    counts = sample.short_counts[exposure]
    H, W, C = counts.shape
    flat = counts.reshape(-1, C)
    flat = np.maximum(flat, 0)
    nmf = NMF(n_components=n_components, init='nndsvd', max_iter=200, random_state=42)
    reconstructed = nmf.inverse_transform(nmf.fit_transform(flat)).reshape(H, W, C)
    return fit_map(sample.axis, reconstructed, sample.meta.get("read_noise_e", 0.0), _nominal(sample))

BASELINES = {
    "raw": baseline_raw,
    "reference": baseline_reference,
    "savgol": baseline_savgol,
    "binning": baseline_binning,
    "spatial_gauss": baseline_spatial_gauss,
    "pca": baseline_pca,
    "nmf": baseline_nmf
}

def tune_baseline(name, samples, exposure=0.1):
    from optiguard.eval.harness import evaluate
    if name in ["raw", "reference"]:
        return BASELINES[name], {}
        
    grid = []
    if name == "savgol":
        grid = [{"window_length": w, "polyorder": p} for w in [5, 9, 15] for p in [2, 3]]
    elif name == "pca":
        # Extend to 16 components: if more is always better the optimum is "don't project"
        grid = [{"n_components": k} for k in [2, 4, 8, 16]]
    elif name == "nmf":
        grid = [{"n_components": k} for k in [2, 4, 8, 16]]
    elif name == "spatial_gauss":
        # sigma=1 -> ~3x3 Gaussian, sigma=2 -> ~5x5; the gain from spatial pooling
        # is the key number — go wide enough to see saturation
        grid = [{"sigma": s} for s in [0.5, 1.0, 1.5, 2.0, 3.0]]
    elif name == "binning":
        # 3x3 box: N_eff=9N; 5x5: N_eff=25N. Defect collapse happens here.
        grid = [{"size": s} for s in [2, 3, 5, 7]]
        
    best_params = None
    best_rmse = float('inf')

    for params in grid:
        captured = dict(params)   # explicit capture — avoids late-binding closure bug
        def method(s, e, p=captured):
            return BASELINES[name](s, e, **p)

        rmse = evaluate(method, samples, exposure).rmse_center
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = captured

    if best_params is None:
        best_params = {}

    def best_method(s, e, p=best_params):
        return BASELINES[name](s, e, **p)

    return best_method, best_params
