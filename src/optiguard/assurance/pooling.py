"""
Effective photon budget calculations for spatial pooling.
Step 9 specification.
"""
from typing import Union, Sequence
import numpy as np


def effective_photon_count(
    weights: Union[Sequence[float], np.ndarray],
    counts: Union[Sequence[float], np.ndarray]
) -> float:
    """Compute effective photon count N_eff for a weighted spatial pool.

    Design rationale:
    Let X_i ~ Poisson(N_i) be independent photon counts across neighbouring pixels.
    The weighted sum S = sum_i (w_i * X_i) with sum(w_i) = 1 has:
        E[S] = sum_i (w_i * N_i)
        Var(S) = sum_i (w_i^2 * N_i)
    The relative variance Var(S) / E[S]^2 equals sum_i(w_i^2 * N_i) / (sum_i w_i * N_i)^2.
    An equivalent single Poisson measurement with count N_eff has relative variance 1 / N_eff.
    Equating relative variances yields:
        N_eff = (sum_i w_i * N_i)^2 / sum_i (w_i^2 * N_i)

    For equal photon expectations N_i = N:
        N_eff = N / sum_i (w_i^2)

    Properties:
    - Uniform pooling over M pixels: w_i = 1/M => N_eff = M * N
    - Single dominant pixel: w = [1, 0, ...] => N_eff = N
    - Any normalized non-negative weighting satisfies N <= N_eff <= M * N

    Args:
        weights: 1D array of non-negative pooling weights
        counts: 1D array of photon counts per pixel in the neighbourhood

    Returns:
        N_eff: scalar effective photon count
    """
    w = np.asarray(weights, dtype=np.float64)
    c = np.asarray(counts, dtype=np.float64)

    if w.ndim != 1 or c.ndim != 1:
        w = w.ravel()
        c = c.ravel()

    if len(w) != len(c):
        raise ValueError(f"Length mismatch: weights ({len(w)}) vs counts ({len(c)})")

    # Normalize weights to sum to 1
    w_sum = np.sum(w)
    if w_sum <= 0:
        raise ValueError("Sum of weights must be positive")
    w = w / w_sum

    numerator = np.sum(w * c) ** 2
    denominator = np.sum((w ** 2) * c)

    if denominator <= 0:
        return 0.0

    return float(numerator / denominator)


def gaussian_pooling_weights_2d(sigma: float, radius: int = None) -> np.ndarray:
    """Compute normalized 2D Gaussian pooling kernel weights.

    Args:
        sigma: Gaussian standard deviation in pixels
        radius: Half-window size in pixels (default: ceil(3 * sigma))

    Returns:
        weights: (2*radius + 1, 2*radius + 1) normalized kernel array
    """
    if radius is None:
        radius = int(np.ceil(3.0 * sigma))

    coords = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    dist_sq = xx ** 2 + yy ** 2
    kernel = np.exp(-0.5 * dist_sq / (sigma ** 2))
    return kernel / np.sum(kernel)


def effective_pooling_multiplier(weights: np.ndarray) -> float:
    """Effective photon multiplication factor M_eff = 1 / sum(w_i^2)."""
    w = np.asarray(weights, dtype=np.float64)
    w_norm = w / np.sum(w)
    return float(1.0 / np.sum(w_norm ** 2))
