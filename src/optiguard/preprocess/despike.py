import numpy as np
from scipy.ndimage import median_filter

def remove_cosmic_rays(spectrum_cube: np.ndarray, threshold: float = 10.0) -> np.ndarray:
    """
    Removes cosmic rays using a simple temporal/spectral median filter.
    Detects spikes that are significantly above the local median.
    """
    # Simple median filter over spectral axis for each pixel
    # For a 3D cube (H, W, C), filter over C.
    med = median_filter(spectrum_cube, size=(1, 1, 3))
    
    # Calculate a rough local noise estimate (MAD)
    abs_diff = np.abs(spectrum_cube - med)
    mad = median_filter(abs_diff, size=(1, 1, 7))
    
    # Spikes are where the difference is large compared to shot noise
    # Shot noise floor scales with sqrt(signal)
    noise_floor = np.maximum(mad, np.sqrt(np.maximum(med, 1.0)))
    spikes = abs_diff > threshold * np.maximum(noise_floor, 5.0)
    
    cleaned = spectrum_cube.copy()
    cleaned[spikes] = med[spikes]
    
    return cleaned
