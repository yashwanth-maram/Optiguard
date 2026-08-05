import numpy as np

def simulate_acquisition(integration_s, signal_rate, read_noise_e, dark_rate_e_per_s, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    signal_counts = rng.poisson(signal_rate * integration_s)
    
    # dark_rate_e_per_s can be scalar or array
    dark_counts = rng.poisson(dark_rate_e_per_s * integration_s)
    
    # Ensure size handles broadcast if needed
    size = np.asarray(signal_rate).shape
    if not size:
        size = 1
    
    read_noise = rng.normal(0, read_noise_e, size=size)
    return signal_counts + dark_counts + read_noise

def estimate_gain_from_ptc(means, variances):
    # Var(ADU) = mean(ADU)/gain + const
    # Fit a straight line: variance = (1/gain) * mean + const
    slope, intercept = np.polyfit(means, variances, 1)
    return 1.0 / slope
