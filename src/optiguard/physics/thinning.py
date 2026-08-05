import numpy as np

def thin_counts(long_counts, t_target, t_source, rng=None):
    if t_target > t_source:
        raise ValueError("t_target must be <= t_source")
    if rng is None:
        rng = np.random.default_rng()
    
    return rng.binomial(long_counts, t_target / t_source)
