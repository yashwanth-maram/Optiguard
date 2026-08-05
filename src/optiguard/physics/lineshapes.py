import numpy as np

def lorentzian(x, center, fwhm, amplitude):
    gamma = fwhm / 2.0
    return amplitude * (gamma**2) / ((x - center)**2 + gamma**2)

def pseudo_voigt(x, center, fwhm, amplitude, eta):
    sigma = fwhm / 2.3548
    g = amplitude * np.exp(-((x - center)**2) / (2 * sigma**2))
    l = lorentzian(x, center, fwhm, amplitude)
    return eta * l + (1 - eta) * g
