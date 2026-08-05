import numpy as np
import yaml
from dataclasses import dataclass
from typing import Dict, Any

from optiguard.physics.detector import simulate_acquisition
from optiguard.physics.lineshapes import lorentzian
from optiguard.physics.thinning import thin_counts
from optiguard.physics.crlb import crlb_peak_position

@dataclass
class Sample:
    rate: np.ndarray
    long_counts: np.ndarray
    short_counts: Dict[float, np.ndarray]
    theta_true: Dict[str, np.ndarray]
    defect_mask: np.ndarray
    defect_shift: np.ndarray
    axis: np.ndarray
    meta: Dict[str, Any]
    
    def difficulty(self, t: float) -> np.ndarray:
        H, W = self.meta["shape"]
        diff = np.zeros((H, W), dtype=np.float32)
        rn = self.meta.get("read_noise_e", 0.0)
        
        for i in range(H):
            for j in range(W):
                if self.defect_mask[i, j]:
                    crlb = crlb_peak_position(
                        axis=self.axis,
                        center=self.theta_true["center"][i, j],
                        fwhm=self.theta_true["fwhm"][i, j],
                        amplitude=self.theta_true["amplitude"][i, j] * t,
                        background=self.theta_true["background"][i, j] * t,
                        read_noise_e=rn
                    )
                    diff[i, j] = np.abs(self.defect_shift[i, j]) / crlb
        return diff

class MapSimulator:
    def __init__(self, config: dict):
        self.config = config
        self.exposures = config["acquisition"]["target_integration_s"]

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)

    def generate(self, index: int, field: str = None, cosmic_rays: bool = False) -> Sample:
        seed = self.config["seed"] + index
        rng = np.random.default_rng(seed)
        
        H, W = self.config["map"]["shape"]
        mat_conf = self.config[self.config["map"]["material"]]
        det_conf = self.config["detector"]
        acq_conf = self.config["acquisition"]
        
        # Axis
        C = det_conf["n_channels"]
        disp = det_conf["dispersion_cm1_per_px"]
        center_cm1 = mat_conf["peak_cm1"]
        axis = np.arange(C, dtype=np.float32) * disp
        axis = axis - axis.mean() + center_cm1
        
        # Stress field
        if field is None:
            field = self.config["stress_field"]["kind"]
            
        stress = np.zeros((H, W), dtype=np.float32)
        if field == "smooth_gradient":
            sr = self.config["stress_field"]["shift_range_cm1"]
            gx = np.linspace(sr[0], sr[1], W)
            gy = np.linspace(sr[0], sr[1], H)
            stress = (gx[None, :] + gy[:, None]) / 2.0
        elif field == "boundary":
            stress[:, W//2:] = 1.0 # Sharp boundary > 0.5 shift
            
        # Background
        # The fitter assumes constant background so we generate constant background
        # We use the mean of the fluorescence_slope to ensure enough noise for the difficulty test
        bg_val = np.mean(self.config["background"]["fluorescence_slope"])
        bg = np.full((H, W), bg_val, dtype=np.float32)
        
        theta_true = {
            "center": np.full((H, W), center_cm1, dtype=np.float32) + stress,
            "fwhm": np.full((H, W), mat_conf["fwhm_cm1"], dtype=np.float32),
            "amplitude": np.full((H, W), mat_conf["peak_photons_per_s"], dtype=np.float32),
            "background": bg
        }
        
        # Defects
        defect_mask = np.zeros((H, W), dtype=bool)
        defect_shift = np.zeros((H, W), dtype=np.float32)
        
        n_defects = rng.integers(self.config["defects"]["count"][0], self.config["defects"]["count"][1] + 1)
        r_range = self.config["defects"]["radius_px"]
        diff_range = self.config["defects"]["difficulty_crlb"]
        ref_t = self.config["defects"]["difficulty_reference_exposure_s"]
        fwhm_range = self.config["defects"]["fwhm_scale"]
        
        for _ in range(n_defects):
            r = rng.integers(r_range[0], r_range[1] + 1)
            cx, cy = rng.integers(0, W), rng.integers(0, H)
            
            # calculate CRLB at (cy, cx) for reference exposure
            crlb = crlb_peak_position(
                axis=axis,
                center=theta_true["center"][cy, cx],
                fwhm=theta_true["fwhm"][cy, cx],
                amplitude=theta_true["amplitude"][cy, cx] * ref_t,
                background=theta_true["background"][cy, cx] * ref_t,
                read_noise_e=det_conf["read_noise_e"]
            )
            
            # sample difficulty log-uniformly
            diff_val = np.exp(rng.uniform(np.log(diff_range[0]), np.log(diff_range[1])))
            shift = diff_val * crlb
            if rng.random() > 0.5: shift = -shift
            
            fwhm_scale = rng.uniform(fwhm_range[0], fwhm_range[1])
            
            y, x = np.ogrid[-cy:H-cy, -cx:W-cx]
            mask = x**2 + y**2 <= r**2
            
            # Prevent out-of-bounds mapping
            mask = mask & (y + cy >= 0) & (y + cy < H) & (x + cx >= 0) & (x + cx < W)
            
            defect_mask[mask] = True
            defect_shift[mask] = shift
            theta_true["center"][mask] += shift
            theta_true["fwhm"][mask] *= fwhm_scale
            
        # Rate (lambda)
        rate = np.zeros((H, W, C), dtype=np.float32)
        for i in range(H):
            for j in range(W):
                rate[i, j] = lorentzian(axis, theta_true["center"][i, j], theta_true["fwhm"][i, j], theta_true["amplitude"][i, j]) + theta_true["background"][i, j]
                
        T = acq_conf["reference_integration_s"]
        
        # Long counts components
        long_signal = rng.poisson(rate * T)
        long_dark = rng.poisson(det_conf["dark_rate_e_per_s"] * T, size=(H, W, C))
        long_read = rng.normal(0, det_conf["read_noise_e"], size=(H, W, C))
        long_counts = (long_signal + long_dark + long_read).astype(np.int32)
        
        short_counts = {}
        for t in self.exposures:
            short_signal = thin_counts(long_signal, t, T, rng)
            short_dark = rng.poisson(det_conf["dark_rate_e_per_s"] * t, size=(H, W, C))
            short_read = rng.normal(0, det_conf["read_noise_e"], size=(H, W, C))
            arr = (short_signal + short_dark + short_read).astype(np.int32)
            
            if cosmic_rays:
                cr_rate = self.config["background"]["cosmic_ray_rate_per_spectrum"]
                n_spikes = rng.poisson(cr_rate * H * W)
                for _ in range(n_spikes):
                    sy, sx, sz = rng.integers(0, H), rng.integers(0, W), rng.integers(0, C)
                    arr[sy, sx, sz] += rng.integers(1000, 50000)
                    
            short_counts[t] = arr
            
        meta = {
            "shape": (H, W),
            "reference_integration_s": T,
            "read_noise_e": det_conf["read_noise_e"],
            "peak_cm1": float(center_cm1)        # physics anchor, not axis geometry
        }
        
        return Sample(
            rate=rate,
            long_counts=long_counts,
            short_counts=short_counts,
            theta_true=theta_true,
            defect_mask=defect_mask,
            defect_shift=defect_shift,
            axis=axis,
            meta=meta
        )
