"""
Pre-generate Synthetic Raman Hyperspectral Dataset Corpus for Step 8 Training.
Usage:
    python scripts/generate_corpus.py --seed 20260806 --n 300 --window 128 --out data/corpus
"""
import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from optiguard.data.simulator import MapSimulator


def generate_corpus(seed: int = 20260806, n: int = 300, window: int = 128, out_dir: str = "data/corpus"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating synthetic corpus: n={n}, window={window}, seed={seed} -> {out_path}")
    
    sim = MapSimulator.from_yaml("configs/simulator.yaml")
    sim.config["seed"] = int(seed)
    
    target_exposures = sim.config["acquisition"].get("target_integration_s", [0.1, 0.25, 0.5, 1.0])
    primary_exp = target_exposures[0] if isinstance(target_exposures, list) else 0.1
    ref_exp = float(sim.config["acquisition"].get("reference_integration_s", 5.0))
    peak_nominal = float(sim.config["silicon"].get("peak_cm1", 520.7))
    
    for i in range(n):
        sample = sim.generate(index=i)
        
        # Determine spectral window crop
        axis = sample.axis
        peak_idx = int(np.argmin(np.abs(axis - peak_nominal)))
        w_start = max(0, peak_idx - window // 2)
        w_end = min(len(axis), peak_idx + window // 2)
        
        # Crop data
        short_counts = sample.short_counts[primary_exp][:, :, w_start:w_end].astype(np.float32)
        clean_rate = sample.rate[:, :, w_start:w_end].astype(np.float32)
        ref_counts = sample.long_counts[:, :, w_start:w_end].astype(np.float32)
        cropped_axis = axis[w_start:w_end].astype(np.float32)
        
        diff_map = sample.difficulty(primary_exp).astype(np.float32)
        defect_mask = sample.defect_mask.astype(bool)
        
        # Compute per-pixel CRLB at primary exposure for normalised centroid loss
        from optiguard.physics.crlb import crlb_peak_position_map
        rn = float(sample.meta.get("read_noise_e", 0.0))
        crlb_map = crlb_peak_position_map(
            axis=sample.axis,
            center=sample.theta_true["center"],
            fwhm=sample.theta_true["fwhm"],
            amplitude=sample.theta_true["amplitude"] * primary_exp,
            background=sample.theta_true["background"] * primary_exp,
            read_noise_e=rn
        ).astype(np.float32)
        
        # True peak centre and defect shift (needed for centroid loss)
        center_true = sample.theta_true["center"].astype(np.float32)        # (H, W) cm^-1
        defect_shift = sample.defect_shift.astype(np.float32)               # (H, W) cm^-1
        
        sample_file = out_path / f"sample_{i:04d}.npz"
        np.savez_compressed(
            sample_file,
            short_counts=short_counts,
            clean_rate=clean_rate,
            reference_counts=ref_counts,
            axis=cropped_axis,
            defect_mask=defect_mask,
            difficulty=diff_map,
            center_true=center_true,
            defect_shift=defect_shift,
            crlb_map=crlb_map,
            exposure=primary_exp,
            reference_exposure=ref_exp,
            w_start=w_start,
            w_end=w_end
        )
        
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  Generated {i + 1}/{n} datacubes ({(i + 1)/n*100:.1f}%)")
            
    info = {
        "n_samples": n,
        "window_channels": window,
        "seed": seed,
        "primary_exposure_s": primary_exp,
        "reference_exposure_s": ref_exp,
        "map_shape": list(sample.rate.shape)
    }
    (out_path / "corpus_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nCorpus generation complete: {n} cubes saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset corpus.")
    parser.add_argument("--seed", type=int, default=20260806, help="Random seed")
    parser.add_argument("--n", type=int, default=300, help="Number of map datacubes to generate")
    parser.add_argument("--window", type=int, default=128, help="Cropped spectral window width in channels")
    parser.add_argument("--out", type=str, default="data/corpus", help="Output directory path")
    args = parser.parse_args()
    
    generate_corpus(seed=args.seed, n=args.n, window=args.window, out_dir=args.out)


if __name__ == "__main__":
    main()
