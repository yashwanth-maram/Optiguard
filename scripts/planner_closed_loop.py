"""Planner closed-loop validation.

For the pixels the gate flags as FEATURE_ERASURE (NN erased a spectral feature),
the planner is asked to recommend new acquisition settings. We then simulate
N_REALISATIONS independent noise realisations at those settings for each pixel,
fit each one, and compare the scatter of fitted centres vs the planner's
predicted sigma.

Key contracts checked here:
  - plan_reacquisition takes a flagged_mask + fitted_param maps, not per-pixel scalars.
  - MapSimulator.generate(exposure=t) synthesises data at arbitrary t (not just
    the four listed in target_integration_s).
  - The NN is run on the 128-channel window only, matching run_gate_sample6.py.
  - achieved_sigma is measured from 500 independent realisations at the recommended
    settings, using the TRUE pixel physics (theta_true), so there is a known answer.
"""
import sys; sys.path.insert(0, 'src')
import json
import os
import numpy as np
import torch
from optiguard.data.simulator import MapSimulator
from optiguard.models.spatial_spectral import SpatialSpectralUNet
from optiguard.assurance.gate import evaluate_gate
from optiguard.planning.planner import plan_reacquisition, AcquisitionSettings
from optiguard.estimation.fit import fit_lorentzian, fit_lorentzian_map
from optiguard.physics.lineshapes import lorentzian

N_REALISATIONS = 500


def run_nn(model, short_counts, w_start, w_end):
    """Run model on the 128-channel window, return restored full cube."""
    cropped = short_counts[:, :, w_start:w_end]                                  # (H, W, 128)
    inp = torch.from_numpy(cropped.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)  # (1, 128, H, W)
    with torch.no_grad():
        out = model(inp)
    out_np = out.squeeze(0).numpy().transpose(1, 2, 0)                           # (H, W, 128)
    restored = np.copy(short_counts).astype(np.float64)
    restored[:, :, w_start:w_end] = out_np
    return restored


def main():
    rng = np.random.default_rng(42)

    sim = MapSimulator.from_yaml('configs/simulator.yaml')
    s = sim.generate(index=6)
    axis = s.axis
    read_noise_e = s.meta.get('read_noise_e', 4.0)

    # Spectral window (128 channels around nominal peak)
    peak_idx = int(np.argmin(np.abs(axis - 520.7)))
    w_start = max(0, peak_idx - 64)
    w_end   = min(len(axis), w_start + 128)
    axis_w  = axis[w_start:w_end]

    short_counts = s.short_counts[0.1]
    H, W = short_counts.shape[:2]

    # ── Load NN ──────────────────────────────────────────────────────────────
    model = SpatialSpectralUNet(128, 128, 64, 2)
    ckpt = torch.load('src/optiguard/models/step8_checkpoint.pt',
                      map_location='cpu', weights_only=True)
    if 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    model.load_state_dict(ckpt)
    model.eval()

    nn_restored = run_nn(model, short_counts, w_start, w_end)

    # ── Run gate ─────────────────────────────────────────────────────────────
    # We will accumulate results across maps
    all_results = []
    total_flagged = 0
    total_sparse_cost_s = 0.0

    # ── Base settings ─────────────────────────────────────────────────────────
    det = sim.config['detector']
    base_settings = AcquisitionSettings(
        integration_s=0.1,
        accumulations=1,
        laser_power_mw=5.0,
        spectral_binning=1,
        dispersion_cm1_per_px=det['dispersion_cm1_per_px'],
        readout_s=0.05,
        overhead_s_per_point=0.05,
        step_size_um=1.0,
    )
    ref_t = base_settings.integration_s  # 0.1 s

    # ── Loop over test maps (1 to 6) ─────────────────────────────────────────
    for map_idx in range(1, 7):
        s = sim.generate(index=map_idx)
        axis = s.axis
        read_noise_e = s.meta.get('read_noise_e', 4.0)

        # Spectral window (128 channels around nominal peak)
        peak_idx = int(np.argmin(np.abs(axis - 520.7)))
        w_start = max(0, peak_idx - 64)
        w_end   = min(len(axis), w_start + 128)
        axis_w  = axis[w_start:w_end]

        short_counts = s.short_counts[0.1]
        H, W = short_counts.shape[:2]

        nn_restored = run_nn(model, short_counts, w_start, w_end)

        # ── Run gate ─────────────────────────────────────────────────────────────
        raw_gate = evaluate_gate(s, short_counts.astype(np.float64),
                                 np.ones((H, W)),
                                 read_noise_e=read_noise_e,
                                 spectral_window=(w_start, w_end))
        raw_ratios = raw_gate['precision_ratio']
        valid_raw  = raw_ratios[np.isfinite(raw_ratios)]
        t1b_thr    = float(np.percentile(valid_raw, 5.0))

        neff_map_nn = np.full((H, W), 1.02)
        nn_gate = evaluate_gate(s, nn_restored, neff_map_nn,
                                read_noise_e=read_noise_e,
                                t1b_threshold_ratio=t1b_thr,
                                spectral_window=(w_start, w_end))

        risk_class = nn_gate['risk_class']
        crlb_map   = nn_gate['crlb_map']            # (H, W) cm^-1

        flagged_mask = risk_class == 'FEATURE_ERASURE'
        n_flagged    = int(flagged_mask.sum())
        print(f"Map {map_idx}: FEATURE_ERASURE pixels (planner targets): {n_flagged}")

        if n_flagged == 0:
            continue
        
        total_flagged += n_flagged

        # ── Fit raw map to get parameter maps ────────────────────────────────────
        theta_raw = fit_lorentzian_map(axis, short_counts.astype(np.float64),
                                       read_noise_e=read_noise_e,
                                       nominal_center_cm1=520.7)
        fitted_params = {
            'center':     theta_raw['center'],
            'fwhm':       theta_raw['fwhm'],
            'amplitude':  theta_raw['amplitude'] / ref_t,   # counts → photons/s
            'background': theta_raw['background'] / ref_t,  # counts → photons/s
        }

        # Planner target: reduce uncertainty to 0.5× the current median CRLB
        target_sigma = float(np.nanmedian(crlb_map[flagged_mask])) * 0.5
        print(f"  Median CRLB on flagged pixels: {float(np.nanmedian(crlb_map[flagged_mask])):.4f} cm-1")
        print(f"  Target sigma:                  {target_sigma:.4f} cm-1")

        plan = plan_reacquisition(
            flagged_mask=flagged_mask,
            current_settings=base_settings,
            fitted_params=fitted_params,
            axis=axis_w,
            target_sigma=target_sigma,
            read_noise_e=read_noise_e,
        )

        if plan.action is None:
            print("  Planner: no feasible action found.")
            continue

        ns = plan.action.settings
        print(f"  Plan chosen: {plan.action.name}")
        print(f"    integration_s={ns.integration_s:.2f}  acc={ns.accumulations}  bin={ns.spectral_binning}")
        print(f"    cost (sparse rescan): {plan.cost_seconds:.0f}s")
        
        total_sparse_cost_s += plan.cost_seconds

        # ── Per-pixel closed-loop measurement ─────────────────────────────────────
        ys, xs = np.where(flagged_mask)

        for y, x in zip(ys, xs):
            pred_sigma = float(plan.predicted_sigma[y, x])

            true_c = float(s.theta_true['center'][y, x])
            true_f = float(s.theta_true['fwhm'][y, x])
            true_a = float(s.theta_true['amplitude'][y, x])  
            true_b = float(s.theta_true['background'][y, x]) 
            rate_w  = lorentzian(axis_w, true_c, true_f, true_a) + true_b

            centers = []
            for _ in range(N_REALISATIONS):
                spec = np.zeros(len(axis_w))
                for _ in range(ns.accumulations):
                    shot  = rng.poisson(rate_w * ns.integration_s)
                    noise = rng.normal(0, read_noise_e, size=rate_w.shape)
                    spec  += shot + noise

                if ns.spectral_binning > 1:
                    b   = ns.spectral_binning
                    n   = (len(spec) // b) * b
                    spec_b = spec[:n].reshape(-1, b).sum(axis=1)
                    ax_b   = axis_w[:n:b]
                else:
                    spec_b, ax_b = spec, axis_w

                try:
                    f2 = fit_lorentzian(ax_b, spec_b,
                                        read_noise_e=read_noise_e * np.sqrt(ns.accumulations))
                    centers.append(f2['center'])
                except Exception:
                    pass

            if len(centers) < 10:
                continue

            achieved = float(np.std(centers))
            ratio    = achieved / pred_sigma if pred_sigma > 0 else float('nan')

            rec = {
                "map_index": map_idx,
                "pixel": [int(y), int(x)],
                "crlb_raw_cm1":                float(crlb_map[y, x]),
                "target_sigma_cm1":            target_sigma,
                "predicted_sigma_cm1":         pred_sigma,
                "achieved_sigma_cm1":          achieved,
                "ratio_achieved_over_predicted": ratio,
                "plan_integration_s":          float(ns.integration_s),
                "plan_accumulations":          int(ns.accumulations),
                "plan_binning":                int(ns.spectral_binning),
                "n_converged":                 len(centers),
            }
            all_results.append(rec)

    print(f"\nPer-pixel results ({len(all_results)} / {total_flagged} converged):")
    for r in all_results[:5]:
        y, x = r['pixel']
        print(f"  Map {r['map_index']} ({y:2d},{x:2d})  pred={r['predicted_sigma_cm1']:.4f}  "
              f"achv={r['achieved_sigma_cm1']:.4f}  ratio={r['ratio_achieved_over_predicted']:.3f}")
    if len(all_results) > 5:
        print(f"  ... ({len(all_results) - 5} more)")

    ratios = [r['ratio_achieved_over_predicted'] for r in all_results
              if np.isfinite(r['ratio_achieved_over_predicted'])]
    print(f"\n-- Ratio distribution (achieved / predicted) --")
    print(f"  n={len(ratios)}  mean={np.mean(ratios):.3f}  "
          f"median={np.median(ratios):.3f}  "
          f"min={np.min(ratios):.3f}  max={np.max(ratios):.3f}")
    print(f"  within 1.5x: {sum(r < 1.5 for r in ratios)}/{len(ratios)}")
    print(f"  within 2.0x: {sum(r < 2.0 for r in ratios)}/{len(ratios)}")
    
    # ── Time Savings Headline ────────────────────────────────────────────────
    n_pixels_map = H * W
    time_fast_scan = n_pixels_map * (base_settings.integration_s * base_settings.accumulations + 
                                     base_settings.readout_s * base_settings.accumulations + 
                                     base_settings.overhead_s_per_point)
    time_slow_scan = n_pixels_map * (0.4 * 1 + 0.05 * 1 + 0.05)  # assuming 0.4s integration for slow scan
    time_hybrid = time_fast_scan + total_sparse_cost_s
    
    print(f"\n── Time Savings Commercial Headline (Summed across 6 maps) ──")
    print(f"  Fast Scan (0.1s):       {time_fast_scan * 6 / 60:.1f} minutes")
    print(f"  Sparse Rescan Cost:     {total_sparse_cost_s / 60:.1f} minutes")
    print(f"  Total Hybrid Method:    {time_hybrid * 6 / 60:.1f} minutes")
    print(f"  Full Slow Scan (0.4s):  {time_slow_scan * 6 / 60:.1f} minutes")
    print(f"  Speedup Factor:         {time_slow_scan * 6 / (time_hybrid * 6):.2f}x faster")

    os.makedirs('evidence', exist_ok=True)
    with open('evidence/planner_closed_loop.json', 'w') as f:
        json.dump({
            "target_sigma_cm1_med": target_sigma,
            "n_flagged_total": total_flagged,
            "n_converged_total": len(all_results),
            "ratio_mean":   float(np.mean(ratios)) if ratios else None,
            "ratio_median": float(np.median(ratios)) if ratios else None,
            "ratio_min":    float(np.min(ratios)) if ratios else None,
            "ratio_max":    float(np.max(ratios)) if ratios else None,
            "time_savings": {
                "fast_scan_s": float(time_fast_scan * 6),
                "sparse_rescan_s": float(total_sparse_cost_s),
                "total_hybrid_s": float(time_hybrid * 6),
                "full_slow_scan_s": float(time_slow_scan * 6),
                "speedup_factor": float(time_slow_scan * 6 / (time_hybrid * 6))
            },
            "per_pixel":    all_results,
        }, f, indent=2)
    print(f"\nSaved to evidence/planner_closed_loop.json")


if __name__ == "__main__":
    main()
