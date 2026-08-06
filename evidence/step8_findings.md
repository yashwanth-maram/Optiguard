# Step 8 Findings: Spatial-Spectral Network Evaluation

## Network Performance

The training run with the physics-constrained gate active completed, but **no checkpoint passed the gate**. 

- The effective exposure gain saturated at ~1.29x by epoch 30.
- The recall at the critical 1.5 CRLB difficulty tier fell to `0.548` (well below the `0.719` raw floor) and stayed there.

## The Danger of Standard Criteria

The most critical finding from this run is that the network improved on nearly every conventional metric:
- **RMSE** dropped from `0.0438` (raw) to `0.0379` cm⁻¹.
- **False-feature rate** dropped from `0.0031` to `0.0005`.
- **Recall at higher difficulties** improved or remained perfect (e.g., recall at 2.5 CRLB improved from `0.76` to `0.83`).

Under standard evaluation criteria (RMSE, overall precision/recall, loss convergence), this model would have been considered a success and shipped. However, it systematically failed in one narrow, critical band: it destroyed faint but previously recoverable peaks near the detection limit (1.5 CRLB). 

### Mechanistically Distinct Failure Modes

The Jacobian probe ($N_{eff} = 1.02$ on real spectroscopic data) reveals that the network
achieved its performance **without spatial pooling** — 99% of the gradient comes from the
central pixel alone. This means its failure mode is purely **spectral smoothing**.

The mechanism is the skip connection. The architecture computes
`out = softplus(x + res)` where `res` is the U-Net's residual output. Near the
training distribution, `res` is small (the network learned to make small corrections),
so `softplus(x + res) ≈ x` pointwise — the Jacobian is dominated by the identity path
and is therefore exactly one pixel wide. With out-of-distribution input such as pure
random noise, the residual branch is no longer suppressed, the convolutional path
activates more strongly, and the spatial footprint widens. This is consistent with the
edge-differential measurement that showed the network's output differed from its input
by only ~1% in the peak region.

The correct prediction from this model is: probe $N_{eff}$ at varying input amplitudes
(or noise levels) and it should rise monotonically as the input moves away from the
training manifold. This is testable and has not yet been run.

> **Probe reconciliation (important for reproducibility):** An earlier full-map run of the
> probe reported Median $N_{eff} = 8.88$. That run used `torch.randn` as input instead of
> real data. Because the network operates near identity on real data (residual branch
> suppressed), probing with random noise puts the network into a different local linear
> region — one where the convolutional path is not suppressed — which widens the spatial
> footprint and inflates $N_{eff}$. This is a piecewise-linear Jacobian effect (the
> architecture uses GroupNorm, not BatchNorm; there are no batch-statistics artefacts).
> The `step9_neff_map.py` script has been fixed to require real data. The 1.02 result,
> validated against the same real-data input in both `jacobian_probe.py` and the fixed
> script, is the operationally correct value.

This makes the physics gate's findings significantly stronger. We now have two mechanistically independent failure modes:
1. **Spatial Pooling** (e.g., Gaussian blur, 16.8× gain): Erases all real features below 2.0 CRLB by averaging them out.
2. **Spectral Smoothing** (The Neural Network, 1.29× gain): Damages only the 1.5 CRLB band by over-smoothing faint peaks.

Both approaches erase real features, and crucially, **neither failure is visible in standard aggregate metrics like RMSE**. The fact that two independent failure mechanisms were both caught by the same recall constraint proves the gate is robust and indispensable.


## Frontiers

![Gain vs Recall](frontier_gain_vs_recall.png)
![Recall vs Difficulty](frontier_recall_vs_difficulty.png)

## The Pattern of Exploits

This marks the third distinct exploit found in this project:
1. **Softmax Quantisation**: Exploited the discretisation of the output to achieve zero loss without fitting the continuous position.
2. **Edge Lever-Arm**: Manipulated noise in the outermost channels to shift the linear Center of Mass with minimal absolute change.
3. **Baseline Wedge**: Generated an asymmetric baseline pedestal beneath the peak to mathematically shift the Center of Mass without touching the peak's shape.

In all three cases, the model drove a plausible-looking training loss near zero while the physics-grounded metric (the physical validation gate) correctly showed no real restoration was happening. Every exploit was caught by an independent evaluator that explicitly models the underlying physics (e.g., Lorentzian fitting with baseline tilt as a free parameter) rather than relying on the training objective. This establishes a clear, verifiable pattern: standard training objectives are incredibly vulnerable to physical loopholes, making an independent, physics-constrained evaluation gate absolutely indispensable for any spectral denoiser.
