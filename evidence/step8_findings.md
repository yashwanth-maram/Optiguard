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

The Jacobian probe ($N_{eff} = 1.02$) reveals that the network achieved its performance **without spatial pooling**—99% of the gradient comes from the central pixel alone. This means its failure mode is purely **spectral smoothing**. 

This makes the physics gate's findings significantly stronger. We now have two mechanistically independent failure modes:
1. **Spatial Pooling** (e.g., Gaussian blur, 16.8x gain): Erases all real features below 2.0 CRLB by averaging them out.
2. **Spectral Smoothing** (The Neural Network, 1.29x gain): Damages only the 1.5 CRLB band by over-smoothing faint peaks.

Both approaches erase real features, and crucially, **neither failure is visible in standard aggregate metrics like RMSE**. The fact that two independent failure mechanisms were both caught by the same recall constraint proves the gate is robust and indispensable.
## Frontiers

![Gain vs Recall](frontier_gain_vs_recall.png)
![Recall vs Difficulty](frontier_recall_vs_difficulty.png)
