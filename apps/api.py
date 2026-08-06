import sys
import os
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from optiguard.data.simulator import MapSimulator
from optiguard.assurance.gate import evaluate_gate
from optiguard.planning.planner import plan_reacquisition, AcquisitionSettings

app = FastAPI(title="Optiguard AI Operator Console API")

# Initialize simulator globally
SIMULATOR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'simulator.yaml'))
sim = MapSimulator.from_yaml(SIMULATOR_PATH)

class ScanRequest(BaseModel):
    material: str
    index: int = 6
    exposure_s: float = 0.1

def _make_serializable(obj):
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        # Handle nan values for JSON, but only for numeric arrays
        if np.issubdtype(obj.dtype, np.number):
            return np.where(np.isnan(obj), None, obj).tolist()
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj

@app.get("/config")
def get_config():
    """Return simulator configuration to populate the UI."""
    return {
        "materials": list(sim.config["materials"].keys()),
        "detector": sim.config["detector"]
    }

@app.post("/scan")
def run_scan(req: ScanRequest):
    """Run a full simulation, gate evaluation, and planning pass."""
    if req.material not in sim.config["materials"]:
        raise HTTPException(status_code=400, detail="Unknown material")
        
    try:
        # 1. Generate map
        s = sim.generate(index=req.index, material=req.material)
        if req.exposure_s not in s.short_counts:
            # Fallback to the first available exposure if not exactly matching
            exposure = list(s.short_counts.keys())[0]
        else:
            exposure = req.exposure_s
            
        short_counts = s.short_counts[exposure]
        H, W = short_counts.shape[:2]
        
        # 2. Extract spectral window (for gating and planning)
        axis = s.axis
        peak_idx = int(np.argmin(np.abs(axis - 520.7)))
        w_start = max(0, peak_idx - 64)
        w_end   = min(len(axis), w_start + 128)
        axis_w  = axis[w_start:w_end]
        
        # 3. Evaluate Gate (which inherently runs OOD physics detectors)
        # Note: We pass short_counts directly instead of NN restored counts for speed in the UI demo,
        # but the physics rules will still fire. We use a neff_map of 1.0.
        neff_map = np.ones((H, W))
        gate_res = evaluate_gate(
            s, 
            short_counts.astype(np.float64),
            neff_map,
            read_noise_e=s.meta.get("read_noise_e", 4.0),
            t1b_threshold_ratio=3.0, # Approximate threshold
            spectral_window=(w_start, w_end)
        )
        
        # 4. Extract data for the UI
        risk_class = gate_res["risk_class"]
        ood_score_map = gate_res["ood_score_map"]
        crlb_map = gate_res["crlb_map"]
        summary = gate_res["summary"]
        
        # Compute some extra top-level stats
        total_pixels = H * W
        flagged_mask = risk_class == "FEATURE_ERASURE"
        n_flagged = int(flagged_mask.sum())
        
        # Base scanning cost
        base_scan_cost = total_pixels * (exposure + 0.05 + 0.05) # int + read + overhead
        
        hybrid_cost = base_scan_cost
        
        # 5. Planning (if needed)
        planner_action = None
        if n_flagged > 0:
            det = sim.config['detector']
            base_settings = AcquisitionSettings(
                integration_s=exposure,
                accumulations=1,
                laser_power_mw=5.0,
                spectral_binning=1,
                dispersion_cm1_per_px=det['dispersion_cm1_per_px'],
                readout_s=0.05,
                overhead_s_per_point=0.05,
                step_size_um=1.0,
            )
            
            from optiguard.estimation.fit import fit_lorentzian_map
            theta_raw = fit_lorentzian_map(axis, short_counts.astype(np.float64),
                                        read_noise_e=s.meta.get("read_noise_e", 4.0),
                                        nominal_center_cm1=520.7)
            fitted_params = {
                'center':     theta_raw['center'],
                'fwhm':       theta_raw['fwhm'],
                'amplitude':  theta_raw['amplitude'] / exposure,
                'background': theta_raw['background'] / exposure,
            }
            
            target_sigma = float(np.nanmedian(crlb_map[flagged_mask])) * 0.5
            
            plan = plan_reacquisition(
                flagged_mask=flagged_mask,
                current_settings=base_settings,
                fitted_params=fitted_params,
                axis=axis_w,
                target_sigma=target_sigma,
                read_noise_e=s.meta.get("read_noise_e", 4.0)
            )
            
            if plan.action:
                planner_action = plan.action.name
                hybrid_cost += plan.cost_seconds
        
        # Slow scan cost (reference: 0.4s integration)
        slow_scan_cost = total_pixels * (0.4 + 0.05 + 0.05)
        
        response = {
            "summary": {
                "pass_rate": summary.get("pass_rate", 0.0),
                "n_pass": summary.get("n_pass", 0),
                "n_feature_erasure": summary.get("n_feature_erasure", 0),
                "n_hallucination": summary.get("n_hallucination", 0),
                "n_exploit": summary.get("n_exploit", 0),
                "ood_rationale": summary.get("ood_rationale", None),
                "planner_action": planner_action,
                "cost_fast_scan_s": base_scan_cost,
                "cost_hybrid_s": hybrid_cost,
                "cost_slow_scan_s": slow_scan_cost,
                "speedup_factor": slow_scan_cost / max(0.1, hybrid_cost)
            },
            "risk_map": _make_serializable(risk_class),
            "ood_score_map": _make_serializable(ood_score_map),
            "crlb_map": _make_serializable(crlb_map),
            "shape": [H, W]
        }
        
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files for the operator console UI
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
