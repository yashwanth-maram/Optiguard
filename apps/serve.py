"""
apps/serve.py — OptiGuard AI Operator Console Service (Step 14)

Usage:
    python apps/serve.py                          # live inference at localhost:8000
    python apps/serve.py --replay evidence/demo_case.json  # replay pre-computed result

Endpoints:
    POST /analyse       cube + acquisition metadata  → full result
    GET  /result/{id}   cached result
    GET  /report/{id}   JSON audit record
    GET  /health        model version, config hash, calibration constants

All inference quantities carry  "simulated": true  in the audit record.
The console is served at / from apps/static/.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path plumbing
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from optiguard.data.simulator import MapSimulator
from optiguard.assurance.gate import evaluate_gate
from optiguard.planning.planner import plan_reacquisition, AcquisitionSettings

# ---------------------------------------------------------------------------
# Git / config fingerprinting (best-effort — never crashes the service)
# ---------------------------------------------------------------------------
def _git_commit() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"

def _file_hash(path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()[:16]
    except Exception:
        return "unknown"

def _data_hash(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr)).hexdigest()[:16]

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
SIMULATOR_PATH = os.path.join(ROOT, "configs", "simulator.yaml")
EVIDENCE_DIR   = os.path.join(ROOT, "evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

sim = MapSimulator.from_yaml(SIMULATOR_PATH)

RESULT_CACHE: dict[str, dict] = {}   # id → full result payload

# CLI replay file (set by --replay before uvicorn starts)
_REPLAY_FILE: Optional[str] = None
_REPLAY_DATA: Optional[dict] = None

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="OptiGuard AI Operator Console",
    version="0.1.0",
    description="Photon-budget measurement assurance — Step 14 service"
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class AnalyseRequest(BaseModel):
    material:   str
    index:      int   = 6
    exposure_s: float = 0.1

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_serializable(obj):
    import math
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.number):
            return np.where(np.isnan(obj) | np.isinf(obj), None, obj).tolist()
        return obj.tolist()
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
    if isinstance(obj, np.generic):
        return _make_serializable(obj.item())
    return obj

def _run_baseline_spatial_gauss(short_counts: np.ndarray, axis: np.ndarray,
                                w_start: int, w_end: int) -> dict:
    """Spatial Gaussian denoiser baseline (Step 7)."""
    from scipy.ndimage import gaussian_filter
    from optiguard.estimation.fit import fit_lorentzian_map
    from optiguard.physics.crlb import crlb_plugin_map

    sg = gaussian_filter(short_counts.astype(np.float64), sigma=(1.0, 1.0, 0))
    theta_sg = fit_lorentzian_map(axis, sg, nominal_center_cm1=520.7)
    theta_raw = fit_lorentzian_map(axis, short_counts.astype(np.float64),
                                   nominal_center_cm1=520.7)

    axis_w = axis[w_start:w_end]
    sg_w   = sg[:, :, w_start:w_end]

    crlb_raw = crlb_plugin_map(axis=axis_w, counts_window=sg_w,
                               fitted_center=theta_sg["center"], read_noise_e=4.0)
    rmse = float(np.sqrt(np.mean(
        (theta_sg["center"] - theta_raw["center"])**2
    )))
    return {
        "rmse_center_cm1": rmse,
        "sg_center_map":   theta_sg["center"].tolist(),
        "sg_crlb_map":     crlb_raw.tolist(),
    }

def _build_result(s, short_counts: np.ndarray, exposure: float,
                  material: str) -> dict:
    """Core analysis pipeline. Returns the full serialisable payload."""
    t0 = time.perf_counter()

    axis = s.axis
    H, W = short_counts.shape[:2]
    peak_idx = int(np.argmin(np.abs(axis - 520.7)))
    w_start = max(0, peak_idx - 64)
    w_end   = min(len(axis), w_start + 128)

    neff_map = np.ones((H, W), dtype=np.float64)
    gate_res = evaluate_gate(
        s,
        short_counts.astype(np.float64),
        neff_map,
        read_noise_e=s.meta.get("read_noise_e", 4.0),
        t1b_threshold_ratio=3.0,
        spectral_window=(w_start, w_end)
    )

    risk_class        = gate_res["risk_class"]
    ood_score_map     = gate_res["ood_score_map"]
    crlb_map          = gate_res["crlb_map"]
    chi2_nu_map       = gate_res["chi2_nu_map"]
    sigma_scaled_map  = gate_res["sigma_scaled_map"]
    precision_ratio   = gate_res["precision_ratio"]
    fail_t1a          = gate_res["fail_t1a"]
    fail_t1b          = gate_res["fail_t1b"]
    fail_t4           = gate_res["fail_t4"]
    confidence_score  = gate_res["confidence_score"]
    summary           = gate_res["summary"]
    center_map        = gate_res["center_map"]
    sigma_center_map  = gate_res["sigma_center_map"]

    # Baseline (Spatial Gauss)
    sg_info = _run_baseline_spatial_gauss(short_counts, axis, w_start, w_end)

    # Planner
    flagged_mask = risk_class == "FEATURE_ERASURE"
    n_flagged    = int(flagged_mask.sum())
    total_pixels = H * W
    planner_action = None
    plan_cost_s    = 0.0

    if n_flagged > 0:
        from optiguard.estimation.fit import fit_lorentzian_map
        det = sim.config["detector"]
        base_settings = AcquisitionSettings(
            integration_s=exposure,
            accumulations=1,
            laser_power_mw=5.0,
            spectral_binning=1,
            dispersion_cm1_per_px=det["dispersion_cm1_per_px"],
            readout_s=0.05,
            overhead_s_per_point=0.05,
            step_size_um=1.0,
        )
        theta_raw = fit_lorentzian_map(axis, short_counts.astype(np.float64),
                                       read_noise_e=s.meta.get("read_noise_e", 4.0),
                                       nominal_center_cm1=520.7)
        fitted_params = {
            "center":     theta_raw["center"],
            "fwhm":       theta_raw["fwhm"],
            "amplitude":  theta_raw["amplitude"] / exposure,
            "background": theta_raw["background"] / exposure,
        }
        target_sigma = float(np.nanmedian(crlb_map[flagged_mask])) * 0.5
        axis_w = axis[w_start:w_end]

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
            plan_cost_s    = float(plan.cost_seconds)

    latency_s = time.perf_counter() - t0

    fast_scan_cost = total_pixels * (exposure + 0.05 + 0.05)
    slow_scan_cost = total_pixels * (0.4 + 0.05 + 0.05)
    hybrid_cost    = fast_scan_cost + plan_cost_s

    return {
        "meta": {
            "material":   material,
            "index":      s.meta.get("shape"),
            "exposure_s": exposure,
            "shape":      [H, W],
            "simulated":  True,
        },
        "summary": {
            "pass_rate":          summary.get("pass_rate", 0.0),
            "n_pass":             summary.get("n_pass", 0),
            "n_feature_erasure":  summary.get("n_feature_erasure", 0),
            "n_hallucination":    summary.get("n_hallucination", 0),
            "n_exploit":          summary.get("n_exploit", 0),
            "n_ood":              summary.get("n_ood", 0),
            "mean_confidence":    summary.get("mean_confidence", 0.0),
            "mean_ood_score":     summary.get("mean_ood_score", 0.0),
            "ood_rationale":      summary.get("ood_rationale"),
            "planner_action":     planner_action,
            "n_flagged_points":   n_flagged,
            "cost_fast_scan_s":   fast_scan_cost,
            "cost_hybrid_s":      hybrid_cost,
            "cost_slow_scan_s":   slow_scan_cost,
            "speedup_factor":     slow_scan_cost / max(0.1, hybrid_cost),
            "latency_s":          latency_s,
        },
        "sg_baseline": sg_info,
        "maps": {
            "risk_class":       _make_serializable(risk_class),
            "ood_score":        _make_serializable(ood_score_map),
            "crlb":             _make_serializable(crlb_map),
            "chi2_nu":          _make_serializable(chi2_nu_map),
            "sigma_scaled":     _make_serializable(sigma_scaled_map),
            "precision_ratio":  _make_serializable(precision_ratio),
            "confidence":       _make_serializable(confidence_score),
            "fail_t1a":         _make_serializable(fail_t1a.astype(int)),
            "fail_t1b":         _make_serializable(fail_t1b.astype(int)),
            "fail_t4":          _make_serializable(fail_t4.astype(int)),
            "center":           _make_serializable(center_map),
            "sigma_center":     _make_serializable(sigma_center_map),
        }
    }

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status":        "ok",
        "model_version": "step8_checkpoint",
        "config_hash":   _file_hash(SIMULATOR_PATH),
        "git_commit":    _git_commit(),
        "calibration":   sim.config.get("detector", {}),
        "replay_mode":   _REPLAY_DATA is not None,
    }

@app.post("/analyse")
def analyse(req: AnalyseRequest):
    if _REPLAY_DATA is not None:
        # Replay: serve pre-computed result, no inference
        result_id = "replay-" + str(uuid.uuid4())[:8]
        RESULT_CACHE[result_id] = _REPLAY_DATA
        return {"id": result_id, **_REPLAY_DATA}

    if req.material not in sim.config["materials"]:
        raise HTTPException(400, f"Unknown material: {req.material}")
    try:
        s = sim.generate(index=req.index, material=req.material)
        avail = list(s.short_counts.keys())
        exposure = req.exposure_s if req.exposure_s in s.short_counts else avail[0]
        short_counts = s.short_counts[exposure]

        result = _build_result(s, short_counts, exposure, req.material)
        result = _make_serializable(result)

        result_id = str(uuid.uuid4())[:8]
        RESULT_CACHE[result_id] = result

        # Audit record
        audit = {
            "result_id":        result_id,
            "git_commit":       _git_commit(),
            "config_hash":      _file_hash(SIMULATOR_PATH),
            "material":         req.material,
            "exposure_s":       exposure,
            "input_hash":       _data_hash(short_counts),
            "decision":         "WITHHELD" if result["summary"]["n_exploit"] > 0 or result["summary"].get("n_ood", 0) > 0
                                else "REVIEW" if result["summary"]["n_feature_erasure"] > 0
                                else "CERTIFIED",
            "runtime_s":        result["summary"]["latency_s"],
            "simulated":        True,
            "thresholds": {
                "t1b_threshold_ratio": 3.0,
                "chi2nu_threshold":    0.5,
                "target_fpr":          0.05,
            },
        }
        audit_path = os.path.join(EVIDENCE_DIR, f"audit_{result_id}.json")
        with open(audit_path, "w") as f:
            json.dump(audit, f, indent=2)

        return {"id": result_id, **result}

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))

@app.get("/result/{result_id}")
def get_result(result_id: str):
    if result_id not in RESULT_CACHE:
        raise HTTPException(404, "Result not found")
    return RESULT_CACHE[result_id]

@app.get("/report/{result_id}")
def get_report(result_id: str):
    audit_path = os.path.join(EVIDENCE_DIR, f"audit_{result_id}.json")
    if not os.path.exists(audit_path):
        raise HTTPException(404, "Audit record not found")
    with open(audit_path) as f:
        return json.load(f)

# Mount static console
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------
def _benchmark_latency():
    """Measure and save CPU latency for 64×64 and 100×100 cubes."""
    results = {}
    for size, mat in [(64, "silicon"), (100, "silicon")]:
        s = sim.generate(index=0, material=mat)
        sc = list(s.short_counts.values())[0]
        # Crop or pad to target size
        H, W, C = sc.shape
        target_counts = sc[:min(H, size), :min(W, size), :]
        # Fake a sample object with matching shape
        import types
        fake = types.SimpleNamespace(
            axis=s.axis,
            short_counts={0.1: target_counts},
            meta=s.meta,
        )
        t0 = time.perf_counter()
        _build_result(fake, target_counts, 0.1, mat)
        elapsed = time.perf_counter() - t0
        results[f"{min(H,size)}x{min(W,size)}"] = {"latency_s": round(elapsed, 3)}
        print(f"  Latency {min(H,size)}×{min(W,size)}: {elapsed:.2f} s")

    lat_path = os.path.join(EVIDENCE_DIR, "latency.json")
    with open(lat_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Latency results saved to {lat_path}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", metavar="FILE",
                        help="Load pre-computed result JSON and serve without inference")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--benchmark", action="store_true",
                        help="Measure and save CPU latency, then exit")
    args = parser.parse_args()

    if args.benchmark:
        print("Running latency benchmark...")
        _benchmark_latency()
        sys.exit(0)

    if args.replay:
        replay_path = os.path.abspath(args.replay)
        if not os.path.exists(replay_path):
            print(f"ERROR: replay file not found: {replay_path}")
            sys.exit(1)
        with open(replay_path) as f:
            _REPLAY_DATA = _make_serializable(json.load(f))
        print(f"REPLAY MODE — serving pre-computed result from: {replay_path}")

    print(f"Starting OptiGuard console at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
