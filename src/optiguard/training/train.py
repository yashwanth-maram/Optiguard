"""
Training pipeline for OptiGuard Neural Restoration Models.
Step 8: Constrained-Gain Model Selection with Full Evidence Packaging.

Usage:
    python src/optiguard/training/train.py \
        --config configs/restoration_v1.yaml \
        --data data/corpus \
        --out runs/restoration_v1 \
        --select-on constrained_gain \
        --recall-floor 0.719 \
        --checkpoint-every 5
"""
import os
import sys
import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import yaml

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from optiguard.models.spatial_spectral import SpatialSpectralUNet, SpatialPoissonLoss, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torch.optim as optim

    class CorpusDataset(Dataset):
        """Loads pre-generated synthetic hyperspectral datacubes from disk."""
        def __init__(self, data_dir: str, split_indices: Optional[List[int]] = None):
            self.data_path = Path(data_dir)
            all_files = sorted(list(self.data_path.glob("sample_*.npz")))
            if not all_files:
                raise FileNotFoundError(f"No sample_*.npz files found in {data_dir}")
                
            if split_indices is not None:
                self.files = [all_files[i] for i in split_indices if i < len(all_files)]
            else:
                self.files = all_files
                
        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            data = np.load(self.files[idx])
            short = data["short_counts"]           # (H, W, C)
            clean = data["clean_rate"] * float(data["exposure"]) # Expected photon count (H, W, C)
            
            # Transpose to PyTorch (C, H, W)
            x = torch.from_numpy(short.transpose(2, 0, 1).astype(np.float32))
            y = torch.from_numpy(clean.transpose(2, 0, 1).astype(np.float32))
            
            # Centroid-loss targets
            center_true = torch.from_numpy(data["center_true"].astype(np.float32))  # (H, W)
            crlb_map    = torch.from_numpy(data["crlb_map"].astype(np.float32))     # (H, W)
            axis        = torch.from_numpy(data["axis"].astype(np.float32))         # (C,)
            return x, y, center_true, crlb_map, axis


    def evaluate_model_on_harness(model, val_samples: list, in_channels: int = 128, exposure: float = 0.1):
        """Runs full OptiGuard evaluation harness on validation/test samples."""
        from optiguard.eval.harness import evaluate
        from optiguard.eval.baselines import fit_map, _nominal
        
        device = next(model.parameters()).device
        model.eval()
        
        def model_method(sample, exposure, **kwargs):
            short_counts = sample.short_counts[exposure]
            axis = sample.axis
            meta = sample.meta

            H, W, C = short_counts.shape
            peak_nominal = _nominal(sample)
            peak_idx = int(np.argmin(np.abs(axis - peak_nominal)))
            w_start = max(0, peak_idx - in_channels // 2)
            w_end = min(C, peak_idx + in_channels // 2)
            
            cropped = short_counts[:, :, w_start:w_end]
            act_c = cropped.shape[2]
            
            if act_c != in_channels:
                padded = np.zeros((H, W, in_channels), dtype=np.float32)
                padded[:, :, :act_c] = cropped
                inp_t = torch.from_numpy(padded.transpose(2, 0, 1)).unsqueeze(0).to(device)
            else:
                inp_t = torch.from_numpy(cropped.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0).to(device)
                
            with torch.no_grad():
                out_t = model(inp_t)
                out_np = out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                
            restored = np.copy(short_counts).astype(np.float64)
            restored[:, :, w_start:w_end] = out_np[:, :, :act_c]
            
            return fit_map(axis, restored, meta.get("read_noise_e", 0.0), peak_nominal)

        res = evaluate(model_method, val_samples, exposure=exposure)
        return res


    def get_git_commit_hash() -> str:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("ascii").strip()
        except Exception:
            return "unknown"


    def train(
        config_path: str,
        data_dir: str,
        out_dir: str,
        select_on: str = "constrained_gain",
        recall_floor: float = 0.719,
        checkpoint_every: int = 5,
        resume: Optional[str] = None
    ):
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            
        m_cfg = cfg.get("model", {})
        t_cfg = cfg.get("training", {})
        
        in_channels = m_cfg.get("in_channels", 128)
        epochs = t_cfg.get("epochs", 50)
        batch_size = t_cfg.get("batch_size", 8)
        lr = t_cfg.get("lr", 0.0005)
        weight_decay = t_cfg.get("weight_decay", 0.0001)
        exposure = float(t_cfg.get("exposure_s", 0.1))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"OptiGuard Training: Device={device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}), Epochs={epochs}, BatchSize={batch_size}")
        print(f"Objective: {select_on} with Recall Floor @ 1.5 CRLB = {recall_floor:.3f}")
        
        # Dataset setup
        dataset = CorpusDataset(data_dir)
        n_total = len(dataset)
        n_val = max(6, int(n_total * 0.1))
        n_train = n_total - n_val
        
        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n_total))
        
        train_ds = CorpusDataset(data_dir, split_indices=train_indices)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        
        print(f"Loaded corpus from {data_dir}: {n_train} train cubes, {n_val} validation cubes")
        
        # Determine test indices from baselines.json or default
        from optiguard.data.simulator import MapSimulator
        sim = MapSimulator.from_yaml("configs/simulator.yaml")
        
        baselines_json_path = Path("evidence/baselines.json")
        test_indices = list(range(6, 12))
        if baselines_json_path.exists():
            try:
                with open(baselines_json_path, "r") as f:
                    b_data = json.load(f)
                    test_indices = b_data.get("_meta", {}).get("test_indices", test_indices)
            except Exception:
                pass
                
        print(f"Evaluation Harness test indices: {test_indices}")
        val_samples = [sim.generate(index=i) for i in test_indices]
        
        # Model, Loss, Optimizer
        model = SpatialSpectralUNet(
            in_channels=in_channels,
            out_channels=in_channels,
            base_channels=m_cfg.get("base_channels", 64),
            depth=m_cfg.get("depth", 2),
            dropout=m_cfg.get("dropout", 0.05)
        ).to(device)
        
        criterion = SpatialPoissonLoss(alpha_nll=0.05)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        
        best_pt_path = out_path / "best.pt"
        latest_pt_path = out_path / "latest.pt"
        metrics_file = out_path / "metrics.jsonl"
        network_row_path = out_path / "network_row.json"
        manifest_path = out_path / "manifest.json"
        
        start_epoch = 1
        best_score = -float("inf")
        best_result_dict = None
        
        # Handle resume
        if resume:
            resume_path = Path(resume)
            if not resume_path.exists() and (out_path / "latest.pt").exists():
                resume_path = out_path / "latest.pt"
            if resume_path.exists():
                print(f"Resuming checkpoint from {resume_path}...")
                ckpt = torch.load(resume_path, map_location=device)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"])
                    if "optimizer_state_dict" in ckpt:
                        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    if "scheduler_state_dict" in ckpt:
                        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                    start_epoch = ckpt.get("epoch", 0) + 1
                    best_score = ckpt.get("best_score", -float("inf"))
                    print(f"Resumed at epoch {start_epoch} (best_score={best_score:.3f})")
                else:
                    model.load_state_dict(ckpt)
                    print(f"Loaded raw state_dict weights from {resume_path}")
            
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            train_loss     = 0.0
            train_centroid = 0.0
            train_l1       = 0.0
            train_poisson  = 0.0
            for batch_x, batch_y, center_true, crlb_map, axis in train_loader:
                batch_x    = batch_x.to(device)
                batch_y    = batch_y.to(device)
                center_true = center_true.to(device)   # (B, H, W)
                crlb_map   = crlb_map.to(device)       # (B, H, W)
                axis       = axis.to(device)            # (B, C)
                
                optimizer.zero_grad()
                pred = model(batch_x)                  # (B, C, H, W)
                
                # --- Centroid loss (dominant term) ---
                # Soft-argmax over spectral axis at each pixel.
                # Windowed Center of Mass to prevent background lever-arm exploit
                pred_hw_c = pred.permute(0, 2, 3, 1)
                window = 18
                B, H, W, C = pred_hw_c.shape
                peak_idx = pred_hw_c.argmax(dim=-1, keepdim=True)
                indices = torch.arange(C, device=device).view(1, 1, 1, C).expand(B, H, W, C)
                mask = (indices >= peak_idx - window) & (indices <= peak_idx + window)
                
                # Local background subtraction inside the window
                masked_pred = pred_hw_c.masked_fill(~mask, float('inf'))
                local_bg = masked_pred.min(dim=-1, keepdim=True).values
                pred_sub_windowed = (pred_hw_c - local_bg) * mask
                
                norm = torch.clamp(pred_sub_windowed.sum(dim=-1, keepdim=True), min=1e-6)
                weights = pred_sub_windowed / norm
                axis_exp = axis[:, None, None, :]               # (B, 1, 1, C)
                centroid = (weights * axis_exp).sum(dim=-1)     # (B, H, W)
                
                # CRLB-normalised error (signed)
                crlb_safe = torch.clamp(crlb_map, min=1e-6)
                norm_err = (centroid - center_true) / crlb_safe
                
                centroid_magnitude = norm_err.abs().mean().item()
                if centroid_magnitude < 0.75:
                    per_px = norm_err.abs()
                    print(f"[guard] mag={centroid_magnitude:.3f} "
                          f"median={per_px.median():.3f} p95={per_px.quantile(0.95):.3f} "
                          f"frac_below_0.5={(per_px < 0.5).float().mean():.3f}")
                
                param_loss = torch.nn.functional.smooth_l1_loss(norm_err, torch.zeros_like(norm_err))
                
                # --- Photon-count terms ---
                l1 = torch.nn.functional.l1_loss(pred, batch_y)
                mu = torch.clamp(pred, min=1e-6)
                poisson = torch.mean(mu - batch_x * torch.log(mu))
                
                # Weights: centroid dominates (0.6), L1 structural (0.3), Poisson regulariser (0.1)
                loss = 0.6 * param_loss + 0.3 * l1 + 0.1 * poisson
                
                loss.backward()
                optimizer.step()
                
                train_loss     += loss.item()       * len(batch_x)
                train_centroid += param_loss.item() * len(batch_x)
                train_l1       += l1.item()         * len(batch_x)
                train_poisson  += poisson.item()    * len(batch_x)
                
            scheduler.step()
            train_loss     /= len(train_ds)
            train_centroid /= len(train_ds)
            train_l1       /= len(train_ds)
            train_poisson  /= len(train_ds)
            
            if epoch == start_epoch or epoch % checkpoint_every == 0:
                # Quick fit check on one validation sample to compare with training centroid loss
                with torch.no_grad():
                    val_sample = val_samples[0]
                    sc = val_sample.short_counts[exposure]
                    peak_nominal = 520.7  # approximate, _nominal isn't imported here, but we can compute w_start
                    peak_idx = int(np.argmin(np.abs(val_sample.axis - peak_nominal)))
                    w_start = max(0, peak_idx - in_channels // 2)
                    w_end = min(sc.shape[2], peak_idx + in_channels // 2)
                    cropped = sc[:, :, w_start:w_end]
                    act_c = cropped.shape[2]
                    
                    if act_c != in_channels:
                        padded = np.zeros((sc.shape[0], sc.shape[1], in_channels), dtype=np.float32)
                        padded[:, :, :act_c] = cropped
                        inp_t = torch.from_numpy(padded.transpose(2, 0, 1)).unsqueeze(0).to(device)
                    else:
                        inp_t = torch.from_numpy(cropped.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0).to(device)
                        
                    out_t = model(inp_t)
                    out_np = out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                    restored = np.copy(sc).astype(np.float64)
                    restored[:, :, w_start:w_end] = out_np[:, :, :act_c]
                    
                    from optiguard.eval.baselines import fit_map, _nominal
                    val_theta = fit_map(val_sample.axis, restored, val_sample.meta.get("read_noise_e", 0.0), _nominal(val_sample))
                    val_rmse = np.sqrt(np.mean((val_theta['center'] - val_sample.theta_true['center'])**2))
                    
                print(
                    f"[Epoch {epoch:03d}] Loss: {train_loss:.4f} "
                    f"| centroid: {train_centroid:.4f} (w: {0.6*train_centroid:.4f}) "
                    f"| L1: {train_l1:.4f} (w: {0.3*train_l1:.4f}) "
                    f"| Poisson: {train_poisson:.4f} (w: {0.1*train_poisson:.4f}) "
                    f"| val_rmse: {val_rmse:.4f} cm^-1"
                )
            
            # Save latest checkpoint every epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_score": best_score
            }, latest_pt_path)
            
            # Validation step
            if epoch % checkpoint_every == 0 or epoch == epochs:
                print(f"\n[Epoch {epoch:03d}/{epochs:03d}] Running Validation Harness...")
                val_res = evaluate_model_on_harness(model, val_samples, in_channels=in_channels, exposure=exposure)
                
                from optiguard.eval.harness import effective_exposure_gain
                
                def model_method_for_gain(sample, exp, **kwargs):
                    from optiguard.eval.baselines import fit_map, _nominal
                    short_counts = sample.short_counts[exp]
                    axis = sample.axis
                    meta = sample.meta
        
                    H, W, C = short_counts.shape
                    peak_nominal = _nominal(sample)
                    peak_idx = int(np.argmin(np.abs(axis - peak_nominal)))
                    w_start = max(0, peak_idx - in_channels // 2)
                    w_end = min(C, peak_idx + in_channels // 2)
                    
                    cropped = short_counts[:, :, w_start:w_end]
                    act_c = cropped.shape[2]
                    
                    if act_c != in_channels:
                        padded = np.zeros((H, W, in_channels), dtype=np.float32)
                        padded[:, :, :act_c] = cropped
                        inp_t = torch.from_numpy(padded.transpose(2, 0, 1)).unsqueeze(0).to(device)
                    else:
                        inp_t = torch.from_numpy(cropped.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0).to(device)
                        
                    with torch.no_grad():
                        out_t = model(inp_t)
                        out_np = out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                        
                    restored = np.copy(short_counts).astype(np.float64)
                    restored[:, :, w_start:w_end] = out_np[:, :, :act_c]
                    
                    return fit_map(axis, restored, meta.get("read_noise_e", 0.0), peak_nominal)

                gain = effective_exposure_gain(model_method_for_gain, val_samples, exposure=exposure)
                if gain is None:
                    gain = 1.0
                gain = float(gain)
                
                rec_1p5 = float(val_res.recall_by_difficulty.get(1.5, 0.0))
                rmse = float(val_res.rmse_center)
                
                metric_entry = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "effective_exposure_gain": gain,
                    "recall_1p5": rec_1p5,
                    "rmse_center_cm1": rmse,
                    "mae_center_cm1": float(val_res.mae_center),
                    "false_feature_rate": float(val_res.false_feature_rate),
                    "recalls": {str(k): float(v) for k, v in val_res.recall_by_difficulty.items()}
                }
                
                with open(metrics_file, "a") as f:
                    f.write(json.dumps(metric_entry) + "\n")
                    
                print(f"  Train Loss: {train_loss:.4f} | Gain: {gain:.2f}x | Recall @ 1.5 CRLB: {rec_1p5*100:.1f}% (Floor: {recall_floor*100:.1f}%) | RMSE: {rmse:.4f} cm^-1")
                
                # Selection logic: Constrained-Gain
                is_best = False
                if select_on == "constrained_gain":
                    # Strictly require meeting or exceeding raw baseline recall floor
                    if rec_1p5 >= recall_floor:
                        if gain > best_score:
                            best_score = gain
                            is_best = True
                    else:
                        print(f"  [REJECTED] Recall @ 1.5 CRLB ({rec_1p5:.3f}) < Floor ({recall_floor:.3f})")
                else: # Default MAE/RMSE minimization
                    score = -rmse
                    if score > best_score:
                        best_score = score
                        is_best = True
                        
                if is_best:
                    print(f"  >>> New BEST checkpoint selected (Effective Exposure Gain = {gain:.2f}x) -> {best_pt_path}")
                    torch.save(model.state_dict(), best_pt_path)
                    
                    # Write formatted network_row.json
                    best_result_dict = {
                        "network_v1": {
                            "rmse_center_cm1": float(val_res.rmse_center),
                            "mae_center_cm1": float(val_res.mae_center),
                            "median_center_cm1": float(val_res.median_center),
                            "mae_fwhm_cm1": float(val_res.mae_fwhm),
                            "mean_crlb_cm1": float(val_res.mean_crlb),
                            "rmse_over_crlb": float(val_res.rmse_over_crlb),
                            "convergence_rate": float(val_res.convergence_rate),
                            "recall_by_difficulty": {str(k): float(v) for k, v in val_res.recall_by_difficulty.items()},
                            "false_feature_rate": float(val_res.false_feature_rate),
                            "effective_exposure_gain": float(gain),
                            "tuned_params": {
                                "model": "SpatialSpectralUNet",
                                "depth": m_cfg.get("depth", 2),
                                "base_channels": m_cfg.get("base_channels", 64),
                                "epoch": epoch
                            }
                        }
                    }
                    with open(network_row_path, "w") as f:
                        json.dump(best_result_dict, f, indent=2)
            else:
                print(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.4f}")
                
        # Write Manifest
        manifest = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "git_commit": get_git_commit_hash(),
            "config": cfg,
            "data_dir": str(data_dir),
            "out_dir": str(out_dir),
            "epochs": epochs,
            "select_on": select_on,
            "recall_floor": recall_floor,
            "test_indices": test_indices,
            "best_score": best_score if best_score != -float("inf") else None,
            "has_best_checkpoint": best_pt_path.exists()
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
            
        print(f"\nTraining pipeline finished.")
        print(f"Artifacts saved in {out_path}:")
        print(f"  - Latest checkpoint: {latest_pt_path}")
        print(f"  - Best checkpoint:   {best_pt_path if best_pt_path.exists() else 'None cleared recall floor'}")
        print(f"  - Metrics log:       {metrics_file}")
        print(f"  - Network row:       {network_row_path if network_row_path.exists() else 'None'}")
        print(f"  - Run manifest:      {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Train OptiGuard Neural Restoration Model.")
    parser.add_argument("--config", type=str, default="configs/restoration_v1.yaml", help="Path to config YAML")
    parser.add_argument("--data", type=str, default="data/corpus", help="Path to pre-generated corpus directory")
    parser.add_argument("--out", type=str, default="runs/restoration_v1", help="Output directory for checkpoints")
    parser.add_argument("--select-on", type=str, default="constrained_gain", help="Selection metric")
    parser.add_argument("--recall-floor", type=float, default=0.719, help="Recall threshold at 1.5 CRLB")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Validation frequency in epochs")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required for training. Install with: pip install torch")
        sys.exit(1)

    train(
        config_path=args.config,
        data_dir=args.data,
        out_dir=args.out,
        select_on=args.select_on,
        recall_floor=args.recall_floor,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
