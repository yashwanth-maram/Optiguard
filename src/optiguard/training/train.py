"""
Step 8: Training pipeline for Deep Spectral Denoising Network on Synthetic Raman Hyperspectral Maps.
Can run locally with CPU/GPU or directly in Google Colab.
"""
import os
import sys
import argparse
from pathlib import Path
import numpy as np

from optiguard.data.simulator import MapSimulator
from optiguard.models.denoiser import SpectralDenoiser, PoissonGaussianLoss, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torch.optim as optim

    class SyntheticSpectralDataset(Dataset):
        """Generates paired (short_counts, clean_rate) spectra on the fly."""
        def __init__(self, simulator: MapSimulator, num_maps: int = 20, exposure: float = 0.1, crop_channels: int = 128):
            self.sim = simulator
            self.exposure = exposure
            self.crop_channels = crop_channels
            
            # Pre-generate maps for fast iteration
            self.inputs = []
            self.targets = []
            
            for i in range(num_maps):
                s = self.sim.generate(index=i)
                counts = s.short_counts[exposure] # (H, W, C)
                clean = s.clean_rate * exposure    # Expected clean counts
                
                # Crop around nominal peak center
                peak_idx = int(np.argmin(np.abs(s.axis - float(s.meta["peak_cm1"]))))
                w_start = max(0, peak_idx - crop_channels // 2)
                w_end = min(counts.shape[2], peak_idx + crop_channels // 2)
                
                counts_c = counts[:, :, w_start:w_end]
                clean_c = clean[:, :, w_start:w_end]
                
                # Reshape to (N, 1, crop_channels)
                H, W, C = counts_c.shape
                self.inputs.append(counts_c.reshape(-1, 1, C).astype(np.float32))
                self.targets.append(clean_c.reshape(-1, 1, C).astype(np.float32))
                
            self.inputs = np.concatenate(self.inputs, axis=0)
            self.targets = np.concatenate(self.targets, axis=0)

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            return torch.from_numpy(self.inputs[idx]), torch.from_numpy(self.targets[idx])


    def train_denoiser(
        config_path: str = "configs/simulator.yaml",
        epochs: int = 10,
        batch_size: int = 128,
        lr: float = 1e-3,
        save_path: str = "runs/spectral_denoiser.pt"
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        sim = MapSimulator.from_yaml(config_path)
        
        print("Synthesizing training and validation datasets...")
        train_dataset = SyntheticSpectralDataset(sim, num_maps=10, exposure=0.1)
        val_dataset = SyntheticSpectralDataset(sim, num_maps=2, exposure=0.1)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model = SpectralDenoiser(in_channels=1, hidden_dim=64, num_blocks=6).to(device)
        criterion = PoissonGaussianLoss(read_noise_e=4.0)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        print(f"Starting training for {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for batch_in, batch_target in train_loader:
                batch_in, batch_target = batch_in.to(device), batch_target.to(device)
                
                optimizer.zero_grad()
                pred_mean, _ = model(batch_in)
                loss = criterion(pred_mean, batch_target, batch_in)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(batch_in)
                
            scheduler.step()
            train_loss /= len(train_dataset)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_in, val_target in val_loader:
                    val_in, val_target = val_in.to(device), val_target.to(device)
                    pred_mean, _ = model(val_in)
                    val_loss += criterion(pred_mean, val_target, val_in).item() * len(val_in)
            val_loss /= len(val_dataset)
            
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
        torch.save(model.state_dict(), save_path)
        print(f"\nModel saved successfully to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-path", type=str, default="runs/spectral_denoiser.pt")
    args = parser.parse_args()

    if TORCH_AVAILABLE:
        train_denoiser(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, save_path=args.save_path)
    else:
        print("PyTorch is required for Step 8 neural network training.")
