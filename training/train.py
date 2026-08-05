"""
Forwarding runner for training/train.py -> src.optiguard.training.train
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from optiguard.training.train import main

if __name__ == "__main__":
    main()
