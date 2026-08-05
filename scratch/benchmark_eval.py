import time
import pytest
from optiguard.data.simulator import MapSimulator
from optiguard.eval.baselines import BASELINES
from optiguard.eval.harness import evaluate

def run():
    print("Generating samples...")
    t0 = time.time()
    sim = MapSimulator.from_yaml("configs/simulator.yaml")
    samples = [sim.generate(index=i) for i in range(6)]
    print(f"Generated samples in {time.time()-t0:.2f}s")
    
    print("Running evaluate...")
    t1 = time.time()
    r = evaluate(BASELINES["savgol"], samples, exposure=0.1)
    print(f"Evaluate in {time.time()-t1:.2f}s")

if __name__ == "__main__":
    run()
