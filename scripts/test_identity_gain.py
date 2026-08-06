import sys; sys.path.insert(0, 'src')
from optiguard.data.simulator import MapSimulator
from optiguard.eval.harness import effective_exposure_gain
from optiguard.eval.baselines import BASELINES

print("Generating 6 test samples...", flush=True)
sim = MapSimulator.from_yaml("configs/simulator.yaml")
# test samples according to generate_baseline_table.py
samples = [sim.generate(index=i) for i in range(6, 12)]

print("Evaluating effective_exposure_gain for BASELINES['raw'] (Identity mapping)...", flush=True)
method = BASELINES["raw"]
gain = effective_exposure_gain(method, samples, exposure=0.1)
print(f"Gain of raw baseline (identity mapping): {gain}")
