import json
import matplotlib.pyplot as plt
import numpy as np

# Load metrics
with open('evidence/step8_metrics.jsonl', 'r') as f:
    lines = [json.loads(line) for line in f if line.strip()]

# Rows 4-9 are indices 3-8
network_runs = lines[3:9]

# Load baselines
with open('evidence/baselines.json', 'r') as f:
    baselines = json.load(f)

# Plot 1: Gain vs Recall@1.5
plt.figure(figsize=(8, 6))

# Plot baselines
for name, marker, color in [('raw', 'o', 'black'), ('binning', 's', 'blue'), ('spatial_gauss', '^', 'green')]:
    b = baselines[name]
    gain = b['effective_exposure_gain']
    recall = b['recall_by_difficulty']['1.5']
    plt.scatter(recall, gain, label=name, marker=marker, color=color, s=100)

# Plot network trajectory
net_gains = [run['effective_exposure_gain'] for run in network_runs]
net_recalls = [run['recall_1p5'] for run in network_runs]
epochs = [run['epoch'] for run in network_runs]

plt.plot(net_recalls, net_gains, 'o-', color='red', label='network (epochs 5-30)', markersize=8)

for i, txt in enumerate(epochs):
    plt.annotate(f'e{txt}', (net_recalls[i], net_gains[i]), textcoords="offset points", xytext=(5,5), ha='left')

plt.axvline(x=baselines['raw']['recall_by_difficulty']['1.5'], color='gray', linestyle='--', label='Raw Recall Floor')

plt.xlabel('Recall at 1.5 CRLB')
plt.ylabel('Effective Exposure Gain (x)')
plt.title('Gain vs Recall Frontier')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('evidence/frontier_gain_vs_recall.png')
plt.close()

# Plot 2: Recall vs Difficulty
plt.figure(figsize=(8, 6))

difficulties = ['0.5', '1.5', '2.5', '4.0', '10.0']
x_vals = [float(d) for d in difficulties]

for name, color in [('raw', 'black'), ('spatial_gauss', 'green')]:
    b = baselines[name]
    recalls = [b['recall_by_difficulty'][d] for d in difficulties]
    plt.plot(x_vals, recalls, 'o-', color=color, label=name)

# Network (last epoch)
last_run = network_runs[-1]
net_recalls_diff = [last_run['recalls'][d] for d in difficulties]
plt.plot(x_vals, net_recalls_diff, 'o-', color='red', label=f'network (epoch 30)')

plt.xlabel('Difficulty (CRLB at 0.1s)')
plt.ylabel('Recall')
plt.title('Recall vs Peak Difficulty')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('evidence/frontier_recall_vs_difficulty.png')
plt.close()
