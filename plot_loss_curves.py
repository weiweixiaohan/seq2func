"""Extract step/loss from training logs and plot loss curves."""
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

SHARED = '/root/shared-nvme'
OUT_DIR = os.path.join(SHARED, 'eval_results', 'loss_curves')
os.makedirs(OUT_DIR, exist_ok=True)

#  Config 
MODELS = {
    'seq200_original': {
        'label': 'seq200 (original)',
        'logs': [f'{SHARED}/train_seq200.log'],
        'color': '#2196F3',
    },
    'seq1000_original': {
        'label': 'seq1000 (original)',
        'logs': [
            f'{SHARED}/train_seq1000_original_resume.log',
            f'{SHARED}/train_seq1000_original_epoch3.log',
            f'{SHARED}/train_seq1000_original_20260606_163521.log',
        ],
        'color': '#FF9800',
    },
    'seq1000_attention': {
        'label': 'seq1000 (attention)',
        'logs': [f'{SHARED}/train_seq1000_attention.log'],
        'color': '#4CAF50',
    },
}

# Regex: "step   469840  loss=0.102" 
STEP_LOSS_RE = re.compile(r'step\s+(\d+)\s+loss=([\d.eE+-]+)')

#  Parse 
def parse_log(filepath):
    steps, losses = [], []
    if not os.path.exists(filepath):
        return np.array([]), np.array([])
    with open(filepath, 'r') as f:
        for line in f:
            m = STEP_LOSS_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return np.array(steps), np.array(losses)

# ── Main 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for model_key, model_info in MODELS.items():
    all_steps = []
    all_losses = []

    for log_path in model_info['logs']:
        s, l = parse_log(log_path)
        if len(s) > 0:
            all_steps.append(s)
            all_losses.append(l)
            print(f"  {os.path.basename(log_path)}: {len(s)} points, steps {s[0]}-{s[-1]}")

    if not all_steps:
        print(f"  {model_key}: no data found!")
        continue

    steps = np.concatenate(all_steps)
    losses = np.concatenate(all_losses)
    # sort by step (in case logs overlap)
    order = np.argsort(steps)
    steps = steps[order]
    losses = losses[order]
    # deduplicate
    _, uniq = np.unique(steps, return_index=True)
    steps = steps[uniq]
    losses = losses[uniq]

    print(f"  {model_key}: total {len(steps)} unique points, steps {steps[0]}-{steps[-1]}")

    # full curve
    ax1.plot(steps, losses, color=model_info['color'], alpha=0.6, linewidth=0.5,
             label=model_info['label'])
    # smoothed (rolling median every 500 points)
    window = 500
    if len(steps) > window:
        smoothed = np.array([np.median(losses[max(0, i-window//2):min(len(losses), i+window//2)])
                             for i in range(len(losses))])
        ax1.plot(steps, smoothed, color=model_info['color'], linewidth=2, alpha=0.9)
        ax2.plot(steps, smoothed, color=model_info['color'], linewidth=2,
                 label=model_info['label'])

ax1.set_xlabel('Step', fontsize=13)
ax1.set_ylabel('Loss', fontsize=13)
ax1.set_title('Training Loss (raw)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Step', fontsize=13)
ax2.set_ylabel('Loss', fontsize=13)
ax2.set_title('Training Loss (smoothed, window=500)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
out_path = os.path.join(OUT_DIR, 'training_loss_curves.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved → {out_path}")
