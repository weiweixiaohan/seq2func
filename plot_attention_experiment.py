"""Experiment 2: Attention mechanism comparison
seq1000_original vs seq1000_attention
- 3 ROC plots (DNase-seq, TF, Histone) + 1 loss curve plot
"""
import os, re, numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SHARED = '/root/shared-nvme'
EVAL_ROOT = os.path.join(SHARED, 'eval_results')
OUT_DIR = os.path.join(EVAL_ROOT, 'attention_experiment')
os.makedirs(OUT_DIR, exist_ok=True)
STEP = 990000
SPLIT = 'val'

MODELS = {
    'seq1000_original':  {'label': 'seq1000 (original)',  'color': '#FF9800', 'ls': '-'},
    'seq1000_attention': {'label': 'seq1000 (attention)', 'color': '#4CAF50', 'ls': '--'},
}

GROUPS = {
    'DNase-seq': (0, 125),
    'TF':       (125, 815),
    'Histone':  (815, 919),
}

# Part A: ROC 
print("=" * 60)
print("Part A: ROC curves")
labels = np.load(os.path.join(EVAL_ROOT, 'seq1000_original', SPLIT, 'labels.npy'))
print(f"Labels: {labels.shape}")

preds = {}
for name, info in MODELS.items():
    p = os.path.join(EVAL_ROOT, name, SPLIT, f'preds-{STEP}.npy')
    arr = np.load(p)
    assert arr.shape == labels.shape, f"{name}: {arr.shape} vs {labels.shape}"
    preds[name] = arr
    print(f"Loaded {name}: {arr.shape}")

for group_name, (start, end) in GROUPS.items():
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, info in MODELS.items():
        y_true = labels[:, start:end].ravel()
        y_score = preds[name][:, start:end].ravel()
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=info['color'], linestyle=info['ls'], linewidth=2.5,
                label=f"{info['label']} (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], 'k:', alpha=0.4, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC — {group_name} (Validation)  |  Attention comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    pth = os.path.join(OUT_DIR, f'roc_{group_name}.png')
    fig.savefig(pth, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {pth}")

# Part B: Loss
print("\n" + "=" * 60)
print("Part B: Loss curves")

MODEL_LOGS = {
    'seq1000_original': [
        f'{SHARED}/train_seq1000_original_resume.log',
        f'{SHARED}/train_seq1000_original_epoch3.log',
        f'{SHARED}/train_seq1000_original_20260606_163521.log',
    ],
    'seq1000_attention': [f'{SHARED}/train_seq1000_attention.log'],
}

STEP_LOSS_RE = re.compile(r'step\s+(\d+)\s+loss=([\d.eE+-]+)')

def parse_logs(log_paths):
    steps, losses = [], []
    for fp in log_paths:
        if not os.path.exists(fp): continue
        with open(fp) as f:
            for line in f:
                m = STEP_LOSS_RE.search(line)
                if m:
                    steps.append(int(m.group(1)))
                    losses.append(float(m.group(2)))
    steps, losses = np.array(steps), np.array(losses)
    if len(steps) == 0: return steps, losses
    order = np.argsort(steps)
    steps, losses = steps[order], losses[order]
    _, uniq = np.unique(steps, return_index=True)
    return steps[uniq], losses[uniq]

fig, ax = plt.subplots(figsize=(10, 6))
window = 500
for name, info in MODELS.items():
    s, l = parse_logs(MODEL_LOGS[name])
    if len(s) == 0: continue
    smoothed = np.array([np.median(l[max(0,i-window//2):min(len(l),i+window//2)]) for i in range(len(l))])
    ax.plot(s, smoothed, color=info['color'], linewidth=2, label=info['label'])
    print(f"  {info['label']}: {len(s)} points, steps {s[0]}-{s[-1]}")
ax.set_xlabel('Step', fontsize=13)
ax.set_ylabel('Loss (smoothed)', fontsize=13)
ax.set_title('Training Loss — Attention mechanism comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
pth = os.path.join(OUT_DIR, 'loss_curves.png')
fig.savefig(pth, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved → {pth}")
print("\nAll plots saved to:", OUT_DIR)
