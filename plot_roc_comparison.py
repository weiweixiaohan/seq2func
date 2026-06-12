#!/usr/bin/env python
"""Compare ROC curves of three models (seq200_original, seq1000_original,
seq1000_attention) on the validation set across three task groups:
  - DNase I-hypersensitive sites  (feature indices 0:125)
  - Transcription factors         (feature indices 125:815)
  - Histone marks                 (feature indices 815:919)

One figure per group; three model curves overlaid.
"""
import os
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Config

EVAL_ROOT = '/root/shared-nvme/eval_results'
STEP = 990000
SPLIT = 'val'

MODELS = {
    'seq200_original':     dict(color='#1f77b4', ls='-',  lw=2.0),
    'seq1000_original':    dict(color='#ff7f0e', ls='--', lw=2.0),
    'seq1000_attention':   dict(color='#2ca02c', ls='-.', lw=2.0),
}

GROUPS = {
    'DNase-seq':   (0, 125),
    'TF':          (125, 815),
    'Histone':     (815, 919),
}

OUT_DIR = os.path.join(EVAL_ROOT, 'comparison_roc')
os.makedirs(OUT_DIR, exist_ok=True)


# Load labels (same for all models on the same split)
labels_path = os.path.join(EVAL_ROOT, 'seq200_original', SPLIT, 'labels.npy')
labels = np.load(labels_path)   # shape (N, 919)
print(f'Labels: {labels.shape}')

# Load predictions for each model
preds = {}
for name in MODELS:
    p = os.path.join(EVAL_ROOT, name, SPLIT, f'preds-{STEP}.npy')
    arr = np.load(p)
    assert arr.shape == labels.shape, f'{name}: {arr.shape} vs {labels.shape}'
    preds[name] = arr
    print(f'Loaded {name}: {arr.shape}')

# Plot per group
for group_name, (start, end) in GROUPS.items():
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name, style in MODELS.items():
        y_true = labels[:, start:end].ravel()
        y_score = preds[model_name][:, start:end].ravel()

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr,
                color=style['color'],
                linestyle=style['ls'],
                linewidth=style['lw'],
                label=f'{model_name} (AUC={roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k:', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC — {group_name} (Validation)', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(OUT_DIR, f'roc_{group_name}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out_path}')

print('\nDone. All three comparison ROC plots are in', OUT_DIR)
