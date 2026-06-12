"""
Compute per-feature AUCs from saved predictions.

Usage:
  python compute_aucs.py --data_dir ../build-deepsea-training-dataset-master/out --seq_len 500 --variant original --global_step 100000
"""
import argparse
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


def _progress(curr, end, msg):
    sys.stdout.write(f"\r>> {msg} {curr/end*100:.1f}%")
    sys.stdout.flush()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='../build-deepsea-training-dataset-master/out',
                   help='directory containing test_labels.npy')
    p.add_argument('--eval_dir', default='eval_results')
    p.add_argument('--seq_len', type=int, default=1000)
    p.add_argument('--variant', default='original')
    p.add_argument('--split', default='test')
    p.add_argument('--global_step', type=int, required=True)
    args = p.parse_args()

    tag = f"seq{args.seq_len}_{args.variant}"
    subdir = os.path.join(args.eval_dir, tag, args.split)
    pred_path = os.path.join(subdir, f'preds-{args.global_step}.npy')

    # Labels from the data directory
    lbl_path = os.path.join(args.data_dir, f'{args.split}_labels.npy')

    if not os.path.exists(pred_path):
        print(f"Predictions not found: {pred_path}")
        print("Run eval with --save_predictions --run_once first.")
        sys.exit(1)

    labels = np.load(lbl_path)
    preds = np.load(pred_path)
    assert len(labels) == len(preds)
    num = labels.shape[1]
    assert num == 919

    aucs = np.zeros(num, dtype=float)
    out_path = os.path.join(subdir, f'aucs-{args.global_step}.txt')
    with open(out_path, 'w') as f:
        for i in range(num):
            try:
                auc = roc_auc_score(labels[:, i], preds[:, i])
                aucs[i] = auc
                f.write(f'{auc:.9f}\n')
            except ValueError:
                f.write('NA (No positive in test set)\n')
            _progress(i + 1, num, 'Computing AUCs')
        print()

    print(f"AUCs → {out_path}\n")
    print("Median AUCs:")
    print(f"  DNase I-hypersensitive sites:  {np.median(aucs[:125]):.4f}")
    print(f"  Transcription factors:         {np.median(aucs[125:815]):.4f}")
    print(f"  Histone marks:                 {np.median(aucs[815:919]):.4f}")
