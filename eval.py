"""
Evaluation script for DeepSEA report experiments.

Usage:
  # Continuous eval on val set
  python eval.py --seq_len 500 --variant original

  # One-shot eval on test set, save predictions
  python eval.py --seq_len 500 --variant original --split test --run_once --save_predictions --global_step 100000
"""
import argparse
import os
import sys
import time
from datetime import datetime
from glob import glob

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import dataset as data_module
from model import DeepSEA, cross_entropy_loss


def get_args():
    p = argparse.ArgumentParser(description='DeepSEA Experiment Evaluation')
    p.add_argument('--data_dir', default='../build-deepsea-training-dataset-master/out',
                   help='directory containing train_data.npy, train_labels.npy, etc.')
    p.add_argument('--train_dir', default='checkpoints', help='checkpoint root dir')
    p.add_argument('--eval_dir', default='eval_results')
    p.add_argument('--seq_len', type=int, default=1000)
    p.add_argument('--variant', default='original',
                   choices=['original', 'residual', 'attention'])
    p.add_argument('--split', default='val')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--global_step', type=int, default=-1, help='-1 = latest')
    p.add_argument('--eval_interval_secs', type=int, default=1000)
    p.add_argument('--run_once', action='store_true')
    p.add_argument('--report_progress', action='store_true')
    p.add_argument('--save_predictions', action='store_true')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def evaluate():
    args = get_args()
    assert args.split in ('val', 'test')

    tag = f"seq{args.seq_len}_{args.variant}"
    ckpt_dir = os.path.join(args.train_dir, tag)
    eval_subdir = os.path.join(args.eval_dir, tag, args.split)
    os.makedirs(eval_subdir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Data
    ds = data_module.DeepSEADataset(
        f'{args.data_dir}/{args.split}_data.npy',
        f'{args.data_dir}/{args.split}_labels.npy',
        args.seq_len)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(args.device == 'cuda'))
    num_examples = len(ds)
    num_steps = (num_examples + args.batch_size - 1) // args.batch_size

    print(f"Eval setup: seq_len={args.seq_len}, variant={args.variant}, split={args.split}")
    print(f"  {num_examples} examples, {num_steps} batches")

    # Model
    model = DeepSEA(seq_len=args.seq_len, variant=args.variant).to(device)
    last_eval_step = None

    while True:
        # Find checkpoint
        if args.global_step > 0:
            path = os.path.join(ckpt_dir, f'ckpt-{args.global_step}.pth')
            if not os.path.exists(path):
                print(f"Checkpoint {path} not found"); return
            gs = args.global_step
        else:
            ckpts = sorted(glob(os.path.join(ckpt_dir, 'ckpt-*.pth')))
            if not ckpts:
                print("No checkpoints yet");
                if args.run_once: return
                time.sleep(args.eval_interval_secs); continue
            gs = int(ckpts[-1].split('-')[-1].replace('.pth', ''))

        if last_eval_step == gs:
            if args.run_once: return
            time.sleep(args.eval_interval_secs); continue
        last_eval_step = gs

        # Load
        ckpt = torch.load(os.path.join(ckpt_dir, f'ckpt-{gs}.pth'), map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        total_loss = 0.0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for step, (seqs, labels) in enumerate(loader):
                seqs, labels = seqs.to(device), labels.to(device)
                logits = model(seqs)
                loss = cross_entropy_loss(logits, labels)
                total_loss += loss.item() * seqs.size(0)
                all_labels.append(labels.cpu().numpy())
                all_preds.append(torch.sigmoid(logits).cpu().numpy())

                if args.report_progress:
                    sys.stdout.write(f"\r>> {args.split} {(step+1)/num_steps*100:.1f}%")
                    sys.stdout.flush()

        if args.report_progress: print()

        mean_loss = total_loss / num_examples
        labels_np = np.concatenate(all_labels)
        preds_np = np.concatenate(all_preds)

        try:
            overall_auc = roc_auc_score(labels_np.ravel(), preds_np.ravel())
        except ValueError:
            overall_auc = 0.5

        print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  [{tag}]  {args.split}  "
              f"step={gs}  loss={mean_loss:.4f}  auc={overall_auc:.4f}")

        if args.save_predictions:
            np_path = os.path.join(eval_subdir, f'preds-{gs}.npy')
            np.save(np_path, preds_np)
            # also copy labels for compute_aucs
            lbl_path = os.path.join(eval_subdir, f'labels.npy')
            if not os.path.exists(lbl_path):
                np.save(lbl_path, labels_np)
            print(f"  Predictions → {np_path}")

        if args.run_once: return
        time.sleep(args.eval_interval_secs)


if __name__ == '__main__':
    evaluate()
