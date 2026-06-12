"""
Training script for DeepSEA report experiments.

Supports:
  --seq_len 200|500|1000   (sequence length experiment)
  --variant original|residual|attention  (architecture experiment)
"""
import argparse
import os
import time
from datetime import datetime
from glob import glob

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

import dataset as data_module
from model import DeepSEA, cross_entropy_loss


def get_args():
    p = argparse.ArgumentParser(description='DeepSEA Experiment Training')
    p.add_argument('--data_dir', default='../build-deepsea-training-dataset-master/out',
                   help='directory containing train_data.npy, train_labels.npy, etc.')
    p.add_argument('--train_dir', default='checkpoints', help='checkpoint dir')
    p.add_argument('--seq_len', type=int, default=1000, help='input sequence length')
    p.add_argument('--variant', default='original',
                   choices=['original', 'residual', 'attention'], help='model variant')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--lr_decay', type=float, default=8e-7)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--l2_lambda', type=float, default=5e-7)
    p.add_argument('--l1_lambda', type=float, default=1e-8)
    p.add_argument('--max_norm', type=float, default=0.9)
    p.add_argument('--log_frequency', type=int, default=10)
    p.add_argument('--summary_frequency', type=int, default=1000)
    p.add_argument('--checkpoint_frequency', type=int, default=10000)
    p.add_argument('--max_epochs', type=int, default=100)
    p.add_argument('--max_steps', type=int, default=0,
                   help='max steps (0 = max_epochs * batches_per_epoch)')
    p.add_argument('--max_to_keep', type=int, default=50)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--shuffle_buffer', type=int, default=5000)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def train():
    args = get_args()

    # Experiment tag for checkpoints/logs
    tag = f"seq{args.seq_len}_{args.variant}"
    ckpt_dir = os.path.join(args.train_dir, tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"{'='*60}")
    print(f"Experiment: seq_len={args.seq_len}, variant={args.variant}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # Data
    train_ds = data_module.DeepSEAIterableDataset(args.data_dir, args.seq_len, args.shuffle_buffer)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda'), drop_last=True)

    steps_per_epoch = data_module.NUM_TRAIN // args.batch_size
    max_steps = args.max_steps if args.max_steps > 0 else args.max_epochs * steps_per_epoch
    print(f"Batches/epoch: {steps_per_epoch}, max_steps: {max_steps}")

    # Model
    model = DeepSEA(seq_len=args.seq_len, variant=args.variant,
                    l1_lambda=args.l1_lambda, max_norm=args.max_norm).to(device)
    print(f"Parameters: {model.param_count:,}")

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.l2_lambda)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda s: 1.0 / (1.0 + args.lr_decay * s))

    # Resume
    global_step = 0
    ckpts = glob(os.path.join(ckpt_dir, 'ckpt-*.pth'))
    if ckpts:
        def _step_from_path(p):
            try:
                return int(os.path.basename(p).split('-')[-1].replace('.pth', ''))
            except Exception:
                return -1
        ckpts = sorted(ckpts, key=_step_from_path, reverse=True)
        latest = None
        ckpt = None
        # Try loading from highest step down, skip corrupted files
        for candidate in ckpts:
            try:
                tmp = torch.load(candidate, map_location=device)
                latest = candidate
                ckpt = tmp
                break
            except Exception:
                print(f"Warning: failed to load checkpoint {candidate}, skipping")
        if ckpt is None:
            print("No valid checkpoints found to resume from")
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        global_step = ckpt['global_step']
        print(f"Resumed from {latest}, step {global_step}")

    writer = SummaryWriter(log_dir=os.path.join(args.train_dir, 'tensorboard', tag))

    # Training loop
    model.train()
    epoch = 0
    start_time = time.time()
    examples_since_log = 0

    while global_step < max_steps:
        epoch += 1
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)

            logits = model(seqs)
            ce_loss = cross_entropy_loss(logits, labels)
            l1_loss = model.l1_sparsity_loss()
            total_loss = ce_loss + l1_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            model._max_norm_constraint()

            global_step += 1
            examples_since_log += args.batch_size

            if global_step % args.log_frequency == 0:
                now = time.time()
                dur = now - start_time
                eps = examples_since_log / dur if dur > 0 else 0
                spb = dur / args.log_frequency if dur > 0 else 0
                print(f"{datetime.now():%H:%M:%S} step {global_step:>9d}  "
                      f"loss={total_loss.item():.3f}  ({eps:.0f} ex/s, {spb:.3f} s/batch)")
                start_time = now
                examples_since_log = 0

            if global_step % args.summary_frequency == 0:
                writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                writer.add_scalar('loss/total', total_loss.item(), global_step)
                writer.add_scalar('loss/ce', ce_loss.item(), global_step)
                writer.add_scalar('loss/l1', l1_loss.item(), global_step)

            if global_step % args.checkpoint_frequency == 0:
                path = os.path.join(ckpt_dir, f'ckpt-{global_step}.pth')
                torch.save({
                    'global_step': global_step,
                    'seq_len': args.seq_len,
                    'variant': args.variant,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, path)
                # Prune old
                ckpts = sorted(glob(os.path.join(ckpt_dir, 'ckpt-*.pth')))
                while len(ckpts) > args.max_to_keep:
                    os.remove(ckpts.pop(0))

            if global_step >= max_steps:
                break

        print(f"--- Epoch {epoch} done, step {global_step} ---")

    writer.close()
    print(f"Training finished at step {global_step}")


if __name__ == '__main__':
    train()
