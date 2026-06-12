"""Attention weight analysis — seq1000_attention model.

Three levels (following the user's framework):
  Level 1: Single-sample heatmaps (8 heads x 53x53)
  Level 2: Per-head distance preference (signed distance, -52 to +52)
  Level 3: Multi-sample aggregation + head specialization dashboard
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from model import DeepSEA, compute_conv_output_len
from dataset import DeepSEADataset

# ══════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════
CKPT_PATH = '/root/shared-nvme/checkpoints/seq1000_attention/ckpt-990000.pth'
DATA_DIR  = '/root/destination/528/out'
OUT_DIR   = '/root/shared-nvme/eval_results/attention_analysis'
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN   = 1000
BATCH     = 8
NUM_SAMPLES = 48
L = compute_conv_output_len(SEQ_LEN)  # 53
H = 8  # num_heads

print(f"Conv3 output length L = {L}")

# ══════════════════════════════════════════════════════════════════
# 1. Load model + data
# ══════════════════════════════════════════════════════════════════
model = DeepSEA(seq_len=SEQ_LEN, variant='attention').to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Model loaded, global_step={ckpt.get('global_step')}")

val_ds = DeepSEADataset(
    f'{DATA_DIR}/valid_data.npy', f'{DATA_DIR}/valid_labels.npy', SEQ_LEN)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

# ══════════════════════════════════════════════════════════════════
# 2. Hook to capture attention
# ══════════════════════════════════════════════════════════════════
attention_weights = []

def attention_hook(module, input, output):
    if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
        attention_weights.append(output[1].detach().cpu())

model.self_attn.register_forward_hook(attention_hook)
_orig_forward = model.self_attn.forward

def patched_forward(query, key, value, **kwargs):
    kwargs['need_weights'] = True
    kwargs['average_attn_weights'] = False
    return _orig_forward(query, key, value, **kwargs)

model.self_attn.forward = patched_forward

# ══════════════════════════════════════════════════════════════════
# 3. Collect attention weights
# ══════════════════════════════════════════════════════════════════
all_attns = []
count = 0
for batch_x, batch_y in val_loader:
    batch_x = batch_x.to(DEVICE)
    with torch.no_grad():
        _ = model(batch_x)
    if attention_weights:
        all_attns.append(attention_weights.pop(0))
    count += batch_x.size(0)
    if count >= NUM_SAMPLES:
        break

attn = torch.cat(all_attns, dim=0)[:NUM_SAMPLES]  # (N, 8, L, L)
print(f"Collected attention: {attn.shape}")

# ══════════════════════════════════════════════════════════════════
# 层次一：单样本热力图
# ══════════════════════════════════════════════════════════════════
print("\n=== Level 1: Single-sample heatmaps ===")

def plot_sample_heatmaps(attn_tensor, sample_idx, out_path):
    """For one sample, plot 8 head heatmaps."""
    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 0.3],
                           height_ratios=[1, 1], hspace=0.35, wspace=0.35)

    for h in range(H):
        ax = fig.add_subplot(gs[h // 4, h % 4])
        mat = attn_tensor[sample_idx, h].numpy()
        vmax_val = max(0.05, mat.max() * 0.8)
        im = ax.imshow(mat, cmap='Blues', aspect='auto',
                        vmin=0, vmax=vmax_val)
        ax.set_title(f'Head {h+1}', fontsize=11)
        if h % 4 == 0:
            ax.set_ylabel('Query position', fontsize=9)
        if h >= 4:
            ax.set_xlabel('Key position', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f'Attention Heatmaps — Sample {sample_idx+1}\n'
                 f'(Q=row, K=col; brighter = stronger attention)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {os.path.basename(out_path)}")

plot_sample_heatmaps(attn, 0, os.path.join(OUT_DIR, 'L1_heatmap_sample1.png'))
plot_sample_heatmaps(attn, 1, os.path.join(OUT_DIR, 'L1_heatmap_sample2.png'))

# ══════════════════════════════════════════════════════════════════
# 层次二：距离偏好分析 (per-head signed distance)
# ══════════════════════════════════════════════════════════════════
print("\n=== Level 2: Distance preference (signed) ===")

def compute_signed_distance_profile(attn_tensor):
    """For each head, compute mean attention at each signed distance.
    
    Signed distance d = key_pos - query_pos.
    d > 0: attending to downstream (right side)
    d < 0: attending to upstream (left side)
    """
    d_min, d_max = -(L - 1), (L - 1)  # -52 to +52
    n_dist = d_max - d_min + 1

    head_profiles = np.zeros((H, n_dist))

    for h in range(H):
        accum = np.zeros(n_dist)
        counts = np.zeros(n_dist)
        for q in range(L):      # query
            for k in range(L):  # key
                d = k - q
                idx = d - d_min
                accum[idx] += attn_tensor[:, h, q, k].mean().item()
                counts[idx] += 1
        head_profiles[h] = accum / counts

    return head_profiles, d_min, d_max

head_profiles, d_min, d_max = compute_signed_distance_profile(attn)
distances = np.arange(d_min, d_max + 1)

# Plot: 8 heads overlaid + near/far ratio inset
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

near_radius = 10  # |d| <= 10 positions (~160bp)
near_far_ratios = []

for h in range(H):
    ax = axes[h]
    prof = head_profiles[h]
    # Signed profile (downstream = positive)
    ax.fill_between(distances, 0, prof, alpha=0.6, color=plt.cm.tab10(h))
    ax.plot(distances, prof, 'k-', linewidth=0.8)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.set_title(f'Head {h+1}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Signed distance (key − query)', fontsize=8)
    ax.set_ylabel('Mean attention', fontsize=8)
    ax.set_xlim(-55, 55)

    # Near/far ratio
    near_mask = np.abs(distances) <= near_radius
    far_mask = np.abs(distances) > near_radius
    near_sum = prof[near_mask].sum()
    far_sum = prof[far_mask].sum()
    ratio = near_sum / (far_sum + 1e-8)
    near_far_ratios.append(ratio)
    ax.text(0.95, 0.95, f'Near/Far = {ratio:.1f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

fig.suptitle('Level 2: Per-Head Distance Preference (signed)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'L2_distance_preference.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved → L2_distance_preference.png")

# Per-head near/far ratio summary
print("\n  Near/Far ratio summary:")
for h in range(H):
    label = "local focus" if near_far_ratios[h] > 2 else ("long-range" if near_far_ratios[h] < 1 else "mixed")
    print(f"    Head {h+1}: {near_far_ratios[h]:.2f}  ({label})")

# ══════════════════════════════════════════════════════════════════
# 层次三：多样本聚合 + 综合仪表板
# ══════════════════════════════════════════════════════════════════
print("\n=== Level 3: Multi-sample aggregation dashboard ===")

# 3a) Average attention matrix (all heads + samples)
avg_all = attn.mean(dim=(0, 1)).numpy()  # (L, L)

# 3b) Per-head average matrices
head_avg = attn.mean(dim=0).numpy()  # (H, L, L)

# --- Figure: comprehensive 3x3 per-head average matrices ---
fig, axes = plt.subplots(3, 3, figsize=(14, 13))
for h in range(H):
    ax = axes[h // 3][h % 3]
    im = ax.imshow(head_avg[h], cmap='inferno', aspect='auto')
    ax.set_title(f'Head {h+1}', fontsize=11, fontweight='bold')
    if h % 3 == 0:
        ax.set_ylabel('Query position', fontsize=9)
    if h >= 6:
        ax.set_xlabel('Key position', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)

# 9th panel: average of all
im = axes[2][2].imshow(avg_all, cmap='inferno', aspect='auto')
axes[2][2].set_title('ALL HEADS AVERAGE', fontsize=11, fontweight='bold', color='darkred')
axes[2][2].set_xlabel('Key position', fontsize=9)
plt.colorbar(im, ax=axes[2][2], fraction=0.046)
fig.suptitle('Level 3: Aggregate Attention Patterns (48 samples)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'L3_aggregate_patterns.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved → L3_aggregate_patterns.png")

# --- Figure: diagonal strength & directionality ---
diagonal_strength = np.zeros((H, L))
for h in range(H):
    for d in range(L):
        vals = [attn[:, h, i, i + d].mean().item() for i in range(L - d)]
        if vals:
            diagonal_strength[h, d] = np.mean(vals)

# directionality: asymmetry between upstream vs downstream
upstream_mask = distances < 0
downstream_mask = distances > 0
directionality = np.zeros(H)
for h in range(H):
    down = head_profiles[h][downstream_mask].sum()
    up = head_profiles[h][upstream_mask].sum()
    directionality[h] = (down - up) / (down + up + 1e-8)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 3c: diagonal decay per head
for h in range(H):
    axes[0].plot(range(L), diagonal_strength[h], linewidth=1.5,
                 label=f'Head {h+1}')
axes[0].set_xlabel('|distance| (Conv3 positions)', fontsize=11)
axes[0].set_ylabel('Mean attention weight', fontsize=11)
axes[0].set_title('Diagonal Decay: Attention vs. Absolute Distance', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=8, ncol=2)
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=10, color='red', linestyle='--', alpha=0.4, label='~160bp')
y_lim = axes[0].get_ylim()
axes[0].fill_between([0, 10], 0, y_lim[1], alpha=0.05, color='orange')
axes[0].set_ylim(y_lim)

# 3d: directionality bar
colors_dir = ['#1f77b4' if v < 0 else '#d62728' for v in directionality]
bars = axes[1].bar(range(1, H+1), directionality, color=colors_dir, edgecolor='black')
axes[1].axhline(y=0, color='black', linewidth=0.8)
axes[1].set_xlabel('Head', fontsize=11)
axes[1].set_ylabel('Directionality (down − up) / total', fontsize=11)
axes[1].set_title('Direction Bias per Head\n(red→downstream, blue→upstream)', fontsize=12, fontweight='bold')
axes[1].set_xticks(range(1, H+1))
for bar, val in zip(bars, directionality):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 0.02 if val >= 0 else -0.08,
                 f'{val:+.2f}', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'L3_diagonal_decay_directionality.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved → L3_diagonal_decay_directionality.png")

# --- Figure: Position importance (center focus analysis) ---
position_importance = avg_all.mean(axis=0)  # how much attention each position receives
position_importance /= position_importance.sum()

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Overlay per-head position importance
for h in range(H):
    pos_imp_h = head_avg[h].mean(axis=0)
    pos_imp_h /= pos_imp_h.sum()
    axes[0].plot(range(L), pos_imp_h, alpha=0.6, linewidth=1.2, label=f'Head {h+1}')
axes[0].fill_between(range(L), 0, position_importance, alpha=0.25, color='grey', label='Mean all heads')
axes[0].set_xlabel('Conv3 position index', fontsize=11)
axes[0].set_ylabel('Normalized attention received', fontsize=11)
axes[0].set_title('Position Importance (which positions are attended to)', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=8, ncol=5)
axes[0].axvline(x=L//2, color='red', linestyle='--', alpha=0.5, label='Center')
axes[0].grid(True, alpha=0.3)

# Total attention received (with center highlight)
axes[1].bar(range(L), position_importance, color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.3)
# Highlight center region (pos 20-33, ~300bp around center)
center_start, center_end = L//2 - 7, L//2 + 7
for i in range(center_start, center_end):
    axes[1].bar(i, position_importance[i], color='coral', alpha=0.8, edgecolor='black', linewidth=0.3)
axes[1].set_xlabel('Conv3 position index', fontsize=11)
axes[1].set_ylabel('Normalized attention received', fontsize=11)
axes[1].set_title('Center Focus: attention concentrates on middle positions', fontsize=12, fontweight='bold')
center_ratio = position_importance[center_start:center_end].sum()
axes[1].text(0.98, 0.95, f'Center ~1/3 positions\nreceives {center_ratio:.1%} of attention',
             transform=axes[1].transAxes, ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'L3_position_importance.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved → L3_position_importance.png")

# --- Figure: Head similarity + specialization dashboard ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Head similarity (correlation between per-head spatial patterns)
head_flat = head_avg.reshape(H, -1)
corr = np.corrcoef(head_flat)
im = axes[0].imshow(corr, cmap='RdYlBu_r', vmin=0, vmax=1)
axes[0].set_xticks(range(H)); axes[0].set_yticks(range(H))
axes[0].set_xticklabels([f'H{h+1}' for h in range(H)])
axes[0].set_yticklabels([f'H{h+1}' for h in range(H)])
axes[0].set_title('Head Pattern Similarity (Pearson r)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=axes[0])
for i in range(H):
    for j in range(H):
        axes[0].text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center', fontsize=7,
                     color='white' if corr[i,j] < 0.5 else 'black')

# Head specialization: entropy + near/far ratio scatter
head_entropy = []
for h in range(H):
    p = head_avg[h].mean(axis=0)
    p = p / p.sum()
    p_pos = p[p > 0]
    entropy = -np.sum(p_pos * np.log(p_pos))
    head_entropy.append(entropy)

sc = axes[1].scatter(near_far_ratios, head_entropy,
                      c=range(H), cmap='tab10', s=200, edgecolors='black', zorder=5)
for h in range(H):
    axes[1].annotate(f'H{h+1}', (near_far_ratios[h], head_entropy[h]),
                     textcoords="offset points", xytext=(8, 4), fontsize=9, fontweight='bold')
axes[1].set_xlabel('Near/Far attention ratio', fontsize=11)
axes[1].set_ylabel('Position entropy (lower = more focused)', fontsize=11)
axes[1].set_title('Head Specialization Map', fontsize=12, fontweight='bold')
axes[1].axhline(y=np.mean(head_entropy), color='grey', linestyle='--', alpha=0.5)
axes[1].axvline(x=1.0, color='grey', linestyle='--', alpha=0.5)
# Quadrant labels
axes[1].text(0.98, 0.02, 'Long-range\nunfocused',
             transform=axes[1].transAxes, ha='right', va='bottom', fontsize=8, alpha=0.6)
axes[1].text(0.02, 0.02, 'Long-range\nfocused',
             transform=axes[1].transAxes, ha='left', va='bottom', fontsize=8, alpha=0.6)
axes[1].text(0.98, 0.98, 'Local\nunfocused',
             transform=axes[1].transAxes, ha='right', va='top', fontsize=8, alpha=0.6)
axes[1].text(0.02, 0.98, 'Local\nfocused',
             transform=axes[1].transAxes, ha='left', va='top', fontsize=8, alpha=0.6)
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'L3_head_similarity_specialization.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved → L3_head_similarity_specialization.png")

# ══════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY — Key observations for reporting:")
print("=" * 60)
diag_mean = avg_all.diagonal().mean()
off_diag = avg_all[~np.eye(L, dtype=bool)]
off_diag_mean = off_diag.mean() if len(off_diag) > 0 else 0
print(f"  1. Diagonal dominance: mean diag = {diag_mean:.4f}, off-diag = {off_diag_mean:.4f}")
print(f"     → Diagonal is {diag_mean/off_diag_mean:.1f}x stronger than off-diagonal"
      if off_diag_mean > 0 else "     → Diagonal is dominant")
center_attn = position_importance[L//3:2*L//3].sum()
print(f"  2. Center focus: middle 1/3 positions receive {center_attn:.1%} of attention")
dir_heads = [h+1 for h in range(H) if abs(directionality[h]) > 0.05]
print(f"  3. Directional heads: {dir_heads} show asymmetric (directional) attention")
specialized = [h+1 for h in range(H) if near_far_ratios[h] > 2 or near_far_ratios[h] < 0.5]
print(f"  4. Specialized heads: {specialized} have extreme near/far ratios")
diverse_pairs = [(i+1, j+1) for i in range(H) for j in range(i+1, H) if corr[i, j] < 0.5]
print(f"  5. Diverse head pairs (r<0.5): {diverse_pairs} — truly different patterns")
print(f"\nAll outputs saved to: {OUT_DIR}")
