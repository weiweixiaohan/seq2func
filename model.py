"""
DeepSEA model with variable input length and architecture variants.

Supports:
  - seq_len: any input length (200, 500, 1000, etc.)
  - variant: 'original' | 'residual' | 'attention'
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


CONV1_OUT = 320
CONV2_OUT = 480
CONV3_OUT = 960
NUM_OUTPUTS = 919
KERNEL_SIZE = 8
POOL_SIZE = 4


def compute_conv_output_len(seq_len: int) -> int:
    """Trace Conv1→Pool1→Conv2→Pool2→Conv3 to get final per-channel length."""
    l = seq_len - KERNEL_SIZE + 1              # Conv1
    l = (l - POOL_SIZE) // POOL_SIZE + 1       # Pool1
    l = l - KERNEL_SIZE + 1                    # Conv2
    l = (l - POOL_SIZE) // POOL_SIZE + 1       # Pool2
    l = l - KERNEL_SIZE + 1                    # Conv3
    return l


def compute_fc_input_dim(seq_len: int) -> int:
    return CONV3_OUT * compute_conv_output_len(seq_len)



# Positional Encoding (used by attention variant)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Conv3 feature positions."""
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, L, C)
        return x + self.pe[:, :x.size(1), :]


import math



# Model
class DeepSEA(nn.Module):
    """
    Args:
        seq_len:  input sequence length (e.g. 200, 500, 1000)
        variant:  'original' | 'residual' | 'attention'
        l1_lambda:  L1 sparsity coefficient
        max_norm:   max-norm constraint for conv kernels (0 to disable)
    """

    def __init__(self, seq_len: int = 1000, variant: str = 'original',
                 l1_lambda: float = 1e-8, max_norm: float = 0.9):
        super().__init__()
        self.seq_len = seq_len
        self.variant = variant
        self.l1_lambda = l1_lambda
        self.max_norm = max_norm

        # --- Convolutional layers ---
        self.conv1 = nn.Conv1d(4, CONV1_OUT, kernel_size=KERNEL_SIZE)
        self.pool1 = nn.MaxPool1d(kernel_size=POOL_SIZE, stride=POOL_SIZE)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(CONV1_OUT, CONV2_OUT, kernel_size=KERNEL_SIZE)
        self.pool2 = nn.MaxPool1d(kernel_size=POOL_SIZE, stride=POOL_SIZE)
        self.dropout2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv1d(CONV2_OUT, CONV3_OUT, kernel_size=KERNEL_SIZE)
        self.dropout3 = nn.Dropout(0.5)

        # --- Residual projections ---
        if variant == 'residual':
            # 1x1 conv to align channel dim for skip connections
            # Skip1: input(4) -> conv1 output channels (320), then pool
            self.skip_conv1 = nn.Conv1d(4, CONV1_OUT, kernel_size=1)
            self.skip_pool1 = nn.MaxPool1d(kernel_size=POOL_SIZE, stride=POOL_SIZE)
            # Skip2: pool1 output (320) -> conv2 output channels (480)
            self.skip_conv2 = nn.Conv1d(CONV1_OUT, CONV2_OUT, kernel_size=1)

        # --- Attention ---
        if variant == 'attention':
            self.pos_encoder = PositionalEncoding(CONV3_OUT)
            self.self_attn = nn.MultiheadAttention(
                embed_dim=CONV3_OUT, num_heads=8, dropout=0.1, batch_first=True)
            self.attn_norm = nn.LayerNorm(CONV3_OUT)

        # --- Fully-connected layers ---
        fc_dim = compute_fc_input_dim(seq_len)
        self.fc4 = nn.Linear(fc_dim, NUM_OUTPUTS)
        self.sigmoid_linear = nn.Linear(NUM_OUTPUTS, NUM_OUTPUTS)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

    def _max_norm_constraint(self):
        if self.max_norm <= 0:
            return
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    w = m.weight.data
                    oc = w.shape[0]
                    w_flat = w.view(oc, -1)
                    norms = w_flat.norm(dim=1, keepdim=True)
                    desired = torch.clamp(norms, max=self.max_norm)
                    w_flat.mul_(desired / (norms + 1e-7))

    def l1_sparsity_loss(self) -> torch.Tensor:
        return self.l1_lambda * torch.sum(torch.abs(self._last_sigmoid_linear))

    def _pad_to_match(self, src, dst_len):
        """Pad or trim src along last dim to match dst_len."""
        cur = src.shape[2]
        if cur < dst_len:
            return F.pad(src, (0, dst_len - cur))
        elif cur > dst_len:
            return src[:, :, :dst_len]
        return src

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ─── Conv1 ───
        c1 = F.relu(self.conv1(x))
        if self.variant == 'residual':
            s1 = self.skip_pool1(self.skip_conv1(x))
            c1 = c1 + self._pad_to_match(s1, c1.shape[2])
        p1 = self.pool1(c1)
        p1 = self.dropout1(p1)

        # ─── Conv2 ───
        c2 = F.relu(self.conv2(p1))
        if self.variant == 'residual':
            s2 = self.skip_conv2(p1)
            c2 = c2 + self._pad_to_match(s2, c2.shape[2])
        p2 = self.pool2(c2)
        p2 = self.dropout2(p2)

        # ─── Conv3 ───
        c3 = F.relu(self.conv3(p2))
        c3 = self.dropout3(c3)

        # ─── Attention (optional) ───
        if self.variant == 'attention':
            # (B, C, L) → (B, L, C)
            c3_t = c3.transpose(1, 2)
            c3_t = self.pos_encoder(c3_t)
            attn_out, _ = self.self_attn(c3_t, c3_t, c3_t)
            c3_t = self.attn_norm(c3_t + attn_out)
            c3 = c3_t.transpose(1, 2)  # back to (B, C, L)

        # ─── FC ───
        x = c3.reshape(c3.size(0), -1)
        x = F.relu(self.fc4(x))
        x = self.sigmoid_linear(x)

        self._last_sigmoid_linear = x
        return x

    @property
    def fc_input_dim(self) -> int:
        return compute_fc_input_dim(self.seq_len)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())



# Loss
def cross_entropy_loss(logits, labels):
    return F.binary_cross_entropy_with_logits(logits, labels)



# Shape trace utility

def print_model_table():
    print(f"{'Seq Len':>8}  {'Conv1':>6}  {'Pool1':>6}  {'Conv2':>6}  "
          f"{'Pool2':>6}  {'Conv3':>6}  {'FC dim':>8}  {'Params':>12}")
    print("-" * 72)
    for sl in [200, 500, 1000]:
        l1 = sl - 8 + 1
        p1 = (l1 - 4) // 4 + 1
        l2 = p1 - 8 + 1
        p2 = (l2 - 4) // 4 + 1
        l3 = p2 - 8 + 1
        fc = 960 * l3
        m = DeepSEA(seq_len=sl)
        print(f"{sl:>8}  {l1:>6}  {p1:>6}  {l2:>6}  "
              f"{p2:>6}  {l3:>6}  {fc:>8}  {m.param_count:>12,}")


if __name__ == '__main__':
    for v in ['original', 'residual', 'attention']:
        print(f"\n=== Variant: {v} ===")
        print_model_table()
