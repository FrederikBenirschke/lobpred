"""PyTorch models for the LOB study: TCN, DeepLOB, and AxialAttentionLOB.

All models share one contract so the training/eval harness is
model-agnostic:

    forward(x) : x is (B, T, F)  ->  (B, out_dim)

``out_dim=1`` is regression on the forward price change (the primary
task); ``out_dim=3`` gives logits for the down/stable/up sign label
(paper-comparable accuracy). The harness picks the loss accordingly.

The TCN is a PyTorch port of the reference architecture (causal dilated
residual stack, kernel 2; Bai et al.). DeepLOB follows Zhang/Zohren/
Roberts (ref [1]). AxialAttentionLOB is the novel
rung: it attends over BOTH the time axis and the *feature* axis, and the
feature-axis attention weights are extractable, a per-feature (hence
per-level) importance map, the "which pockets matter" view a fixed CNN
kernel can't give. Treat that map as a hypothesis to confirm with
permutation importance before trusting it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── TCN (paper) ─────────────────────────────────────────────


class _CausalConv1d(nn.Module):
    """1D conv with left-only padding (no future leakage)."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class _TCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.c1 = _CausalConv1d(c_in, c_out, kernel, dilation)
        self.c2 = _CausalConv1d(c_out, c_out, kernel, dilation)
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(F.relu(self.c1(x)))
        y = self.drop(F.relu(self.c2(y)))
        res = x if self.down is None else self.down(x)
        return F.relu(y + res)


class TCN(nn.Module):
    """Causal dilated temporal CNN (Bai et al.; the reference paper's model)."""

    def __init__(
        self,
        n_features: int,
        out_dim: int = 1,
        channels: int = 48,
        levels: int = 6,
        kernel: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        blocks = []
        c_in = n_features
        for i in range(levels):
            blocks.append(_TCNBlock(c_in, channels, kernel, dilation=2 ** i, dropout=dropout))
            c_in = channels
        self.net = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, F)
        h = self.net(x.transpose(1, 2))         # (B, C, T)
        return self.head(h[:, :, -1])           # last timestep → (B, out_dim)


# ── DeepLOB (ref [1]) ───────────────────────────────────────


class _Inception(nn.Module):
    """DeepLOB inception module: parallel 1x1, 3x1, 5x1 + pooled branch."""

    def __init__(self, c_in: int, c: int):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(c_in, c, 1), nn.LeakyReLU(0.01),
                                nn.Conv2d(c, c, (3, 1), padding=(1, 0)), nn.LeakyReLU(0.01))
        self.b2 = nn.Sequential(nn.Conv2d(c_in, c, 1), nn.LeakyReLU(0.01),
                                nn.Conv2d(c, c, (5, 1), padding=(2, 0)), nn.LeakyReLU(0.01))
        self.b3 = nn.Sequential(nn.MaxPool2d((3, 1), stride=1, padding=(1, 0)),
                                nn.Conv2d(c_in, c, 1), nn.LeakyReLU(0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)


class DeepLOB(nn.Module):
    """CNN + inception + LSTM (Zhang, Zohren, Roberts 2019).

    Treats the (T, F) window as a 1-channel image and convolves across the
    feature axis first (collapsing it), then runs an LSTM over time. The
    feature-axis conv strides assume an interleaved price/size layout; on
    our stationary feature set it still works as a learned feature mixer,
    without the hand-designed level semantics.
    """

    def __init__(self, n_features: int, out_dim: int = 1, lstm_hidden: int = 64):
        super().__init__()
        c = 16
        # Collapse the feature axis in stages (stride 2 on width).
        self.conv = nn.Sequential(
            nn.Conv2d(1, c, (1, 2), stride=(1, 2)), nn.LeakyReLU(0.01),
            nn.Conv2d(c, c, (4, 1), padding=(0, 0)), nn.LeakyReLU(0.01),
            nn.Conv2d(c, c, (1, 2), stride=(1, 2)), nn.LeakyReLU(0.01),
            nn.Conv2d(c, c, (4, 1), padding=(0, 0)), nn.LeakyReLU(0.01),
        )
        self.inception = _Inception(c, 32)
        # LayerNorm before the LSTM tames the unbounded conv/inception
        # features that otherwise detonate the recurrence (grad-norm 3e17
        # in diagnostics). Standard fix for the recurrence.
        self.pre_lstm_norm = nn.LayerNorm(96)
        self.lstm = nn.LSTM(input_size=96, hidden_size=lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, F)
        h = self.conv(x.unsqueeze(1))           # (B, c, T', F')
        h = self.inception(h)                    # (B, 96, T', F')
        h = h.mean(dim=3)                        # collapse remaining feature axis → (B, 96, T')
        h = h.transpose(1, 2)                    # (B, T', 96)
        h = self.pre_lstm_norm(h)
        out, _ = self.lstm(h)
        return self.head(out[:, -1])             # (B, out_dim)


# ── AxialAttentionLOB (the novel rung) ──────────────────────


class _PosEnc(nn.Module):
    """Learned positional embedding for a sequence of given length."""

    def __init__(self, n_pos: int, dim: int):
        super().__init__()
        self.emb = nn.Parameter(torch.zeros(1, n_pos, dim))
        nn.init.normal_(self.emb, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.emb[:, : x.size(1)]


class AxialAttentionLOB(nn.Module):
    """Attention over the time axis AND the feature/level axis.

    Two encoders:
      * **temporal**: tokens are the T snapshots (each embedded from F
        features); learns which past moments matter.
      * **feature**: tokens are the F features (each embedded from its
        T-length history); learns which features/levels matter. When the
        features are per-level (relpx/relsz at level k), the feature-axis
        attention is a direct read on *which book levels carry signal*,
        the "pockets" view. Retrieve it with ``last_feature_attn``.

    Kept small (data is scarce and the source books' deep
    levels are thin), heavy dropout, few heads. ``feature_mask``
    (B, F) can zero
    out absent levels so attention doesn't find structure in
    padding.
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        out_dim: int = 1,
        d_model: int = 32,
        n_heads: int = 2,
        depth: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_features = n_features
        # temporal stream: embed F → d_model per timestep
        self.t_in = nn.Linear(n_features, d_model)
        self.t_pos = _PosEnc(seq_len, d_model)
        t_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.t_enc = nn.TransformerEncoder(t_layer, depth)
        # feature stream: embed T → d_model per feature token
        self.f_in = nn.Linear(seq_len, d_model)
        self.f_pos = _PosEnc(n_features, d_model)
        self.f_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.f_norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )
        self.last_feature_attn: torch.Tensor | None = None  # (B, F) after forward

    def forward(self, x: torch.Tensor, feature_mask: torch.Tensor | None = None) -> torch.Tensor:
        # temporal stream
        t = self.t_pos(self.t_in(x))            # (B, T, d)
        t = self.t_enc(t).mean(dim=1)           # pool time → (B, d)
        # feature stream: tokens = features, each carries its T-history
        f_tok = self.f_pos(self.f_in(x.transpose(1, 2)))   # (B, F, d)
        f_out, attn_w = self.f_attn(f_tok, f_tok, f_tok,
                                    need_weights=True, average_attn_weights=True)
        f_out = self.f_norm(f_out + f_tok)
        # per-feature importance = how much every other token attends TO it
        self.last_feature_attn = attn_w.mean(dim=1).detach()   # (B, F)
        if feature_mask is not None:
            f_out = f_out * feature_mask.unsqueeze(-1)
        f = f_out.mean(dim=1)                    # (B, d)
        return self.head(torch.cat([t, f], dim=1))


# ── PerLevelLOB (the per-level stationary representation) ───


class PerLevelLOB(nn.Module):
    """Spatial-temporal net for the per-level stationary tensor.

    Input is the flat (B, T, L*C) window from ``features.add_perlevel_features``
    (C channels per book level, level-major). It reshapes to (B, T, L, C),
    convolves across the **level** axis at each timestep, learning cross-level
    patterns a tree or flat-scalar model can't represent, then collapses
    levels and runs an LSTM over time. This is the DeepLOB philosophy applied
    to stationary per-level inputs (cf. Kolm et al., multi-level OFI → RNN);
    the fair, structured way to give a deep model the order book.
    """

    def __init__(self, n_features: int, seq_len: int, out_dim: int = 1,
                 channels: int = 3, hidden: int = 32, lstm_hidden: int = 48, dropout: float = 0.2):
        super().__init__()
        if n_features % channels != 0:
            raise ValueError(f"n_features {n_features} not divisible by channels {channels}")
        self.channels = channels
        self.levels = n_features // channels
        self.conv = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=(1, 3), padding=(0, 1)), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, hidden, kernel_size=(1, 3), padding=(0, 1)), nn.ReLU(),
        )
        # LayerNorm before the LSTM tames the unbounded conv features that
        # otherwise explode the recurrence on heavy-tailed targets (the
        # DeepLOB fix). Without it, grad norms blow up to ~1e6 on real books.
        self.pre_lstm_norm = nn.LayerNorm(hidden)
        self.lstm = nn.LSTM(hidden, lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, L*C)
        b, t, _ = x.shape
        x = x.view(b, t, self.levels, self.channels).permute(0, 3, 1, 2)  # (B, C, T, L)
        h = self.conv(x)                 # (B, hidden, T, L)
        h = h.mean(dim=3)                # collapse levels → (B, hidden, T)
        h = h.transpose(1, 2)            # (B, T, hidden)
        h = self.pre_lstm_norm(h)        # stabilize before the recurrence
        out, _ = self.lstm(h)
        return self.head(out[:, -1])     # (B, out_dim)


# ── SeqLSTM (the no-pool control) ───────────────────────────


class SeqLSTM(nn.Module):
    """Plain LSTM over the feature sequence, the no-pool control.

    Unlike ``PerLevelLOB``/``DeepLOB``, which convolve across the feature
    axis and then *collapse* it (mean over levels) before the recurrence,
    this keeps every feature channel as a direct LSTM input. On the
    stationary per-level set, the collapse loses the signal. Pooling over the
    few real book levels averages away the structure, so this minimal model is
    the strongest deep baseline here
    (see README, "Learning deep learning on the LOB"). LayerNorm on the
    input keeps the recurrence stable on heavy-tailed targets.
    """

    def __init__(self, n_features: int, seq_len: int | None = None, out_dim: int = 1,
                 hidden: int = 128, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(n_features)
        self.lstm = nn.LSTM(n_features, hidden, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, F)
        out, _ = self.lstm(self.norm(x))
        return self.head(self.drop(out[:, -1]))           # (B, out_dim)


# ── registry ────────────────────────────────────────────────


def build_model(name: str, *, n_features: int, seq_len: int, out_dim: int = 1) -> nn.Module:
    name = name.lower()
    if name == "tcn":
        return TCN(n_features, out_dim=out_dim)
    if name == "deeplob":
        return DeepLOB(n_features, out_dim=out_dim)
    if name in ("attention", "axial", "axialattentionlob"):
        return AxialAttentionLOB(n_features, seq_len, out_dim=out_dim)
    if name in ("perlevel", "perlevellob"):
        return PerLevelLOB(n_features, seq_len, out_dim=out_dim)
    if name in ("seqlstm", "lstm"):
        return SeqLSTM(n_features, seq_len, out_dim=out_dim)
    raise ValueError(f"unknown model {name!r}; choose tcn | deeplob | attention | perlevel | seqlstm")
