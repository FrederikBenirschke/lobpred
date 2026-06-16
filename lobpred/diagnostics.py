"""Diagnostic plots: training/gradient curves, error analysis, attention.

Three families, each answering a different question:

  * **training curves**, train vs val loss + gradient-norm traces.
    Diagnoses under/over-fitting and optimization health (vanishing /
    exploding / spiking gradients).
  * **error analysis**, residual distribution, predicted-vs-actual,
    error-vs-move-size, and calibration. Diagnoses *where* a model is
    wrong and by how much.
  * **attention pockets**, the AxialAttentionLOB per-feature attention
    averaged over a sample; on grid features this is an attention-by-
    book-depth profile (the "which pockets matter" view). A finder, not
    a verdict, confirm with permutation importance.

Uses the Agg backend (no display); every function writes a PNG and
returns its path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_training_curves(histories: dict[str, dict], outpath: Path, title: str = "") -> Path:
    """Loss curves + gradient norms per model.

    histories: {model_name: history dict from evaluate.train_torch}.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, h in histories.items():
        ep = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(ep, h["train_loss"], "-", label=f"{name} train")
        axes[0].plot(ep, h["val_loss"], "--", label=f"{name} val")
        axes[1].plot(ep, h["grad_norm_mean"], "-o", ms=3, label=f"{name} mean")
        axes[2].plot(h["grad_steps"], lw=0.6, alpha=0.8, label=name)
    axes[0].set(title="loss (MSE)", xlabel="epoch", ylabel="MSE"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].set(title="gradient global-norm (per-epoch mean)", xlabel="epoch", ylabel="‖g‖₂")
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    axes[2].set(title="gradient norm per step", xlabel="optimizer step", ylabel="‖g‖₂")
    axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=110); plt.close(fig)
    return outpath


def plot_error_analysis(y_true: np.ndarray, y_pred: np.ndarray, outpath: Path,
                        title: str = "") -> Path:
    """Residuals, pred-vs-actual, error-vs-move, calibration."""
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    resid = y_true - y_pred
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))

    # residual histogram
    ax[0, 0].hist(resid, bins=120, color="steelblue", alpha=0.8)
    ax[0, 0].axvline(0, color="k", lw=0.8)
    ax[0, 0].set(title=f"residuals (mean={resid.mean():.2e}, std={resid.std():.2e})",
                 xlabel="y_true − y_pred")

    # predicted vs actual (hexbin)
    hb = ax[0, 1].hexbin(y_pred, y_true, gridsize=60, bins="log", cmap="viridis")
    lim = np.percentile(np.abs(np.concatenate([y_true, y_pred])), 99)
    ax[0, 1].plot([-lim, lim], [-lim, lim], "r--", lw=1)
    ax[0, 1].set(title="predicted vs actual", xlabel="y_pred", ylabel="y_true",
                 xlim=(-lim, lim), ylim=(-lim, lim))
    fig.colorbar(hb, ax=ax[0, 1], label="log count")

    # abs error vs |move| (binned)
    mv = np.abs(y_true); ae = np.abs(resid)
    order = np.argsort(mv); nbin = 20
    edges = np.linspace(0, len(mv), nbin + 1).astype(int)
    bx, by = [], []
    for i in range(nbin):
        sl = order[edges[i]:edges[i + 1]]
        if len(sl):
            bx.append(mv[sl].mean()); by.append(ae[sl].mean())
    ax[1, 0].plot(bx, by, "-o", ms=4, color="darkorange")
    ax[1, 0].set(title="abs error vs |move| (20 quantile bins)",
                 xlabel="|y_true|", ylabel="mean |error|")
    ax[1, 0].grid(alpha=0.3)

    # calibration: decile of y_pred → mean pred vs mean actual
    q = np.quantile(y_pred, np.linspace(0, 1, 11))
    q[-1] += 1e-9
    idx = np.clip(np.digitize(y_pred, q) - 1, 0, 9)
    cx = [y_pred[idx == b].mean() for b in range(10) if (idx == b).any()]
    cy = [y_true[idx == b].mean() for b in range(10) if (idx == b).any()]
    ax[1, 1].plot(cx, cy, "-o", ms=5, color="seagreen", label="binned")
    clim = max(abs(min(cx)), abs(max(cx)))
    ax[1, 1].plot([-clim, clim], [-clim, clim], "r--", lw=1, label="y=x")
    ax[1, 1].set(title="calibration (pred deciles)", xlabel="mean y_pred", ylabel="mean y_true")
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=110); plt.close(fig)
    return outpath


def plot_attention_pockets(attn: np.ndarray, feat_cols: list[str], outpath: Path,
                           title: str = "", grid_k: int | None = None) -> Path:
    """Per-feature attention (sample-averaged) as an importance profile.

    ``attn`` is (F,) mean attention received by each feature token. When
    the features are the price grid, this reads as attention-by-depth: a
    spike at offset o is the model saying "the pocket o ticks from touch
    matters". The non-grid scalars are shown as separate bars.
    """
    attn = np.asarray(attn, float)
    is_grid = np.array(["_grid_" in c for c in feat_cols])
    fig, ax = plt.subplots(1, 2 if is_grid.any() else 1, figsize=(13, 4.5), squeeze=False)
    ax = ax[0]

    if is_grid.any() and grid_k:
        bid = np.array([attn[feat_cols.index(f"bid_grid_{o}")]
                        if f"bid_grid_{o}" in feat_cols else 0.0 for o in range(grid_k)])
        ask = np.array([attn[feat_cols.index(f"ask_grid_{o}")]
                        if f"ask_grid_{o}" in feat_cols else 0.0 for o in range(grid_k)])
        off = np.arange(grid_k)
        ax[0].bar(-off - 0.5, bid, width=0.9, color="green", alpha=0.7, label="bid side")
        ax[0].bar(off + 0.5, ask, width=0.9, color="red", alpha=0.7, label="ask side")
        ax[0].axvline(0, color="k", lw=0.8)
        ax[0].set(title="attention by book depth (pockets)",
                  xlabel="tick offset from touch  (← bid | ask →)", ylabel="mean attention")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
        # scalars panel
        sc = [c for c in feat_cols if "_grid_" not in c]
        sv = [attn[feat_cols.index(c)] for c in sc]
        ax[1].barh(range(len(sc)), sv, color="slateblue", alpha=0.8)
        ax[1].set_yticks(range(len(sc))); ax[1].set_yticklabels(sc, fontsize=8)
        ax[1].set(title="attention on flow scalars", xlabel="mean attention"); ax[1].grid(alpha=0.3)
    else:
        order = np.argsort(attn)[::-1]
        ax[0].bar(range(len(attn)), attn[order], color="slateblue", alpha=0.8)
        ax[0].set_xticks(range(len(attn)))
        ax[0].set_xticklabels([feat_cols[i] for i in order], rotation=90, fontsize=6)
        ax[0].set(title="per-feature attention", ylabel="mean attention")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=110); plt.close(fig)
    return outpath
