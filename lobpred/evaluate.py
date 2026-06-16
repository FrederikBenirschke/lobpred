"""Metrics, the PyTorch training loop, and the phase gap table.

These metrics measure forecast quality. The study reports no
trading PnL. On a class-imbalanced sign label, accuracy flatters a model
that always predicts "stable", so we report several complementary views:

  * **R²** and **correlation** of the forward-change regression (OOS).
  * **sign hit-rate** and **size-weighted hit-rate** (a correct call on a
    big move counts more than on a tiny one).
  * 3-class **accuracy** and **macro-F1** for the sign head.

The harness trains one model per walk-forward fold and averages, so a
single lucky split can't carry a verdict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("lobpred.evaluate")


# ── metrics ─────────────────────────────────────────────────


@dataclass
class RegMetrics:
    r2: float
    corr: float
    hit_rate: float
    weighted_hit: float
    n: int


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegMetrics:
    """Score forward-change predictions (R², correlation, sign hit-rates)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 0 else 0.0

    nonzero = y_true != 0
    sign_ok = np.sign(y_pred) == np.sign(y_true)
    hit = float(sign_ok[nonzero].mean()) if nonzero.any() else float("nan")
    w = np.abs(y_true)
    weighted_hit = (
        float((sign_ok * w)[nonzero].sum() / w[nonzero].sum()) if nonzero.any() else float("nan")
    )
    return RegMetrics(r2, corr, hit, weighted_hit, len(y_true))


@dataclass
class ClsMetrics:
    accuracy: float
    macro_f1: float
    hit_rate: float       # directional accuracy on directional-truth rows
    weighted_hit: float
    n: int


def classification_metrics(y_cls_true: np.ndarray, probs: np.ndarray,
                           y_fwd_true: np.ndarray) -> ClsMetrics:
    """Score a 3-class sign model.

    ``probs`` is (N,3) over classes [-1, 0, +1] (sorted order). ``hit_rate``
    is directional accuracy on rows the model called directional (argmax≠0)
    with a nonzero true move; ``weighted_hit`` weights by |move|.
    """
    from sklearn.metrics import f1_score
    y_cls_true = np.asarray(y_cls_true)
    pred = np.argmax(probs, axis=1) - 1                # {-1,0,+1}
    acc = float((pred == y_cls_true).mean())
    f1 = float(f1_score(y_cls_true, pred, average="macro", labels=[-1, 0, 1], zero_division=0))

    directional = pred != 0
    nz = directional & (y_fwd_true != 0)
    hit = float((np.sign(y_fwd_true[nz]) == pred[nz]).mean()) if nz.any() else float("nan")
    w = np.abs(y_fwd_true)
    whit = (
        float(((np.sign(y_fwd_true) == pred) * w)[nz].sum() / w[nz].sum())
        if nz.any() else float("nan")
    )
    return ClsMetrics(acc, f1, hit, whit, len(pred))


# ── torch training ──────────────────────────────────────────


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 4
    val_frac: float = 0.15      # tail of train (chronological) for early stop
    device: str | None = None   # autodetect mps/cuda/cpu
    seed: int = 0
    max_grad_norm: float = 1.0  # gradient clipping (RNN stability; 0 disables)
    loss: str = "mse"           # regression loss: "mse" or "huber" (robust to tails)
    huber_delta: float = 1.0    # Huber transition point (in standardized-target units)
    target_clip: float | None = None  # winsorize standardized target to ±this (tames jumps)
    class_weight: str | None = None   # 'balanced' => inverse-freq CE weights (cls only);
    #                                   counters majority-class collapse on imbalanced labels


def _device(cfg: TrainConfig):
    import torch
    if cfg.device:
        return torch.device(cfg.device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _grad_global_norm(model) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum())
    return sq ** 0.5


def train_torch(model, Xtr, ytr, Xte, cfg: TrainConfig, *,
                task: str = "reg", collect_history: bool = True):
    """Train with chronological early-stop, gradient clipping, and (for
    regression) target standardization.

    ``task='reg'``, MSE on a **standardized** target: the net trains on
    ``(Δ−μ)/σ`` (train stats) and predictions are inverted to original units
    before return, so the net optimizes in unit-variance space (matching its
    init/LR) while metrics stay in price units. This is the output-side
    mirror of input scaling, invertible, no distributional assumption.

    ``task='cls'``, cross-entropy on a 3-class sign label; ``ytr`` must be
    int in {0,1,2}. Returns class **probabilities** (N,3).

    Gradients are clipped to ``cfg.max_grad_norm`` (RNN stability). The
    history records the *true* (pre-clip) grad norm so a diagnostic still
    shows any instability. The last ``val_frac`` of the time-sorted train
    block is the early-stop validation set, never shuffled across the
    boundary, so selection can't leak the future. Returns
    ``(preds_or_probs, model, history)``.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(cfg.seed)
    dev = _device(cfg)
    model = model.to(dev)
    is_cls = task == "cls"

    if is_cls:
        ymu, ysd = 0.0, 1.0
        y_use = ytr.astype(np.int64)
        if cfg.class_weight == "balanced":
            # sklearn 'balanced' weights: w_c = N / (K * count_c). Up-weights
            # rare directional classes so the net can't minimize loss by
            # collapsing to the majority ("stable") class.
            counts = np.bincount(y_use, minlength=int(y_use.max()) + 1)
            w = len(y_use) / (len(counts) * np.maximum(counts, 1))
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(w, dtype=torch.float32, device=dev))
        elif cfg.class_weight is not None:
            raise ValueError(f"unknown class_weight {cfg.class_weight!r}; use 'balanced' or None")
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
    else:
        ymu = float(np.mean(ytr))
        ysd = float(np.std(ytr)) or 1.0
        ysd = ysd if ysd > 1e-12 else 1.0
        y_use = ((ytr - ymu) / ysd).astype(np.float32)
        if cfg.target_clip is not None:                 # winsorize standardized target
            y_use = np.clip(y_use, -cfg.target_clip, cfg.target_clip).astype(np.float32)
        loss_fn = torch.nn.HuberLoss(delta=cfg.huber_delta) if cfg.loss == "huber" else torch.nn.MSELoss()

    n = len(Xtr)
    n_val = max(1, int(cfg.val_frac * n))
    tr_sl, val_sl = slice(0, n - n_val), slice(n - n_val, n)

    def _loader(X, y, shuffle):
        yt = torch.from_numpy(y).long() if is_cls else torch.from_numpy(y).float()
        ds = TensorDataset(torch.from_numpy(X).float(), yt)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle)

    tr_dl = _loader(Xtr[tr_sl], y_use[tr_sl], True)
    va_dl = _loader(Xtr[val_sl], y_use[val_sl], False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val, best_state, bad = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": [], "grad_norm_mean": [],
               "grad_norm_max": [], "grad_steps": []}

    def _fwd_loss(xb, yb):
        out = model(xb)
        return loss_fn(out if is_cls else out.squeeze(-1), yb)

    for ep in range(cfg.epochs):
        model.train()
        ttot, tn, gnorms = 0.0, 0, []
        for xb, yb in tr_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = _fwd_loss(xb, yb)
            loss.backward()
            if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                g = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm))
            else:
                g = _grad_global_norm(model)
            if collect_history:
                gnorms.append(g); history["grad_steps"].append(g)
            opt.step()
            ttot += loss.item() * len(yb); tn += len(yb)
        model.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(dev), yb.to(dev)
                vtot += float(_fwd_loss(xb, yb)) * len(yb); vn += len(yb)
        vloss = vtot / max(vn, 1)
        if collect_history:
            history["train_loss"].append(ttot / max(tn, 1))
            history["val_loss"].append(vloss)
            history["grad_norm_mean"].append(float(np.mean(gnorms)) if gnorms else 0.0)
            history["grad_norm_max"].append(float(np.max(gnorms)) if gnorms else 0.0)
        if vloss < best_val - 1e-12:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                logger.info("early stop at epoch %d (val loss %.3e)", ep, best_val)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    outs = []
    te_dl = _loader(Xte, np.zeros(len(Xte), np.int64 if is_cls else np.float32), False)
    with torch.no_grad():
        for xb, _ in te_dl:
            o = model(xb.to(dev))
            if is_cls:
                outs.append(torch.softmax(o, dim=1).cpu().numpy())
            else:
                outs.append(o.squeeze(-1).cpu().numpy() * ysd + ymu)
    return np.concatenate(outs), model, history


# ── gap table ───────────────────────────────────────────────


@dataclass
class PhaseResult:
    phase: str
    model: str
    metrics: RegMetrics


@dataclass
class ClsPhaseResult:
    phase: str
    model: str
    metrics: ClsMetrics


def format_gap_table(results: list[PhaseResult]) -> str:
    """Regression attribution table (one row per phase×model)."""
    hdr = f"{'phase':<10} {'model':<12} {'R2':>8} {'corr':>7} {'hit':>6} {'wHit':>6} {'n':>9}"
    lines = ["REGRESSION", hdr, "-" * len(hdr)]
    for r in results:
        m = r.metrics
        lines.append(
            f"{r.phase:<10} {r.model:<12} {m.r2:>8.4f} {m.corr:>7.3f} "
            f"{m.hit_rate:>6.3f} {m.weighted_hit:>6.3f} {m.n:>9d}"
        )
    return "\n".join(lines)


def format_cls_table(results: list[ClsPhaseResult]) -> str:
    """Classification attribution table (one row per phase×model)."""
    hdr = f"{'phase':<10} {'model':<12} {'acc':>7} {'maF1':>7} {'hit':>6} {'wHit':>6} {'n':>9}"
    lines = ["CLASSIFICATION", hdr, "-" * len(hdr)]
    for r in results:
        m = r.metrics
        lines.append(
            f"{r.phase:<10} {r.model:<12} {m.accuracy:>7.3f} {m.macro_f1:>7.3f} "
            f"{m.hit_rate:>6.3f} {m.weighted_hit:>6.3f} {m.n:>9d}"
        )
    return "\n".join(lines)
