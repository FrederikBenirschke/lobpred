"""Train DL models on one phase WITH history, emit diagnostic plots.

    python -m lobpred.diagnose --roots experiments/synthetic \
      --phase 3 --horizon 30 --seq-len 24 --epochs 12 --out experiments/diag

Writes, under --out:
  training_curves.png        loss + gradient norms, all DL models
  error_<model>.png          residuals / pred-vs-actual / error-vs-move / calibration
  attention_pockets.png      attention-by-book-depth (the pockets view)

Separate from run.py (the gap-table CLI) so the heavy history capture and
plotting don't slow the headline feature comparison. Requires the ``plot`` extra
(matplotlib).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from . import dataset as D
from . import diagnostics as G
from . import evaluate as E
from . import models as M
from .run import _build_phase

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("lobpred.diagnose")


def main() -> int:
    args = _parse_args()
    out = Path(args.out)
    roots = tuple(Path(r) for r in args.roots)
    pool_base = D.load_pool(D.LoadConfig(roots=roots, min_rows=args.min_rows))
    pool_base = D.add_activity_tier(pool_base, n_tiers=args.n_tiers, min_mid_moves=args.min_mid_moves)
    pool_base = pool_base.filter(pool_base["act_tier"] == args.n_tiers - 1)
    pool, feats = _build_phase(pool_base, args.phase, args)
    logger.info("phase %s: %d rows, %d features", args.phase, pool.height, len(feats))

    folds = D.walk_forward_splits(pool["timestamp_ns"].to_numpy(),
                                  n_folds=args.n_folds, horizon_s=args.horizon)
    tr0, _ = folds[0]
    mask = np.zeros(pool.height, bool); mask[tr0] = True
    sc = D.fit_scaler(pool, feats, mask)
    W = D.make_windows(pool, feats, seq_len=args.seq_len, scaler=sc)
    wf = D.walk_forward_splits(W.ts, n_folds=args.n_folds, horizon_s=args.horizon)
    tr, te = wf[-1]
    if args.subsample and len(tr) > args.subsample:
        rng = np.random.default_rng(args.seed); tr = rng.choice(tr, args.subsample, replace=False)
    Xtr, ytr, Xte, yte = W.X[tr], W.y[tr], W.X[te], W.y[te]
    logger.info("train %d  test %d", len(Xtr), len(Xte))

    tc = E.TrainConfig(epochs=args.epochs, batch_size=args.batch_size)
    histories: dict[str, dict] = {}
    attn_model = None
    print(f"\n{'model':<12} {'R2':>8} {'corr':>7} {'hit':>6} {'wHit':>6}")
    for name in args.models:
        mdl = M.build_model(name, n_features=len(feats), seq_len=args.seq_len, out_dim=1)
        preds, mdl, hist = E.train_torch(mdl, Xtr, ytr, Xte, tc, collect_history=True)
        histories[name] = hist
        m = E.regression_metrics(yte, preds)
        print(f"{name:<12} {m.r2:>8.4f} {m.corr:>7.3f} {m.hit_rate:>6.3f} {m.weighted_hit:>6.3f}")
        G.plot_error_analysis(yte, preds, out / f"error_{name}.png",
                              title=f"phase{args.phase} {name}")
        if name in ("attention", "axial", "axialattentionlob"):
            attn_model = mdl

    p1 = G.plot_training_curves(histories, out / "training_curves.png",
                                title=f"phase{args.phase} (h={args.horizon}s, seq={args.seq_len})")
    logger.info("wrote %s", p1)

    if attn_model is not None:
        import torch
        dev = E._device(tc)
        attn_model.eval()
        with torch.no_grad():
            attn_model(torch.from_numpy(Xte[: args.attn_sample]).float().to(dev))
        attn = attn_model.last_feature_attn.mean(0).cpu().numpy()
        p2 = G.plot_attention_pockets(attn, feats, out / "attention_pockets.png",
                                      title=f"phase{args.phase} attention", grid_k=args.grid_k)
        logger.info("wrote %s", p2)

    print(f"\nplots written to {out}/")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roots", nargs="+", required=True, help="dataset root dir(s) of canonical parquet")
    p.add_argument("--phase", default="3", choices=["0", "1", "2", "3", "4"])
    p.add_argument("--min-mid-moves", type=int, default=10)
    p.add_argument("--n-tiers", type=int, default=3)
    p.add_argument("--horizon", type=float, default=30.0)
    p.add_argument("--max-stale", type=float, default=120.0)
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--grid-k", type=int, default=20)
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--min-rows", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--alpha", type=float, default=0.002)
    p.add_argument("--subsample", type=int, default=100000)
    p.add_argument("--attn-sample", type=int, default=4096)
    p.add_argument("--models", nargs="+", default=["tcn", "deeplob", "attention"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="experiments/diag")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
