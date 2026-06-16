"""Target-design study: point-exit vs smoothed-hold, across horizons.

Answers two questions on the price-active dataset:

  1. Is a *holdable* target more predictable than the next move? For each
     horizon W we compare a POINT target (microprice at t+W minus now, exit
     at one instant) against a SMOOTHED target (mean microprice over (t, t+W]
     minus now, hold ~W s and exit on a TWAP). Smoothing reduces noise
     and matches a hold-for-seconds view, so we expect it to be more learnable.
  2. How much of the headline 3-class accuracy is the LABEL + class balancing?
     ``--balance`` downsamples the Stable class so the 3-class accuracy is
     comparable across studies (otherwise Stable dominates and inflates it).

Reports, per (horizon, target, model): regression corr/hit and 3-class
acc/macro-F1. lgbm = strong simple baseline; tcn = a deep model.

    python -m lobpred.target_study --roots experiments/synthetic --horizons 2 5 10 30
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from . import baselines as B
from . import dataset as D
from . import evaluate as E
from . import features as Fx
from . import models as M

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("lobpred.target_study")


def _prep(args):
    roots = tuple(Path(r) for r in args.roots)
    pool = D.load_pool(D.LoadConfig(roots=roots, min_rows=args.min_rows))
    pool, gcols = D.add_grid_features(pool, k=args.grid_k)
    pool, scols = D.add_scalar_features(pool)
    feats = list(gcols + scols)
    if not args.base_features_only:
        pool, fcols = Fx.add_flow_features(pool)
        pool, hcols = Fx.add_history_features(pool)
        pool, pcols = Fx.add_shape_features(pool)
        pool, ecols = Fx.add_activity_spread_features(pool)
        pool, icols = Fx.add_impact_features(pool)
        pool, smcols = Fx.add_smoothed_features(pool)
        feats += fcols + hcols + pcols + ecols + icols + smcols
    if args.with_trades:
        trades = D.load_trades(roots)
        if trades.height:
            pool, tcols = Fx.add_trade_features(pool, trades)
            feats += tcols
    pool = D.add_activity_tier(pool, n_tiers=args.n_tiers, min_mid_moves=args.min_mid_moves)
    pool = pool.filter(pool["act_tier"] == args.n_tiers - 1)
    logger.info("active tier: %d markets, %d features", pool["market_id"].n_unique(), len(feats))
    return pool, feats


def _balance_idx(y_cls: np.ndarray, ts: np.ndarray, seed: int) -> np.ndarray:
    """Downsample the Stable (0) class to the larger directional class (paper-style)."""
    rng = np.random.default_rng(seed)
    up, dn, st = np.where(y_cls == 1)[0], np.where(y_cls == -1)[0], np.where(y_cls == 0)[0]
    keep_stable = min(len(st), max(len(up), len(dn)))
    st_keep = rng.choice(st, keep_stable, replace=False) if keep_stable < len(st) else st
    idx = np.concatenate([up, dn, st_keep])
    return idx[np.argsort(ts[idx])]  # keep chronological order for walk-forward


def _eval_cell(pool, feats, mode, H, args):
    if mode == "point":
        p = D.add_forward_target(pool, horizon_s=H, price_col="microprice", max_stale_s=args.max_stale)
    else:
        p = D.add_forward_target(pool, avg_window_s=H, price_col="microprice")
    p = D.add_sign_label(p, alpha=args.alpha)
    med_wall = float(np.nanmedian(p["fwd_dt_s"].to_numpy()))
    embargo = max(H, 1.0)
    folds = D.walk_forward_splits(p["timestamp_ns"].to_numpy(), n_folds=args.n_folds, horizon_s=embargo)
    tr0, _ = folds[0]
    mask = np.zeros(p.height, bool); mask[tr0] = True
    sc = D.fit_scaler(p, feats, mask)
    W = D.make_windows(p, feats, seq_len=args.seq_len, scaler=sc)
    wf = D.walk_forward_splits(W.ts, n_folds=args.n_folds, horizon_s=embargo)
    acc = {k: [] for k in ("lgbm_corr", "lgbm_hit", "tcn_corr", "tcn_hit",
                            "lgbm_acc", "lgbm_f1", "tcn_acc", "tcn_f1")}
    for tr, te in wf:
        if args.subsample and len(tr) > args.subsample:
            rng = np.random.default_rng(args.seed); tr = rng.choice(tr, args.subsample, replace=False)
        Xtr, ytr, Xte, yte = W.X[tr], W.y[tr], W.X[te], W.y[te]
        # regression
        pr, _ = B.lgbm_fit_predict(Xtr, ytr, Xte)
        m = E.regression_metrics(yte, pr); acc["lgbm_corr"].append(m.corr); acc["lgbm_hit"].append(m.hit_rate)
        mdl = M.build_model("tcn", n_features=len(feats), seq_len=args.seq_len, out_dim=1)
        pr, _, _ = E.train_torch(mdl, Xtr, ytr, Xte, E.TrainConfig(epochs=args.epochs, patience=args.patience),
                                 task="reg", collect_history=False)
        m = E.regression_metrics(yte, pr); acc["tcn_corr"].append(m.corr); acc["tcn_hit"].append(m.hit_rate)
        # classification (optionally Stable-balanced)
        ytr_c, yte_c = W.y_cls[tr], W.y_cls[te]
        if args.balance:
            bi = _balance_idx(ytr_c, W.ts[tr], args.seed); Xc, yc = Xtr[bi], (ytr_c[bi] + 1).astype("int64")
            bte = _balance_idx(yte_c, W.ts[te], args.seed); Xcte, ycte = Xte[bte], yte_c[bte]
        else:
            Xc, yc, Xcte, ycte = Xtr, (ytr_c + 1).astype("int64"), Xte, yte_c
        pr, _ = B.lgbm_cls_proba(Xc, yc - 1, Xcte)
        m = E.classification_metrics(ycte, pr, np.zeros(len(ycte))); acc["lgbm_acc"].append(m.accuracy); acc["lgbm_f1"].append(m.macro_f1)
        mdl = M.build_model("tcn", n_features=len(feats), seq_len=args.seq_len, out_dim=3)
        pr, _, _ = E.train_torch(mdl, Xc, yc, Xcte, E.TrainConfig(epochs=args.epochs, patience=args.patience),
                                 task="cls", collect_history=False)
        m = E.classification_metrics(ycte, pr, np.zeros(len(ycte))); acc["tcn_acc"].append(m.accuracy); acc["tcn_f1"].append(m.macro_f1)
    a = lambda k: float(np.nanmean(acc[k])) if acc[k] else float("nan")
    return {k: a(k) for k in acc}, med_wall, len(W.y)


def main() -> int:
    args = _parse_args()
    pool, feats = _prep(args)
    tag = "BALANCED (paper-style)" if args.balance else "natural distribution"
    print(f"\n=== TARGET STUDY, active tier, {len(feats)} feats, cls={tag}, alpha={args.alpha} ===")
    print(f"{'horizon':>8} {'target':>7} {'wall_s':>7} | {'lgbm_corr':>9} {'tcn_corr':>9} "
          f"{'lgbm_hit':>8} {'tcn_hit':>8} | {'lgbm_acc':>8} {'tcn_acc':>8} {'lgbm_f1':>8} {'tcn_f1':>8}")
    for H in args.horizons:
        for mode in ("point", "smooth"):
            try:
                r, mw, n = _eval_cell(pool, feats, mode, H, args)
                print(f"{H:>8.0f} {mode:>7} {mw:>7.1f} | {r['lgbm_corr']:>9.3f} {r['tcn_corr']:>9.3f} "
                      f"{r['lgbm_hit']:>8.3f} {r['tcn_hit']:>8.3f} | {r['lgbm_acc']:>8.3f} {r['tcn_acc']:>8.3f} "
                      f"{r['lgbm_f1']:>8.3f} {r['tcn_f1']:>8.3f}")
            except Exception as exc:  # noqa: BLE001
                logger.warning("H=%s %s failed: %s", H, mode, exc)
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roots", nargs="+", required=True, help="dataset root dir(s) of canonical parquet")
    p.add_argument("--horizons", nargs="+", type=float, default=[2, 5, 10, 30])
    p.add_argument("--min-mid-moves", type=int, default=10)
    p.add_argument("--n-tiers", type=int, default=3)
    p.add_argument("--grid-k", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--min-rows", type=int, default=2000)
    p.add_argument("--max-stale", type=float, default=120.0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--alpha", type=float, default=0.002)
    p.add_argument("--subsample", type=int, default=60000)
    p.add_argument("--balance", action="store_true", help="downsample Stable for the cls metric (paper-style)")
    p.add_argument("--with-trades", action="store_true", help="add trade-tape features (needs *.trades.parquet)")
    p.add_argument("--base-features-only", action="store_true", help="grid+scalar only (skip flow/history/shape)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
