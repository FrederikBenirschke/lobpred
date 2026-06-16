"""Run the staged feature comparison and print the attribution gap table.

    python -m lobpred.run --roots experiments/synthetic \
        --phases 0 1 2 3 --horizon 30 --seq-len 24 --n-folds 3 --epochs 12

Each phase changes ONE thing vs the previous, so any accuracy gain
is attributable. The gap table is the deliverable.
The feature set is a real axis, so the reference paper's price-levels run
goes head-to-head with the cross-instrument grid:

  phase 0 : target=mid,        features=levels   (the reference paper's inputs)
  phase 1 : target=microprice, features=levels   (isolates the target)
  phase 2 : target=microprice, features=grid     (cross-instrument grid)
  phase 3 : target=microprice, features=grid+scalar (+ flow: OFI/imbalance)

Set ``KMP_DUPLICATE_LIB_OK=TRUE`` (torch + lightgbm both link libomp).
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
logger = logging.getLogger("lobpred.run")

_PHASES = {
    "0": ("mid", "levels"),                # reference paper: standardized price levels
    "1": ("microprice", "levels"),         # + microprice target
    "2": ("microprice", "grid"),           # cross-instrument grid representation
    "3": ("microprice", "grid+scalar"),    # + flow scalars
    "4": ("microprice", "extended"),       # + flow / return-vol / book-shape / activity-spread
    "5": ("microprice", "extended+trades"),# + trade-tape features
}


def _build_phase(pool_base, phase: str, args, trades=None):
    """Return (pool_with_features_and_target, feature_cols) for a phase."""
    price_col, fset = _PHASES[phase]
    pool = pool_base
    feats: list[str] = []
    if fset == "levels":
        pool, feats = D.add_paper_features(pool)
    else:  # grid-based sets
        pool, gcols = D.add_grid_features(pool, k=args.grid_k)
        feats = list(gcols)
    # scalars carry microprice/ofi; add when requested OR when the target is
    # microprice (which add_scalar_features computes).
    if fset != "levels" or price_col == "microprice":
        pool, scols = D.add_scalar_features(pool)
        if fset != "levels":
            feats += scols
    if fset in ("extended", "extended+trades"):
        pool, fcols = Fx.add_flow_features(pool)
        pool, hcols = Fx.add_history_features(pool)
        pool, pcols = Fx.add_shape_features(pool)
        pool, ecols = Fx.add_activity_spread_features(pool)
        pool, icols = Fx.add_impact_features(pool)
        pool, smcols = Fx.add_smoothed_features(pool)
        feats += fcols + hcols + pcols + ecols + icols + smcols
    if fset == "extended+trades":
        if trades is None or trades.height == 0:
            raise ValueError("phase 5 needs trades; none found under roots (*.trades.parquet)")
        pool, tcols = Fx.add_trade_features(pool, trades)
        feats += tcols
    pool = D.add_forward_target(pool, horizon_s=args.horizon, price_col=price_col, max_stale_s=args.max_stale)
    pool = D.add_sign_label(pool, alpha=args.alpha)  # for the classification track
    return pool, feats


# per-model LR overrides (DeepLOB's LSTM wants a gentler step)
_MODEL_LR = {"deeplob": 3e-4}


def _cfg_for(name: str, args) -> E.TrainConfig:
    return E.TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                         lr=_MODEL_LR.get(name, 1e-3), patience=args.patience)


def _run_fold(W, tr, te, feats, args):
    """Return (reg_metrics_by_model, cls_metrics_by_model) for one fold."""
    if args.subsample and len(tr) > args.subsample:
        rng = np.random.default_rng(args.seed)
        tr = rng.choice(tr, size=args.subsample, replace=False)
    Xtr, Xte = W.X[tr], W.X[te]
    ytr, yte = W.y[tr], W.y[te]
    reg: dict[str, E.RegMetrics] = {}
    cls: dict[str, E.ClsMetrics] = {}
    dl_models = [m for m in ("tcn", "deeplob", "attention") if m in args.models]

    if "reg" in args.tasks:
        reg["persistence"] = E.regression_metrics(yte, B.persistence_predict(len(yte)))
        p, _ = B.ridge_fit_predict(Xtr, ytr, Xte)
        reg["ridge"] = E.regression_metrics(yte, p)
        if "lgbm" in args.models:
            p, _ = B.lgbm_fit_predict(Xtr, ytr, Xte)
            reg["lgbm"] = E.regression_metrics(yte, p)
        for name in dl_models:
            mdl = M.build_model(name, n_features=len(feats), seq_len=args.seq_len, out_dim=1)
            p, _, _ = E.train_torch(mdl, Xtr, ytr, Xte, _cfg_for(name, args),
                                    task="reg", collect_history=False)
            reg[name] = E.regression_metrics(yte, p)

    if "cls" in args.tasks and W.y_cls is not None:
        ytr_c = W.y_cls[tr]                 # labels {-1,0,+1}
        yte_c = W.y_cls[te]
        cls["majority"] = E.classification_metrics(yte_c, B.majority_proba(ytr_c, len(te)), yte)
        pr, _ = B.logistic_proba(Xtr, ytr_c, Xte)
        cls["logistic"] = E.classification_metrics(yte_c, pr, yte)
        if "lgbm" in args.models:
            pr, _ = B.lgbm_cls_proba(Xtr, ytr_c, Xte)
            cls["lgbm"] = E.classification_metrics(yte_c, pr, yte)
        for name in dl_models:
            mdl = M.build_model(name, n_features=len(feats), seq_len=args.seq_len, out_dim=3)
            pr, _, _ = E.train_torch(mdl, Xtr, (ytr_c + 1).astype("int64"), Xte,
                                     _cfg_for(name, args), task="cls", collect_history=False)
            cls[name] = E.classification_metrics(yte_c, pr, yte)
    return reg, cls


def _avg_reg(ms: list[E.RegMetrics]) -> E.RegMetrics:
    a = lambda f: float(np.nanmean([getattr(m, f) for m in ms]))
    return E.RegMetrics(a("r2"), a("corr"), a("hit_rate"), a("weighted_hit"),
                        int(np.nansum([m.n for m in ms])))


def _avg_cls(ms: list[E.ClsMetrics]) -> E.ClsMetrics:
    a = lambda f: float(np.nanmean([getattr(m, f) for m in ms]))
    return E.ClsMetrics(a("accuracy"), a("macro_f1"), a("hit_rate"), a("weighted_hit"),
                        int(np.nansum([m.n for m in ms])))


def main() -> int:
    args = _parse_args()
    roots = tuple(Path(r) for r in args.roots)
    # PRICE-ACTIVE books, top activity tier by mid-move rate, same population
    # selection as analyze.py.
    pool_base = D.load_pool(D.LoadConfig(roots=roots, min_rows=args.min_rows))
    pool_base = D.add_activity_tier(pool_base, n_tiers=args.n_tiers, min_mid_moves=args.min_mid_moves)
    pool_base = pool_base.filter(pool_base["act_tier"] == args.n_tiers - 1)
    logger.info("active tier: %d markets, %.2f mid-moves/min median",
                pool_base["market_id"].n_unique(), float(pool_base["mid_moves_per_min"].median()))

    trades = D.load_trades(roots) if "5" in args.phases else None
    if "5" in args.phases:
        logger.info("loaded %d trades for phase 5", 0 if trades is None else trades.height)

    reg_results: list[E.PhaseResult] = []
    cls_results: list[E.ClsPhaseResult] = []
    for phase in args.phases:
        pool, feats = _build_phase(pool_base, phase, args, trades=trades)
        logger.info("phase %s: %d rows, %d features", phase, pool.height, len(feats))

        folds = D.walk_forward_splits(pool["timestamp_ns"].to_numpy(),
                                      n_folds=args.n_folds, horizon_s=args.horizon)
        tr0, _ = folds[0]
        mask = np.zeros(pool.height, bool); mask[tr0] = True
        sc = D.fit_scaler(pool, feats, mask)
        W = D.make_windows(pool, feats, seq_len=args.seq_len, scaler=sc)
        wfolds = D.walk_forward_splits(W.ts, n_folds=args.n_folds, horizon_s=args.horizon)

        reg_pm: dict[str, list[E.RegMetrics]] = {}
        cls_pm: dict[str, list[E.ClsMetrics]] = {}
        for fi, (tr, te) in enumerate(wfolds):
            logger.info("  phase %s fold %d/%d (train %d, test %d)",
                        phase, fi + 1, len(wfolds), len(tr), len(te))
            reg_out, cls_out = _run_fold(W, tr, te, feats, args)
            for k, v in reg_out.items():
                reg_pm.setdefault(k, []).append(v)
            for k, v in cls_out.items():
                cls_pm.setdefault(k, []).append(v)
        for model, ms in reg_pm.items():
            reg_results.append(E.PhaseResult(f"phase{phase}", model, _avg_reg(ms)))
        for model, ms in cls_pm.items():
            cls_results.append(E.ClsPhaseResult(f"phase{phase}", model, _avg_cls(ms)))

    if reg_results:
        print("\n" + E.format_gap_table(reg_results))
    if cls_results:
        print("\n" + E.format_cls_table(cls_results))
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roots", nargs="+", required=True, help="dataset root dir(s) of canonical parquet")
    p.add_argument("--min-mid-moves", type=int, default=10, help="active-market threshold")
    p.add_argument("--n-tiers", type=int, default=3, help="activity tiers (keeps the top)")
    p.add_argument("--phases", nargs="+", default=["0", "1", "2", "3"], choices=list(_PHASES))
    p.add_argument("--horizon", type=float, default=30.0, help="forward horizon (s)")
    p.add_argument("--max-stale", type=float, default=120.0)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--grid-k", type=int, default=20, help="tick-offsets per side in the price grid")
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--min-rows", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--tasks", nargs="+", default=["reg", "cls"], choices=["reg", "cls"])
    p.add_argument("--alpha", type=float, default=0.0,
                   help="deadband for the 3-class sign label (price units, e.g. 0.002)")
    p.add_argument("--subsample", type=int, default=0, help="cap train windows/fold (0=all)")
    p.add_argument("--models", nargs="+",
                   default=["lgbm", "tcn", "deeplob", "attention"],
                   help="which models to run (persistence+ridge always on)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
