"""Predictive-model analysis: which model, keyed on what.

Fixes the most price-active tier and one event-horizon, then compares
every model on PREDICTIVE metrics only, regression (corr, hit, R²) and
classification (acc, macro-F1, hit). Then attributes the signal to feature
groups via LightGBM gain importance and group permutation importance
(Δcorr when a group's columns are shuffled), the "what is it keying on"
view.

    python -m lobpred.analyze --roots experiments/synthetic --horizon-events 50

(Generate the synthetic data first: ``python -m lobpred.data.synthetic``.)
Set ``KMP_DUPLICATE_LIB_OK=TRUE`` when torch + lightgbm share libomp.
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
logger = logging.getLogger("lobpred.analyze")


def _feature_groups(feats: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(feats):
        if c.startswith("bid_grid_"):
            g = "grid_bid"
        elif c.startswith("ask_grid_"):
            g = "grid_ask"
        elif c.startswith("mg_ewma"):
            g = "smooth_microgap"
        elif c.startswith("obi_ewma"):
            g = "smooth_obi"
        elif c.startswith(("ofi_ewma", "ofidepth_ewma")):
            g = "smooth_ofi"
        elif c.startswith(("kyle_lambda_", "amihud_")):
            g = "price_impact"
        elif c.startswith(("add_rate_", "cancel_rate_")):
            g = "orderflow_decomp"
        elif c.startswith(("tfi_", "trades_", "vol_", "vwapgap_")):
            g = "trade_flow"
        elif c.startswith(("ofi_roll_", "ofi_ev_", "ofi_l")):
            g = "ofi_dynamics"
        elif c == "ofi":
            g = "ofi"
        elif c.startswith(("ret_", "rv_")):
            g = "return/vol"
        elif c.startswith(("mid_moves_", "updates_")):
            g = "activity"
        elif c == "rel_spread" or (c.startswith("spread_") and c != "spread_ticks"):
            g = "spread_dyn"
        elif c.startswith("imbalance_") or c.startswith("obi_roll_"):
            g = "imbalance"
        elif c.startswith(("touch_conc_", "depth_log_ratio", "log_depth_", "micro_gap_l")):
            g = "book_shape"
        elif c in ("micro_gap_ticks", "spread_ticks"):
            g = "spread/micro_gap"
        else:
            g = "other"
        groups.setdefault(g, []).append(i)
    return groups


def main() -> int:
    args = _parse_args()
    roots = tuple(Path(r) for r in args.roots)
    # PRICE-ACTIVE books only, tiered by mid-move rate (not update-rate churn).
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
            logger.info("trade features added (%d trades)", trades.height)
        else:
            logger.warning("--with-trades set but no *.trades.parquet found under roots")
    pool = D.add_activity_tier(pool, n_tiers=args.n_tiers, min_mid_moves=args.min_mid_moves)
    pool = pool.filter(pool["act_tier"] == args.n_tiers - 1)  # most price-active tier
    logger.info("active tier: %d markets, %.2f mid-moves/min median",
                pool["market_id"].n_unique(), float(pool["mid_moves_per_min"].median()))

    pool = D.add_forward_target(pool, horizon_events=args.horizon_events, price_col="microprice")
    pool = D.add_sign_label(pool, alpha=args.alpha)
    med_wall = float(pool["fwd_dt_s"].median())
    embargo = max(med_wall, 1.0)

    folds = D.walk_forward_splits(pool["timestamp_ns"].to_numpy(), n_folds=args.n_folds, horizon_s=embargo)
    tr0, _ = folds[0]
    mask = np.zeros(pool.height, bool); mask[tr0] = True
    sc = D.fit_scaler(pool, feats, mask)
    W = D.make_windows(pool, feats, seq_len=args.seq_len, scaler=sc)
    wf = D.walk_forward_splits(W.ts, n_folds=args.n_folds, horizon_s=embargo)
    logger.info("ev_h=%d → median wall %.1fs, %d windows, %d folds",
                args.horizon_events, med_wall, len(W.y), len(wf))

    # ── model comparison (predictive metrics only) ──
    reg_acc: dict[str, list] = {}
    cls_acc: dict[str, list] = {}
    last_lgbm = None  # keep a fitted lgbm + test slice for importance
    for tr, te in wf:
        if args.subsample and len(tr) > args.subsample:
            rng = np.random.default_rng(args.seed); tr = rng.choice(tr, args.subsample, replace=False)
        Xtr, ytr, Xte, yte = W.X[tr], W.y[tr], W.X[te], W.y[te]
        ytr_c, yte_c = W.y_cls[tr], W.y_cls[te]
        tc = E.TrainConfig(epochs=args.epochs, batch_size=args.batch_size, patience=args.patience)

        p, _ = B.ridge_fit_predict(Xtr, ytr, Xte)
        reg_acc.setdefault("ridge", []).append(E.regression_metrics(yte, p))
        p, lgbm_m = B.lgbm_fit_predict(Xtr, ytr, Xte)
        reg_acc.setdefault("lgbm", []).append(E.regression_metrics(yte, p))
        last_lgbm = (lgbm_m, B.last_step(Xte), yte)
        for name in ("tcn", "deeplob", "attention"):
            if name not in args.models:
                continue
            mdl = M.build_model(name, n_features=len(feats), seq_len=args.seq_len, out_dim=1)
            lr = 3e-4 if name == "deeplob" else 1e-3
            p, _, _ = E.train_torch(mdl, Xtr, ytr, Xte, E.TrainConfig(
                epochs=args.epochs, batch_size=args.batch_size, lr=lr, patience=args.patience),
                task="reg", collect_history=False)
            reg_acc.setdefault(name, []).append(E.regression_metrics(yte, p))

        pr, _ = B.logistic_proba(Xtr, ytr_c, Xte)
        cls_acc.setdefault("logistic", []).append(E.classification_metrics(yte_c, pr, yte))
        pr, _ = B.lgbm_cls_proba(Xtr, ytr_c, Xte)
        cls_acc.setdefault("lgbm", []).append(E.classification_metrics(yte_c, pr, yte))
        for name in ("tcn", "attention"):
            if name not in args.models:
                continue
            mdl = M.build_model(name, n_features=len(feats), seq_len=args.seq_len, out_dim=3)
            p, _, _ = E.train_torch(mdl, Xtr, (ytr_c + 1).astype("int64"), Xte, tc,
                                    task="cls", collect_history=False)
            cls_acc.setdefault(name, []).append(E.classification_metrics(yte_c, p, yte))

    am = lambda ms, f: float(np.nanmean([getattr(m, f) for m in ms]))
    print(f"\n=== MODEL COMPARISON (active tier, ev_h={args.horizon_events} ≈ {med_wall:.1f}s) ===")
    print("REGRESSION (predictive only)")
    print(f"{'model':<12} {'corr':>7} {'hit':>7} {'wHit':>7} {'R2':>9}")
    for m, ms in reg_acc.items():
        print(f"{m:<12} {am(ms,'corr'):>7.3f} {am(ms,'hit_rate'):>7.3f} {am(ms,'weighted_hit'):>7.3f} {am(ms,'r2'):>9.3f}")
    print("CLASSIFICATION (predictive only)")
    print(f"{'model':<12} {'acc':>7} {'maF1':>7} {'hit':>7}")
    for m, ms in cls_acc.items():
        print(f"{m:<12} {am(ms,'accuracy'):>7.3f} {am(ms,'macro_f1'):>7.3f} {am(ms,'hit_rate'):>7.3f}")

    # ── feature-group importance (lgbm) ──
    groups = _feature_groups(feats)
    m, Xte_last, yte = last_lgbm
    base = m.predict(Xte_last)
    base_corr = float(np.corrcoef(base, yte)[0, 1])
    gain = np.asarray(m.feature_importances_, float)
    print(f"\n=== FEATURE-GROUP IMPORTANCE (lgbm, base corr {base_corr:.3f}) ===")
    print(f"{'group':<18} {'gain%':>7} {'perm Δcorr':>11}")
    rng = np.random.default_rng(args.seed)
    rows = []
    for g, idx in groups.items():
        gain_pct = 100 * gain[idx].sum() / gain.sum()
        Xp = Xte_last.copy()
        perm = rng.permutation(len(Xp))
        for j in idx:
            Xp[:, j] = Xp[perm, j]
        permuted = m.predict(Xp)
        dcorr = base_corr - float(np.corrcoef(permuted, yte)[0, 1])
        rows.append((g, gain_pct, dcorr))
    for g, gp, dc in sorted(rows, key=lambda r: -r[2]):
        print(f"{g:<18} {gp:>7.1f} {dc:>11.4f}")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roots", nargs="+", required=True, help="dataset root dir(s) of canonical parquet")
    p.add_argument("--min-mid-moves", type=int, default=10, help="active-market threshold")
    p.add_argument("--horizon-events", type=int, default=50)
    p.add_argument("--n-tiers", type=int, default=3)
    p.add_argument("--grid-k", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--min-rows", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--alpha", type=float, default=0.002)
    p.add_argument("--subsample", type=int, default=80000)
    p.add_argument("--with-trades", action="store_true",
                   help="add trade-tape features (needs *.trades.parquet)")
    p.add_argument("--base-features-only", action="store_true",
                   help="grid+scalar only (skip flow/history/shape)")
    p.add_argument("--models", nargs="+", default=["tcn", "deeplob", "attention"])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
