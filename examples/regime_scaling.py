"""Regime-scaling demo: does MORE DATA let the deep net catch the tree?

The published DeepLOB result (FI-2010: ~1e6+ samples of the RAW 10-level
book) shows a CNN-LSTM beating classical baselines. This study found the
OPPOSITE on medium-scale, *engineered* (stationary per-level) features:
LightGBM wins. Two things differ between those regimes: the data SCALE and
the input REPRESENTATION (raw price/size vs engineered stationary features).

This script isolates the **scale** axis on shippable synthetic data: it
sweeps the training size and reports deep (SeqLSTM) vs tree (LightGBM)
correlation at each N, on the *same* per-level features / target /
walk-forward split. If the deep net closes the gap as N grows, scale is part
of the story; if the gap stays flat, scale alone is not enough on engineered
features, which points at the *other* axis (raw input at large scale, i.e.
the FI-2010 regime; see the README's regime-boundary section).

Fairness: the tree is given the FULL flattened window (so it has the same
history the LSTM sees) rather than the last snapshot. We don't hand the net an
edge that is history access. The snapshot-only tree is shown too.

Runs on the synthetic generator (no downloads). Full sweep ~30-40 min:

    PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE python examples/regime_scaling.py
"""

from __future__ import annotations

import logging

import numpy as np

from lobpred import baselines as B
from lobpred import dataset as D
from lobpred import evaluate as E
from lobpred import features as Fx
from lobpred import models as M
from lobpred.data import synthetic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("regime")

SEQ_LEN, HORIZON_EVENTS, N_FOLDS, EMBARGO_S = 32, 20, 3, 2.0
SIZES = [10_000, 40_000, 160_000, 640_000, None]   # None -> all available train
SEEDS = [0, 1]


def tree_window(Xtr, ytr, Xte):
    """LightGBM on the FULL flattened window (history-aware tree baseline)."""
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                          subsample=0.8, colsample_bytree=0.8, min_child_samples=100,
                          n_jobs=-1, verbosity=-1)
    m.fit(Xtr.reshape(len(Xtr), -1), ytr)
    return m.predict(Xte.reshape(len(Xte), -1))


def main() -> int:
    import torch

    # 1) a large synthetic pool (planted AR(1) imbalance signal -> next move)
    log.info("generating synthetic pool ...")
    pool = synthetic.generate_pool(n_markets=40, minutes=900.0, base_rate_per_min=60.0,
                                   kappa=0.8, seed=0)
    pool, feats = Fx.add_perlevel_features(pool)
    pool = D.add_forward_target(pool, horizon_events=HORIZON_EVENTS, price_col="mid")

    folds = D.walk_forward_splits(pool["timestamp_ns"].to_numpy(), n_folds=N_FOLDS, horizon_s=EMBARGO_S)
    tr0, _ = folds[0]
    mask = np.zeros(pool.height, bool); mask[tr0] = True
    scaler = D.fit_scaler(pool, feats, mask)                       # fit on TRAIN ROWS ONLY
    W = D.make_windows(pool, feats, seq_len=SEQ_LEN, scaler=scaler)
    del pool
    wf = D.walk_forward_splits(W.ts, n_folds=N_FOLDS, horizon_s=EMBARGO_S)
    tr_all, te = wf[-1]                                            # last fold: most train history
    Xte, yte = W.X[te], W.y[te]
    rng = np.random.default_rng(0)
    log.info("windows=%d  train_pool=%d  test=%d  feat_dim=%s",
             len(W.y), len(tr_all), len(te), tuple(W.X.shape[1:]))

    cfg = dict(epochs=25, patience=6, lr=1e-3, weight_decay=1e-4,
               loss="huber", huber_delta=1.0, target_clip=5.0)
    sizes = sorted({min(s or len(tr_all), len(tr_all)) for s in SIZES})
    rows = []
    for N in sizes:
        tr = np.sort(rng.choice(tr_all, N, replace=False)) if N < len(tr_all) else tr_all
        Xtr, ytr = W.X[tr], W.y[tr]
        snap = E.regression_metrics(yte, B.lgbm_fit_predict(Xtr, ytr, Xte)[0]).corr
        win = E.regression_metrics(yte, tree_window(Xtr, ytr, Xte)).corr
        preds = []
        for s in SEEDS:
            torch.manual_seed(s)
            mdl = M.build_model("seqlstm", n_features=len(feats), seq_len=SEQ_LEN, out_dim=1)
            pr, _, _ = E.train_torch(mdl, Xtr, ytr, Xte, E.TrainConfig(seed=s, **cfg),
                                     task="reg", collect_history=False)
            preds.append(pr)
        net_mean = float(np.mean([E.regression_metrics(yte, p).corr for p in preds]))
        net_ens = E.regression_metrics(yte, np.mean(np.stack(preds), axis=0)).corr
        rows.append((N, snap, win, net_mean, net_ens))
        log.info("N=%-8d lgbm_snap=%.3f lgbm_win=%.3f seqlstm=%.3f ens=%.3f",
                 N, snap, win, net_mean, net_ens)

    print("\n=== REGIME SCALING (synthetic; deep vs tree as data grows) ===")
    print(f"{'N_train':>9} {'lgbm_snap':>10} {'lgbm_win':>9} {'seqlstm':>8} {'ens':>7} {'gap(win-ens)':>13}")
    for N, sp, wn, nm, en in rows:
        print(f"{N:>9} {sp:>10.3f} {wn:>9.3f} {nm:>8.3f} {en:>7.3f} {wn - en:>13.3f}")
    g0, g1 = rows[0][2] - rows[0][4], rows[-1][2] - rows[-1][4]
    print(f"\ngap (lgbm_win - seqlstm_ens) at smallest N: {g0:+.3f}   at largest N: {g1:+.3f}")
    print("READING:", "scale CLOSES the gap; deep benefits from more data"
          if g1 < g0 - 0.02 else
          "gap ~FLAT with scale; on engineered features, more data alone does not "
          "rescue the deep net. The DL win in the literature (FI-2010) comes from "
          "raw book input at much larger scale.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
