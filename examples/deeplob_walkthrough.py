"""DeepLOB walkthrough: a self-contained teaching example.

Run (from the repo root, with the env that has torch + lightgbm):

    PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE python examples/deeplob_walkthrough.py

It generates a synthetic limit order book with a KNOWN planted signal, builds
a PER-LEVEL STATIONARY representation (OBI / OFI / rel-size per book level,
the input real LOB deep models use), trains a PerLevelLOB (convolves
across levels + LSTM over time), prints the learning curve, and compares
against a LightGBM baseline on the same data.

The point is to learn the moving parts of deep-learning-on-an-order-book:
  1. how the raw book becomes a (sequence, features) tensor,
  2. leak-safe forward labels + train-only normalization,
  3. the training loop: target standardization, early stopping, grad clipping,
  4. the lesson: at this data scale a simple model matches/beats the deep one,
     and *more epochs makes the deep model worse* (overfitting).

Everything runs on synthetic data in ~1 minute with no downloads. Swap the data
loader for `lobpred.data.fi2010` to repeat on the standard FI-2010 benchmark.
"""

from __future__ import annotations

import numpy as np

from lobpred import baselines as B
from lobpred import dataset as D
from lobpred import evaluate as E
from lobpred import models as M

SEQ_LEN = 32          # how many past book updates the model sees (event-time)
HORIZON_EVENTS = 20   # predict the microprice change 20 events ahead
N_FOLDS = 3


def main() -> None:
    # ── 1. DATA ────────────────────────────────────────────────────────
    # A synthetic book whose next move follows noisy order-flow
    # imbalance, so there IS a real signal to find, and we know its source.
    print("1) generating synthetic LOB (known planted signal) ...")
    from lobpred.data import synthetic
    pool = synthetic.generate_pool(n_markets=10, minutes=120, seed=0, kappa=1.0)

    # ── 2. FEATURES: PER-LEVEL STATIONARY TENSOR (the fair DL input) ────
    # For each book level ℓ: OBI_ℓ (size imbalance), OFI_ℓ (order flow),
    # rel_size_ℓ (share of depth), all STATIONARY (so pooling across markets
    # is valid) but with the per-level STRUCTURE kept. PerLevelLOB reshapes the
    # flat (T, L*3) window to (T, L, 3) and convolves across levels. This is
    # what real LOB deep models use (DeepLOB; Kolm et al. multi-level OFI),
    # neither raw prices nor flat hand-crafted scalars.
    from lobpred import features as Fx
    pool, feats = Fx.add_perlevel_features(pool)
    nlev = len(feats) // Fx.PERLEVEL_CHANNELS
    print(f"   per-level features: {len(feats)} = {nlev} levels x {Fx.PERLEVEL_CHANNELS} channels")
    print(f"   (level-major: L1={feats[:3]}  L2={feats[3:6]} ...)")

    # ── 3. LABEL + WINDOWS ──────────────────────────────────────────────
    # Forward mid-price change (leak-safe: searched forward in time).
    pool = D.add_forward_target(pool, horizon_events=HORIZON_EVENTS, price_col="mid")
    pool = D.add_sign_label(pool, alpha=0.0)

    # Walk-forward split by wall-clock with an embargo, train-only z-scoring,
    # then build per-market causal windows of the last SEQ_LEN updates.
    embargo_s = 2.0
    folds = D.walk_forward_splits(pool["timestamp_ns"].to_numpy(), n_folds=N_FOLDS, horizon_s=embargo_s)
    tr0, _ = folds[0]
    mask = np.zeros(pool.height, bool); mask[tr0] = True
    scaler = D.fit_scaler(pool, feats, mask)                  # fit on TRAIN ROWS ONLY
    W = D.make_windows(pool, feats, seq_len=SEQ_LEN, scaler=scaler)
    wf = D.walk_forward_splits(W.ts, n_folds=N_FOLDS, horizon_s=embargo_s)
    tr, te = wf[-1]                                            # last fold: most train history
    Xtr, ytr, Xte, yte = W.X[tr], W.y[tr], W.X[te], W.y[te]
    print(f"   windows: train={len(Xtr):,}  test={len(Xte):,}  shape={Xtr.shape[1:]}")

    # ── 4. PerLevelLOB ──────────────────────────────────────────────────
    # Reshapes (T, L*3) → (T, L, 3), convolves ACROSS LEVELS at each timestep
    # (learning cross-level patterns a tree can't), collapses levels, then
    # LSTM over time. train_torch does the real work: it z-scores the TARGET
    # (so the net optimizes in unit variance, then inverts), early-stops on a
    # chronological validation tail, and clips gradients.
    print("\n2) training PerLevelLOB (conv-across-levels + LSTM) ...")
    mdl = M.build_model("perlevel", n_features=len(feats), seq_len=SEQ_LEN, out_dim=1)
    preds, mdl, hist = E.train_torch(
        mdl, Xtr, ytr, Xte, E.TrainConfig(epochs=25, patience=6, lr=3e-4),
        task="reg", collect_history=True,
    )
    print(f"   {'epoch':>5} {'train_loss':>11} {'val_loss':>10}")
    for ep, (tl, vl) in enumerate(zip(hist["train_loss"], hist["val_loss"]), 1):
        print(f"   {ep:>5} {tl:>11.4f} {vl:>10.4f}")
    deep = E.regression_metrics(yte, preds)

    # ── 5. BASELINE: LightGBM on the same windows ───────────────────────
    # The tree sees only the last snapshot of each window (last_step), yet
    # that is often enough to match or beat the deep net at this data scale.
    print("\n3) training LightGBM baseline ...")
    p, _ = B.lgbm_fit_predict(Xtr, ytr, Xte)
    gbm = E.regression_metrics(yte, p)

    # ── 6. THE LESSON ───────────────────────────────────────────────────
    print("\n=== RESULT (synthetic; predicting forward mid move) ===")
    print(f"{'model':<12} {'corr':>7} {'hit':>7} {'R2':>8}")
    print(f"{'PerLevelLOB':<12} {deep.corr:>7.3f} {deep.hit_rate:>7.3f} {deep.r2:>8.3f}")
    print(f"{'lgbm':<12} {gbm.corr:>7.3f} {gbm.hit_rate:>7.3f} {gbm.r2:>8.3f}")
    print(
        "\nWhat to take away:\n"
        "  • Both should find the planted signal (corr > 0), so the pipeline works,\n"
        "    and PerLevelLOB gets the structured per-level book.\n"
        "  • At this scale the simple model still matches/beats the deep one. That\n"
        "    is the expected tabular-regime result. The right input\n"
        "    representation helps the net, but doesn't beat the data-scale wall.\n"
        "  • Try epochs=80: PerLevelLOB's val_loss will start RISING while\n"
        "    train_loss keeps falling, which is overfitting. More compute ≠ more signal.\n"
        "  • To give DL a real edge you need (a) far more data (~1e7+ windows)\n"
        "    AND (b) this structured input across MANY instruments, so it learns\n"
        "    cross-level + cross-asset structure a tree can't. See the README.\n"
        "  • Swap synthetic -> lobpred.data.fi2010 to repeat on the FI-2010 benchmark."
    )


if __name__ == "__main__":
    main()
