"""End-to-end smoke tests on synthetic data (no torch, no downloads).

The synthetic generator plants an imbalance-driven signal; these tests
assert the harness (a) runs end-to-end and (b) recovers that planted edge
with a simple baseline. If a model can't find a signal we put there,
something in the pipeline is broken.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from lobpred import baselines as B
from lobpred import dataset as D
from lobpred import evaluate as E
from lobpred.data import fi2010, synthetic


def _windows(pool, feats, *, seq_len=16, horizon_events=20, n_folds=3):
    pool = D.add_forward_target(pool, horizon_events=horizon_events, price_col="microprice")
    pool = D.add_sign_label(pool, alpha=0.0)
    folds = D.walk_forward_splits(pool["timestamp_ns"].to_numpy(), n_folds=n_folds, horizon_s=1.0)
    tr0, _ = folds[0]
    mask = np.zeros(pool.height, bool); mask[tr0] = True
    sc = D.fit_scaler(pool, feats, mask)
    return D.make_windows(pool, feats, seq_len=seq_len, scaler=sc)


def test_synthetic_pool_schema():
    pool = synthetic.generate_pool(n_markets=4, minutes=20, seed=1)
    for c in ("market_id", "timestamp_ns", "bid_price", "ask_price", "mid", "spread",
              "bids_price", "bids_size", "asks_price", "asks_size"):
        assert c in pool.columns
    # books are two-sided and mid is between the touch quotes
    assert (pool["ask_price"] >= pool["bid_price"]).all()
    assert pool["bids_price"].dtype == pl.List(pl.Float64)


def test_activity_tier_filters_and_ranks():
    pool = synthetic.generate_pool(n_markets=10, minutes=40, seed=2)
    tiered = D.add_activity_tier(pool, n_tiers=3, min_mid_moves=5)
    assert "act_tier" in tiered.columns
    assert tiered["act_tier"].min() >= 0
    assert tiered["act_tier"].max() <= 2
    # higher tier => higher median mid-move rate
    rates = (tiered.group_by("act_tier")
             .agg(pl.col("mid_moves_per_min").median())
             .sort("act_tier"))
    r = rates["mid_moves_per_min"].to_list()
    assert r[0] <= r[-1]


def test_forward_target_modes():
    pool = synthetic.generate_pool(n_markets=3, minutes=20, seed=3)
    pool, _ = D.add_scalar_features(pool)
    point = D.add_forward_target(pool, horizon_s=5.0, price_col="microprice")
    smooth = D.add_forward_target(pool, avg_window_s=5.0, price_col="microprice")
    assert "y_fwd" in point.columns and "y_fwd" in smooth.columns
    assert point["y_fwd"].is_finite().all()
    assert smooth["y_fwd"].is_finite().all()


def test_baseline_recovers_planted_signal():
    pool = synthetic.generate_pool(n_markets=8, minutes=60, seed=4, kappa=1.0)
    pool, scols = D.add_scalar_features(pool)
    W = _windows(pool, scols)
    wf = D.walk_forward_splits(W.ts, n_folds=3, horizon_s=1.0)
    corrs = []
    for tr, te in wf:
        pred, _ = B.lgbm_fit_predict(W.X[tr], W.y[tr], W.X[te])
        corrs.append(E.regression_metrics(W.y[te], pred).corr)
    # the planted imbalance signal should be recoverable well above zero
    assert np.nanmean(corrs) > 0.15


def test_perlevel_features_and_model():
    import pytest
    torch = pytest.importorskip("torch")
    from lobpred import features as Fx
    from lobpred import models as M
    pool = synthetic.generate_pool(n_markets=3, minutes=15, seed=7)
    pool, feats = Fx.add_perlevel_features(pool)
    assert len(feats) == D.LEVELS * Fx.PERLEVEL_CHANNELS          # level-major (obi,ofi,relsz)/level
    for c in feats:
        assert pool[c].is_finite().all(), c
    assert pool["pl_obi_1"].min() >= -1.0 - 1e-9 and pool["pl_obi_1"].max() <= 1.0 + 1e-9
    assert (pool["pl_relsz_1"] >= 0).all()
    # PerLevelLOB reshapes the flat (B,T,L*C) window to (B,T,L,C) and runs
    mdl = M.build_model("perlevel", n_features=len(feats), seq_len=8, out_dim=1)
    out = mdl(torch.zeros(4, 8, len(feats)))
    assert tuple(out.shape) == (4, 1)


def test_seqlstm_builds_and_forwards():
    import pytest
    torch = pytest.importorskip("torch")
    from lobpred import models as M
    # SeqLSTM = the no-pool control: plain LSTM over (B,T,F), no level collapse.
    mdl = M.build_model("seqlstm", n_features=15, seq_len=8, out_dim=1)
    assert tuple(mdl(torch.zeros(4, 8, 15)).shape) == (4, 1)
    # alias + 3-class directional head
    clf = M.build_model("lstm", n_features=15, seq_len=8, out_dim=3)
    assert tuple(clf(torch.zeros(2, 8, 15)).shape) == (2, 3)


def test_fi2010_loader_on_synthetic_matrix():
    # FI-2010 matrix: features as ROWS (149, N); first 40 = 10 levels x 4.
    rng = np.random.default_rng(0)
    n = 300
    mat = np.zeros((149, n))
    # build a sane book per event in the interleaved [P_a,V_a,P_b,V_b] order
    for lvl in range(10):
        mat[4 * lvl + 0] = 100.0 + 0.1 * (lvl + 1)       # ask price ascends
        mat[4 * lvl + 1] = rng.uniform(1, 5, n)           # ask size
        mat[4 * lvl + 2] = 100.0 - 0.1 * (lvl + 1)        # bid price descends
        mat[4 * lvl + 3] = rng.uniform(1, 5, n)           # bid size
    df = fi2010.load_matrix(mat, market_id="TEST")
    assert df.height == n
    assert df["bids_price"].dtype == pl.List(pl.Float64)
    assert len(df["bids_price"][0]) == fi2010.N_LEVELS
    assert (df["ask_price"] >= df["bid_price"]).all()


def test_extended_features_build():
    from lobpred import features as Fx
    pool = synthetic.generate_pool(n_markets=4, minutes=30, seed=5)
    pool, _ = D.add_scalar_features(pool)
    pool, fcols = Fx.add_flow_features(pool)
    pool, hcols = Fx.add_history_features(pool)
    pool, pcols = Fx.add_shape_features(pool)
    pool, ecols = Fx.add_activity_spread_features(pool)
    pool, icols = Fx.add_impact_features(pool)
    pool, smcols = Fx.add_smoothed_features(pool)
    for c in fcols + hcols + pcols + ecols + icols + smcols:
        assert c in pool.columns, c
        assert pool[c].is_finite().all(), c
    # order-flow decomposition: add/cancel intensities are non-negative
    assert (pool["add_rate_5s"] >= 0).all()
    assert (pool["cancel_rate_5s"] >= 0).all()
    assert "kyle_lambda_5s" in pool.columns
    # EWMA smoothed signals present and bounded like their source
    assert "mg_ewma_5s" in pool.columns and "ofi_ewma_30s" in pool.columns
    assert pool["obi_ewma_5s"].min() >= -1.0 - 1e-9 and pool["obi_ewma_5s"].max() <= 1.0 + 1e-9
    # queue concentration is a fraction in [0, 1]; resting-volume present
    assert pool["touch_conc_bid"].min() >= 0.0
    assert pool["touch_conc_bid"].max() <= 1.0 + 1e-9
    assert "log_depth_bid" in pool.columns
    # activity counts are non-negative and the spread vol is non-negative
    assert (pool["mid_moves_5s"] >= 0).all()
    assert (pool["updates_5s"] >= 1).all()        # current event always counts
    assert (pool["spread_std_30s"] >= 0).all()
    # rolling OBI is a mean of L1 imbalance ∈ [-1, 1]; relative spread ≥ 0
    assert pool["obi_roll_5s"].min() >= -1.0 - 1e-9
    assert pool["obi_roll_5s"].max() <= 1.0 + 1e-9
    assert (pool["rel_spread"] >= 0).all()


def test_trade_features_build_and_bounded():
    from lobpred import features as Fx
    book, trades = synthetic.generate_with_trades(n_markets=4, minutes=30, seed=6)
    assert trades.height > 0 and set(("market_id", "price", "size", "side")).issubset(trades.columns)
    book, _ = D.add_scalar_features(book)
    book, tcols = Fx.add_trade_features(book, trades)
    for c in tcols:
        assert c in book.columns and book[c].is_finite().all(), c
    for w in (5, 30):                       # TFI is a signed fraction in [-1, 1]
        col = book[f"tfi_{w}s"]
        assert col.min() >= -1.0 - 1e-9 and col.max() <= 1.0 + 1e-9
        assert (book[f"vol_{w}s"] >= 0).all()      # traded-volume magnitude (log1p)
        assert (book[f"amihud_{w}s"] >= 0).all()   # Amihud illiquidity ≥ 0


def test_make_windows_stale_filters():
    """Window-level staleness guards: drop windows that straddle a quiet
    wall-clock gap or contain no real price action."""
    import pytest
    n = 40
    ts = np.arange(n, dtype=np.int64) * 1_000_000_000        # 1s apart
    ts[20:] += np.int64(3600) * 1_000_000_000                # 1h gap before the 2nd half
    mid = np.concatenate([np.linspace(0.50, 0.60, 20),       # first half moves every step
                          np.full(20, 0.60)])                # second half frozen
    pool = pl.DataFrame({
        "market_id": ["M"] * n,
        "timestamp_ns": ts,
        "mid": mid,
        "f0": np.arange(n, dtype=float),
        "y_fwd": np.zeros(n, dtype=float),
    })
    feats = ["f0"]
    base = D.make_windows(pool, feats, seq_len=8)
    # permissive thresholds == no filtering
    allk = D.make_windows(pool, feats, seq_len=8, max_window_span_s=1e9, min_window_mid_moves=0)
    assert allk.X.shape[0] == base.X.shape[0]
    # span cap drops the windows straddling the 1h gap (but keeps the dense ones)
    span = D.make_windows(pool, feats, seq_len=8, max_window_span_s=60.0)
    assert 0 < span.X.shape[0] < base.X.shape[0]
    assert span.X.shape[0] == span.y.shape[0] == span.ts.shape[0]   # arrays stay aligned
    # mid-move floor drops the frozen-tail windows
    mv = D.make_windows(pool, feats, seq_len=8, min_window_mid_moves=1)
    assert 0 < mv.X.shape[0] < base.X.shape[0]
    # the mid-move guard needs a mid column — loud, not silent (Rule #0.5)
    with pytest.raises(KeyError):
        D.make_windows(pool.drop("mid"), feats, seq_len=8, min_window_mid_moves=1)
