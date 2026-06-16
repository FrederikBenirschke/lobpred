"""Extended feature families, all leak-safe and backward-looking.

These build on the base sets in ``dataset.py`` (grid and scalar). Every
feature uses only information available at or before the decision time ``t``;
the windows look backward, mirroring the forward-target idiom.

  * ``add_flow_features`` (A): rolling and multi-level order-flow imbalance
  * ``add_history_features`` (B): lagged returns and realized volatility
  * ``add_shape_features`` (C): book shape, depth, resting volume
  * ``add_trade_features`` (D): trade-tape flow (TFI, intensity, traded
                                volume, VWAP gap); needs a trade stream
  * ``add_activity_spread_features`` (E): mid-change and update intensity over
                                the last seconds, plus rolling spread mean and vol
  * ``add_impact_features`` (F): order-flow decomposition (limit add and cancel
                                intensity from book diffs) plus a rolling
                                Kyle-λ price-impact estimate

A/B/C assume ``add_scalar_features`` has run, so ``microprice``, ``ofi``, and
the depth ``microprice_l*`` / ``imbalance_l*`` columns exist.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .dataset import LEVELS, NS_PER_S, TICK

# Per-level representation: channels per book level (OBI, OFI, rel-size), in
# this order. The model reshapes (T, L*PERLEVEL_CHANNELS) -> (T, L, C), so the
# feature builder must emit columns level-major in this channel order.
PERLEVEL_CHANNELS = 3


# ── leak-safe backward-window helpers ───────────────────────


def _back_window(ts: np.ndarray, vals: np.ndarray, win_ns: int) -> tuple[np.ndarray, np.ndarray]:
    """For each i: (sum, count) of vals[j] with ts[i]-win < ts[j] <= ts[i]."""
    csum = np.concatenate([[0.0], np.cumsum(vals)])
    hi = np.searchsorted(ts, ts, side="right")          # first j with ts[j] > ts[i]
    lo = np.searchsorted(ts, ts - win_ns, side="right")  # first j with ts[j] > ts[i]-win
    return csum[hi] - csum[lo], (hi - lo).astype(float)


def _back_value_at(ts: np.ndarray, vals: np.ndarray, lag_ns: int) -> np.ndarray:
    """Value of vals at the last event <= ts[i]-lag (vals[i] itself if none)."""
    idx = np.searchsorted(ts, ts - lag_ns, side="right") - 1
    out = vals.copy()
    ok = idx >= 0
    out[ok] = vals[idx[ok]]
    return out


def _ofi_levels(bp, bs, ap, as_) -> np.ndarray:
    """Per-tick Cont/Kukanov OFI for one book level (first tick = 0)."""
    n = len(bp)
    ofi = np.zeros(n)
    if n < 2:
        return ofi
    pbp, pbs, pap, pas = bp[:-1], bs[:-1], ap[:-1], as_[:-1]
    cbp, cbs, cap, cas = bp[1:], bs[1:], ap[1:], as_[1:]
    dbid = np.where(cbp > pbp, cbs, np.where(cbp == pbp, cbs - pbs, -pbs))
    dask = np.where(cap < pap, -cas, np.where(cap == pap, -(cas - pas), pas))
    ofi[1:] = dbid + dask
    return ofi


# ── A, flow dynamics ───────────────────────────────────────


def add_flow_features(
    pool: pl.DataFrame,
    *,
    time_windows_s: tuple[float, ...] = (1.0, 5.0, 30.0),
    event_windows: tuple[int, ...] = (10, 50),
    levels: int = 3,
) -> tuple[pl.DataFrame, list[str]]:
    """Order-flow and book-imbalance dynamics.

    Requires per-tick ``ofi`` and ``imbalance_l1`` (from
    ``add_scalar_features``). Adds:
      * ``ofi_roll_{w}s``, OFI summed over the last w s
      * ``ofi_ev_{k}``: OFI summed over the last k events
      * ``ofi_l{m}``: per-tick OFI at deeper levels (m=2..levels)
      * ``obi_roll_{w}s``, rolling MEAN of L1 order-book imbalance (OBI) over
        the last w s (the smoothed-OBI analogue of rolling OFI)
    """
    if "ofi" not in pool.columns or "imbalance_l1" not in pool.columns:
        raise KeyError("add_flow_features needs 'ofi' and 'imbalance_l1'; run add_scalar_features first")
    # extract deeper level columns once (vectorized) for multi-level OFI
    lvl_exprs = []
    for m in range(2, levels + 1):
        lvl_exprs += [
            pl.col("bids_price").list.get(m - 1, null_on_oob=True).fill_null(pl.col("bid_price")).alias(f"_bp{m}"),
            pl.col("bids_size").list.get(m - 1, null_on_oob=True).fill_null(0.0).alias(f"_bs{m}"),
            pl.col("asks_price").list.get(m - 1, null_on_oob=True).fill_null(pl.col("ask_price")).alias(f"_ap{m}"),
            pl.col("asks_size").list.get(m - 1, null_on_oob=True).fill_null(0.0).alias(f"_as{m}"),
        ]
    pool = pool.with_columns(lvl_exprs)

    cols = ([f"ofi_roll_{int(w)}s" for w in time_windows_s]
            + [f"ofi_ev_{k}" for k in event_windows]
            + [f"ofi_l{m}" for m in range(2, levels + 1)]
            + [f"obi_roll_{int(w)}s" for w in time_windows_s])

    out = []
    for (_mid,), g in pool.group_by(["market_id"], maintain_order=True):
        ts = g["timestamp_ns"].to_numpy()
        ofi = np.nan_to_num(g["ofi"].to_numpy().astype(float))
        obi = np.nan_to_num(g["imbalance_l1"].to_numpy().astype(float))
        new = {}
        for w in time_windows_s:
            s, _ = _back_window(ts, ofi, int(w * NS_PER_S))
            new[f"ofi_roll_{int(w)}s"] = s
            s_obi, cnt = _back_window(ts, obi, int(w * NS_PER_S))
            new[f"obi_roll_{int(w)}s"] = s_obi / np.maximum(cnt, 1.0)   # rolling MEAN OBI
        cs = np.concatenate([[0.0], np.cumsum(ofi)])
        n = len(ts)
        for k in event_windows:
            lo = np.maximum(np.arange(n) - k + 1, 0)
            new[f"ofi_ev_{k}"] = cs[np.arange(n) + 1] - cs[lo]
        for m in range(2, levels + 1):
            new[f"ofi_l{m}"] = _ofi_levels(
                g[f"_bp{m}"].to_numpy().astype(float), g[f"_bs{m}"].to_numpy().astype(float),
                g[f"_ap{m}"].to_numpy().astype(float), g[f"_as{m}"].to_numpy().astype(float))
        out.append(g.with_columns([pl.Series(c, v) for c, v in new.items()]))

    pool = pl.concat(out, how="vertical").drop(
        [c for m in range(2, levels + 1) for c in (f"_bp{m}", f"_bs{m}", f"_ap{m}", f"_as{m}")])
    return pool, cols


# ── B, return / volatility history ─────────────────────────


def add_history_features(
    pool: pl.DataFrame,
    *,
    event_lags: tuple[int, ...] = (1, 5, 10),
    time_windows_s: tuple[float, ...] = (5.0, 30.0),
    tick: float = TICK,
) -> tuple[pl.DataFrame, list[str]]:
    """Lagged microprice returns (event + time) and realized volatility.

    Adds ``ret_ev_{k}`` (microprice change over last k events, in ticks),
    ``ret_roll_{w}s`` (change over last w s), and ``rv_{w}s`` (RMS of
    per-event returns over last w s, in ticks). Requires ``microprice``.
    """
    if "microprice" not in pool.columns:
        raise KeyError("add_history_features needs 'microprice'; run add_scalar_features first")
    cols = ([f"ret_ev_{k}" for k in event_lags]
            + [f"ret_roll_{int(w)}s" for w in time_windows_s]
            + [f"rv_{int(w)}s" for w in time_windows_s])
    out = []
    for (_mid,), g in pool.group_by(["market_id"], maintain_order=True):
        ts = g["timestamp_ns"].to_numpy()
        mp = g["microprice"].to_numpy().astype(float)
        mp = np.nan_to_num(mp)
        n = len(ts)
        new = {}
        for k in event_lags:
            prev = np.empty(n); prev[:k] = mp[:k]; prev[k:] = mp[:-k] if k else mp
            new[f"ret_ev_{k}"] = (mp - prev) / tick
        r = np.zeros(n); r[1:] = np.diff(mp)            # per-event return
        for w in time_windows_s:
            win = int(w * NS_PER_S)
            new[f"ret_roll_{int(w)}s"] = (mp - _back_value_at(ts, mp, win)) / tick
            s2, cnt = _back_window(ts, r * r, win)
            new[f"rv_{int(w)}s"] = np.sqrt(s2 / np.maximum(cnt, 1.0)) / tick
        out.append(g.with_columns([pl.Series(c, v) for c, v in new.items()]))
    return pl.concat(out, how="vertical"), cols


# ── C, book shape / depth (stationary) ─────────────────────


def add_shape_features(pool: pl.DataFrame, *, tick: float = TICK) -> tuple[pl.DataFrame, list[str]]:
    """Book-shape, depth, and resting-volume features.

    Queue concentration, depth log-ratio, deeper microprice gaps, deep
    imbalance, and **resting volume** (log total depth per side). Requires
    the depth columns from ``add_microstructure`` (run via
    ``add_scalar_features``): ``microprice_l3/l5``, ``imbalance_l10``.
    """
    need = ["microprice_l3", "microprice_l5", "imbalance_l10"]
    missing = [c for c in need if c not in pool.columns]
    if missing:
        raise KeyError(f"add_shape_features needs {missing}; run add_scalar_features first")
    bid_tot = pl.col("bids_size").list.sum()
    ask_tot = pl.col("asks_size").list.sum()
    pool = pool.with_columns(
        (pl.col("bid_size") / pl.when(bid_tot > 0).then(bid_tot).otherwise(None)).fill_null(0.0).alias("touch_conc_bid"),
        (pl.col("ask_size") / pl.when(ask_tot > 0).then(ask_tot).otherwise(None)).fill_null(0.0).alias("touch_conc_ask"),
        ((1.0 + bid_tot).log() - (1.0 + ask_tot).log()).alias("depth_log_ratio"),
        (1.0 + bid_tot).log().alias("log_depth_bid"),   # resting volume (scale-compressed)
        (1.0 + ask_tot).log().alias("log_depth_ask"),
        ((pl.col("microprice_l3") - pl.col("mid")) / tick).alias("micro_gap_l3"),
        ((pl.col("microprice_l5") - pl.col("mid")) / tick).alias("micro_gap_l5"),
    )
    cols = ["touch_conc_bid", "touch_conc_ask", "depth_log_ratio",
            "log_depth_bid", "log_depth_ask",
            "micro_gap_l3", "micro_gap_l5", "imbalance_l10"]
    return pool, cols


# ── D, trade-tape features (needs a trade stream) ──────────


def add_trade_features(
    book: pl.DataFrame,
    trades: pl.DataFrame,
    *,
    windows_s: tuple[float, ...] = (5.0, 30.0),
    tick: float = TICK,
) -> tuple[pl.DataFrame, list[str]]:
    """Trade-flow features as-of joined onto each book row (past only).

    ``trades`` schema: ``market_id, timestamp_ns, price, size, side`` with
    ``side`` = +1 taker-buy / -1 taker-sell. For each book row at t and
    window w, aggregates trades in (t-w, t]:

      * ``tfi_{w}s``: signed-volume imbalance Σ(side·size)/Σ size ∈ [-1,1]
      * ``trades_{w}s``, trade count (intensity)
      * ``vol_{w}s``: traded volume log1p(Σ size) (magnitude, not signed)
      * ``vwapgap_{w}s``, (VWAP over the window − mid) / tick
      * ``amihud_{w}s``, Amihud illiquidity Σ|Δmid| / Σ traded volume

    Markets with no trades get zeros (a real "no trade flow" state).
    """
    for c in ("market_id", "timestamp_ns", "price", "size", "side"):
        if c not in trades.columns:
            raise KeyError(f"trades missing column {c!r}")
    cols = ([f"tfi_{int(w)}s" for w in windows_s]
            + [f"trades_{int(w)}s" for w in windows_s]
            + [f"vol_{int(w)}s" for w in windows_s]
            + [f"vwapgap_{int(w)}s" for w in windows_s]
            + [f"amihud_{int(w)}s" for w in windows_s])

    tr_by: dict[object, dict] = {}
    for (mid,), g in trades.sort("market_id", "timestamp_ns").group_by(["market_id"], maintain_order=True):
        ts = g["timestamp_ns"].to_numpy()
        sz = g["size"].to_numpy().astype(float)
        sgn = g["side"].to_numpy().astype(float)
        px = g["price"].to_numpy().astype(float)
        tr_by[mid] = {
            "ts": ts,
            "c_signed": np.concatenate([[0.0], np.cumsum(sgn * sz)]),
            "c_size": np.concatenate([[0.0], np.cumsum(sz)]),
            "c_pxsz": np.concatenate([[0.0], np.cumsum(px * sz)]),
        }

    out = []
    for (mid,), g in book.group_by(["market_id"], maintain_order=True):
        bts = g["timestamp_ns"].to_numpy()
        mid_px = g["mid"].to_numpy().astype(float)
        n = len(bts)
        admid = np.abs(np.concatenate([[0.0], np.diff(mid_px)]))   # |Δmid| per book event
        new = {c: np.zeros(n) for c in cols}
        t = tr_by.get(mid)
        if t is not None:
            for w in windows_s:
                win = int(w * NS_PER_S)
                hi = np.searchsorted(t["ts"], bts, side="right")
                lo = np.searchsorted(t["ts"], bts - win, side="right")
                signed = t["c_signed"][hi] - t["c_signed"][lo]
                vol = t["c_size"][hi] - t["c_size"][lo]
                pxsz = t["c_pxsz"][hi] - t["c_pxsz"][lo]
                cnt = (hi - lo).astype(float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    new[f"tfi_{int(w)}s"] = np.where(vol > 0, signed / np.maximum(vol, 1e-12), 0.0)
                    vwap = np.where(vol > 0, pxsz / np.maximum(vol, 1e-12), mid_px)
                new[f"trades_{int(w)}s"] = cnt
                new[f"vol_{int(w)}s"] = np.log1p(np.maximum(vol, 0.0))
                new[f"vwapgap_{int(w)}s"] = (vwap - mid_px) / tick
                sdm, _ = _back_window(bts, admid, win)             # Amihud: Σ|Δmid| / traded volume
                new[f"amihud_{int(w)}s"] = np.where(vol > 0, sdm / np.maximum(vol, 1e-9), 0.0)
        out.append(g.with_columns([pl.Series(c, v) for c, v in new.items()]))
    return pl.concat(out, how="vertical"), cols


# ── E, activity & spread dynamics ──────────────────────────


def add_activity_spread_features(
    pool: pl.DataFrame,
    *,
    time_windows_s: tuple[float, ...] = (1.0, 5.0, 30.0),
    spread_windows_s: tuple[float, ...] = (5.0, 30.0),
) -> tuple[pl.DataFrame, list[str]]:
    """Realized activity intensity and rolling spread statistics.

    Per market, over the last w seconds:
      * ``mid_moves_{w}s``, number of mid-price CHANGES (price-discovery
        intensity, the activity axis that matters, not raw quote churn).
      * ``updates_{w}s``: number of book updates (quote intensity).
      * ``spread_mean_{w}s`` / ``spread_std_{w}s``, rolling spread level and
        volatility (ticks). Requires ``spread_ticks`` (add_scalar_features).
    Plus the instantaneous ``rel_spread`` = spread / mid (spread normalized by
    price level, complementing the tick-normalized ``spread_ticks``).
    """
    if "spread_ticks" not in pool.columns:
        raise KeyError("add_activity_spread_features needs 'spread_ticks'; run add_scalar_features first")
    pool = pool.with_columns(
        (pl.col("spread") / pl.when(pl.col("mid") > 0).then(pl.col("mid")).otherwise(None))
        .fill_null(0.0).alias("rel_spread")
    )
    cols = (["rel_spread"]
            + [f"mid_moves_{int(w)}s" for w in time_windows_s]
            + [f"updates_{int(w)}s" for w in time_windows_s]
            + [f"spread_mean_{int(w)}s" for w in spread_windows_s]
            + [f"spread_std_{int(w)}s" for w in spread_windows_s])
    out = []
    for (_mid,), g in pool.group_by(["market_id"], maintain_order=True):
        ts = g["timestamp_ns"].to_numpy()
        mid = g["mid"].to_numpy().astype(float)
        sp = np.nan_to_num(g["spread_ticks"].to_numpy().astype(float))
        n = len(ts)
        moved = np.zeros(n)
        moved[1:] = (np.abs(np.diff(mid)) > 1e-12).astype(float)
        new = {}
        for w in time_windows_s:
            win = int(w * NS_PER_S)
            s_moved, cnt = _back_window(ts, moved, win)
            new[f"mid_moves_{int(w)}s"] = s_moved
            new[f"updates_{int(w)}s"] = cnt
        for w in spread_windows_s:
            win = int(w * NS_PER_S)
            s_sp, cnt = _back_window(ts, sp, win)
            s_sp2, _ = _back_window(ts, sp * sp, win)
            mean = s_sp / np.maximum(cnt, 1.0)
            var = np.maximum(s_sp2 / np.maximum(cnt, 1.0) - mean * mean, 0.0)
            new[f"spread_mean_{int(w)}s"] = mean
            new[f"spread_std_{int(w)}s"] = np.sqrt(var)
        out.append(g.with_columns([pl.Series(c, v) for c, v in new.items()]))
    return pl.concat(out, how="vertical"), cols


# ── F, order-flow decomposition & price impact ─────────────


def _prev(a: np.ndarray) -> np.ndarray:
    p = a.copy()
    p[1:] = a[:-1]
    return p


def add_impact_features(
    pool: pl.DataFrame,
    *,
    time_windows_s: tuple[float, ...] = (5.0, 30.0),
) -> tuple[pl.DataFrame, list[str]]:
    """Order-flow decomposition and a rolling Kyle-λ price-impact estimate.

    From consecutive book snapshots, the same-price change in resting size at
    each touch decomposes into limit-order **additions** (size up) and
    **cancellations** (size down). Per market, over the last w s:

      * ``add_rate_{w}s``: Σ positive same-price touch-size changes (adds)
      * ``cancel_rate_{w}s``, Σ |negative same-price touch-size changes|
        (cancellations; book-only, so it also absorbs touch executions, a
        trade-matched split would subtract traded volume at the touch)
      * ``kyle_lambda_{w}s``, price impact: OLS slope of Δmid on OFI through
        the origin, ``Σ(Δmid·ofi) / Σ(ofi²)`` over the window. Higher = a
        given order-flow imbalance moves price more (less depth/liquidity).

    Requires ``ofi`` (from ``add_scalar_features``).
    """
    if "ofi" not in pool.columns:
        raise KeyError("add_impact_features needs 'ofi'; run add_scalar_features first")
    cols = ([f"add_rate_{int(w)}s" for w in time_windows_s]
            + [f"cancel_rate_{int(w)}s" for w in time_windows_s]
            + [f"kyle_lambda_{int(w)}s" for w in time_windows_s])
    out = []
    for (_mid,), g in pool.group_by(["market_id"], maintain_order=True):
        ts = g["timestamp_ns"].to_numpy()
        bp = g["bid_price"].to_numpy().astype(float); bs = g["bid_size"].to_numpy().astype(float)
        ap = g["ask_price"].to_numpy().astype(float); a_s = g["ask_size"].to_numpy().astype(float)
        mid = g["mid"].to_numpy().astype(float)
        ofi = np.nan_to_num(g["ofi"].to_numpy().astype(float))
        # same-price touch-size deltas → adds (+) / cancels (-)
        dqb = np.where(bp == _prev(bp), bs - _prev(bs), 0.0)
        dqa = np.where(ap == _prev(ap), a_s - _prev(a_s), 0.0)
        adds = np.maximum(dqb, 0.0) + np.maximum(dqa, 0.0)
        cancels = np.maximum(-dqb, 0.0) + np.maximum(-dqa, 0.0)
        dm = np.zeros(len(mid)); dm[1:] = np.diff(mid)
        a_imp = dm * ofi
        b_imp = ofi * ofi
        new = {}
        for w in time_windows_s:
            win = int(w * NS_PER_S)
            new[f"add_rate_{int(w)}s"], _ = _back_window(ts, adds, win)
            new[f"cancel_rate_{int(w)}s"], _ = _back_window(ts, cancels, win)
            sa, _ = _back_window(ts, a_imp, win)
            sb, _ = _back_window(ts, b_imp, win)
            new[f"kyle_lambda_{int(w)}s"] = sa / np.maximum(sb, 1e-12)
        out.append(g.with_columns([pl.Series(c, v) for c, v in new.items()]))
    return pl.concat(out, how="vertical"), cols


# ── G, smoothed / EWMA signals ─────────────────────────────


def add_smoothed_features(
    pool: pl.DataFrame,
    *,
    halflives_s: tuple[float, ...] = (5.0, 30.0),
) -> tuple[pl.DataFrame, list[str]]:
    """Time-decayed (EWMA) versions of the core signals, smoother and better
    matched to a multi-second horizon than the hard rolling windows.

    Uses polars' time-aware ``ewm_mean_by`` (proper irregular-time decay), per
    market. For each halflife: EWMA-mean of ``micro_gap`` and L1 ``OBI`` (the
    top predictors, which previously had no smoothing / only a hard window),
    and EWMA-mean of ``OFI`` and depth-normalized OFI. An EWMA *mean* of OFI is
    a recency-weighted average per event, churn-normalized, unlike
    the rolling *sum* ``ofi_roll``. Requires ``micro_gap_ticks``, ``ofi``,
    ``imbalance_l1`` (from ``add_scalar_features``).
    """
    for c in ("micro_gap_ticks", "ofi", "imbalance_l1"):
        if c not in pool.columns:
            raise KeyError(f"add_smoothed_features needs {c!r}; run add_scalar_features first")
    depth = pl.col("bids_size").list.sum() + pl.col("asks_size").list.sum()
    pool = pool.with_columns(
        pl.col("timestamp_ns").cast(pl.Datetime("ns")).alias("_dt"),
        (pl.col("ofi") / (1.0 + depth)).alias("_ofi_dn"),
    )
    specs = [("mg_ewma", "micro_gap_ticks"), ("obi_ewma", "imbalance_l1"),
             ("ofi_ewma", "ofi"), ("ofidepth_ewma", "_ofi_dn")]
    exprs, cols = [], []
    for hl in halflives_s:
        hs = f"{int(hl)}s"
        for name, src in specs:
            col = f"{name}_{hs}"
            exprs.append(
                pl.col(src).ewm_mean_by(by="_dt", half_life=hs).over("market_id").fill_null(0.0).alias(col)
            )
            cols.append(col)
    pool = pool.with_columns(exprs).drop("_dt", "_ofi_dn")
    return pool, cols


# ── per-level stationary tensor (the DeepLOB / Kolm representation) ──


def add_perlevel_features(pool: pl.DataFrame, *, levels: int = LEVELS) -> tuple[pl.DataFrame, list[str]]:
    """Per-level **stationary** book features, shaped for a spatial-temporal net.

    For each of ``levels`` book levels ℓ, three channels, all stationary, so
    pooling across markets is valid AND the per-level structure is preserved
    (unlike flat scalars or non-stationary raw prices):

      * ``pl_obi_ℓ``: order-book imbalance at level ℓ: (bsz−asz)/(bsz+asz)
      * ``pl_ofi_ℓ``: Cont/Kukanov OFI computed at level ℓ's quotes
      * ``pl_relsz_ℓ``, level ℓ's size as a fraction of total in-book depth

    Columns are returned **level-major** in (obi, ofi, relsz) order, so the
    flat (T, L*3) window reshapes to (T, L, 3) for ``PerLevelLOB``. This is the
    representation deep LOB models use (DeepLOB normalizes raw levels;
    Kolm et al. feed multi-level OFI), the fair, structured input for DL.
    """
    lvl = []
    for l in range(levels):
        lvl += [
            pl.col("bids_price").list.get(l, null_on_oob=True).fill_null(pl.col("bid_price")).alias(f"_bp{l}"),
            pl.col("bids_size").list.get(l, null_on_oob=True).fill_null(0.0).alias(f"_bs{l}"),
            pl.col("asks_price").list.get(l, null_on_oob=True).fill_null(pl.col("ask_price")).alias(f"_ap{l}"),
            pl.col("asks_size").list.get(l, null_on_oob=True).fill_null(0.0).alias(f"_as{l}"),
        ]
    pool = pool.with_columns(lvl).with_columns(
        (pl.col("bids_size").list.sum() + pl.col("asks_size").list.sum()).alias("_tot"))
    # OBI_ℓ and relsz_ℓ (vectorized)
    static = []
    for l in range(levels):
        b, a = pl.col(f"_bs{l}"), pl.col(f"_as{l}")
        den = b + a
        static.append(pl.when(den > 0).then((b - a) / den).otherwise(0.0).alias(f"pl_obi_{l + 1}"))
        static.append((den / pl.when(pl.col("_tot") > 0).then(pl.col("_tot")).otherwise(None))
                      .fill_null(0.0).alias(f"pl_relsz_{l + 1}"))
    pool = pool.with_columns(static)
    # OFI_ℓ (per-tick Cont/Kukanov at each level; per-market scan)
    out = []
    for (_m,), g in pool.group_by(["market_id"], maintain_order=True):
        new = {}
        for l in range(levels):
            new[f"pl_ofi_{l + 1}"] = _ofi_levels(
                g[f"_bp{l}"].to_numpy().astype(float), g[f"_bs{l}"].to_numpy().astype(float),
                g[f"_ap{l}"].to_numpy().astype(float), g[f"_as{l}"].to_numpy().astype(float))
        out.append(g.with_columns([pl.Series(c, v) for c, v in new.items()]))
    pool = pl.concat(out, how="vertical").drop(
        [c for l in range(levels) for c in (f"_bp{l}", f"_bs{l}", f"_ap{l}", f"_as{l}")] + ["_tot"])
    # level-major order: (obi, ofi, relsz) per level, must match PERLEVEL_CHANNELS
    cols = [f"pl_{ch}_{l}" for l in range(1, levels + 1) for ch in ("obi", "ofi", "relsz")]
    return pool, cols
