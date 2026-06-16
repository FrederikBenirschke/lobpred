"""Load, feature-build, label, and window canonical LOB data.

Single source of truth for the data contract shared by every study and
every model. The framework-specific code (PyTorch models, LightGBM
baselines) consumes the numpy arrays this module emits; it never re-reads
parquet or re-derives features, so leakage discipline lives in one place.

Canonical schema (one row per book update), produced by ``lobpred.data``:

    market_id, timestamp_ns,
    bid_price, bid_size, ask_price, ask_size, mid, spread,
    bids_price (list), bids_size (list), asks_price (list), asks_size (list)

The six things this module guarantees
-------------------------------------
1. **Pooling.** Markets are pooled (a single market has too few rows to
   train a deep model). ``market_id`` tags every row so a market never
   straddles a train/test split.
2. **Three feature sets, same rows.** ``add_paper_features`` (raw L-level
   px+size, the reference paper's inputs), ``add_grid_features`` (resting
   size on a fixed tick grid, comparable across instruments), and
   ``add_scalar_features`` (OFI + imbalance + micro-gap).
3. **Leak-safe forward target.** ``add_forward_target`` builds the
   forward price change by a *forward* search per market, point exit,
   event-horizon, or a smoothed hold (TWAP) window.
4. **Activity selection by price discovery, not churn.**
   ``add_activity_tier`` ranks markets by *mid-move rate*
   (``mid_moves_per_min``), not quote-update rate, a book that flickers
   500 quotes/min with 2 mid moves is not active. (See the README's
   "churn ≠ discovery".)
5. **Train-only normalization.** ``fit_scaler`` z-scores using statistics
   from the training rows alone.
6. **Walk-forward splits with a purge gap.** ``walk_forward_splits``
   partitions by global wall-clock time and embargoes a gap >= the
   horizon around every boundary so no label straddles it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from .microstructure import add_microstructure
from .ofi import add_ofi

logger = logging.getLogger("lobpred.dataset")

# Default ladder depth used by the level/grid feature builders. Books with
# fewer populated levels pad with 0; deeper books use whatever is present.
LEVELS = 5

# Default price increment. Relative prices are expressed in ticks so a
# "2 ticks from mid" feature means the same thing on every market. The
# grid feature builder quantizes to this; override per dataset if needed.
TICK = 0.01

NS_PER_S = 1_000_000_000


# ── loading ─────────────────────────────────────────────────


@dataclass(frozen=True)
class LoadConfig:
    """Knobs for assembling the pooled book panel."""

    roots: tuple[Path, ...]
    min_rows: int = 2000               # drop markets too short to learn from
    market_ids: tuple[str, ...] | None = None  # optional whitelist
    require_two_sided: bool = True     # drop one-sided / dead books
    # Optional cleaning for *bounded-price* venues (e.g. prediction markets
    # priced in [0,1]): drop rows whose mid sits within ``corner_eps`` of a
    # bound (the endgame teaches "→ corner", not microstructure) and the
    # final ``tail_drop_s`` seconds of each market. Both OFF by default,
    # a general (unbounded-price) LOB has no settlement geometry.
    price_bounds: tuple[float, float] | None = None
    corner_eps: float = 0.0
    tail_drop_s: float = 0.0


def discover_book_parquets(roots: tuple[Path, ...]) -> list[Path]:
    """All book parquets under the roots (skipping ``_system`` and trade files)."""
    out: list[Path] = []
    for r in roots:
        out.extend(p for p in Path(r).rglob("*.parquet")
                   if p.parent.name != "_system" and not p.name.endswith(".trades.parquet"))
    if not out:
        raise FileNotFoundError(f"no book parquets under {[str(r) for r in roots]}")
    return sorted(out)


# Trade-tape schema (optional second stream, consumed by features.add_trade_features).
_TRADE_COLS = ("market_id", "timestamp_ns", "price", "size", "side")


def discover_trade_parquets(roots: tuple[Path, ...]) -> list[Path]:
    """All ``*.trades.parquet`` files under the roots."""
    out: list[Path] = []
    for r in roots:
        out.extend(p for p in Path(r).rglob("*.trades.parquet") if p.parent.name != "_system")
    return sorted(out)


def load_trades(roots: tuple[Path, ...], *, market_ids: tuple[str, ...] | None = None) -> pl.DataFrame:
    """Load the trade tape (``*.trades.parquet``) into the canonical trade schema.

    Returns an empty frame (with the right columns) if no trade files exist,
    trade features are optional, so absence is a valid state.
    """
    paths = discover_trade_parquets(roots)
    if not paths:
        return pl.DataFrame(schema={"market_id": pl.Utf8, "timestamp_ns": pl.Int64,
                                    "price": pl.Float64, "size": pl.Float64, "side": pl.Int64})
    frames = []
    for p in paths:
        try:
            frames.append(pl.read_parquet(p, columns=list(_TRADE_COLS)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping unreadable trades %s: %s", p, exc)
    trades = pl.concat(frames, how="vertical_relaxed")
    if market_ids is not None:
        trades = trades.filter(pl.col("market_id").is_in(list(market_ids)))
    return trades.sort("market_id", "timestamp_ns")


_REQUIRED_COLS = (
    "market_id", "timestamp_ns",
    "bid_price", "bid_size", "ask_price", "ask_size", "mid", "spread",
    "bids_price", "bids_size", "asks_price", "asks_size",
)


def load_pool(cfg: LoadConfig) -> pl.DataFrame:
    """Concatenate book parquets into one pooled, cleaned panel.

    Returns a sorted DataFrame with one row per book update. Markets
    shorter than ``cfg.min_rows`` (after cleaning) are dropped. Raises if
    no market survives, a silent empty frame would hide a config bug.
    """
    paths = discover_book_parquets(cfg.roots)
    frames: list[pl.DataFrame] = []
    for p in paths:
        try:
            df = pl.read_parquet(p, columns=list(_REQUIRED_COLS))
        except Exception as exc:  # noqa: BLE001, one bad file shouldn't kill the pool; logged loudly
            logger.warning("skipping unreadable %s: %s", p, exc)
            continue
        if df.height:
            frames.append(df)
    if not frames:
        raise FileNotFoundError("every book parquet was empty or unreadable")

    pool = pl.concat(frames, how="vertical_relaxed")
    if cfg.market_ids is not None:
        pool = pool.filter(pl.col("market_id").is_in(list(cfg.market_ids)))
    pool = _clean(pool, cfg)

    counts = pool.group_by("market_id").len()
    keep = counts.filter(pl.col("len") >= cfg.min_rows)["market_id"]
    pool = pool.filter(pl.col("market_id").is_in(keep))
    if pool.height == 0:
        raise ValueError(
            f"no market has >= {cfg.min_rows} clean rows; lower min_rows "
            f"or widen the roots"
        )
    pool = pool.sort("market_id", "timestamp_ns")
    logger.info("pooled %d rows across %d markets (>= %d rows each)",
                pool.height, pool["market_id"].n_unique(), cfg.min_rows)
    return pool


def _clean(pool: pl.DataFrame, cfg: LoadConfig) -> pl.DataFrame:
    """Drop dead books and (for bounded-price venues) settlement corners/tails."""
    pool = pool.filter(pl.col("mid").is_not_null() & pl.col("spread").is_not_null())
    if cfg.require_two_sided:
        pool = pool.filter((pl.col("bid_size") > 0) & (pl.col("ask_size") > 0))
    if cfg.price_bounds is not None and cfg.corner_eps > 0:
        lo, hi = cfg.price_bounds
        pool = pool.filter(
            (pl.col("mid") > lo + cfg.corner_eps) & (pl.col("mid") < hi - cfg.corner_eps)
        )
    if cfg.tail_drop_s > 0:
        tail_ns = int(cfg.tail_drop_s * NS_PER_S)
        pool = pool.with_columns(
            pl.col("timestamp_ns").max().over("market_id").alias("_end_ns")
        )
        pool = pool.filter(
            pl.col("timestamp_ns") <= pl.col("_end_ns") - tail_ns
        ).drop("_end_ns")
    return pool


# ── feature sets ────────────────────────────────────────────


def add_paper_features(pool: pl.DataFrame, levels: int = LEVELS) -> tuple[pl.DataFrame, list[str]]:
    """Raw L-level price+size, the reference paper's 4*L inputs.

    Each level k contributes bid_px_k, bid_sz_k, ask_px_k, ask_sz_k.
    Short books (fewer than ``levels`` populated) pad with 0, a vacated
    level has zero size, and the price slot is filled with the
    touch price so it carries no spurious distance signal after z-scoring.
    These are NON-stationary (absolute prices) by design; that is the
    weakness the grid representation fixes.
    """
    exprs: list[pl.Expr] = []
    cols: list[str] = []
    for side, px_list, sz_list, touch in (
        ("bid", "bids_price", "bids_size", "bid_price"),
        ("ask", "asks_price", "asks_size", "ask_price"),
    ):
        for k in range(levels):
            pc, sc = f"{side}_px_{k+1}", f"{side}_sz_{k+1}"
            exprs.append(
                pl.col(px_list).list.get(k, null_on_oob=True)
                .fill_null(pl.col(touch)).alias(pc)
            )
            exprs.append(
                pl.col(sz_list).list.get(k, null_on_oob=True)
                .fill_null(0.0).alias(sc)
            )
            cols.extend([pc, sc])
    return pool.with_columns(exprs), cols


def add_grid_features(
    pool: pl.DataFrame, *, k: int = 20, tick: float = TICK, normalize: bool = True
) -> tuple[pl.DataFrame, list[str]]:
    """Resting size on a FIXED tick grid relative to each side's touch.

    Rank-based level features (``bid_px_3`` etc.) are not comparable across
    instruments with different book geometry: one book may pack 5 levels
    into 4 ticks while another spreads them over 18 (gappy). "Level 3" is
    then a different *price distance* on each, so pooling on rank teaches
    instrument identity, not microstructure.

    This builds, per side, the resting size at a *fixed* offset from the
    best quote: ``bid_grid_o`` = size at (best_bid - o ticks),
    ``ask_grid_o`` = size at (best_ask + o ticks), for o in 0..k-1. Now
    offset o is the same price distance everywhere; gaps appear as explicit
    zeros; and a "pocket" is a nonzero bucket at a specific
    offset. Requires a meaningful fixed ``tick`` (works on raw venue data
    and the synthetic generator; not on pre-normalized data like FI-2010,
    use the level/scalar features there).

    With ``normalize`` (default), buckets are divided by total in-grid depth
    (both sides) so the profile is scale-free while preserving imbalance.
    Size beyond ``k`` ticks is dropped; the dropped fraction is logged.
    """
    n = pool.height
    pool = pool.with_columns(pl.int_range(0, n, dtype=pl.Int64).alias("_rid"))
    cols: list[str] = []

    def _side_grid(px_list: str, sz_list: str, touch: str, sign: int, prefix: str):
        long = (
            pool.select("_rid", touch, px_list, sz_list)
            .explode([px_list, sz_list])
            .drop_nulls([px_list, sz_list])
            .with_columns(
                (sign * (pl.col(px_list) - pl.col(touch)) / tick)
                .round().cast(pl.Int64).alias("_off")
            )
        )
        total = long["_off"].len()
        in_grid = long.filter((pl.col("_off") >= 0) & (pl.col("_off") < k))
        clipped = 1.0 - (in_grid.height / total) if total else 0.0
        agg = in_grid.group_by("_rid", "_off").agg(pl.col(sz_list).sum().alias("_sz"))
        wide = agg.pivot(values="_sz", index="_rid", on="_off").fill_null(0.0)
        rename = {}
        for o in range(k):
            src = str(o)
            name = f"{prefix}_grid_{o}"
            if src not in wide.columns:
                wide = wide.with_columns(pl.lit(0.0).alias(name))
            else:
                rename[src] = name
            cols.append(name)
        wide = wide.rename(rename)
        wide = wide.with_columns(pl.col("_rid").cast(pl.Int64))  # pivot may widen to f64
        return wide, clipped

    bid_w, bid_clip = _side_grid("bids_price", "bids_size", "bid_price", -1, "bid")
    ask_w, ask_clip = _side_grid("asks_price", "asks_size", "ask_price", +1, "ask")
    logger.info("grid k=%d: clipped depth bid=%.1f%% ask=%.1f%% (size beyond %d ticks)",
                k, 100 * bid_clip, 100 * ask_clip, k)

    pool = pool.join(bid_w, on="_rid", how="left").join(ask_w, on="_rid", how="left")
    pool = pool.with_columns([pl.col(c).fill_null(0.0) for c in cols])

    if normalize:
        total = pool.select(cols).sum_horizontal()
        denom = pl.when(total > 0).then(total).otherwise(None)
        pool = pool.with_columns([(pl.col(c) / denom).fill_null(0.0).alias(c) for c in cols])

    return pool.drop("_rid"), cols


def add_scalar_features(pool: pl.DataFrame, *, tick: float = TICK) -> tuple[pl.DataFrame, list[str]]:
    """Instrument-comparable microstructure scalars (flow + imbalance).

    OFI (Cont/Kukanov), depth imbalance at L1/3/5, microprice gap from mid,
    and spread, all invariant to absolute price, so they pool across
    markets without the rank problem the grid fixes for shape.
    """
    lf = add_microstructure(pool.lazy())
    lf = add_ofi(lf)
    pool = lf.collect()
    pool = pool.with_columns(
        ((pl.col("microprice") - pl.col("mid")) / tick).alias("micro_gap_ticks"),
        (pl.col("spread") / tick).alias("spread_ticks"),
    )
    feats = ["ofi", "imbalance_l1", "imbalance_l3", "imbalance_l5",
             "micro_gap_ticks", "spread_ticks"]
    feats = [c for c in feats if c in pool.columns]
    return pool, feats


def add_stationary_features(
    pool: pl.DataFrame, levels: int = LEVELS, *, tick: float = TICK
) -> tuple[pl.DataFrame, list[str]]:
    """OFI + imbalance + per-level relative price/size (a richer scalar set)."""
    lf = add_microstructure(pool.lazy())
    lf = add_ofi(lf)
    pool = lf.collect()

    base = ["imbalance_l1", "imbalance_l3", "imbalance_l5",
            "microprice", "microprice_l3", "microprice_l5", "ofi"]
    rel_exprs = [
        ((pl.col("microprice") - pl.col("mid")) / tick).alias("micro_gap_ticks"),
        (pl.col("spread") / tick).alias("spread_ticks"),
    ]
    feat = ["micro_gap_ticks", "spread_ticks"]
    for side, px_list, sz_list in (
        ("bid", "bids_price", "bids_size"),
        ("ask", "asks_price", "asks_size"),
    ):
        total_sz = pl.col(sz_list).list.sum()
        for k in range(levels):
            rc, rs = f"{side}_relpx_{k+1}", f"{side}_relsz_{k+1}"
            rel_exprs.append(
                ((pl.col(px_list).list.get(k, null_on_oob=True) - pl.col("mid")) / tick)
                .fill_null(0.0).alias(rc)
            )
            rel_exprs.append(
                (pl.col(sz_list).list.get(k, null_on_oob=True)
                 / pl.when(total_sz > 0).then(total_sz).otherwise(None))
                .fill_null(0.0).alias(rs)
            )
            feat.extend([rc, rs])
    pool = pool.with_columns(rel_exprs)
    feat = base + feat
    feat = [c for c in feat if c in pool.columns]
    return pool, feat


# ── forward target (leak-safe) ──────────────────────────────


def add_forward_target(
    pool: pl.DataFrame,
    *,
    horizon_s: float | None = None,
    horizon_events: int | None = None,
    avg_window_s: float | None = None,
    price_col: str = "microprice",
    max_stale_s: float | None = None,
) -> pl.DataFrame:
    """Append ``y_fwd`` (forward price change) + ``fwd_dt_s``, per market.

    Three target modes (choose one):

    * **point, wall-clock** (``horizon_s``): ``price(t+h) - price(t)`` at the
      first book update at or after ``t + horizon_s``. ``max_stale_s`` rejects
      a match too far past the target (the market went quiet).
    * **point, event** (``horizon_events``): ``price`` N rows ahead,
      regardless of wall-clock. Invariant in #events, not in time, pair with
      activity tiers so each tier's horizon is internally comparable.
    * **smoothed window** (``avg_window_s``): ``mean(price over (t, t+W]) -
      price(t)``, the average over a forward holding window of W seconds, NOT
      a single exit instant. The DeepLOB/paper-style smoothed target: "if I
      hold ~W s and exit on a TWAP, which way does the average go?" Less
      noisy than a point target and aligned with a hold-for-seconds strategy,
      it is easier, so report it as a holdable-horizon target.

    ``fwd_dt_s`` records the realized wall-clock (window length for the
    smoothed mode). Uses a per-market ``searchsorted``/``cumsum``, never any
    backward rolling-window column.
    """
    modes = sum(x is not None for x in (horizon_s, horizon_events, avg_window_s))
    if modes != 1:
        raise ValueError("pass exactly one of horizon_s / horizon_events / avg_window_s")
    if price_col not in pool.columns:
        raise KeyError(
            f"price_col {price_col!r} not in pool; build features first "
            f"(microprice comes from add_scalar_features / add_stationary_features)"
        )
    stale_ns = None if max_stale_s is None else int(max_stale_s * NS_PER_S)

    out_parts: list[pl.DataFrame] = []
    for (_mid,), g in pool.group_by(["market_id"], maintain_order=True):
        ts = g["timestamp_ns"].to_numpy()
        px = g[price_col].to_numpy().astype(float)
        n = len(ts)
        if avg_window_s is not None:
            win_ns = int(avg_window_s * NS_PER_S)
            csum = np.concatenate([[0.0], np.cumsum(px)])
            lo = np.searchsorted(ts, ts, side="right")            # first row after t
            hi = np.searchsorted(ts, ts + win_ns, side="right")   # first row after t+W
            cnt = hi - lo
            valid = cnt > 0
            y = np.full(n, np.nan)
            y[valid] = (csum[hi[valid]] - csum[lo[valid]]) / cnt[valid] - px[valid]
            dt = np.full(n, np.nan); dt[valid] = avg_window_s
            out_parts.append(g.with_columns(pl.Series("y_fwd", y), pl.Series("fwd_dt_s", dt)))
            continue
        if horizon_events is not None:
            idx = np.arange(n) + horizon_events
            valid = idx < n
            target_t = None
        else:
            target_t = ts + int(horizon_s * NS_PER_S)
            idx = np.searchsorted(ts, target_t, side="left")
            valid = idx < n
        safe_idx = np.where(valid, idx, 0)
        matched_ts = ts[safe_idx]
        y = np.full(n, np.nan)
        dt = np.full(n, np.nan)
        y_calc = px[safe_idx] - px
        if target_t is not None and stale_ns is not None:
            valid = valid & ((matched_ts - target_t) <= stale_ns)
        y[valid] = y_calc[valid]
        dt[valid] = (matched_ts[valid] - ts[valid]) / NS_PER_S
        out_parts.append(g.with_columns(pl.Series("y_fwd", y), pl.Series("fwd_dt_s", dt)))

    out = pl.concat(out_parts, how="vertical")
    # NaN (no future quote / too stale) is distinct from null in polars;
    # filter both, else tail rows survive and get mislabeled class 0.
    out = out.filter(pl.col("y_fwd").is_not_null() & pl.col("y_fwd").is_not_nan())
    return out


def add_sign_label(pool: pl.DataFrame, alpha: float = 0.0) -> pl.DataFrame:
    """3-class sign of ``y_fwd`` with a deadband.

    ``y_cls`` in {-1, 0, +1}: +1 if y_fwd > alpha, -1 if y_fwd < -alpha,
    else 0 (stable). ``alpha`` in price units. Regression on ``y_fwd`` is
    the primary task; this is for the paper-comparable accuracy number.
    """
    return pool.with_columns(
        pl.when(pl.col("y_fwd") > alpha).then(1)
        .when(pl.col("y_fwd") < -alpha).then(-1)
        .otherwise(0).alias("y_cls")
    )


# ── activity / liquidity tiers ──────────────────────────────


def add_liquidity_tier(pool: pl.DataFrame, n_tiers: int = 3) -> pl.DataFrame:
    """Tag each market with ``updates_per_min`` and a ``liq_tier`` rank.

    ``liq_tier`` 0 = least … n_tiers-1 = most, by equal-count quantiles of
    per-market book-UPDATE rate. This is *quote churn*, the rate that
    converts an event-horizon into wall-clock, NOT a measure of price
    discovery. Use it only for event-horizon↔wall-clock reasoning; select
    "active" markets with ``add_activity_tier`` instead.
    """
    rate = pool.group_by("market_id").agg(
        (pl.len() /
         ((pl.col("timestamp_ns").max() - pl.col("timestamp_ns").min()) / NS_PER_S / 60.0)
         .clip(lower_bound=1e-9)).alias("updates_per_min")
    )
    rate = rate.with_columns(
        ((pl.col("updates_per_min").rank() - 1) / pl.len() * n_tiers)
        .floor().clip(upper_bound=n_tiers - 1).cast(pl.Int32).alias("liq_tier")
    )
    return pool.join(rate, on="market_id", how="left")


def market_activity(pool: pl.DataFrame, *, eps: float = 1e-12) -> pl.DataFrame:
    """Per-market price-activity summary computed inline from the pool.

    Returns one row per ``market_id`` with ``n_mid_moves`` (count of
    consecutive mid changes) and ``mid_moves_per_min`` (real price changes
    per minute), the churn-free activity measure. No external manifest.
    """
    g = pool.sort("market_id", "timestamp_ns").with_columns(
        (pl.col("mid").diff().over("market_id").abs() > eps).alias("_moved")
    )
    return g.group_by("market_id").agg(
        pl.col("_moved").sum().alias("n_mid_moves"),
        ((pl.col("timestamp_ns").max() - pl.col("timestamp_ns").min()) / NS_PER_S / 60.0)
        .alias("_minutes"),
    ).with_columns(
        (pl.col("n_mid_moves") / pl.col("_minutes").clip(lower_bound=1e-9))
        .alias("mid_moves_per_min")
    )


def add_activity_tier(
    pool: pl.DataFrame, n_tiers: int = 3, *, min_mid_moves: int = 10
) -> pl.DataFrame:
    """Filter to price-ACTIVE markets and tag an activity tier.

    ``act_tier`` 0 = least … n_tiers-1 = most price-active, by equal-count
    quantiles of ``mid_moves_per_min`` (real price changes per minute), the
    correct liquidity axis (NOT quote churn; see ``add_liquidity_tier``).
    Markets with fewer than ``min_mid_moves`` total mid changes are dropped
    (inner join), so a flickering-but-frozen book never enters the study.
    """
    act = market_activity(pool).filter(pl.col("n_mid_moves") >= min_mid_moves)
    if act.height == 0:
        raise ValueError(
            f"no market has >= {min_mid_moves} mid moves; lower min_mid_moves "
            f"or widen the data"
        )
    act = act.with_columns(
        ((pl.col("mid_moves_per_min").rank() - 1) / pl.len() * n_tiers)
        .floor().clip(upper_bound=n_tiers - 1).cast(pl.Int32).alias("act_tier")
    )
    return pool.join(
        act.select("market_id", "n_mid_moves", "mid_moves_per_min", "act_tier"),
        on="market_id", how="inner",
    )


# ── normalization (train-only) ──────────────────────────────


@dataclass
class Scaler:
    """Per-feature mean/std fit on training rows only."""

    cols: list[str]
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


def fit_scaler(pool: pl.DataFrame, feat_cols: list[str], train_mask: np.ndarray) -> Scaler:
    """Fit z-score statistics on ``pool[train_mask]`` only."""
    X = pool.select(feat_cols).to_numpy().astype(float)
    Xtr = X[train_mask]
    mean = np.nanmean(Xtr, axis=0)
    std = np.nanstd(Xtr, axis=0)
    std = np.where(std < 1e-12, 1.0, std)  # guard constant features
    return Scaler(cols=list(feat_cols), mean=mean, std=std)


# ── event-time sequence windows ─────────────────────────────


@dataclass
class Windows:
    """Materialized sequence dataset, ready for a model."""

    X: np.ndarray             # (N, seq_len, F) float32
    y: np.ndarray             # (N,) float32, forward price change
    y_cls: np.ndarray | None  # (N,) int, sign label if present
    ts: np.ndarray            # (N,) int64, decision-time timestamp_ns
    market_id: np.ndarray     # (N,) object, for group-aware splitting
    feat_cols: list[str] = field(default_factory=list)


def make_windows(
    pool: pl.DataFrame,
    feat_cols: list[str],
    *,
    seq_len: int,
    scaler: Scaler | None = None,
    max_window_span_s: float | None = None,
    min_window_mid_moves: int | None = None,
) -> Windows:
    """Build per-market causal windows of the last ``seq_len`` updates.

    A window ending at row i uses features from rows [i-seq_len+1 .. i]
    (inclusive, causal) and is labeled with that row's ``y_fwd``. Windows
    never cross a market boundary. Nulls are filled with 0 *after* optional
    scaling (0 == the feature mean under z-score). Pass ``scaler`` (fit on
    train rows) to normalize; omit for tree baselines that don't need it.

    **Stale-window guards** (both optional; off by default so existing
    callers are unchanged). The market-level activity tier drops frozen
    *markets*; these drop frozen *windows* inside otherwise-active markets,
    a quiet patch where the book didn't move is ~one stale snapshot repeated
    ``seq_len`` times and dilutes the temporal signal:

    * ``max_window_span_s``, drop a window whose ``seq_len`` events span
      more than this many wall-clock seconds (book went quiet mid-window).
    * ``min_window_mid_moves``, drop a window with fewer than this many
      mid-price changes inside it (needs a ``mid`` column); real price
      action, not quote churn.

    Dropped counts are logged, never silent (Rule #0.5).
    """
    has_cls = "y_cls" in pool.columns
    need_mid = min_window_mid_moves is not None
    if need_mid and "mid" not in pool.columns:
        raise KeyError(
            "min_window_mid_moves set but 'mid' not in pool; build it "
            "(add_scalar_features) before windowing"
        )
    Xs, ys, yc, tss, mids = [], [], [], [], []
    n_raw = n_kept = 0
    for (mid_val,), g in pool.group_by(["market_id"], maintain_order=True):
        if g.height < seq_len:
            continue
        feats = g.select(feat_cols).to_numpy().astype(np.float64)
        if scaler is not None:
            feats = scaler.transform(feats)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        y = g["y_fwd"].to_numpy().astype(np.float64)
        ts = g["timestamp_ns"].to_numpy()
        cls = g["y_cls"].to_numpy() if has_cls else None
        win = np.lib.stride_tricks.sliding_window_view(feats, seq_len, axis=0)
        win = np.transpose(win, (0, 2, 1))           # → (T-seq_len+1, seq_len, F)
        end = np.arange(seq_len - 1, g.height)        # decision row per window
        start = end - (seq_len - 1)                   # first row of each window
        n_raw += len(end)

        keep = np.ones(len(end), dtype=bool)
        if max_window_span_s is not None:             # window spanning a quiet stretch = stale
            span_s = (ts[end] - ts[start]) / NS_PER_S
            keep &= span_s <= max_window_span_s
        if need_mid:                                  # require real price action inside the window
            mid = g["mid"].to_numpy().astype(float)
            moved = np.zeros(g.height, dtype=np.int64)
            moved[1:] = (np.abs(np.diff(mid)) > 1e-12).astype(np.int64)
            cmoves = np.concatenate([[0], np.cumsum(moved)])     # cmoves[k] = moves in rows [0..k-1]
            moves_in = cmoves[end + 1] - cmoves[start + 1]       # moves inside (start, end]
            keep &= moves_in >= min_window_mid_moves
        if not keep.any():
            continue
        win, end = win[keep], end[keep]
        n_kept += len(end)
        Xs.append(win.astype(np.float32))
        ys.append(y[end].astype(np.float32))
        tss.append(ts[end])
        mids.append(np.full(end.shape, mid_val, dtype=object))
        if cls is not None:
            yc.append(cls[end])
    if not Xs:
        raise ValueError(f"no market had >= seq_len ({seq_len}) rows after cleaning")
    if (max_window_span_s is not None or need_mid) and n_raw:
        logger.info(
            "make_windows: kept %d/%d windows after stale filter (%.1f%%; "
            "max_span_s=%s min_mid_moves=%s)",
            n_kept, n_raw, 100.0 * n_kept / n_raw, max_window_span_s, min_window_mid_moves,
        )
    return Windows(
        X=np.concatenate(Xs),
        y=np.concatenate(ys),
        y_cls=np.concatenate(yc) if yc else None,
        ts=np.concatenate(tss),
        market_id=np.concatenate(mids),
        feat_cols=list(feat_cols),
    )


# ── walk-forward splits ─────────────────────────────────────


def walk_forward_splits(
    ts: np.ndarray,
    *,
    n_folds: int,
    horizon_s: float,
    embargo_mult: float = 1.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Time-ordered (train_idx, test_idx) folds with a purge embargo.

    Splits the global wall-clock span into ``n_folds`` contiguous test
    blocks; train is everything before the test block. A purge gap
    of ``embargo_mult * horizon_s`` is removed from the end of train so no
    training label's [t, t+h] window overlaps the test block (the López de
    Prado embargo, adapted to a forward label). Operates on window
    decision-times ``ts`` (int64 ns).
    """
    t0, t1 = ts.min(), ts.max()
    edges = np.linspace(t0, t1, n_folds + 1).astype(np.int64)
    embargo_ns = int(embargo_mult * horizon_s * NS_PER_S)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(1, n_folds + 1):  # skip fold 0 as test (no train before it)
        test_lo, test_hi = edges[k - 1], edges[k]
        test = (ts >= test_lo) & (ts < test_hi if k < n_folds else ts <= test_hi)
        train = ts < (test_lo - embargo_ns)
        tr_idx, te_idx = np.where(train)[0], np.where(test)[0]
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue
        folds.append((tr_idx, te_idx))
    if not folds:
        raise ValueError("no usable walk-forward fold; widen the data span or reduce n_folds")
    return folds
