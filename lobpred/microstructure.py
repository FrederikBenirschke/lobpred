"""Per-book microstructure features.

Input: a Polars LazyFrame over canonical book snapshots (one row per
book update), with columns:

    market_id, timestamp_ns,
    bid_price, bid_size, ask_price, ask_size, mid, spread,
    bids_price (list), bids_size (list), asks_price (list), asks_size (list)

Output: same rows + L1/L3/L5/L10 imbalance, microprice, depth totals,
size-weighted depth midprice.

All features tolerate missing levels, depth-N values use whatever is
available up to N, and become null only if a side is empty.
"""

from __future__ import annotations

import polars as pl

DEPTH_LEVELS = (3, 5, 10)


def add_microstructure(df: pl.LazyFrame) -> pl.LazyFrame:
    """Append microprice + imbalance + depth features."""
    # Defensive cast: a book with no depth can arrive as List[Null],
    # which makes list arithmetic panic in Polars. Force the element type
    # so list.head + sum work.
    df = df.with_columns(
        pl.col("bids_price").cast(pl.List(pl.Float64)),
        pl.col("bids_size").cast(pl.List(pl.Float64)),
        pl.col("asks_price").cast(pl.List(pl.Float64)),
        pl.col("asks_size").cast(pl.List(pl.Float64)),
    )
    df = df.with_columns(
        _microprice_l1().alias("microprice"),
        _imbalance_l1().alias("imbalance_l1"),
    )
    for n in DEPTH_LEVELS:
        df = df.with_columns(
            _sum_first_n("bids_size", n).alias(f"bid_size_l{n}"),
            _sum_first_n("asks_size", n).alias(f"ask_size_l{n}"),
            _depth_imbalance(n).alias(f"imbalance_l{n}"),
            _weighted_mid(n).alias(f"microprice_l{n}"),
        )
    return df


# ── building blocks ─────────────────────────────────────────


def _microprice_l1() -> pl.Expr:
    """(bid_px * ask_sz + ask_px * bid_sz) / (bid_sz + ask_sz).

    Weights each side's price by the *opposite* side's size, heavier
    ask depth pulls the microprice toward the bid. Null when either
    side is missing or total size is zero.
    """
    denom = pl.col("bid_size") + pl.col("ask_size")
    num = pl.col("bid_price") * pl.col("ask_size") + pl.col("ask_price") * pl.col(
        "bid_size"
    )
    return pl.when(denom > 0).then(num / denom).otherwise(None)


def _imbalance_l1() -> pl.Expr:
    """(bid_sz - ask_sz) / (bid_sz + ask_sz). Range [-1, 1]."""
    denom = pl.col("bid_size") + pl.col("ask_size")
    return (
        pl.when(denom > 0)
        .then((pl.col("bid_size") - pl.col("ask_size")) / denom)
        .otherwise(None)
    )


def _sum_first_n(col: str, n: int) -> pl.Expr:
    return pl.col(col).list.head(n).list.sum()


def _depth_imbalance(n: int) -> pl.Expr:
    bid = pl.col("bids_size").list.head(n).list.sum()
    ask = pl.col("asks_size").list.head(n).list.sum()
    denom = bid + ask
    return pl.when(denom > 0).then((bid - ask) / denom).otherwise(None)


def _weighted_mid(n: int) -> pl.Expr:
    """Size-weighted mid using first N levels of each side.

    sum(bid_px * bid_sz, ask_px * ask_sz) / sum(bid_sz, ask_sz).
    Tracks the price at which an aggressor would clear N levels of
    liquidity on each side.
    """
    bid_px = pl.col("bids_price").list.head(n)
    bid_sz = pl.col("bids_size").list.head(n)
    ask_px = pl.col("asks_price").list.head(n)
    ask_sz = pl.col("asks_size").list.head(n)
    num = (
        (bid_px * bid_sz).list.sum()
        + (ask_px * ask_sz).list.sum()
    )
    denom = bid_sz.list.sum() + ask_sz.list.sum()
    return pl.when(denom > 0).then(num / denom).otherwise(None)
