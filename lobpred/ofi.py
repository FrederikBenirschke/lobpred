"""Order Flow Imbalance (Cont/Kukanov 2014).

OFI is the signed contribution of best-bid/ask *changes* between
consecutive book snapshots, expressed as a per-tick flow variable.
Sign convention: positive OFI = net buy pressure.

Per-tick contributions:

    Δbid =
        + bid_size                           if bid_price > prev_bid_price   (new better bid)
        + (bid_size - prev_bid_size)         if bid_price == prev_bid_price  (size grew/shrunk at same price)
        - prev_bid_size                      if bid_price < prev_bid_price   (best bid pulled back)

    Δask = symmetric, sign flipped:
        - ask_size                           if ask_price < prev_ask_price   (new better ask)
        - (ask_size - prev_ask_size)         if ask_price == prev_ask_price  (size grew/shrunk)
        + prev_ask_size                      if ask_price > prev_ask_price   (best ask pulled back)

    ofi = Δbid + Δask

A growing best bid (more depth, or higher price) and a shrinking best
ask (less depth, or higher price) both indicate buy pressure → +OFI.
"""

from __future__ import annotations

import polars as pl


def add_ofi(df: pl.LazyFrame) -> pl.LazyFrame:
    """Append the per-tick ``ofi`` flow variable.

    Input columns required: ``market_id``, ``timestamp_ns``,
    ``bid_price``, ``bid_size``, ``ask_price``, ``ask_size``. Output adds
    ``delta_bid``, ``delta_ask``, ``ofi`` (per-tick).

    OFI for the first tick of each ``market_id`` is null, there is no
    previous snapshot to diff against.
    """
    df = df.sort("market_id", "timestamp_ns")
    prev_bid_p = pl.col("bid_price").shift(1).over("market_id")
    prev_bid_s = pl.col("bid_size").shift(1).over("market_id")
    prev_ask_p = pl.col("ask_price").shift(1).over("market_id")
    prev_ask_s = pl.col("ask_size").shift(1).over("market_id")

    delta_bid = (
        pl.when(pl.col("bid_price") > prev_bid_p).then(pl.col("bid_size"))
        .when(pl.col("bid_price") == prev_bid_p).then(pl.col("bid_size") - prev_bid_s)
        .when(pl.col("bid_price") < prev_bid_p).then(-prev_bid_s)
        .otherwise(None)
    )
    delta_ask = (
        pl.when(pl.col("ask_price") < prev_ask_p).then(-pl.col("ask_size"))
        .when(pl.col("ask_price") == prev_ask_p).then(-(pl.col("ask_size") - prev_ask_s))
        .when(pl.col("ask_price") > prev_ask_p).then(prev_ask_s)
        .otherwise(None)
    )
    df = df.with_columns(
        delta_bid.alias("delta_bid"),
        delta_ask.alias("delta_ask"),
    ).with_columns(
        (pl.col("delta_bid") + pl.col("delta_ask")).alias("ofi"),
    )
    return df
