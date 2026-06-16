"""Synthetic limit-order-book generator with a known, modest signal.

Lets the whole pipeline run with zero external data and gives a *planted*
edge the harness should recover, a useful sanity check (if the models
can't find a signal you put there, something is wrong) and a fast
substrate for tests.

Generative model, per market:

  * A latent order-flow imbalance ``I_t`` follows an AR(1) process,
    squashed to (-1, 1). It is persistent (autocorrelated), so its current
    value carries information about the near future.
  * The fair value evolves as ``f_{t} = f_{t-1} + κ·I_{t-1}·tick + σ·tick·ε``,
    so imbalance leads the price move by one event. The i.i.d. noise keeps the
    relationship real but far from deterministic.
  * The visible book encodes ``I_t`` in its top-of-book sizes (more bid
    size when ``I_t>0``), so the microprice gap and L1 imbalance are
    observable proxies for the latent signal. Deeper levels step one tick
    apart with decaying size.
  * A **trade tape** is generated alongside: trades arrive as a Poisson
    process and the taker side is biased by ``I_t`` (more buys when
    ``I_t>0``), so trade-flow imbalance (TFI) is also predictive, exercising
    the trade-feature path with planted signal.

Because the signal lives in imbalance / micro-gap / trade flow (not in raw
price levels), a model on stationary features should beat one on absolute
levels, mirroring the real finding. Timestamps are irregular and rescaled
so every market spans the same wall-clock window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

NS_PER_S = 1_000_000_000


def _gen_one(m: int, *, minutes, base_rate_per_min, trade_rate_per_min, levels,
             tick, f0, kappa, sigma, beta, phi, seed) -> tuple[pl.DataFrame, pl.DataFrame]:
    mr = np.random.default_rng(seed * 1000 + m + 1)
    span_ns = int(minutes * 60 * NS_PER_S)
    rate = base_rate_per_min * float(np.exp(mr.normal(0, 0.6)))  # heavy-tailed activity
    n = max(levels + 5, int(rate * minutes))

    # latent AR(1) imbalance, squashed to (-1, 1)
    eps = mr.normal(0, 1, n)
    z = np.empty(n)
    z[0] = eps[0]
    for t in range(1, n):
        z[t] = phi * z[t - 1] + eps[t]
    z /= (np.std(z) or 1.0)
    I = np.tanh(0.7 * z)

    # fair value: imbalance leads the next increment by one event
    df = np.empty(n)
    df[0] = 0.0
    df[1:] = kappa * I[:-1] * tick + sigma * tick * mr.normal(0, 1, n - 1)
    fair = f0 + np.cumsum(df)

    mid_ticks = np.round(fair / tick).astype(np.int64)
    hb = mr.integers(1, 3, n)
    ha = mr.integers(1, 3, n)
    best_bid = (mid_ticks - hb) * tick
    best_ask = (mid_ticks + ha) * tick

    base_sz = 100.0 * np.exp(mr.normal(0, 0.3, n))
    bid_l1 = base_sz * np.exp(beta * I)
    ask_l1 = base_sz * np.exp(-beta * I)
    decay = 0.6 ** np.arange(levels)

    bids_price, bids_size, asks_price, asks_size = [], [], [], []
    for k in range(levels):
        bids_price.append(best_bid - k * tick)
        asks_price.append(best_ask + k * tick)
        bids_size.append(bid_l1 * decay[k])
        asks_size.append(ask_l1 * decay[k])
    bids_price = np.stack(bids_price, axis=1)
    asks_price = np.stack(asks_price, axis=1)
    bids_size = np.stack(bids_size, axis=1)
    asks_size = np.stack(asks_size, axis=1)

    gaps = mr.exponential(1.0, n)
    ts = np.cumsum(gaps)
    ts = ((ts - ts[0]) / (ts[-1] - ts[0] + 1e-9) * span_ns).astype(np.int64)

    book = pl.DataFrame({
        "market_id": [f"SYN{m:02d}"] * n,
        "timestamp_ns": ts,
        "bid_price": best_bid, "bid_size": bid_l1,
        "ask_price": best_ask, "ask_size": ask_l1,
        "mid": (best_bid + best_ask) / 2.0, "spread": best_ask - best_bid,
        "bids_price": bids_price.tolist(), "bids_size": bids_size.tolist(),
        "asks_price": asks_price.tolist(), "asks_size": asks_size.tolist(),
    })

    # trade tape: Poisson arrivals, taker side biased by I (the planted signal)
    n_tr = max(1, int(trade_rate_per_min * minutes))
    tr_t = np.sort(mr.integers(int(ts[0]), int(ts[-1]) + 1, n_tr))
    bidx = np.clip(np.searchsorted(ts, tr_t, side="right") - 1, 0, n - 1)
    p_buy = 0.5 + 0.4 * I[bidx]
    side = np.where(mr.random(n_tr) < p_buy, 1, -1)
    tsize = mr.exponential(5.0, n_tr)
    tprice = np.where(side > 0, best_ask[bidx], best_bid[bidx])
    trades = pl.DataFrame({
        "market_id": [f"SYN{m:02d}"] * n_tr,
        "timestamp_ns": tr_t.astype(np.int64),
        "price": tprice, "size": tsize, "side": side.astype(np.int64),
    })
    return book, trades


def generate_with_trades(
    *, n_markets: int = 12, minutes: float = 120.0, base_rate_per_min: float = 60.0,
    trade_rate_per_min: float = 20.0, levels: int = 5, tick: float = 0.01, f0: float = 100.0,
    kappa: float = 0.8, sigma: float = 1.0, beta: float = 1.2, phi: float = 0.92, seed: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate (book panel, trade tape), both canonical schema."""
    books, trades = [], []
    for m in range(n_markets):
        b, t = _gen_one(m, minutes=minutes, base_rate_per_min=base_rate_per_min,
                        trade_rate_per_min=trade_rate_per_min, levels=levels, tick=tick,
                        f0=f0, kappa=kappa, sigma=sigma, beta=beta, phi=phi, seed=seed)
        books.append(b); trades.append(t)
    return pl.concat(books, how="vertical"), pl.concat(trades, how="vertical")


def generate_pool(**kw) -> pl.DataFrame:
    """Generate the book panel (canonical schema)."""
    return generate_with_trades(**kw)[0]


def generate(out_dir: Path | str = "experiments/synthetic", *,
             with_trades: bool = True, **kw) -> Path:
    """Generate and write a synthetic dataset (one book + one trade file per market).

    Returns the output directory (a valid ``LoadConfig.roots`` entry).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    books, trades = generate_with_trades(**kw)
    for mid, g in books.group_by("market_id"):
        name = mid[0] if isinstance(mid, tuple) else mid
        g.write_parquet(out_dir / f"{name}.parquet")
    if with_trades:
        for mid, g in trades.group_by("market_id"):
            name = mid[0] if isinstance(mid, tuple) else mid
            g.write_parquet(out_dir / f"{name}.trades.parquet")
    return out_dir


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Generate a synthetic LOB dataset (canonical schema).")
    p.add_argument("--out", default="experiments/synthetic")
    p.add_argument("--n-markets", type=int, default=12)
    p.add_argument("--minutes", type=float, default=120.0)
    p.add_argument("--rate", type=float, default=60.0, help="base book events/min")
    p.add_argument("--trade-rate", type=float, default=20.0, help="trades/min")
    p.add_argument("--kappa", type=float, default=0.8, help="signal strength (imbalance→move)")
    p.add_argument("--no-trades", action="store_true", help="skip the trade tape")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    d = generate(a.out, with_trades=not a.no_trades, n_markets=a.n_markets, minutes=a.minutes,
                 base_rate_per_min=a.rate, trade_rate_per_min=a.trade_rate, kappa=a.kappa, seed=a.seed)
    print(f"wrote synthetic dataset to {d}/")
