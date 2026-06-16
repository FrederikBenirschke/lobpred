"""Loader for the public FI-2010 LOB benchmark → canonical schema.

FI-2010 (Ntakaris, Magris, Kanniainen, Gabbouj, Iosifidis 2018,
"Benchmark dataset for mid-price forecasting of limit order book data
with machine learning methods") is the standard academic LOB dataset:
10 levels of a Nasdaq Nordic book for 5 stocks over 10 days.

Get it from the authors' archive (search "FI-2010 benchmark dataset",
Etsin/CSC record `73eb48d7-4dbc-4a10-a52a-da745b47a649`). The common
distribution ships fixed-event text matrices, e.g.::

    Train_Dst_NoAuction_DecPre_CF_7.txt
    Test_Dst_NoAuction_DecPre_CF_9.txt

File layout (the standard one this loader assumes):

  * shape ``(149, n_events)``, features are ROWS, events are COLUMNS.
  * rows 0..39  : the LOB, interleaved per level as
                  ``[P_ask, V_ask, P_bid, V_bid]`` for level 1, then level
                  2, … level 10  (40 = 10 levels × 4).
  * rows 40..143: handcrafted features (ignored here, we build our own).
  * rows 144..148: 5 sign labels for horizons k ∈ {10,20,30,50,100}.

NOTE ON VARIANTS: the *normalized* variants (``ZScore``, ``MinMax``) can
emit negative "sizes", which makes size-weighted features (microprice,
imbalance) meaningless and trips ``require_two_sided``. Prefer a
non-normalized variant (``DecPre``) for the microstructure/grid feature
sets; the raw level features tolerate any variant. FI-2010 has no real
timestamps (it is event-indexed), so a synthetic 1-event-per-``dt_s``
clock is assigned; use the EVENT-horizon target on this data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

NS_PER_S = 1_000_000_000
N_LEVELS = 10
_LOB_ROWS = 40  # 10 levels × [P_ask, V_ask, P_bid, V_bid]


def load_matrix(
    mat: np.ndarray,
    *,
    market_id: str = "FI2010",
    dt_s: float = 1.0,
    t_start_ns: int = 0,
) -> pl.DataFrame:
    """Convert one FI-2010 feature matrix into the canonical schema.

    ``mat`` is the raw ``(149, n_events)`` array (features as rows). Returns
    a canonical book panel with ``N_LEVELS`` levels and a synthetic clock
    (one event every ``dt_s`` seconds, starting at ``t_start_ns``).
    """
    if mat.shape[0] < _LOB_ROWS:
        raise ValueError(
            f"expected >= {_LOB_ROWS} feature rows, got {mat.shape[0]}; "
            f"is this an FI-2010 matrix with features as ROWS?"
        )
    lob = mat[:_LOB_ROWS, :].T                      # (n_events, 40)
    n = lob.shape[0]
    block = lob.reshape(n, N_LEVELS, 4)             # per level: [P_a, V_a, P_b, V_b]
    ask_p, ask_v = block[:, :, 0], block[:, :, 1]
    bid_p, bid_v = block[:, :, 2], block[:, :, 3]

    best_bid, best_ask = bid_p[:, 0], ask_p[:, 0]
    ts = (t_start_ns + (np.arange(n) * dt_s * NS_PER_S)).astype(np.int64)

    return pl.DataFrame({
        "market_id": [market_id] * n,
        "timestamp_ns": ts,
        "bid_price": best_bid,
        "bid_size": bid_v[:, 0],
        "ask_price": best_ask,
        "ask_size": ask_v[:, 0],
        "mid": (best_bid + best_ask) / 2.0,
        "spread": best_ask - best_bid,
        "bids_price": bid_p.tolist(),
        "bids_size": bid_v.tolist(),
        "asks_price": ask_p.tolist(),
        "asks_size": ask_v.tolist(),
    })


def load_file(path: Path | str, *, market_id: str | None = None, dt_s: float = 1.0,
              t_start_ns: int = 0) -> pl.DataFrame:
    """Load one FI-2010 ``.txt`` matrix file into the canonical schema."""
    path = Path(path)
    mat = np.loadtxt(path)
    return load_matrix(mat, market_id=market_id or path.stem, dt_s=dt_s, t_start_ns=t_start_ns)


def load(paths: Path | str | list, *, dt_s: float = 1.0) -> pl.DataFrame:
    """Load one or more FI-2010 files (or a directory of ``.txt``) → pool.

    Each file becomes its own ``market_id`` (so files never straddle a
    walk-forward split), all placed on a shared synthetic clock.
    """
    if isinstance(paths, (str, Path)) and Path(paths).is_dir():
        files = sorted(Path(paths).glob("*.txt"))
    elif isinstance(paths, (str, Path)):
        files = [Path(paths)]
    else:
        files = [Path(p) for p in paths]
    if not files:
        raise FileNotFoundError(f"no FI-2010 .txt files at {paths}")
    return pl.concat([load_file(f, dt_s=dt_s) for f in files], how="vertical")


def write_parquets(paths: Path | str | list, out_dir: Path | str, *, dt_s: float = 1.0) -> Path:
    """Convert FI-2010 file(s) to canonical parquet(s) under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = load(paths, dt_s=dt_s)
    for mid, g in pool.group_by("market_id"):
        name = mid[0] if isinstance(mid, tuple) else mid
        g.write_parquet(out_dir / f"{name}.parquet")
    return out_dir
