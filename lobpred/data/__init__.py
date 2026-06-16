"""Public data sources that emit the canonical LOB schema.

Two ways to get runnable data without any proprietary data:

  * ``synthetic``, a self-contained generator with a known, modest,
    imbalance-driven signal. Runs the whole pipeline end-to-end with zero
    downloads; useful for tests and for sanity-checking that the harness
    recovers a planted edge.
  * ``fi2010``, a loader for the public FI-2010 benchmark (Ntakaris et
    al. 2018), the standard academic LOB dataset.

Import the submodule you need, e.g. ``from lobpred.data import synthetic``.
Both return / write the canonical schema consumed by ``lobpred.dataset``:

    market_id, timestamp_ns,
    bid_price, bid_size, ask_price, ask_size, mid, spread,
    bids_price, bids_size, asks_price, asks_size   (lists, L levels)
"""

__all__ = ["synthetic", "fi2010"]
