# lobpred: short-horizon price prediction on prediction-market order books

A prediction market trades contracts that pay one dollar if an event occurs and
nothing otherwise, so each price lies in [0, 1] and behaves as the market's
probability of that event. Kalshi and Polymarket maintain live limit order books
for thousands of such contracts.

`lobpred` asks whether the next few seconds of price movement are predictable
from the current book, and benchmarks deep sequence models against gradient
boosted trees on that question. It is a prediction study: it reports forecast
quality, not trading PnL.

The package loads order-book recordings into one schema, builds microstructure
features, trains linear, tree and deep models on a shared target and split, and
scores them. A synthetic generator with a planted
signal and a loader for the public FI-2010 benchmark let every result run
without private data.

## What it finds

**The predictand changes the score more than the model does.** Predicting the
microprice scores roughly twice as high as predicting the mid on identical rows
(0.53 against 0.22 for the best model). The microprice is defined by the same
touch sizes the features report and moves whenever those sizes move. Against
474,449 trade prints it is closer to the execution price than the mid on 40% of
them.

**The horizon changes the score more than the model does.** Across 1 to 60
seconds the network's correlation falls from 0.58 to 0.44 while the tree stays
nearly flat. Horizon sensitivity is a property of the sequence model rather
than of the data, so a comparison run at one horizon can rank models
differently than a swept one.

**Model class changes it least.** Sequence models read a window of history
while a tree reads a single snapshot. Given the same 32-tick window, LightGBM
comes within 0.02 of the best network on microprice and matches it on mid.

**The input representation changes it most.** Adding the raw price and size
ladder on top of engineered features lifts every model, tree and network alike.
The lift survives scoring within price-decile buckets, so it is not an artifact
of the bounded [0, 1] price scale.

## Quickstart

```bash
pip install -e ".[deep,plot,dev]"     # deep = torch; plot = matplotlib; dev = pytest

# generate a synthetic dataset with a known, modest, planted signal
python -m lobpred.data.synthetic --out experiments/synthetic --n-markets 12 --minutes 120

# model comparison + feature-group importance
KMP_DUPLICATE_LIB_OK=TRUE python -m lobpred.analyze \
    --roots experiments/synthetic --horizon-events 50

# the staged feature-comparison table: each phase adds one feature group
KMP_DUPLICATE_LIB_OK=TRUE python -m lobpred.run \
    --roots experiments/synthetic --phases 0 1 2 3 4 5 --horizon 30

# point-exit vs smoothed-hold target across horizons
KMP_DUPLICATE_LIB_OK=TRUE python -m lobpred.target_study \
    --roots experiments/synthetic --horizons 2 5 10 30

# diagnostic plots for one phase (loss/grad curves, error, attention pockets)
KMP_DUPLICATE_LIB_OK=TRUE python -m lobpred.diagnose \
    --roots experiments/synthetic --phase 3 --out experiments/diag

# tests (no torch, no downloads)
pytest -q
```

`KMP_DUPLICATE_LIB_OK=TRUE` is needed only because PyTorch and LightGBM both
link libomp in one process.

To run on the **FI-2010** benchmark, download it (Ntakaris et al. 2018) and
convert to the canonical schema:

```python
from lobpred.data import fi2010
fi2010.write_parquets("BenchmarkDatasets/.../*.txt", "experiments/fi2010")
# then: python -m lobpred.analyze --roots experiments/fi2010 --horizon-events 50
```

## Data

Every module reads one contract, a single row per book update:

```
market_id, timestamp_ns,
bid_price, bid_size, ask_price, ask_size, mid, spread,
bids_price, bids_size, asks_price, asks_size   # lists, L levels each
```

Anything that can be put in this shape works, so the same pipeline runs on the
synthetic generator, on FI-2010, and on live venue recordings.

The results below come from live Kalshi and Polymarket order books: 12.0M book
events across 371 daily-temperature markets recorded over three months, and an
earlier 923-market recording spanning sports and weather. The raw recordings
contain trader usernames and stay out of the repository; the collected dataset
is available on request at benirschke.math@gmail.com.

## Features

Feature families are built on the same rows, so they compare on equal footing.
Every family except **levels** is price-invariant, which permits pooling across
instruments.

| family | builder | what it is |
|---|---|---|
| **levels** | `add_paper_features` | raw L-level price + size (non-stationary; the reference paper's inputs) |
| **grid** | `add_grid_features` | resting size on a fixed tick grid from touch |
| **scalar** | `add_scalar_features` | order-flow imbalance, book imbalance at L1/3/5, micro-gap, spread |
| **flow (A)** | `add_flow_features` | rolling and multi-level order-flow imbalance, smoothed book imbalance |
| **history (B)** | `add_history_features` | lagged returns (event and time) + realized volatility |
| **shape (C)** | `add_shape_features` | queue concentration, depth log-ratio, resting depth, deeper micro-gaps |
| **trade (D)** | `add_trade_features` | trade-flow imbalance, intensity, volume, VWAP−mid, Amihud illiquidity |
| **activity (E)** | `add_activity_spread_features` | mid-change and update counts, rolling spread mean and volatility |
| **impact (F)** | `add_impact_features` | limit-add and cancel intensity from book diffs, rolling Kyle-λ |
| **smoothed (G)** | `add_smoothed_features` | time-aware EWMA of micro-gap, imbalance and order flow |

All families read the book only, except **trade (D)**, which needs a trade tape
(`--with-trades`). Every feature looks backward through the same idiom as the
forward target, so no window sees past `t`.

### Fixed grid over rank-based levels

Instruments differ in book geometry: one packs 5 levels into 4 ticks, another
spreads them over 18. "Level 3" then sits at a different price distance on each,
so ranking by level teaches the model which instrument it is looking at.
`add_grid_features` records resting size at a fixed tick offset from touch, so
the same offset means the same price distance everywhere.

## How it works

### Targets

Two forms of forward move, both computed strictly forward of `t`: a **point**
move, `price(t+h) − price(t)`, and a **smoothed** move, the average price over
the hold minus the price at `t`. Either can be measured against the mid or the
microprice, and either can use a time horizon or an event horizon.

The harness trains two heads. The regression head predicts the forward change
and is scored on correlation and sign-hit rate. The three-class sign head reports
accuracy and macro-F1 for comparison with the published literature.

### Splits

Train and test are separated by wall-clock time, with a gap of at least one
horizon removed on each side of the boundary so no label straddles it.
Normalization statistics are fit on training rows only.

The split is purely temporal, so a market that trades on both sides of the
boundary appears in both. That is the intended design, since the question is
whether the next few seconds are predictable rather than whether skill transfers
to unseen markets, but it means these numbers say nothing about cross-market
generalization.

### Market selection

Markets are ranked by mid-price changes per minute rather than by quote count.
The two diverge: a book can post several hundred quotes per minute while its mid
moves twice.

### Models

Baselines are ridge, LightGBM and multinomial logistic, all reading a
single snapshot unless given a flattened window. The sequence models are:

- **TCN** — dilated causal convolutions over the feature window.
- **DeepLOB** — the Zhang–Zohren–Roberts CNN-LSTM: convolutions across price
  levels, then an inception block, then an LSTM over time.
- **AxialAttentionLOB** — attention along the time and level axes separately.
- **PerLevelLOB** — convolves across levels then runs an LSTM, on a per-level
  stationary tensor rather than raw prices.
- **SeqLSTM** — a plain LSTM, as a no-pooling control.

Regression targets are standardized for the networks and mapped back to price
units afterward. DeepLOB needs a LayerNorm before its LSTM and gradient
clipping to train stably.

## Results

Measured on 12.0M book events across 371 live markets, with a chronological
split, 400,000 train and ~447,000 test windows, sequence length 32.

Re-running one configuration across five seeds gives a standard deviation of
0.0036, so differences below about 0.007 are not distinguishable from run-to-run
noise.

### Predictand

Same rows, same features, same split; only the predicted price changes.

| predictand | ridge | LightGBM | TCN |
|---|---|---|---|
| microprice | 0.250 | 0.454 | **0.528** |
| mid | 0.179 | 0.170 | **0.221** |

Microprice scores roughly twice the mid on every model. It is an interpolation
between bid and ask weighted by the touch sizes, so it moves whenever those
sizes move, with no trade and no price level clearing — and the features that
predict it include the sizes that define it.

Joining 474,449 trade prints to the book state at the same millisecond, the
microprice is closer to the execution price than the mid on 40.15% of them,
against 50% for no information. It is closer only where the spread is 0-2
ticks, which covers 3.3% of prints; half of all prints occur at spreads of 8
ticks or more.

### Horizon

Correlation against forward horizon. Time horizons use the smoothed target,
event horizons a point change; compare within an axis, not across.

| horizon | microprice: ridge / lgbm / tcn | mid: ridge / lgbm / tcn |
|---|---|---|
| 1 s | 0.250 / 0.452 / **0.577** | 0.158 / 0.143 / 0.233 |
| 5 s | 0.248 / 0.459 / 0.541 | 0.178 / 0.165 / **0.249** |
| 10 s | 0.251 / 0.441 / 0.533 | 0.183 / 0.191 / 0.234 |
| 19 s | 0.250 / 0.454 / 0.528 | 0.179 / 0.170 / 0.221 |
| 60 s | 0.245 / 0.450 / 0.444 | 0.183 / 0.150 / 0.126 |
| 20 ev | 0.200 / 0.339 / 0.411 | 0.182 / 0.138 / 0.244 |
| 50 ev | 0.212 / 0.352 / 0.407 | 0.175 / 0.120 / 0.190 |
| 100 ev | 0.215 / 0.344 / 0.383 | 0.180 / 0.126 / 0.167 |

The network decays monotonically along the time axis while LightGBM is nearly
flat. On mid at 60 seconds the network (0.126) falls below ridge (0.183).

The smoothed target averages fewer ticks at a short horizon, so it is noisier
per observation as correlation rises. A 1-second horizon is also not reachable
by a strategy that has to cross a spread.

### Lookback against architecture

The tree is given progressively more of what the network already sees.
"Flattened" hands it all 32 ticks as 1,120 features.

| view | microprice | mid |
|---|---|---|
| LightGBM, snapshot (35 feat) | 0.455 | 0.149 |
| LightGBM, flattened (1,120 feat) | 0.501 | **0.231** |
| LightGBM, flattened, larger | 0.503 | 0.225 |
| LightGBM, flattened, engineered only | 0.308 | 0.113 |
| TCN (32 x 35) | **0.524** | **0.234** |
| *flattened − snapshot* | *+0.046* | *+0.082* |
| *best flattened − TCN* | *−0.021* | *−0.003* |

Lookback buys +0.046 and +0.082; the residual model-class gap is −0.021 and
−0.003. The larger tree is no better than the smaller one, so the flattened tree
is not capacity-starved. The flattened tree on engineered features alone stays
far below the flattened tree with the raw book, so lookback and representation
are complements rather than substitutes.

### Representation

19-second smoothed microprice change, 371 markets.

| model | engineered (15) | + raw book (35) | lift |
|---|---|---|---|
| ridge | 0.204 | 0.247 | +0.043 |
| LightGBM | 0.254 | 0.449 | +0.195 |
| TCN | 0.306 | **0.533** | +0.227 |
| DeepLOB | 0.301 | 0.354 | +0.053 |

Scored within price-decile buckets, where the absolute price is roughly constant
and cannot drive the result, 94% of the lift survives and it is positive in all
ten deciles. On the mid target it is smaller and noisier but still positive in
eight of ten.

### Per-market spread

Pooled correlation averages markets that differ by more than 2x. On the earlier
923-market corpus LightGBM scored 0.56 on tennis, 0.36 on weather and 0.23 on
NBA. A model trained on one segment matches the pooled model on that segment's
own test windows, so pooling heterogeneous markets does not reduce per-segment
accuracy. The pooled figure is an average over segments of differing
difficulty.

### Further results

Measured on the earlier 923-market corpus:

- **Target shape beats the model.** A smoothed hold reaches 0.407 correlation
  against 0.245 for a point exit at the same 10-second horizon, a wider gap than
  between any two models.
- **A frequently cited 71% three-class accuracy is a class-balance artifact.**
  Balancing the majority "stable" class drops it to 0.44-0.50, consistent with
  the same source's reported Up/Down F1 of about 0.50.
- **The signal concentrates in the spread and micro-gap.** Permutation
  importance gives Δcorr 0.177 for spread and micro-gap against 0.013 for
  imbalance and −0.001 for order-flow imbalance. At five levels the depth-pocket
  thesis fails the test.
- **Per-market normalization destroys the raw signal**, dropping correlation
  from 0.470 to 0.324: centering each market removes the absolute price level,
  which carries cross-market information.
- **Scale alone does not rescue the network on engineered features.** From 300K
  to 2M windows the LSTM moved 0.262 to 0.278 and the tree 0.323 to 0.330.
- **Stacking the network into the tree adds nothing:** a frozen network's
  prediction contributed +0.004 correlation and its embedding +0.001.

## Modules

```
lobpred/
  dataset.py        load, pool, base features, forward targets, activity tiers, windows, splits
  features.py       extended feature families + per-level tensor for the deep models
  baselines.py      persistence / ridge / lgbm / logistic / majority
  models.py         TCN, DeepLOB, AxialAttentionLOB, PerLevelLOB, SeqLSTM
  evaluate.py       metrics + the torch training loop + gap tables
  diagnostics.py    training-curve, error-analysis and attention plots
  analyze.py        model comparison + feature-group importance
  run.py            the staged feature-comparison table
  target_study.py   point vs smoothed target across horizons
  diagnose.py       train one phase with history, write plots
  microstructure.py microprice / imbalance / depth
  ofi.py            Cont-Kukanov order-flow imbalance
  data/
    synthetic.py    self-contained generator with a planted signal
    fi2010.py       loader for the public FI-2010 benchmark
examples/
  deeplob_walkthrough.py   per-level tensor -> PerLevelLOB -> comparison to LightGBM
  regime_scaling.py        deep vs tree as training size grows
```

`examples/deeplob_walkthrough.py` runs the deep-learning path end to end in
about a minute on synthetic data, with no downloads.

## Where deep learning wins

DeepLOB reports F1 ≈ 0.83 at the shortest horizon on FI-2010, ahead of the
classical baselines in that paper. Its CNN learns features off the raw 10-level
ladder that hand-engineered scalars discard, and a tree discards them too. Two
caveats: FI-2010's baselines are linear and TABL networks, so no gradient-boosted
tree was tuned against DeepLOB there, and it comes from a different venue at a
different scale.

That maps a regime over input representation, scale and depth rather than a
head-to-head on one dataset:

- **Engineered features at medium scale** (here): a snapshot tree matches or
  beats the sequence networks, and more data widens the tree's lead.
- **Raw book at large scale and depth** (FI-2010): the CNN-LSTM leads on
  10-level input.

This venue's books sit between the two: the raw book lifts every model, but at
five levels and this scale no network pulls ahead of a tree given the same
input.

## Limitations

- **No confidence intervals.** The smoothed label windows overlap heavily, so
  consecutive rows share almost their entire label and the effective sample is
  far below the row count. A naive interval on that row count would be roughly
  an order of magnitude too tight. A block bootstrap over markets or
  market-days would be the appropriate estimator; it has not been run.
- **Correlation is tail-decided on this corpus.** The forward-return
  distribution has excess kurtosis 55.5: the middle 50% of moves span ±0.0014
  while the 1st and 99th percentiles are ±0.088. Correlation is therefore
  largely a statement about rare large moves, and where it disagrees with
  hit-rate, hit-rate is the more robust comparison.
- **No costed backtest.** Every number is forecast quality against a price
  change, not PnL. Half of all trade prints occur at spreads of 8 ticks or more,
  so a taker signal would likely lose most of this edge to the spread.
- **The primary corpus is single-venue and single-instrument.** All 371 markets
  are Kalshi daily-temperature contracts, so any cross-corpus comparison is
  confounded with the venue and instrument mix.
- **The horizon sweep confounds horizon with target smoothing**, since the time
  axis uses a smoothed target. The same sweep with a point target has not been
  run.
- **The five-level depth limit.** These books run five levels deep, so the
  depth-attention thesis is untestable past L5.
- **FI-2010 ships pre-normalized and event-indexed**, so it supports the level
  and scalar feature sets and the event-horizon target, but not the fixed-tick
  grid, which needs a real tick.
- **The synthetic generator plants its signal**, so its absolute numbers check
  the pipeline rather than measure anything.

## License

MIT. See [LICENSE](LICENSE).
