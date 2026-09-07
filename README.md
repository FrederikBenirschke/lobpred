# lobpred: predicting short-horizon order-book moves on prediction markets

A prediction market trades contracts that pay one dollar if an event occurs and
nothing otherwise, so each price lies in [0, 1] and behaves as the market's
probability of that event. Kalshi and Polymarket maintain live limit order books
for thousands of such contracts, covering NBA games, World Cup matches, and
daily high-temperature settlements. This study asks whether the next few seconds
of price movement are predictable from the current book, and what actually
drives that prediction.

Deep sequence models are widely reported to beat gradient-boosted trees on
order-book data. This benchmark tests that claim while varying four things, not
one: how the order book is **represented**, which **model** reads it, how far
ahead the **horizon** looks, and which **price** is being predicted. The
representations are a set of scale-free microstructure features (order-flow
imbalance, book imbalance, relative depth) and the raw order book itself (bid
and ask prices and sizes across five levels). The models are linear and tree
baselines (ridge, lasso, LightGBM, logistic) and deep sequence networks (a TCN,
DeepLOB, an axial-attention model, and an LSTM).

One confound recurs in such comparisons and is controlled here throughout: a
sequence network reads a window of history while a tree reads a single
snapshot, so the tree is given the same window before any model is judged.

**The headline is that the model class is the least important of the four
choices.** Handing the tree the same 32-tick window closes almost the entire
apparent gap between it and the best network. Meanwhile two settings that are
usually inherited rather than measured — the prediction horizon and the choice
of microprice versus mid as the predictand — each move the score further than
swapping the model does. A study that fixes those two and sweeps only the
architecture is measuring the smallest available effect.

> **The data.** This is a prediction study: it reports forecast quality, not
> trading PnL. The primary corpus is 12.0M order-book events across 371 Kalshi
> daily-temperature markets, recorded 2026-06-14 to 2026-09-04 and reduced to
> one row per venue event. An earlier corpus of 923 markets across Kalshi and
> PolymarketUS — MLB (138), NBA (63), tennis (71), WNBA (7), World Cup soccer
> (102), 172 daily-temperature, 26 other, plus 344 on PolymarketUS — is
> retained for cross-corpus comparison and reported separately, since it
> carries a known recorder defect described below. The raw recordings contain
> trader usernames and stay out of the repository. The synthetic generator and
> the FI-2010 loader reproduce every pipeline and result. The collected dataset
> is available on request at benirschke.math@gmail.com.

## Summary

**The predictand matters more than the model.** The literature's default target
is the microprice, a size-weighted interpolation between the best bid and ask.
Predicting it scores roughly twice as high as predicting the mid on the same
rows (TCN 0.53 versus 0.22). That gap is not good news: microprice moves
whenever the touch *sizes* move, with no trade and no change in either quoted
price, so the book features that predict it include the sizes that define it.
Checked against 474,449 actual trade prints, the microprice is closer to the
execution price than the mid on only **40.15%** of them — below the 50% line a
coin flip would give. It is the easier target and the worse description of
where trading happens. The mid number is the defensible one.

**The horizon matters more than the model.** Sweeping 1 to 60 seconds, the
network's correlation falls monotonically from 0.58 to 0.44; the inherited
19-second horizon sits near the bottom of that range. Choosing 1 second instead
would have bought more than the entire architecture comparison does. The tree,
by contrast, is nearly flat across the same axis, so horizon sensitivity is a
property of the sequence model rather than of the data.

**Model class has little effect once the inputs match.** Given the same 32-tick
window, LightGBM comes within 0.02 of the best network on microprice and ties
it on mid (0.231 versus 0.234 — a difference well inside the +/-0.0036
run-to-run noise, so a tie in the sense of *unresolved*, not *proven equal*). The network leads only when the comparison
grants it history the snapshot tree never receives. This reproduces on a second, independently
recorded corpus with a different venue and instrument mix (which carries a known
recorder defect — see Limitations) and on both predictands.

**Representation is the one lever that clearly pays.** Adding the raw price and
size ladder on top of the engineered features lifts every model, tree and
network alike. The lift is not an artifact of the bounded [0, 1] price scale:
scored within price-decile buckets, where the absolute price is roughly
constant and cannot drive the result, 94% of it survives and it is positive in
all ten deciles.

Market selection follows price activity rather than quote count. A book that
posts hundreds of quotes per minute while its mid price moves twice contributes
noise, so the ranking uses the mid-move rate.

## Quickstart

```bash
pip install -e ".[deep,plot,dev]"     # deep = torch; plot = matplotlib; dev = pytest

# 1) generate a synthetic dataset with a known, modest, planted signal
python -m lobpred.data.synthetic --out experiments/synthetic --n-markets 12 --minutes 120

# 2) model comparison + feature-group importance
KMP_DUPLICATE_LIB_OK=TRUE python -m lobpred.analyze \
    --roots experiments/synthetic --horizon-events 50

# 3) the staged feature-comparison gap table (phases 4–5 add the extended + trade features)
KMP_DUPLICATE_LIB_OK=TRUE python -m lobpred.run \
    --roots experiments/synthetic --phases 0 1 2 3 4 5 --horizon 30

#    feature-importance with the trade-tape families included
KMP_DUPLICATE_LIB_OK=TRUE python -m lobpred.analyze \
    --roots experiments/synthetic --horizon-events 50 --with-trades

# 4) point-exit vs smoothed-hold target across horizons
KMP_DUPLICATE_LIB_OK=TRUE python -m lobpred.target_study \
    --roots experiments/synthetic --horizons 2 5 10 30

# 5) diagnostic plots for one phase (loss/grad curves, error, attention pockets)
KMP_DUPLICATE_LIB_OK=TRUE python -m lobpred.diagnose \
    --roots experiments/synthetic --phase 3 --out experiments/diag

# tests (no torch / no downloads needed)
pytest -q
```

`KMP_DUPLICATE_LIB_OK=TRUE` is only needed because PyTorch and LightGBM
both link libomp in one process.

To run on the **FI-2010** benchmark instead, download it (search "FI-2010
benchmark dataset", Ntakaris et al. 2018) and convert to the canonical
schema:

```python
from lobpred.data import fi2010
fi2010.write_parquets("BenchmarkDatasets/.../*.txt", "experiments/fi2010")
# then: python -m lobpred.analyze --roots experiments/fi2010 --horizon-events 50
```

## Data: one canonical schema

Every module reads one contract, a single row per book update:

```
market_id, timestamp_ns,
bid_price, bid_size, ask_price, ask_size, mid, spread,
bids_price, bids_size, asks_price, asks_size   # lists, L levels each
```

Every feature family is built on the same rows, so they compare on equal
footing. The first three form the base set; the rest are stationary extensions
(`features.py`). Every family except **levels** is price-invariant, which permits
pooling across instruments.

| family | builder | what it is |
|---|---|---|
| **levels** | `add_paper_features` | raw L-level px+size (non-stationary; reference paper inputs) |
| **grid** | `add_grid_features` | resting size on a **fixed tick grid** from touch, comparable across instruments |
| **scalar** | `add_scalar_features` | OFI, L1/3/5 order-book imbalance (OBI), micro-gap, spread |
| **flow (A)** | `add_flow_features` | rolling OFI (1/5/30 s + 10/50-event) + multi-level OFI (L2–L3) + **rolling OBI** (smoothed L1 book imbalance) |
| **history (B)** | `add_history_features` | lagged microprice returns (event + time) + realized vol |
| **shape (C)** | `add_shape_features` | queue concentration, depth log-ratio, **log resting depth (volume)**, deeper micro-gaps (L3/L5), L10 imbalance |
| **trade (D)** | `add_trade_features` | TFI, trade intensity, **traded volume**, VWAP−mid, **Amihud illiquidity** over 5/30 s (needs a trade stream) |
| **activity/spread (E)** | `add_activity_spread_features` | **mid-change count** & update count over last 1/5/30 s + **rolling spread mean / volatility** + **relative spread** (spread/mid) |
| **impact (F)** | `add_impact_features` | order-flow decomposition (**limit-add / cancel intensity** from book diffs) + rolling **Kyle-λ** price impact (Δmid on OFI) |
| **smoothed (G)** | `add_smoothed_features` | time-aware **EWMA** of micro-gap, OBI, OFI, depth-normalized OFI (irregular-time `ewm_mean_by`; smoother + horizon-matched vs the hard rolling windows) |

A/B/C/E/F/G read the book only and run by default in `analyze` and
`target_study`; D needs the trade tape (`--with-trades`, requires
`*.trades.parquet`; the synthetic generator emits one, FI-2010 has none). Every
feature looks backward through the same searchsorted/cumsum idiom as the forward
target, so no window peeks past `t`. `mid_moves_{w}s` counts realized price
changes per row, the per-row version of the discovery axis that drives activity
tiering.

### Fixed grid over rank-based levels

Instruments differ in book geometry. One packs 5 levels into 4 ticks; another
spreads them over 18. "Level 3" then sits at a different price distance on each,
so ranking by level teaches the model which instrument it sees.
`add_grid_features` records resting size at a fixed tick-offset from touch, so
offset `o` marks the same price distance everywhere and a "pocket" becomes a
nonzero bucket at a set offset.

## Methodology

### Guarding against lookahead

Three rules keep future information out of the training data:

- **The target is computed forward.** It is never taken from a backward-looking
  window. Two forms are used: a point move, `price(t+h) − price(t)`, and a
  smoothed move, `mean(price over (t, t+h]) − price(t)` (the average price over
  the hold). Every price entering the target falls after `t`, so neither form
  can observe its own outcome.
- **Normalization is fit on the training set only.** The z-score statistics are
  estimated on the train rows and then applied to the test rows.
- **The split is by time, with an embargo.** Train and test are separated by
  wall-clock time, and a gap of at least one horizon is removed on each side of
  the boundary so no label straddles it (the purged walk-forward split of López
  de Prado). The split is purely temporal: markets are pooled across the
  boundary, so a market that trades on both sides of it appears in train and
  test alike. That is the intended design — the question is whether the next
  few seconds are predictable, not whether skill transfers to unseen markets —
  but it means these numbers do not bound cross-market generalization.

### Measuring activity by real price moves

A market is active when its price actually moves. `add_activity_tier` ranks
markets by `mid_moves_per_min` — the number of genuine mid-price changes — and
ignores `updates_per_min`, which only counts quotes being posted and pulled. A
book that posts 500 quotes a minute while its mid moves twice contributes mostly
noise. Ranking by quote count carries that noise into every downstream result,
so the ranking uses real price moves instead.

### Adding features one group at a time

`run.py` builds up the feature set in stages and reports the change in score at
each step: `phase1−phase0` (microprice target), `phase2−phase1` (fixed grid vs.
raw levels), `phase3−phase2` (order-flow scalars), `phase4−phase3` (flow
dynamics + returns/volatility + book shape + activity/spread), `phase5−phase4`
(trade tape), plus deep vs. simple model within each stage. The step-by-step
changes are the result of interest: absolute scores move with the fold, but the
differences between consecutive stages hold steady.

### Two prediction targets

The harness trains two output heads, which fail in different ways. The
regression head predicts the forward price change and is scored on correlation
and hit-rate; R² is left out, since cross-fold regime shift dominates it. The
three-class sign head reports accuracy and macro-F1 for comparison with the
published literature. The regression target is standardized for the neural
networks and the predictions are mapped back to price units afterward, which
keeps outputs in range. DeepLOB needs a LayerNorm before its LSTM and gradient
clipping, otherwise its gradient norm blows up to ~1e17; the code documents both
fixes.

## Results (development dataset)

These numbers come from the development data: live Kalshi and PolymarketUS
order books (the raw recordings stay private). The synthetic and FI-2010 paths
reproduce the pipeline and the qualitative conclusions.

**Primary corpus.** 11,962,182 book events, 371 Kalshi daily-temperature
markets, 2026-06-14 to 2026-09-04. One row per venue event: Kalshi's protocol
emits one message per price level and the client re-emits the whole ladder
after each, so raw rows over-count events by ~1.28x and are collapsed on the
venue's own event stamp. Markets are the 400 most price-active by mid-move
rate, capped at 12M rows. **The corpus is single-venue** — PolymarketUS is
present in the recording but no PolyUS market survived the activity cut — so
every cross-corpus comparison below is confounded with the venue mix.

**Protocol.** Single chronological split at the 66th percentile, train-only
z-scoring, embargo equal to the label horizon, 400,000 train and ~447,000 test
windows, sequence length 32. Scored on correlation and sign-hit; R² is omitted
because cross-fold regime shift dominates it.

**How precise are these numbers? +/-0.0036.** Every table below is a single
split with a single seed. Re-running the same configuration across 5 seeds
(varying the window subsample and the train/test draw) gives sd **0.0036**,
range **0.0087**, on the snapshot-LightGBM eng+raw cell. **Differences below
~0.007 are not distinguishable from run-to-run noise** — so read Finding 3's
mid-target "tie" (-0.003) as unresolved rather than as a demonstrated equality,
while its microprice gap (-0.021, ~6 sd) is a real ordering.

No confidence intervals are computed anywhere in this study, and that is a real
limitation rather than an oversight: the 19-second forward-average label windows
overlap heavily, so consecutive rows share almost their entire label and the
effective sample is far below the ~447,000 row count. A naive CI on that row
count would be roughly an order of magnitude too tight. The honest estimator is
a block bootstrap over markets or market-days; it has not been run.

**Read correlation with care on this corpus.** The forward-return distribution
has excess kurtosis 55.5 — the middle 50% of moves span ±0.0014 while the 1st
and 99th percentiles are ±0.088, a factor of 60. Nearly all the variance sits
in a small fraction of rows, so correlation is largely a statement about the
model's grip on rare large moves. Where correlation and hit-rate disagree, the
hit-rate is the more robust comparison.

### Finding 1: the predictand outranks the model

Same rows, same features, same split; only the predicted price changes.
19-second forward-average target, 35 features.

| predictand | ridge | LightGBM | TCN |
|---|---|---|---|
| microprice | 0.250 | 0.454 | **0.528** |
| mid | 0.179 | 0.170 | **0.221** |

Microprice scores roughly twice the mid on every model. Two mechanisms are
consistent with that and this test does not separate them: microprice may be
partly *self*-predictable, since it is an interpolation weighted by the very
touch sizes the features report, and moves whenever those sizes move without
any trade; or the mid may simply be harder, since it only changes when a price
level clears.

What settles the practical question is a separate measurement. Joining 474,449
distinct trade prints to the book state at the same millisecond (2,353,064
joined rows — one trade matches several book rows sharing its millisecond), the
microprice is closer to the actual execution price than the mid on **40.15%** — below the 50%
no-information line. It wins only where the spread is 0-2 ticks, which is 3.3%
of prints; half of all prints occur at spreads of 8 ticks or more. Microprice is
an equity-microstructure construct being applied to a book that does not look
like an equity book.

So the higher number is measured against the less faithful target. **Quote the
mid result.**

### Finding 2: the horizon outranks the model

Correlation against forward horizon, 35 features, both predictands. Time
horizons use the forward-average target; event horizons use a point change.

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

The network decays monotonically along the time axis while **LightGBM is nearly
flat** (0.441 to 0.459 across the whole sweep). Horizon sensitivity is therefore
a property of the sequence model, not of the corpus — which also means a
single-horizon comparison can rank models differently than a swept one.

The 19-second horizon used in the tables below was inherited, not chosen. It is
not a maximum: 1 second scores +0.049 higher on microprice, and the mid target
peaks at 5 seconds. On mid at 60 seconds the network (0.126) falls *below*
ridge (0.183), losing its advantage entirely.

Shorter is not automatically better. The target is a forward average, so a
shorter window averages fewer ticks and is noisier per observation even as
correlation rises, and a 1-second horizon is not reachable by a strategy that
must cross a spread. The point is narrower: the horizon axis carries more
signal than the model axis, and it is usually not swept.

Event horizons sit uniformly below time horizons because a forward average is
smoother than a point change. Compare within an axis, never across.

### Finding 3: with matched inputs, the tree catches the network

The apparent architecture gap is an input-width gap. Below, the tree is given
progressively more of what the network already sees; "flattened" hands it all
32 ticks as 1,120 features.

| view | microprice | mid |
|---|---|---|
| LightGBM, snapshot (35 feat) | 0.455 | 0.149 |
| LightGBM, flattened (1,120 feat) | 0.501 | **0.231** |
| LightGBM, flattened, larger | 0.503 | 0.225 |
| LightGBM, flattened, engineered only | 0.308 | 0.113 |
| TCN (32 x 35) | **0.524** | **0.234** |
| *flattened − snapshot* | *+0.046* | *+0.082* |
| *best flattened − TCN* | *−0.021* | *−0.003* |

Lookback is worth more than architecture: handing the tree the window buys
+0.046 and +0.082, while the residual model-class gap is −0.021 and −0.003. On
the mid target the two are a **tie**.

This reproduces the same test on the earlier 923-market corpus (snapshot 0.454,
flattened 0.504, TCN 0.499) across a different venue mix and a second
predictand. The larger tree is no better than the smaller one, so the flattened
tree is not capacity-starved. And the flattened tree on engineered features
alone (0.308 / 0.113) stays far below the flattened tree with the raw book, so
lookback and representation are complements, not substitutes.

### Finding 4: the raw book lifts every model, and the lift is real skill

19-second forward-average microprice change, 371 markets, single split.

| model | engineered (15) | + raw book (35) | lift |
|---|---|---|---|
| ridge | 0.204 | 0.247 | +0.043 |
| lasso | 0.204 | 0.246 | +0.042 |
| LightGBM | 0.254 | 0.449 | +0.195 |
| TCN | 0.306 | **0.533** | +0.227 |
| DeepLOB | 0.301 | 0.354 | +0.053 |
| attention | 0.278 | **0.533** | +0.255 |
| LSTM | 0.261 | 0.446 | +0.185 |

A bounded [0, 1] price drifts back toward the interior on its own, so some of
this lift could be mechanical reversion rather than skill. Scored within price
deciles, where the absolute price is roughly constant and cannot drive the
result, the lift survives: **+0.179 of the +0.191 pooled** (94%), positive in
all ten deciles. On the mid target it is smaller and noisier — +0.085 within
buckets against +0.064 pooled, positive in eight of ten — but still not a
price-level artifact.

One caveat the correlation column hides: on **hit-rate** the raw book is a
slight *loss* for most models (TCN 0.704 to 0.679) even as correlation rises
sharply. Given kurtosis 55.5, that is the expected signature — the raw book buys
grip on the large moves that dominate the variance, at a small cost to the sign
of the typical small move. Three-class accuracy rises across the board (TCN
0.543 to 0.591), which points the same way. "Raw helps" is therefore
metric-dependent, and this is a narrower claim than a correlation table alone
suggests.

Ridge and lasso agree to three decimals on the engineered set (0.204 / 0.204).
The two were run precisely to separate joint shrinkage from variable selection
across five collinear levels; that they land identically indicates the deeper
levels carry no signal a linear model can isolate from level one.

### Finding 5: pooled correlation hides a large per-market spread

On the earlier 923-market corpus, pooled correlation (0.33 tree, 0.28 net)
averages segments that differ by more than 2x: LightGBM scores 0.56 on tennis,
0.36 on weather, and 0.23 on NBA. A model trained on one segment matches the
pooled model on that segment's own test windows, so pooling heterogeneous
markets costs nothing — the low aggregate is an average of one easy segment and
several hard ones, not evidence of interference.

### Findings from the earlier corpus

These predate the current corpus and are retained because they are not
superseded by it, only rescoped. They were measured on the 923-market
Kalshi + PolymarketUS recording, whose known defect is described under
Limitations.

**The target specification beats the model.** LightGBM correlation, point exit
versus smoothed hold:

| hold W | point | smoothed |
|---|---|---|
| 2 s | 0.296 | 0.339 |
| 5 s | 0.268 | 0.394 |
| 10 s | 0.245 | **0.407** |
| 30 s | 0.220 | 0.390 |

The point exit decays with horizon; the smoothed hold peaks near 10 s. Moving
from point to smoothed at 10 s adds 0.16 correlation while the tree-network gap
is 0.13. This is the same lesson Findings 1 and 2 reach from two other
directions: target and horizon choices dominate architecture choices.

**A frequently cited 71% three-class accuracy is a class-balance artifact.**
Balancing the majority "stable" class in train and test drops accuracy to
0.44-0.50, consistent with the same source's reported Up/Down F1 of about 0.50.
On directional metrics the gap between models stays small. This is a
cross-dataset comparison with a different label deadband, so the transferable
part is the accuracy collapse under balancing, not the exact figure.

**The signal concentrates in the spread and micro-gap.** Feature-group
permutation importance on the base feature set (grid + scalars, 46 features,
~185-198 markets) gave Δcorr 0.177 for spread/micro-gap against 0.013 for
imbalance, ~0.010 for grid depth and −0.001 for OFI. At five levels the
depth-"pocket" thesis fails the permutation test — reported here as a negative
result. This ran on the base feature set and an earlier market selection, so it
is not directly comparable to the tables above; it has not been re-run on the
current corpus.

**Per-market normalization destroys the raw signal.** A per-market expanding
z-score drops raw-feature correlation from 0.470 to 0.324, because centering
each market removes the absolute price level, which carries cross-market
information. Global normalization is correct for raw features. An earlier
conclusion that "raw lifts only the tree" came from this normalization choice
combined with testing only an LSTM.

**Scale alone does not rescue the network on engineered features.** From 300K
to 2M windows the LSTM moved 0.262 to 0.278 while the tree moved 0.323 to
0.330; the gap held near 0.05 across 8.7x more data. This is scoped to the
LSTM on engineered features — Finding 3 shows the picture changes once the
input is raw and the model is a TCN.

**Stacking the network into the tree adds nothing.** A frozen network's
prediction contributed +0.004 correlation and its 128-dimensional embedding
+0.001, so the network's representation holds nothing the tree cannot reach
from the same window.

## Modules

```
lobpred/
  dataset.py        load + pool + base features + leak-safe targets + activity tiers + windows + walk-forward + trade loader
  features.py       extended families A–G + add_perlevel_features (per-level stationary (T,L,C) tensor for DL)
  baselines.py      persistence / ridge / lgbm / logistic / majority
  models.py         TCN, DeepLOB, AxialAttentionLOB, PerLevelLOB (conv-across-levels + LSTM), SeqLSTM (no-pool control)
  evaluate.py       prediction metrics (corr/hit/acc/F1) + the torch training loop + gap tables
  diagnostics.py    training-curve / error-analysis / attention-pocket plots
  analyze.py        model comparison + feature-group importance
  run.py            the staged feature-comparison gap table (phases 0–5)
  target_study.py   point vs smoothed target across horizons
  diagnose.py       train one phase with history, write plots
  microstructure.py vendored: microprice / imbalance / depth
  ofi.py            vendored: Cont/Kukanov order-flow imbalance
  data/
    synthetic.py    self-contained generator (book + aligned trade tape) with a planted signal (tested)
    fi2010.py       loader for the public FI-2010 benchmark
examples/
    deeplob_walkthrough.py   teaching script: per-level tensor → PerLevelLOB → vs lgbm
    regime_scaling.py        deep vs tree as training size grows (the scale axis of the regime boundary)
```

### Learning deep learning on the LOB

`examples/deeplob_walkthrough.py` shows the mechanics in one script:
synthetic book → **per-level stationary tensor**
(`add_perlevel_features`: OBI/OFI/rel-size per level) → **PerLevelLOB**
(convolves across levels + LSTM over time, the DeepLOB/Kolm representation)
→ training curve → comparison to LightGBM. Runs in ~1 min, no downloads.

A network trains stably on ~10⁵ windows, generated in seconds. More data alone
does not rescue the network on engineered features.
`examples/regime_scaling.py` sweeps the training size and compares SeqLSTM
against LightGBM on the same features, target, and walk-forward split:

| train N | LightGBM | SeqLSTM | gap (tree − net) |
|---|---|---|---|
| 10 K | 0.501 | **0.511** | −0.011 |
| 40 K | 0.520 | 0.515 | +0.005 |
| 160 K | 0.526 | 0.515 | +0.010 |
| 640 K | 0.527 | 0.515 | +0.012 |
| 1.7 M | 0.528 | 0.507 | +0.021 |

(corr on the held-out fold; SeqLSTM is a 2-seed ensemble; the synthetic signal
is planted, so the trend matters more than the level.) The network leads at 10 K.
As data grows, the tree keeps improving while the network plateaus, so the gap
moves the wrong way for "DL just needs scale." Engineered features leave the
network nothing extra to learn, and the new rows sharpen the tree.

### The regime boundary

Deep learning wins on the other axis: raw book input at large scale. DeepLOB
(Zhang, Zohren, Roberts) reports F1 ≈ 0.83 at the shortest horizon on FI-2010,
ahead of the classical baselines in that paper. Its CNN learns features off the
raw 10-level price/size ladder that hand-engineered scalars discard, and a tree
discards them too. Two caveats keep this honest. FI-2010's baselines are linear
and TABL nets, so no GBDT was tuned against DeepLOB there. FI-2010 also comes
from a different venue at a different scale. It maps a regime over input
representation, scale, and signal richness, not a head-to-head on one dataset.

The map has two cells:

- engineered features at medium scale (here): a snapshot tree matches or beats
  the sequence nets, and extra data widens the tree's lead. Switch to the raw
  book with matched inputs and the tree and the best net tie instead.
- raw book at large scale and depth (FI-2010): the CNN-LSTM earns its keep on
  10-level input that this venue's 5-level books cannot supply.

This venue's data sits between the cells: the raw book lifts every model, but at
five levels and this scale no net pulls ahead of a tree with the same input.

## Honest limitations

- **No confidence intervals, anywhere.** Every number is a single split and a
  single seed. Measured run-to-run variation is sd 0.0036 (range 0.0087), so
  differences below ~0.007 are noise. Worse, the label windows overlap heavily,
  so the effective sample is far below the row count and even a naive CI would
  be far too tight — the honest estimator is a block bootstrap over markets or
  market-days, which has not been run. The neural rows are single-seed, so the
  60-second collapse in Finding 2 may be an optimization failure rather than a
  result.
- **The horizon sweep confounds horizon with target smoothing.** The forward
  average is taken over the horizon itself, so as the horizon grows the target
  becomes smoother and its variance composition changes. Finding 2's decay
  therefore cannot separate "the network degrades at long horizons" from "the
  network is worse at heavily-averaged targets". The discriminating run — the
  same sweep with a point target — has not been done.
- **No costed backtest has run.** Every number here is forecast quality against
  a price change, not PnL. The spread on this book is wide — half of all trade
  prints occur at 8 ticks or more — so a taker signal would likely lose most of
  this to the spread. Nothing here is evidence of a tradeable edge, and the
  discriminating test (enter at ask, exit at bid, net of fees) is not run.
- **Correlation is tail-decided on this corpus.** Excess kurtosis is 55.5, so
  correlation mostly measures the model's grip on rare large moves. Hit-rate is
  the more robust statistic, and the two disagree in places — notably the raw
  book, which raises correlation while slightly lowering hit-rate.
- **The primary corpus is single-venue.** PolymarketUS is in the recording but
  no PolyUS market survived the activity cut, so cross-corpus comparisons are
  confounded with venue mix, instrument type and date range simultaneously.
- **Why the corpus changed.** The earlier 923-market recording carries a
  reconnect defect: the socket watchdog stamped liveness only from data frames,
  so quiet sockets were killed roughly every 100 seconds (one shard 237 times in
  a single session). Fixed three months after those recordings were made, and
  not repairable from the tape. Its numbers are reported separately for that
  reason, not merged.
- **A clock bug preceded both corpora.** The recorded parquet carried two
  timestamp columns both populated from the *receive* clock, while one was named
  for the venue clock. Since Kalshi frames from one engine event share a venue
  timestamp and differ in receive time by microseconds, any grouping on that
  column grouped nothing. Both corpora were re-derived from the raw tape after
  the fix; every number here postdates it.
- **The 60-second collapse on mid is undiagnosed.** The network falls to 0.126,
  below ridge's 0.183. Either a real regime change or an optimization failure in
  that cell; one seed, not investigated.
- The development books run five levels deep, so any depth thesis is untestable
  past L5.
- R² swings with cross-fold regime shift, so direction (correlation, hit-rate)
  and the three-class metrics are preferred over R².
- The split is temporal, not by market: a market trading on both sides of the
  boundary appears in train and test alike. That is intended — the question is
  whether the next few seconds are predictable — but these numbers do not bound
  cross-market generalization.
- FI-2010 ships pre-normalized and event-indexed. It supports the level and
  scalar feature sets and the event-horizon target; the fixed-tick grid needs a
  real tick, which the synthetic generator and raw venue data carry.
- The synthetic generator plants its signal, so its absolute numbers sanity-
  check the pipeline. The development-data tables carry the empirical result.

## License

MIT. See [LICENSE](LICENSE).
