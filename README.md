# lobpred: predicting short-horizon order-book moves on prediction markets

On a prediction market you buy a contract that pays a dollar if something
happens and nothing if it doesn't, so its price sits between 0 and 1 and reads
as the crowd's probability. Kalshi and Polymarket run live order books for
thousands of these: every NBA game, every World Cup match, tomorrow's high
temperature in Chicago. I recorded those books for five days and asked one
question. From the shape of the book right now, can you tell where the price
drifts over the next few seconds?

I kept hearing that deep sequence models beat plain gradient-boosted trees on
order-book data, so I built both on the same recordings and held them to one
setup: the same features, the same target, the same split. Most comparisons
quietly cheat here, because the deep model reads a window of history while the
tree reads a single snapshot. I give the tree that same window before I call a
winner.

The short answer is that the representation matters more than the model. Hand
every model the raw order book instead of engineered ratios and all of them
improve. Once the tree and the net read the same window, they finish level.
Every number below says which markets it came from and how many samples sit
behind it.

> **The data.** This is a prediction study, so it reports forecast quality and
> no trading PnL. Over five days I recorded about 33M order-book updates on
> Kalshi and PolymarketUS and kept the 923 most active markets, 21.7M updates,
> with their matching trade tape. The Kalshi side spans MLB (138 markets), NBA
> (63), tennis (71), WNBA (7), World Cup soccer (102), 172 daily-temperature
> markets, and 26 others; PolymarketUS adds 344 on the same events. The raw
> recordings carry trader usernames, so they stay out of the repo. The synthetic
> generator and the FI-2010 loader reproduce every pipeline and finding here.
> Want the collected dataset? Email benirschke.math@gmail.com.

## What I found

The raw order book is where the gains come from. Adding it on top of the
engineered ratios lifts correlation by 0.12 to 0.19 for every model, tree and
net alike. That lift is real order-book skill, not a quirk of bounded prices:
hold the price level fixed by scoring inside price-decile buckets and almost all
of it survives, +0.11 of a +0.12 pooled lift.

The model class barely matters once the inputs match. Give LightGBM the same
32-tick window the nets read and it ties the best net near 0.50 correlation. The
net only pulls ahead when the comparison hands it history the snapshot tree
never sees.

Two smaller results held up. The target you pick beats the model you pick: a
smoothed exit over a few seconds reaches 0.41 correlation where a single future
instant reaches 0.25, a wider gap than any two models show. And a widely-cited
71% three-class accuracy turns out to be a class-balance artifact, since
balancing the "stable" majority drops it to 0.46-0.50, in line with the source's
own Up/Down F1.

One choice shaped all of it: I call a market "active" by how often its mid price
moves, not how often it posts a quote. A book that fires hundreds of quotes a
minute and moves price twice is mostly noise.

## Quickstart

```bash
pip install -e ".[deep,plot,dev]"     # deep = torch; plot = matplotlib; dev = pytest

# 1) generate a synthetic dataset with a known, modest, planted signal
python -m lobpred.data.synthetic --out experiments/synthetic --n-markets 12 --minutes 120

# 2) model comparison + feature-group importance (Findings 1 & 3)
KMP_DUPLICATE_LIB_OK=TRUE python -m lobpred.analyze \
    --roots experiments/synthetic --horizon-events 50

# 3) the staged ablation gap table (phases 4–5 add the extended + trade features)
KMP_DUPLICATE_LIB_OK=TRUE python -m lobpred.run \
    --roots experiments/synthetic --phases 0 1 2 3 4 5 --horizon 30

#    feature-importance with the trade-tape families included
KMP_DUPLICATE_LIB_OK=TRUE python -m lobpred.analyze \
    --roots experiments/synthetic --horizon-events 50 --with-trades

# 4) point-exit vs smoothed-hold target across horizons (Finding 2)
KMP_DUPLICATE_LIB_OK=TRUE python -m lobpred.target_study \
    --roots experiments/synthetic --horizons 2 5 10 30

# 5) diagnostic plots for one phase (loss/grad curves, error, attention pockets)
python -m lobpred.diagnose --roots experiments/synthetic --phase 3 --out experiments/diag

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

I build every feature family on the same rows, so they compare on equal
footing. The first three form the base set; the rest are stationary extensions
(`features.py`). Every family except **levels** is price-invariant, which lets
me pool across instruments.

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

### Leakage discipline

- **Forward target, built here.** I never read it from a backward rolling
  window. Two forms: point `price(t+h) − price(t)` and smoothed
  `mean(price over (t, t+h]) − price(t)` (TWAP over the hold). Every averaged
  price falls after `t`, so neither form leaks.
- **Train-only normalization.** I fit z-score statistics on the train rows and
  apply them to test.
- **Walk-forward with a purge embargo.** I split by global wall-clock time and
  drop a gap of one horizon or more around each boundary, so no label straddles
  it (López de Prado). A market stays inside one split.

### Churn vs discovery

An active market moves its price. `add_activity_tier` ranks markets by
`mid_moves_per_min`, the count of real mid changes, and ignores
`updates_per_min`, which counts quote churn. A book that posts 500 quotes a
minute with 2 mid moves adds noise. Rank on churn and you poison every
downstream number, so the selection path ranks on discovery.

### The ablation (one change per rung)

`run.py` reports the gap at each rung: `phase1−phase0` (microprice target),
`phase2−phase1` (grid vs levels), `phase3−phase2` (flow scalars),
`phase4−phase3` (flow-dynamics + return/vol + book-shape + activity/spread),
`phase5−phase4` (trade-tape), and deep−simple within each phase. Read the gaps.
The absolute numbers shift with the fold; the gaps hold.

### Dual head

I run two heads, because they fail differently. The regression head predicts
the forward change, judged on corr and hit-rate; I skip R², which cross-fold
regime shift dominates. The 3-class sign head gives paper-comparable accuracy
and macro-F1. I standardize the regression target for the net and invert the
predictions back to price units, which holds the net's outputs in range.
DeepLOB needs a LayerNorm before the LSTM and gradient clipping, or its
grad-norm climbs to ~1e17. The code documents both fixes.

## Results (development dataset)

These numbers come from the development data: Kalshi and PolymarketUS order
books, price-active top tier (the raw recordings stay private). The synthetic
and FI-2010 paths reproduce the pipeline and the qualitative conclusions (on
engineered features a snapshot tree matches or beats the nets, smoothed beats
point, signal in the micro-gap).

> These four tables were measured on the **base** feature set (grid +
> scalar, 46 features). The extended families (flow-dynamics, return/vol,
> book-shape, trade-tape) are newer; their marginal contribution is exactly
> what the `phase4−phase3` and `phase5−phase4` ablation gaps measure. I have
> not folded that run into the tables below, and I claim no numbers for it here.

**Population (stated, per the project's discipline):** price-active books
only, top activity tier by mid-move rate, ~185–198 markets, ~2.7M
event-time windows, walk-forward folds with embargo, grid k=20 + scalars
(46 features) unless noted.

### Finding 1: on engineered features, the snapshot tree matches or beats the nets

Liquid tier, event-horizon 50 ≈ 8.4 s, engineered base set, tree on the last
snapshot:

| model | reg corr | reg hit | cls acc | cls maF1 |
|---|---|---|---|---|
| ridge | 0.363 | 0.572 | n/a | n/a |
| **lgbm** | **0.368** | **0.577** | 0.544 | 0.471 |
| logistic | n/a | n/a | **0.600** | **0.490** |
| tcn | 0.279 | 0.573 | 0.558 | 0.485 |
| deeplob | 0.091 | 0.540 | n/a | n/a |
| attention | 0.308 | 0.564 | 0.565 | 0.481 |

This is the engineered-feature, snapshot-input regime. Finding 5 gives every
model the raw book and the full 32-tick window; the gap closes and the tree and
best net tie near 0.50.

### Finding 2: the target beats the model

lgbm corr, point exit vs smoothed hold:

| hold W | point | smoothed |
|---|---|---|
| 2 s | 0.296 | 0.339 |
| 5 s | 0.268 | 0.394 |
| 10 s | 0.245 | **0.407** |
| 30 s | 0.220 | 0.390 |

The point exit decays as the horizon grows. The smoothed hold rises and peaks
near 10 s. Moving from point to smoothed at 10 s adds 0.16 corr, while the
lgbm-tcn gap is 0.13. Best operating point: hold about 10 s and predict the
TWAP move, for 0.41 corr and 0.59 sign-hit.

### Finding 3: the signal lives in the spread

Feature-group permutation importance (Δcorr when shuffled):

| group | perm Δcorr |
|---|---|
| spread / micro_gap | **0.177** |
| imbalance | 0.013 |
| grid depth (bid+ask) | ~0.010 |
| ofi | −0.001 |

The spread and micro-gap hold the skill. Depth shape and OFI add ≈0. At five
levels the depth-"pockets" thesis fails the permutation test, and I report that
negative.

### Finding 4: the 71% headline is a balance artifact

A well-known result reports ~71% 3-class accuracy. Balance the Stable class
(`--balance`, train and test) and accuracy drops to 0.44-0.50 here, so an easy
majority inflates the 71% rather than data volume (the same source reports
Up/Down F1 ≈ 0.50). On the directional metric (macro-F1 ~0.46-0.50, sign-hit
0.55-0.59) the gap stays small. Caveat: this is cross-dataset with a different
label deadband and features. The accuracy collapse reproduces here, which is
the robust part.

### Finding 5: raw book lifts every model, and the lift is real skill

*Population:* the 923-market pooled corpus (21.7M updates), regression on the
19s forward-average microprice change, single-split OOS.

I gave every model the raw price and size at five levels next to the engineered
features. All of them rose: LightGBM 0.34 to 0.46, the TCN 0.32 to 0.51,
attention 0.35 to 0.48, the LSTM 0.31 to 0.44. The raw lift runs +0.12 to +0.19.

Two controls keep the read honest. The lift is not the bounded-price reversion:
bucket the test rows by current price into deciles and recompute corr inside each
bucket, where absolute price holds roughly constant, and the raw lift survives
(+0.11 of the +0.12 pooled), spread across every decile. So the raw book carries
within-price order-book skill. And the net does not beat the tree on this input:
give LightGBM the same 32-tick window the nets see (flattened to 1,120 features)
and it reaches 0.50, level with the TCN. The earlier read that "raw lifts only
the tree" came from normalizing raw features per-market, which removes the price
level and craters raw (0.47 to 0.32), and from testing only an LSTM.

Tradeability stays a separate, open question: these corrs score microprice
change, which drifts within a fixed bid and ask, so a taker signal likely falls
to the spread until a costed backtest says otherwise.

### Finding 6: pooled corr hides a 2x per-market spread

The pooled corr (0.33 for LightGBM, 0.28 for the net) averages markets that
differ by more than 2x. Per segment, LightGBM scores 0.56 on tennis, 0.36 on
weather, and 0.23 on NBA. A model trained on one segment matches the pooled
model on that segment's own test windows, so pooling heterogeneous markets
costs nothing. The net ties the tree on tennis (0.53 vs 0.56) and loses on the
mixed segments.

## Modules

```
lobpred/
  dataset.py        load + pool + base features + leak-safe targets + activity tiers + windows + walk-forward + trade loader
  features.py       extended families A–G + add_perlevel_features (per-level stationary (T,L,C) tensor for DL)
  baselines.py      persistence / ridge / lgbm / logistic / majority
  models.py         TCN, DeepLOB, AxialAttentionLOB, PerLevelLOB (conv-across-levels + LSTM), SeqLSTM (no-pool control)
  evaluate.py       prediction metrics (corr/hit/acc/F1) + the torch training loop + gap tables
  diagnostics.py    training-curve / error-analysis / attention-pocket plots
  analyze.py        model comparison + feature-group importance     (Findings 1 & 3)
  run.py            the staged ablation gap table (phases 0–5)
  target_study.py   point vs smoothed target across horizons        (Finding 2)
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

Open `examples/deeplob_walkthrough.py` to see the mechanics in one script:
synthetic book → **per-level stationary tensor**
(`add_perlevel_features`: OBI/OFI/rel-size per level) → **PerLevelLOB**
(convolves across levels + LSTM over time, the DeepLOB/Kolm representation)
→ training curve → comparison to LightGBM. Runs in ~1 min, no downloads.

You can train a net stably on ~10⁵ windows, which you generate in seconds.
More data alone does not rescue the net on engineered features.
`examples/regime_scaling.py` sweeps the training size and pits SeqLSTM against
LightGBM on the same features, target, and walk-forward split:

| train N | LightGBM | SeqLSTM | gap (tree − net) |
|---|---|---|---|
| 10 K | 0.501 | **0.511** | −0.011 |
| 40 K | 0.520 | 0.515 | +0.005 |
| 160 K | 0.526 | 0.515 | +0.010 |
| 640 K | 0.527 | 0.515 | +0.012 |
| 1.7 M | 0.528 | 0.507 | +0.021 |

(corr on the held-out fold; SeqLSTM is a 2-seed ensemble; the synthetic signal
is planted, so read the trend rather than the level.) The net leads at 10 K. As
data grows, the tree keeps improving while the net plateaus, so the gap moves
the wrong way for "DL just needs scale." Engineered features leave the net
nothing extra to learn, and the new rows sharpen the tree.

### The regime boundary

Deep learning wins on the other axis: raw book input at large scale. DeepLOB
(Zhang, Zohren, Roberts) reports F1 ≈ 0.83 at the shortest horizon on FI-2010,
ahead of the classical baselines in that paper. Its CNN learns features off the
raw 10-level price/size ladder that hand-engineered scalars discard, and a tree
discards them too. Two caveats keep this honest. FI-2010's baselines are linear
and TABL nets, so nobody there tuned a GBDT against DeepLOB. FI-2010 also comes
from a different venue at a different scale. Read it as a regime map over input
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

- The development books run five levels deep, so I cannot test the
  depth-attention thesis past L5. Finding 3 is scoped to those five levels.
- R² swings with cross-fold regime shift, so I trust direction (corr, hit) and
  the 3-class metrics over R².
- FI-2010 ships pre-normalized and event-indexed. Use the level and scalar
  feature sets and the event-horizon target on it; the fixed-tick grid needs a
  real tick, which the synthetic generator and raw venue data carry.
- The synthetic generator plants its signal, so its absolute numbers sanity-
  check the pipeline. The development-data tables carry the empirical result.

## License

MIT. See [LICENSE](LICENSE).
