"""lobpred, short-horizon limit-order-book move prediction.

A compact, leak-disciplined study of whether short-horizon moves in a
limit order book are predictable from stationary microstructure features,
and whether deep sequence models (TCN, DeepLOB, axial attention) beat
simple baselines (ridge / LightGBM / logistic) on the *same* features,
target, and walk-forward split.

The public package ships a synthetic LOB generator and an FI-2010 loader
so every result reproduces with no proprietary data. See the README.
"""

__version__ = "0.1.0"
