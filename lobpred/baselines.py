"""Simple baselines, the bar every deep model must clear.

A DL model only earns its complexity if it beats these on the *same*
features, target, and walk-forward split. The harness reports
``DL - GBM``, ``GBM - linear``, ``linear - persistence`` so each gain is
attributable.

  * **persistence**, predict Δ=0. Class imbalance makes raw accuracy
    lie; this is the real floor.
  * **ridge / logistic**, linear on the most recent snapshot's features
    (Cont/Kukanov: OFI is near-linearly predictive).
  * **lightgbm**, gradient boosting on the same snapshot features.

All consume the *last timestep* of each window (B, F), the standard,
cheap, fair comparison point. (A sequence model that can't beat a linear
fit on the current snapshot has learned nothing from history.)
"""

from __future__ import annotations

import numpy as np


def last_step(X: np.ndarray) -> np.ndarray:
    """(N, T, F) -> (N, F): the decision-time snapshot features."""
    return X[:, -1, :]


# ── regression baselines ────────────────────────────────────


def persistence_predict(n: int) -> np.ndarray:
    """Predict zero forward change."""
    return np.zeros(n, dtype=np.float64)


def ridge_fit_predict(Xtr, ytr, Xte, alpha: float = 1.0):
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=alpha)
    m.fit(last_step(Xtr), ytr)
    return m.predict(last_step(Xte)), m


def lgbm_fit_predict(Xtr, ytr, Xte, **kw):
    import lightgbm as lgb
    params = dict(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=100,
        n_jobs=-1, verbosity=-1,
    )
    params.update(kw)
    m = lgb.LGBMRegressor(**params)
    m.fit(last_step(Xtr), ytr)
    return m.predict(last_step(Xte)), m


# ── classification baselines (sign label) ───────────────────


def logistic_proba(Xtr, ytr_cls, Xte):
    """Multinomial logistic on snapshot features → class probabilities (N,3).

    ``predict_proba`` columns follow sorted classes [-1, 0, +1], matching
    the (argmax − 1) convention in ``classification_metrics``.
    """
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=500, C=1.0)  # multinomial by default in sklearn ≥1.7
    m.fit(last_step(Xtr), ytr_cls)
    return m.predict_proba(last_step(Xte)), m


def lgbm_cls_proba(Xtr, ytr_cls, Xte, **kw):
    import lightgbm as lgb
    params = dict(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=100,
        n_jobs=-1, verbosity=-1,
    )
    params.update(kw)
    m = lgb.LGBMClassifier(**params)
    m.fit(last_step(Xtr), ytr_cls)
    return m.predict_proba(last_step(Xte)), m


def majority_proba(ytr_cls, n: int) -> np.ndarray:
    """One-hot the training-majority class for every test row (N,3 floor)."""
    vals, counts = np.unique(ytr_cls, return_counts=True)
    maj = int(vals[np.argmax(counts)])
    probs = np.zeros((n, 3), dtype=np.float64)
    probs[:, maj + 1] = 1.0
    return probs
