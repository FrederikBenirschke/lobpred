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


def _to_3class(probs: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Reindex sklearn's probabilities onto the fixed [-1, 0, +1] basis.

    ``predict_proba`` returns one column per class PRESENT IN THE TRAINING
    FOLD, in sorted order — not always three. With a sign deadband (``alpha``)
    a fold can contain no +1 rows at all, and sklearn then returns (N, 2).
    Consuming that directly is silently wrong: ``classification_metrics`` maps
    predictions with ``argmax − 1``, so column 1 would decode as class 0
    instead of +1 and the +1 class becomes unreachable. No exception is
    raised and accuracy/macro-F1 are quietly incorrect.

    Mapping each column by its actual class label makes the basis explicit.
    """
    out = np.zeros((len(probs), 3), dtype=np.float64)
    for j, c in enumerate(classes):
        c = int(c)
        if c not in (-1, 0, 1):
            raise ValueError(f"unexpected sign class {c!r}; expected -1/0/+1")
        out[:, c + 1] = probs[:, j]
    return out


def logistic_proba(Xtr, ytr_cls, Xte):
    """Multinomial logistic on snapshot features → class probabilities (N,3).

    Always returns three columns on the [-1, 0, +1] basis that
    ``classification_metrics`` decodes with (argmax − 1), even when a class is
    absent from the training fold — see ``_to_3class``.
    """
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=500, C=1.0)  # multinomial by default in sklearn ≥1.7
    m.fit(last_step(Xtr), ytr_cls)
    return _to_3class(m.predict_proba(last_step(Xte)), m.classes_), m


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
    return _to_3class(m.predict_proba(last_step(Xte)), m.classes_), m


def majority_proba(ytr_cls, n: int) -> np.ndarray:
    """One-hot the training-majority class for every test row (N,3 floor)."""
    vals, counts = np.unique(ytr_cls, return_counts=True)
    maj = int(vals[np.argmax(counts)])
    probs = np.zeros((n, 3), dtype=np.float64)
    probs[:, maj + 1] = 1.0
    return probs
