"""Centralised baselines: every training row pooled in one place.

* Logistic regression (scikit-learn, class_weight='balanced') - main baseline in the report.
* Random forest (scikit-learn) - stronger non-linear reference ("upper envelope").
* SGD logistic regression (numpy, same code as the federated clients) trained on the pooled
  data - isolates the effect of data locality from the effect of the optimiser.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .federated import LogRegParams, local_sgd, predict_proba


def train_logistic_regression(X, y, seed: int = 42, class_weight="balanced",
                              max_iter: int = 1000, C: float = 1.0) -> LogisticRegression:
    model = LogisticRegression(class_weight=class_weight, max_iter=max_iter, C=C,
                               random_state=seed)
    return model.fit(X, y)


def train_random_forest(X, y, seed: int = 42, n_estimators: int = 300, max_depth=None,
                        class_weight="balanced") -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   class_weight=class_weight, random_state=seed, n_jobs=-1)
    return model.fit(X, y)


def train_pooled_sgd(X, y, epochs: int, lr: float, batch_size: int, l2: float,
                     class_weight: str | None, seed: int = 42) -> LogRegParams:
    """Run the federated client optimiser once, on all training rows pooled together."""
    rng = np.random.default_rng(seed)
    params = LogRegParams.zeros(X.shape[1])
    return local_sgd(params, X, y, epochs=epochs, lr=lr, batch_size=batch_size, l2=l2,
                     rng=rng, class_weight=class_weight)


__all__ = ["train_logistic_regression", "train_random_forest", "train_pooled_sgd",
           "predict_proba"]
