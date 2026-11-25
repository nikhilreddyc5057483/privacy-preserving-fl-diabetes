"""Hand-written federated learning simulation (numpy) for logistic regression.

Everything runs on one machine, but each simulated client only ever touches its own
rows: the server receives model parameters (weights + bias), never raw data.

Glossary
--------
local SGD : each client starts from the current global model and runs E epochs of
            mini-batch stochastic gradient descent on its own rows only.
FedAvg    : the server replaces the global model with the average of the client models,
            weighted by how many training rows each client holds (McMahan et al., 2017).
FedProx   : same as FedAvg, but each client adds a proximal term (mu/2)*||w - w_global||^2
            to its local loss, which stops clients drifting too far on skewed data
            (Li et al., 2020).
DP-style  : before sending, each client clips its model *update* (local - global) to an
            L2 norm of C and adds Gaussian noise N(0, (sigma*C)^2). This is the mechanism
            used by DP-FedAvg, but no formal (epsilon, delta) accounting is done here, so
            it is a utility/noise comparison, not a privacy guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import compute_metrics


# --------------------------------------------------------------------------- model
@dataclass
class LogRegParams:
    """Parameters of a logistic-regression model: p(y=1|x) = sigmoid(x.w + b)."""
    w: np.ndarray
    b: float

    @classmethod
    def zeros(cls, n_features: int) -> "LogRegParams":
        return cls(w=np.zeros(n_features), b=0.0)

    def to_vector(self) -> np.ndarray:
        return np.concatenate([self.w, [self.b]])

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "LogRegParams":
        v = np.asarray(v, dtype=float)
        return cls(w=v[:-1].copy(), b=float(v[-1]))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def predict_proba(params: LogRegParams, X: np.ndarray) -> np.ndarray:
    return sigmoid(X @ params.w + params.b)


def _sample_weights(y: np.ndarray, class_weight: str | None) -> np.ndarray:
    """'balanced' -> n / (2 * n_class), computed from THIS client's labels only."""
    if class_weight != "balanced":
        return np.ones(len(y))
    n = len(y)
    weights = np.ones(n)
    for c in (0, 1):
        n_c = int((y == c).sum())
        if n_c:
            weights[y == c] = n / (2.0 * n_c)
    return weights


# --------------------------------------------------------------------- local SGD
def local_sgd(params: LogRegParams, X: np.ndarray, y: np.ndarray, *, epochs: int,
              lr: float, batch_size: int, l2: float, rng: np.random.Generator,
              class_weight: str | None = "balanced", mu: float = 0.0,
              global_params: LogRegParams | None = None) -> LogRegParams:
    """Run E epochs of mini-batch SGD on one client's data.

    Loss per batch = weighted log-loss + (l2/2)||w||^2 [+ (mu/2)||theta - theta_global||^2].
    mu = 0 gives plain FedAvg local training; mu > 0 gives FedProx.
    """
    w, b = params.w.copy(), float(params.b)
    sw = _sample_weights(y, class_weight)
    g_w = global_params.w if global_params is not None else None
    g_b = global_params.b if global_params is not None else 0.0
    n = len(y)
    for _ in range(epochs):
        order = rng.permutation(n)                     # reshuffle every epoch
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            err = (sigmoid(X[idx] @ w + b) - y[idx]) * sw[idx]   # d(log-loss)/d(logit)
            grad_w = X[idx].T @ err / len(idx) + l2 * w
            grad_b = err.mean()
            if mu > 0 and g_w is not None:             # FedProx proximal pull to global
                grad_w = grad_w + mu * (w - g_w)
                grad_b = grad_b + mu * (b - g_b)
            w -= lr * grad_w
            b -= lr * grad_b
    return LogRegParams(w=w, b=b)


# ------------------------------------------------------------------ partitioning
def partition_iid(y: np.ndarray, n_clients: int, rng: np.random.Generator) -> list[np.ndarray]:
    """IID shards: shuffle each class and deal it round-robin, so every client
    gets roughly the same size and the same diabetes / non-diabetes ratio."""
    shards: list[list[int]] = [[] for _ in range(n_clients)]
    offset = 0
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        for i, row in enumerate(idx):
            shards[(i + offset) % n_clients].append(int(row))
        offset += len(idx)
    return [np.array(sorted(s)) for s in shards]


def partition_label_skew(y: np.ndarray, n_clients: int, rng: np.random.Generator,
                         alpha: float = 0.5, min_size: int = 20,
                         max_tries: int = 1000) -> list[np.ndarray]:
    """Non-IID shards via Dirichlet label skew: for each class, draw client proportions
    from Dirichlet(alpha) and split that class's rows accordingly. Small alpha means
    some clients get mostly diabetes cases and others mostly non-diabetes cases."""
    for _ in range(max_tries):
        shards: list[list[int]] = [[] for _ in range(n_clients)]
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0])
            props = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx, cuts)):
                shards[k].extend(part.tolist())
        if min(len(s) for s in shards) >= min_size:
            return [np.array(sorted(s)) for s in shards]
    raise RuntimeError("could not build label-skew shards meeting min_size")


# ------------------------------------------------------------- DP-style update
def clip_and_noise(update: np.ndarray, clip_norm: float, noise_multiplier: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Clip an update to L2 norm <= clip_norm, then add N(0, (noise_multiplier*clip_norm)^2)."""
    norm = np.linalg.norm(update)
    clipped = update * min(1.0, clip_norm / (norm + 1e-12))
    if noise_multiplier > 0:
        clipped = clipped + rng.normal(0.0, noise_multiplier * clip_norm, size=update.shape)
    return clipped


# --------------------------------------------------------------- aggregation
def fedavg_aggregate(vectors: list[np.ndarray], num_examples: list[int]) -> np.ndarray:
    """FedAvg: average of client vectors weighted by client sample counts."""
    weights = np.asarray(num_examples, dtype=float)
    stacked = np.stack([np.asarray(v, dtype=float) for v in vectors])
    return (weights[:, None] * stacked).sum(axis=0) / weights.sum()


# ------------------------------------------------------------- full simulation
@dataclass
class FLResult:
    params: LogRegParams
    history: list[dict] = field(default_factory=list)   # one entry per round
    client_sizes: list[int] = field(default_factory=list)
    client_label_counts: list[dict] = field(default_factory=list)


def run_federated(X_train, y_train, X_test, y_test, *, n_clients: int, partition: str,
                  strategy: str = "fedavg", rounds: int = 15, local_epochs: int = 5,
                  lr: float = 0.05, batch_size: int = 16, l2: float = 1e-4,
                  class_weight: str | None = "balanced", mu: float = 0.0,
                  alpha: float = 0.5, min_client_size: int = 20,
                  dp_clip: float | None = None, dp_noise: float = 0.0,
                  seed: int = 42) -> FLResult:
    """Simulate `rounds` of federated training and log held-out metrics each round."""
    rng = np.random.default_rng(seed)
    if partition == "iid":
        shards = partition_iid(y_train, n_clients, rng)
    elif partition == "non_iid":
        shards = partition_label_skew(y_train, n_clients, rng, alpha, min_client_size)
    else:
        raise ValueError(partition)
    client_data = [(X_train[s], y_train[s]) for s in shards]   # each client keeps its rows
    sizes = [len(s) for s in shards]
    prox_mu = mu if strategy == "fedprox" else 0.0

    global_params = LogRegParams.zeros(X_train.shape[1])
    history = []
    for rnd in range(1, rounds + 1):
        g_vec = global_params.to_vector()
        updates = []
        for Xc, yc in client_data:                       # --- on each client ---
            local = local_sgd(global_params, Xc, yc, epochs=local_epochs, lr=lr,
                              batch_size=batch_size, l2=l2, rng=rng,
                              class_weight=class_weight, mu=prox_mu,
                              global_params=global_params)
            delta = local.to_vector() - g_vec            # only parameters leave the client
            if dp_clip is not None:
                delta = clip_and_noise(delta, dp_clip, dp_noise, rng)
            updates.append(delta)
        # --- on the server: weighted FedAvg of the client updates ---
        global_params = LogRegParams.from_vector(g_vec + fedavg_aggregate(updates, sizes))
        m = compute_metrics(y_test, predict_proba(global_params, X_test))
        history.append({"round": rnd, **{k: m[k] for k in
                        ("accuracy", "precision", "recall", "f1", "roc_auc")}})
    labels = [{"n0": int((yc == 0).sum()), "n1": int((yc == 1).sum())} for _, yc in client_data]
    return FLResult(params=global_params, history=history, client_sizes=sizes,
                    client_label_counts=labels)
