import numpy as np
import pytest

from src.federated import (LogRegParams, clip_and_noise, fedavg_aggregate, local_sgd,
                           partition_iid, partition_label_skew, run_federated)


@pytest.fixture
def labels():
    rng = np.random.default_rng(0)
    return (rng.random(614) < 0.35).astype(int)


@pytest.mark.parametrize("k", [3, 5])
def test_iid_shards_disjoint_and_complete(labels, k):
    shards = partition_iid(labels, k, np.random.default_rng(42))
    allidx = np.concatenate(shards)
    assert len(allidx) == len(labels) == len(set(allidx.tolist()))   # disjoint + complete
    for s in shards:                                                 # same label mix
        assert abs(labels[s].mean() - labels.mean()) < 0.03


@pytest.mark.parametrize("k", [3, 5])
def test_non_iid_shards_disjoint_and_skewed(labels, k):
    shards = partition_label_skew(labels, k, np.random.default_rng(42), alpha=0.5, min_size=20)
    allidx = np.concatenate(shards)
    assert len(allidx) == len(labels) == len(set(allidx.tolist()))
    assert min(len(s) for s in shards) >= 20
    rates = [labels[s].mean() for s in shards]
    assert max(rates) - min(rates) > 0.3          # strong label skew between clients


def test_fedavg_weighted_average():
    v = [np.array([1.0, 2.0]), np.array([3.0, 6.0]), np.array([0.0, 0.0])]
    n = [10, 30, 60]
    expected = (10 * v[0] + 30 * v[1] + 60 * v[2]) / 100
    np.testing.assert_allclose(fedavg_aggregate(v, n), expected)
    np.testing.assert_allclose(fedavg_aggregate([v[0], v[1]], [1, 1]), [2.0, 4.0])


def test_fedavg_matches_flower_aggregate():
    flwr_agg = pytest.importorskip("flwr.server.strategy.aggregate")
    rng = np.random.default_rng(1)
    vecs = [rng.normal(size=9) for _ in range(5)]
    sizes = [123, 123, 123, 123, 122]
    ours = fedavg_aggregate(vecs, sizes)
    theirs = flwr_agg.aggregate([([v], n) for v, n in zip(vecs, sizes)])[0]
    np.testing.assert_allclose(ours, theirs)


def test_dp_clipping_and_noise():
    rng = np.random.default_rng(0)
    u = np.full(9, 10.0)
    clipped = clip_and_noise(u, clip_norm=1.0, noise_multiplier=0.0, rng=rng)
    assert np.linalg.norm(clipped) == pytest.approx(1.0)
    noisy = clip_and_noise(u, 1.0, 0.5, np.random.default_rng(0))
    assert not np.allclose(noisy, clipped)                   # noise actually applied
    draws = np.stack([clip_and_noise(np.zeros(9), 1.0, 0.5, np.random.default_rng(s))
                      for s in range(400)])
    assert draws.std() == pytest.approx(0.5, rel=0.1)        # std = sigma * C


def test_fedprox_term_pulls_towards_global():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3)); y = (X[:, 0] > 0).astype(int)
    g = LogRegParams.zeros(3)
    plain = local_sgd(g, X, y, epochs=5, lr=0.1, batch_size=10, l2=0,
                      rng=np.random.default_rng(1), mu=0.0, global_params=g)
    prox = local_sgd(g, X, y, epochs=5, lr=0.1, batch_size=10, l2=0,
                     rng=np.random.default_rng(1), mu=1.0, global_params=g)
    assert np.linalg.norm(prox.to_vector()) < np.linalg.norm(plain.to_vector())


def test_run_federated_history_and_dp_changes_model():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4)); y = (X[:, 0] + 0.5 * rng.normal(size=300) > 0).astype(int)
    kw = dict(n_clients=3, partition="iid", rounds=4, local_epochs=1, seed=3)
    r = run_federated(X[:240], y[:240], X[240:], y[240:], **kw)
    assert [h["round"] for h in r.history] == [1, 2, 3, 4]
    assert sum(r.client_sizes) == 240
    r_dp = run_federated(X[:240], y[:240], X[240:], y[240:], dp_clip=0.5, dp_noise=1.0, **kw)
    assert not np.allclose(r.params.to_vector(), r_dp.params.to_vector())
