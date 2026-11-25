"""Optional cross-check: run FedAvg / FedProx through the Flower (flwr) simulation engine.

The main experiments use the hand-written loop in src/federated.py (fully deterministic and
easy to read). This module re-runs selected settings with Flower's own FedAvg/FedProx
strategies, using the SAME client optimiser (local_sgd) and the SAME data shards, to show
the hand-written loop behaves like a standard Flower deployment.

Usage:  python -m src.flower_sim      (writes results/flower_crosscheck.json)
Needs:  pip install "flwr[simulation]"  (pulls in Ray)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def run_flower(Xtr, ytr, Xte, yte, *, n_clients, partition, strategy, cfg, seed=42):
    from flwr.client import ClientApp, NumPyClient
    from flwr.common import Context, ndarrays_to_parameters
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
    from flwr.server.strategy import FedAvg, FedProx
    from flwr.simulation import run_simulation

    from .federated import (LogRegParams, local_sgd, partition_iid, partition_label_skew,
                            predict_proba)
    from .metrics import compute_metrics

    f = cfg["federated"]
    rng = np.random.default_rng(seed)
    if partition == "iid":
        shards = partition_iid(ytr, n_clients, rng)
    else:
        shards = partition_label_skew(ytr, n_clients, rng, f["non_iid"]["alpha"],
                                      f["non_iid"]["min_client_size"])
    mu = f["fedprox_mu"] if strategy == "fedprox" else 0.0
    history = []

    class PimaClient(NumPyClient):
        def __init__(self, cid):
            self.cid = cid
            self.X, self.y = Xtr[shards[cid]], ytr[shards[cid]]   # rows never leave here

        def fit(self, parameters, config):
            g = LogRegParams.from_vector(parameters[0])
            rnd = int(config.get("round", 0))
            local = local_sgd(g, self.X, self.y, epochs=f["local_epochs"],
                              lr=f["learning_rate"], batch_size=f["batch_size"], l2=f["l2"],
                              rng=np.random.default_rng(seed * 1000 + rnd * 10 + self.cid),
                              class_weight=f["class_weight"], mu=mu, global_params=g)
            return [local.to_vector()], len(self.y), {}

    def client_fn(context: Context):
        return PimaClient(int(context.node_config["partition-id"])).to_client()

    def evaluate_fn(server_round, parameters, config):
        p = LogRegParams.from_vector(parameters[0])
        m = compute_metrics(yte, predict_proba(p, Xte))
        if server_round > 0:
            history.append({"round": server_round, **{k: m[k] for k in
                            ("accuracy", "precision", "recall", "f1", "roc_auc")}})
        return 0.0, m

    init = ndarrays_to_parameters([np.zeros(Xtr.shape[1] + 1)])
    common = dict(fraction_fit=1.0, fraction_evaluate=0.0, min_fit_clients=n_clients,
                  min_available_clients=n_clients, initial_parameters=init,
                  evaluate_fn=evaluate_fn, on_fit_config_fn=lambda r: {"round": r})
    strat = FedProx(proximal_mu=mu, **common) if strategy == "fedprox" else FedAvg(**common)

    def server_fn(context: Context):
        return ServerAppComponents(strategy=strat, config=ServerConfig(num_rounds=f["rounds"]))

    run_simulation(server_app=ServerApp(server_fn=server_fn),
                   client_app=ClientApp(client_fn=client_fn), num_supernodes=n_clients,
                   backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}})
    return history


def main():
    from .data_prep import load_pima, prepare_data
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    df = load_pima(ROOT / cfg["data"]["csv_path"])
    d = prepare_data(df, cfg["data"]["test_size"], cfg["seed"], cfg["data"]["zero_as_missing"])
    out = {}
    for k, part, strat in [(5, "iid", "fedavg"), (5, "non_iid", "fedprox")]:
        hist = run_flower(d.X_train, d.y_train, d.X_test, d.y_test, n_clients=k,
                          partition=part, strategy=strat, cfg=cfg, seed=cfg["seed"])
        out[f"flower_fl{k}_{part}_{strat}"] = {"final": hist[-1], "history": hist}
        print(k, part, strat, hist[-1])
    (ROOT / "results" / "flower_crosscheck.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
