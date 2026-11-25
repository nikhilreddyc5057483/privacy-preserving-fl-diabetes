"""Run every experiment in the dissertation and export metrics, models and figures.

Usage:  python run_experiments.py [--config config.yaml]
Writes: results/metrics.csv, results/metrics.json, results/round_history.csv,
        results/dp_sweep.csv, results/seed_repeats.csv, results/figures/*.png,
        models/*.json (used by the Streamlit app, so it never retrains on launch).
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml

from src.centralised import (predict_proba, train_logistic_regression, train_pooled_sgd,
                             train_random_forest)
from src.data_prep import FEATURES, load_pima, prepare_data
from src.federated import run_federated
from src.metrics import METRIC_KEYS, compute_metrics
from src import plots

ROOT = Path(__file__).resolve().parent


def fl_kwargs(cfg):
    f = cfg["federated"]
    return dict(rounds=f["rounds"], local_epochs=f["local_epochs"], lr=f["learning_rate"],
                batch_size=f["batch_size"], l2=f["l2"], class_weight=f["class_weight"],
                alpha=f["non_iid"]["alpha"], min_client_size=f["non_iid"]["min_client_size"])


def main(config_path: str = "config.yaml") -> dict:
    t0 = time.time()
    cfg = yaml.safe_load((ROOT / config_path).read_text())
    seed = cfg["seed"]
    res_dir, fig_dir, model_dir = (ROOT / cfg["outputs"][k] for k in
                                   ("results_dir", "figures_dir", "models_dir"))
    for d in (res_dir, fig_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- data
    df = load_pima(ROOT / cfg["data"]["csv_path"])
    data = prepare_data(df, cfg["data"]["test_size"], seed, cfg["data"]["zero_as_missing"])
    Xtr, Xte, ytr, yte = data.X_train, data.X_test, data.y_train, data.y_test
    print(f"train={len(ytr)} test={len(yte)} | test positives={int(yte.sum())}")

    runs, probs, histories, label_counts = [], {}, {}, {}

    def record(run_id, label, family, prob, extra=None):
        m = compute_metrics(yte, prob)
        row = {"run_id": run_id, "label": label, "family": family,
               **{k: m[k] for k in METRIC_KEYS}, "confusion_matrix": m["confusion_matrix"],
               "n_test": m["n_test"], **(extra or {})}
        runs.append(row)
        probs[run_id] = (label, prob, m["roc_auc"])
        print(f"{label:<32} " + " ".join(f"{k}={m[k]:.3f}" for k in METRIC_KEYS))
        return row

    # ------------------------------------------------------ centralised
    c = cfg["centralised"]
    lr_model = train_logistic_regression(Xtr, ytr, seed, **c["logistic_regression"])
    record("central_lr", "Central LR", "centralised", lr_model.predict_proba(Xte)[:, 1])
    rf_model = train_random_forest(Xtr, ytr, seed, **c["random_forest"])
    record("central_rf", "Central RF", "centralised", rf_model.predict_proba(Xte)[:, 1])
    f = cfg["federated"]
    sgd_params = train_pooled_sgd(Xtr, ytr, epochs=c["sgd_logreg"]["epochs"],
                                  lr=f["learning_rate"], batch_size=f["batch_size"],
                                  l2=f["l2"], class_weight=f["class_weight"], seed=seed)
    record("central_sgd_lr", "Central SGD-LR", "centralised", predict_proba(sgd_params, Xte))

    # ---------------------------------------------------------- federated
    kw = fl_kwargs(cfg)
    dp = f["dp"]
    fl_specs = []
    for k in f["client_counts"]:
        for part in ("iid", "non_iid"):
            fl_specs.append((k, part, "fedavg", False))
            fl_specs.append((k, part, "fedprox", False))
    fl_specs.append((5, "iid", "fedavg", True))
    fl_specs.append((5, "non_iid", "fedavg", True))

    fl_models = {}
    for k, part, strat, use_dp in fl_specs:
        run_id = f"fl{k}_{part}_{strat}" + ("_dp" if use_dp else "")
        label = (f"FL{k} {'IID' if part == 'iid' else 'non-IID'} "
                 f"{'FedAvg' if strat == 'fedavg' else 'FedProx'}" + (" + DP" if use_dp else ""))
        r = run_federated(Xtr, ytr, Xte, yte, n_clients=k, partition=part, strategy=strat,
                          mu=f["fedprox_mu"], dp_clip=dp["clip_norm"] if use_dp else None,
                          dp_noise=dp["noise_multiplier"] if use_dp else 0.0, seed=seed, **kw)
        record(run_id, label, "federated", predict_proba(r.params, Xte), {
            "n_clients": k, "partition": part, "strategy": strat, "dp": use_dp,
            "client_sizes": r.client_sizes, "client_label_counts": r.client_label_counts})
        histories[run_id] = (label, r.history)
        if strat == "fedavg" and not use_dp:
            label_counts[f"{k} clients, {'IID' if part == 'iid' else 'non-IID'}"] = \
                r.client_label_counts
        fl_models[run_id] = r.params

    # ------------------------------------------------------ DP noise sweep
    sweep = []
    for sigma in dp["sweep"]:
        r = run_federated(Xtr, ytr, Xte, yte, n_clients=5, partition="iid", strategy="fedavg",
                          dp_clip=dp["clip_norm"], dp_noise=sigma, seed=seed, **kw)
        m = compute_metrics(yte, predict_proba(r.params, Xte))
        sweep.append({"noise_multiplier": sigma, "clip_norm": dp["clip_norm"],
                      **{k: m[k] for k in METRIC_KEYS}})
    pd.DataFrame(sweep).to_csv(res_dir / "dp_sweep.csv", index=False)

    # -------------------- robustness: repeat FL randomness over 5 seeds (split fixed)
    seed_rows = []
    for s in range(seed, seed + 5):
        for k, part, strat in [(5, "iid", "fedavg"), (5, "non_iid", "fedavg"),
                               (5, "non_iid", "fedprox")]:
            r = run_federated(Xtr, ytr, Xte, yte, n_clients=k, partition=part, strategy=strat,
                              mu=f["fedprox_mu"], seed=s, **kw)
            m = compute_metrics(yte, predict_proba(r.params, Xte))
            seed_rows.append({"seed": s, "run_id": f"fl{k}_{part}_{strat}",
                              **{kk: m[kk] for kk in METRIC_KEYS}})
    seeds_df = pd.DataFrame(seed_rows)
    seeds_df.to_csv(res_dir / "seed_repeats.csv", index=False)
    seed_summary = (seeds_df.groupby("run_id")[METRIC_KEYS].agg(["mean", "std"]).round(4))
    seed_summary.to_csv(res_dir / "seed_repeats_summary.csv")

    # ------------------------------------------------------------ export
    flat = pd.DataFrame([{k: v for k, v in r.items()
                          if k not in ("confusion_matrix", "client_sizes", "client_label_counts")}
                         | {f"cm_{kk}": vv for kk, vv in r["confusion_matrix"].items()}
                         for r in runs])
    flat.to_csv(res_dir / "metrics.csv", index=False)
    hist_rows = [{"run_id": rid, "label": lab, **h} for rid, (lab, hist) in histories.items()
                 for h in hist]
    pd.DataFrame(hist_rows).to_csv(res_dir / "round_history.csv", index=False)
    pd.DataFrame({"y_true": yte, **{rid: p for rid, (_, p, _) in probs.items()}}) \
        .to_csv(res_dir / "test_predictions.csv", index=False)

    meta = {
        "generated_unix": int(time.time()),
        "runtime_seconds": round(time.time() - t0, 1),
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "dataset": {"rows": int(len(df)), "features": FEATURES,
                    "class_counts": {str(k): int(v) for k, v in df.Outcome.value_counts().items()},
                    "n_train": int(len(ytr)), "n_test": int(len(yte)),
                    "source_url": cfg["data"]["source_url"],
                    "kaggle_listing": cfg["data"]["kaggle_listing"],
                    "train_medians": data.imputer.medians_},
        "config": cfg,
        "implementation": "hand-written numpy FedAvg/FedProx simulation (see src/federated.py); "
                          "aggregation cross-checked against flwr.server.strategy.aggregate in tests",
    }
    out = {"meta": meta, "runs": runs,
           "history": {rid: {"label": lab, "rounds": h} for rid, (lab, h) in histories.items()},
           "dp_sweep": sweep, "seed_repeats": seed_rows}
    (res_dir / "metrics.json").write_text(json.dumps(out, indent=2))

    # models for the Streamlit "Try a prediction" tab (plain JSON, no pickles)
    prep = {"features": FEATURES, "medians": data.imputer.medians_,
            "zero_as_missing": data.imputer.columns,
            "scaler_mean": data.scaler.mean_.tolist(), "scaler_scale": data.scaler.scale_.tolist()}
    models = {"preprocessing": prep,
              "central_lr": {"label": "Central LR", "w": lr_model.coef_[0].tolist(),
                             "b": float(lr_model.intercept_[0])},
              "fl5_iid_fedavg": {"label": "FL5 IID FedAvg",
                                 "w": fl_models["fl5_iid_fedavg"].w.tolist(),
                                 "b": float(fl_models["fl5_iid_fedavg"].b)}}
    (model_dir / "models.json").write_text(json.dumps(models, indent=2))

    # ------------------------------------------------------------ figures
    plots.plot_class_balance(df.Outcome, fig_dir / "class_balance.png")
    by_id = {r["run_id"]: r for r in runs}
    plots.plot_confusion(by_id["central_lr"]["confusion_matrix"],
                         "Central LR (n_test=154)", fig_dir / "cm_central_lr.png")
    plots.plot_confusion(by_id["fl5_iid_fedavg"]["confusion_matrix"],
                         "FL5 IID FedAvg (n_test=154)", fig_dir / "cm_fl5_iid_fedavg.png")
    plots.plot_confusion(by_id["fl5_non_iid_fedprox"]["confusion_matrix"],
                         "FL5 non-IID FedProx (n_test=154)", fig_dir / "cm_fl5_non_iid_fedprox.png")
    roc_ids = ["central_lr", "central_rf", "fl3_iid_fedavg", "fl5_iid_fedavg",
               "fl5_non_iid_fedavg", "fl5_non_iid_fedprox", "fl5_iid_fedavg_dp"]
    plots.plot_roc({probs[i][0]: (probs[i][1], probs[i][2]) for i in roc_ids}, yte,
                   fig_dir / "roc_curves.png")
    ref = ("Central LR AUC", by_id["central_lr"]["roc_auc"])
    plots.plot_round_curves({lab: h for rid, (lab, h) in histories.items() if "_dp" not in rid},
                            fig_dir / "fl_auc_per_round.png", reference=ref)
    plots.plot_round_curves({lab: h for rid, (lab, h) in histories.items() if "_dp" not in rid},
                            fig_dir / "fl_f1_per_round.png", metric="f1",
                            reference=("Central LR F1", by_id["central_lr"]["f1"]))
    plots.plot_metric_bars(runs, fig_dir / "metric_comparison.png")
    plots.plot_dp_sweep(sweep, fig_dir / "dp_effect.png",
                        reference=by_id["fl5_iid_fedavg"]["roc_auc"])
    plots.plot_client_shards(label_counts, fig_dir / "client_shards.png")
    plots.plot_feature_importance(FEATURES, rf_model.feature_importances_,
                                  fig_dir / "rf_feature_importance.png")
    print(f"done in {time.time() - t0:.1f}s -> {res_dir}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
