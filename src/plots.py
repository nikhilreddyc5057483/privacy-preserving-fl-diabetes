"""Figures for the report and the Streamlit demo (matplotlib only)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

PALETTE = ["#1f4e79", "#c0504d", "#4f81bd", "#9bbb59", "#8064a2", "#f79646",
           "#4bacc6", "#7f7f7f", "#2e8b57", "#b8860b", "#d2691e"]


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_balance(y, path):
    counts = np.bincount(np.asarray(y), minlength=2)
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    bars = ax.bar(["0 = no diabetes", "1 = diabetes"], counts, color=["#4f81bd", "#c0504d"])
    ax.bar_label(bars)
    ax.set_ylabel("Number of rows")
    ax.set_title(f"Pima outcome classes (n = {counts.sum()})")
    ax.set_ylim(0, counts.max() * 1.15)
    return _save(fig, path)


def plot_confusion(cm: dict, title, path):
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    im = ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=13,
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, path)


def plot_roc(curves: dict, y_test, path):
    """curves = {label: predicted probabilities on the test set}."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for k, (label, (prob, auc)) in enumerate(curves.items()):
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, color=PALETTE[k % len(PALETTE)], lw=1.6,
                label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curves on held-out test set (n = {len(y_test)})")
    ax.legend(fontsize=7.5, loc="lower right")
    return _save(fig, path)


def plot_round_curves(histories: dict, path, metric="roc_auc", reference=None):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for k, (label, hist) in enumerate(histories.items()):
        rounds = [h["round"] for h in hist]
        ax.plot(rounds, [h[metric] for h in hist], marker="o", ms=3,
                color=PALETTE[k % len(PALETTE)], label=label)
    if reference is not None:
        ax.axhline(reference[1], color="k", ls="--", lw=1, label=reference[0])
    ax.set_xlabel("Communication round")
    ax.set_ylabel(metric.upper().replace("_", "-"))
    ax.set_title(f"Federated {metric.upper().replace('_', '-')} per round (held-out test set)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_metric_bars(rows: list[dict], path):
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    width = 0.16
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 4.4))
    for k, key in enumerate(keys):
        ax.bar(x + (k - 2) * width, [r[key] for r in rows], width, label=key.replace("_", "-"),
               color=PALETTE[k])
    ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Held-out test metrics by model (n_test = 154)")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def plot_dp_sweep(sweep: list[dict], path, reference=None):
    fig, ax = plt.subplots(figsize=(5.8, 4))
    sig = [s["noise_multiplier"] for s in sweep]
    ax.plot(sig, [s["roc_auc"] for s in sweep], marker="o", label="ROC-AUC", color=PALETTE[0])
    ax.plot(sig, [s["f1"] for s in sweep], marker="s", label="F1", color=PALETTE[1])
    ax.plot(sig, [s["accuracy"] for s in sweep], marker="^", label="Accuracy", color=PALETTE[3])
    if reference is not None:
        ax.axhline(reference, color="k", ls="--", lw=1, label="No clipping/noise AUC")
    ax.set_xlabel("Noise multiplier sigma (noise std = sigma x clip norm)")
    ax.set_ylabel("Score on held-out test set")
    ax.set_title("Effect of DP-style clipping + noise (FL5 IID FedAvg)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_client_shards(label_counts: dict, path):
    """label_counts = {setting: [ {n0, n1}, ... per client ]}."""
    n = len(label_counts)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, counts) in zip(axes, label_counts.items()):
        ids = [f"C{i+1}" for i in range(len(counts))]
        n0 = [c["n0"] for c in counts]
        n1 = [c["n1"] for c in counts]
        ax.bar(ids, n0, color="#4f81bd", label="Outcome 0")
        ax.bar(ids, n1, bottom=n0, color="#c0504d", label="Outcome 1")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Training rows on client")
    axes[0].legend(fontsize=7)
    return _save(fig, path)


def plot_feature_importance(names, importances, path):
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.barh(np.array(names)[order], np.array(importances)[order], color="#1f4e79")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Random forest feature importances (n_train = 614)")
    return _save(fig, path)
