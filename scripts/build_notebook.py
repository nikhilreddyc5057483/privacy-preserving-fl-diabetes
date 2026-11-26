"""Builds notebooks/fl_diabetes_experiments.ipynb (then execute it with nbconvert)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Privacy-Preserving Diabetes Prediction with Federated Learning
**MSc Computing dissertation notebook – Nikhil Reddy Meesala (C5057483), Sheffield Hallam University**

This notebook walks through the whole experiment on the Pima Indians Diabetes Database:
data overview → preprocessing → centralised baselines → federated learning (FedAvg / FedProx,
3 and 5 clients, IID and non-IID) → results comparison → effect of DP-style noise.

All code lives in `src/`; the notebook calls it so the numbers match `run_experiments.py` exactly.
""")
md("## 1. Setup")
code("""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, yaml
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from src.data_prep import load_pima, prepare_data, FEATURES, ZERO_AS_MISSING
from src.centralised import train_logistic_regression, train_random_forest, train_pooled_sgd, predict_proba
from src.federated import run_federated, partition_iid, partition_label_skew
from src.metrics import compute_metrics, METRIC_KEYS

cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
SEED = cfg["seed"]
pd.set_option("display.precision", 3)
print("seed =", SEED, "| rounds =", cfg["federated"]["rounds"], "| local epochs =", cfg["federated"]["local_epochs"])
""")
md("## 2. Data overview\nThe dataset has 768 rows, 8 clinical features and a binary `Outcome` (1 = diabetes).")
code("""
df = load_pima(ROOT / cfg["data"]["csv_path"])
print("shape:", df.shape)
print("class counts:", df["Outcome"].value_counts().to_dict())
df.head()
""")
code("""
zero_counts = (df[FEATURES] == 0).sum().rename("rows with value 0")
summary = df.describe().T[["mean", "std", "min", "max"]].join(zero_counts)
summary
""")
code("""
counts = df["Outcome"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(4.5, 3.2))
bars = ax.bar(["0 = no diabetes", "1 = diabetes"], counts, color=["#4f81bd", "#c0504d"])
ax.bar_label(bars); ax.set_ylabel("Rows"); ax.set_title("Outcome classes (n = 768)")
plt.show()
""")
md("""
## 3. Preprocessing
* stratified 80/20 split with `random_state = 42` → 614 train / 154 test rows
* zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are impossible values, so they
  are treated as missing and filled with the **training-set median**
* `StandardScaler` is **fitted on training rows only**, then applied to the test rows
""")
code("""
data = prepare_data(df, cfg["data"]["test_size"], SEED, cfg["data"]["zero_as_missing"])
Xtr, Xte, ytr, yte = data.X_train, data.X_test, data.y_train, data.y_test
print(f"train = {len(ytr)} rows ({ytr.mean():.1%} diabetes) | test = {len(yte)} rows ({yte.mean():.1%} diabetes)")
print("training medians used for imputation:", {k: round(v, 1) for k, v in data.imputer.medians_.items()})
print("scaled train mean ≈", np.round(Xtr.mean(axis=0), 3))
""")
md("## 4. Centralised baselines\nAll 614 training rows in one place. Logistic regression is the main baseline; random forest is a stronger non-linear reference; SGD-LR is the exact federated client optimiser run on pooled data.")
code("""
rows, probs = [], {}
def evaluate(label, prob):
    m = compute_metrics(yte, prob); probs[label] = prob
    rows.append({"Model": label, **{k: m[k] for k in METRIC_KEYS}}); return m

c, f = cfg["centralised"], cfg["federated"]
lr = train_logistic_regression(Xtr, ytr, SEED, **c["logistic_regression"])
rf = train_random_forest(Xtr, ytr, SEED, **c["random_forest"])
sgd = train_pooled_sgd(Xtr, ytr, epochs=c["sgd_logreg"]["epochs"], lr=f["learning_rate"],
                       batch_size=f["batch_size"], l2=f["l2"], class_weight=f["class_weight"], seed=SEED)
m_lr = evaluate("Central LR", lr.predict_proba(Xte)[:, 1])
evaluate("Central RF", rf.predict_proba(Xte)[:, 1])
evaluate("Central SGD-LR", predict_proba(sgd, Xte))
pd.DataFrame(rows).set_index("Model")
""")
md("## 5. Federated clients: IID vs non-IID shards\nIID shards keep the same label mix on every client. Non-IID shards use Dirichlet label skew (α = 0.5), so some clients hold mostly diabetes cases and others mostly non-diabetes cases. Shards are disjoint: no row is on two clients.")
code("""
a = cfg["federated"]["non_iid"]["alpha"]
shards = {"5 clients IID": partition_iid(ytr, 5, np.random.default_rng(SEED)),
          "5 clients non-IID": partition_label_skew(ytr, 5, np.random.default_rng(SEED), a, 20)}
table = pd.DataFrame({name: [f"{len(s)} rows, {ytr[s].mean():.0%} diabetes" for s in sh]
                      for name, sh in shards.items()}, index=[f"Client {i+1}" for i in range(5)])
table
""")
md("""
## 6. Federated training (15 rounds)
Each round: the server sends the global model → every client runs **E = 5 epochs of local
mini-batch SGD** on its own rows → clients send back only their parameter update → the server
takes a **weighted average (FedAvg)**. FedProx adds a proximal term μ/2·‖w − w_global‖² to each
client's loss (μ = 0.1).
""")
code("""
kw = dict(rounds=f["rounds"], local_epochs=f["local_epochs"], lr=f["learning_rate"],
          batch_size=f["batch_size"], l2=f["l2"], class_weight=f["class_weight"],
          alpha=f["non_iid"]["alpha"], min_client_size=f["non_iid"]["min_client_size"], seed=SEED)
histories = {}
for k in f["client_counts"]:
    for part in ("iid", "non_iid"):
        for strat in ("fedavg", "fedprox"):
            label = f"FL{k} {'IID' if part=='iid' else 'non-IID'} {'FedAvg' if strat=='fedavg' else 'FedProx'}"
            r = run_federated(Xtr, ytr, Xte, yte, n_clients=k, partition=part, strategy=strat,
                              mu=f["fedprox_mu"], **kw)
            evaluate(label, predict_proba(r.params, Xte)); histories[label] = r.history
print(f"trained {len(histories)} federated configurations")
""")
code("""
fig, ax = plt.subplots(figsize=(7, 4))
for label, h in histories.items():
    ax.plot([x["round"] for x in h], [x["roc_auc"] for x in h], marker="o", ms=3, label=label)
ax.axhline(m_lr["roc_auc"], color="k", ls="--", lw=1, label="Central LR")
ax.set_xlabel("Communication round"); ax.set_ylabel("ROC-AUC (held-out)")
ax.set_title("Federated ROC-AUC per round"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)
plt.show()
""")
md("## 7. Results comparison (held-out test set, n = 154)")
code("""
results = pd.DataFrame(rows).set_index("Model")
results["AUC gap vs Central LR"] = results["roc_auc"] - m_lr["roc_auc"]
results.round(3)
""")
md("## 8. Effect of DP-style clipping + Gaussian noise\nEach client clips its update to L2 norm C = 1.0 and adds noise N(0, (σ·C)²) before sending. No formal (ε, δ) accounting is done, so this shows the utility cost of noise rather than a privacy guarantee.")
code("""
dp = cfg["federated"]["dp"]; sweep = []
for sigma in dp["sweep"]:
    r = run_federated(Xtr, ytr, Xte, yte, n_clients=5, partition="iid", strategy="fedavg",
                      dp_clip=dp["clip_norm"], dp_noise=sigma, **kw)
    m = compute_metrics(yte, predict_proba(r.params, Xte))
    sweep.append({"noise σ": sigma, **{k: m[k] for k in METRIC_KEYS}})
sweep = pd.DataFrame(sweep).set_index("noise σ")
display(sweep.round(3))
ax = sweep[["roc_auc", "f1", "accuracy"]].plot(marker="o", figsize=(5.5, 3.6), grid=True)
ax.set_ylabel("score"); ax.set_title("More noise → lower utility (FL5 IID FedAvg)"); plt.show()
""")
md("## 9. ROC curves, confusion matrices and feature importances\nThe curves and matrices below use the same held-out test set (n = 154) and the probabilities stored in `probs` above. The noise run uses the headline setting from `config.yaml` (clip norm C = 1.0, noise multiplier σ = 0.5).")
code("""
from sklearn.metrics import roc_curve
PALETTE = ["#1f4e79", "#c0504d", "#4f81bd", "#9bbb59", "#8064a2", "#f79646", "#4bacc6"]
r_dp = run_federated(Xtr, ytr, Xte, yte, n_clients=5, partition="iid", strategy="fedavg",
                     dp_clip=dp["clip_norm"], dp_noise=dp["noise_multiplier"], **kw)
probs["FL5 IID FedAvg + DP"] = predict_proba(r_dp.params, Xte)
roc_models = ["Central LR", "Central RF", "FL3 IID FedAvg", "FL5 IID FedAvg",
              "FL5 non-IID FedAvg", "FL5 non-IID FedProx", "FL5 IID FedAvg + DP"]
fig, ax = plt.subplots(figsize=(6, 5))
for k, label in enumerate(roc_models):
    fpr, tpr, _ = roc_curve(yte, probs[label])
    auc = compute_metrics(yte, probs[label])["roc_auc"]
    ax.plot(fpr, tpr, color=PALETTE[k], lw=1.6, label=f"{label} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title(f"ROC curves on held-out test set (n = {len(yte)})"); ax.legend(fontsize=7.5, loc="lower right")
plt.show()
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
for ax, label in zip(axes, ["Central LR", "FL5 IID FedAvg"]):
    cm = compute_metrics(yte, probs[label])["confusion_matrix"]
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    im = ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=13,
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"]); ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    ax.set_title(f"{label} (n_test = {len(yte)})", fontsize=10); fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout(); plt.show()
""")
code("""
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
ax = imp.plot.barh(color="#1f4e79", figsize=(5.5, 3.6))
ax.set_xlabel("Mean decrease in impurity")
ax.set_title(f"Random forest feature importances (n_train = {len(ytr)})"); plt.show()
print(imp.sort_values(ascending=False).round(3).to_dict())
""")
md("## 10. Cross-checks\n**(a)** the notebook numbers equal the exported `results/metrics.json`; **(b)** a Flower (flwr) simulation of the same settings gives very similar results.")
code("""
exported = {r["label"]: r for r in json.loads((ROOT / "results/metrics.json").read_text())["runs"]}
diff = max(abs(exported[m][k] - results.loc[m, k]) for m in results.index for k in METRIC_KEYS)
print(f"largest difference vs results/metrics.json: {diff:.2e}")
fl = ROOT / "results/flower_crosscheck.json"
if fl.exists():
    for name, v in json.loads(fl.read_text()).items():
        print(name, {k: round(v["final"][k], 3) for k in METRIC_KEYS})
""")
md("""
## 11. Summary
* Federated training reaches ROC-AUC close to the centralised logistic regression when client
  data are IID, without any raw row leaving its client.
* Strong label skew (non-IID) lowers performance; FedProx gives at most a small change here.
* Adding DP-style noise trades ROC-AUC for protection of individual updates.
* This is a demonstration on one small public dataset, **not** a clinical tool.
""")
nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
Path(__file__).resolve().parents[1].joinpath("notebooks/fl_diabetes_experiments.ipynb").write_text(nbf.writes(nb))
print("notebook written")
