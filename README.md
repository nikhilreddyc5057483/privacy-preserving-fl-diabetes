# Privacy-Preserving Diabetes Prediction: Federated Learning Prototype with Differential Privacy

MSc Computing research project, Sheffield Hallam University
**Nikhil Reddy Meesala (C5057483)** · Supervisor: Kostas Domdouzis

This repository holds the full code for the dissertation. It trains diabetes-risk models on the
**Pima Indians Diabetes Database** in two ways: centrally (all rows pooled) and with
**federated learning (FL)**, where each simulated client keeps its own rows and shares only model
updates. It then compares the two on one held-out test set. A **Streamlit demo** presents the
results to peers and collects their usability feedback, using the questionnaire approved by the
ethics committee (UREC2 Annex D).

> ⚠️ This is a research and teaching prototype, **not a diagnostic device**.

## What is inside

| Part | What it does |
|---|---|
| `src/data_prep.py` | Loads the CSV, makes a stratified 80/20 split (seed 42), fills implausible zeros with **training** medians and fits the scaler on **training** rows only |
| `src/centralised.py` | Centralised baselines: logistic regression, random forest, and pooled SGD logistic regression |
| `src/federated.py` | Hand-written FL simulation: IID and Dirichlet label-skew shards, **local SGD**, **FedAvg**, **FedProx**, and **DP-style clipping + Gaussian noise** |
| `src/flower_sim.py` | Optional cross-check that reruns FedAvg and FedProx with the **Flower (flwr)** simulation engine |
| `src/metrics.py`, `src/plots.py` | Accuracy, precision, recall, F1, ROC-AUC and confusion matrix; all figures |
| `run_experiments.py` | Runs everything and writes `results/` and `models/` |
| `notebooks/fl_diabetes_experiments.ipynb` | Executed walk-through notebook with outputs |
| `app/streamlit_app.py` | Peer-evaluation demo: Home, Results, Privacy explained, Try a prediction, Feedback |
| `tests/` | 19 pytest tests (log in `results/test_log.txt`) |
| `report_figures/` | Cropped code, notebook and plot images for the report, with `captions.md` |

## Federated learning implementation: what was actually used

All 13 reported runs use the **hand-written numpy simulation** in `src/federated.py`. It keeps
every step visible (local SGD → update → weighted average) and gives bit-for-bit reproducible
results. **Flower was used for two things**:
1. `tests/test_federated.py` checks that our `fedavg_aggregate` gives the same result as Flower's
   `flwr.server.strategy.aggregate.aggregate`.
2. `python -m src.flower_sim` reruns FL5 IID FedAvg and FL5 non-IID FedProx through Flower's own
   `FedAvg` and `FedProx` strategies (`run_simulation`, Ray backend), using the same client
   optimiser and the same shards. The results are in `results/flower_crosscheck.json`.

| Setting | Hand-written loop ROC-AUC | Flower simulation ROC-AUC |
|---|---|---|
| FL5 IID FedAvg | 0.810 | 0.810 |
| FL5 non-IID FedProx | 0.767 | 0.771 |

The small non-IID difference comes from the order of random mini-batch shuffles; Flower
calls clients in its own order.

## Setup

Python 3.11 or newer (built and tested on 3.13.5).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # experiments + Streamlit app
pip install -r requirements-dev.txt                    # tests, notebook, Flower, screenshots (optional)
```

## How to run

```bash
python data/download_data.py          # (optional) re-download the CSV; a copy is already in data/
python run_experiments.py             # about 6 s: writes results/*.csv|json, results/figures/, models/
python -m src.flower_sim              # optional Flower cross-check (about 30 s, needs requirements-dev)
pytest -v                             # 19 tests
jupyter nbconvert --to notebook --execute --inplace notebooks/fl_diabetes_experiments.ipynb
streamlit run app/streamlit_app.py    # opens the demo at http://localhost:8501
python scripts/make_report_figures.py # rebuilds report_figures/ (needs: python -m playwright install chromium)
```

To capture app screenshots, start the app, then run `python scripts/capture_screenshots.py`.

## Deploying the demo on Streamlit Community Cloud

The app runs entirely from committed files (`results/`, `models/models.json`) and **never
retrains on launch**. All paths are relative to the repository root.

1. Push this repository to GitHub.
2. At <https://share.streamlit.io>, choose **Create app** and select the repo, branch `main`, and
   main file path **`app/streamlit_app.py`**.
3. Under *Advanced settings*, choose Python **3.12 or 3.13**. `requirements.txt` at the repo root
   is installed automatically. It deliberately leaves out Jupyter, Flower and Ray to keep the
   build small.
4. (Optional) In *Secrets*, add the live questionnaire link:
   ```toml
   MS_FORM_URL = "https://forms.office.com/r/your-form-id"
   ```
   See `.streamlit/secrets.toml.example`. Without it, the button shows a placeholder.

**Feedback storage:** the Feedback tab appends answers to `feedback/responses.csv`. That folder
is git-ignored because peer responses are research data and belong on the SHU F: drive, not
GitHub. On Streamlit Cloud the file system resets on restart, so for hosted sessions use the
**Microsoft Form** link as the record, or have each tester download their CSV copy.

## Dataset, source and licence

* **Pima Indians Diabetes Database**: 768 rows, 8 features and `Outcome` (500 non-diabetes, 268 diabetes).
* Downloaded from the public mirror
  <https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv>
  (SHA-256 `6bfe5d0f…f154af`). Column names are added by `data/download_data.py`.
* The Kaggle listing named in the ethics form is
  <https://www.kaggle.com/datasets/jamaltariqcheema/pima-indians-diabetes-dataset>, licensed
  **CC0 1.0 Public Domain** (<https://creativecommons.org/publicdomain/zero/1.0/>), so the CSV
  can be redistributed here. The data originally comes from the US National Institute of Diabetes
  and Digestive and Kidney Diseases (Smith et al., 1988). See `data/SOURCE.txt`.
* The Kaggle copy needs a login to download, so the identical public mirror is used for a
  scripted, reproducible download.

## Results (held-out test set, n = 154, seed 42)

All values below come from `results/metrics.csv`, produced by `python run_experiments.py`.
Positive class = diabetes, decision threshold = 0.5.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Central LR | 0.734 | 0.603 | 0.704 | 0.650 | 0.813 |
| Central RF | 0.747 | 0.623 | 0.704 | 0.661 | 0.831 |
| Central SGD-LR | 0.734 | 0.603 | 0.704 | 0.650 | 0.812 |
| FL3 IID FedAvg | 0.727 | 0.597 | 0.685 | 0.638 | 0.812 |
| FL3 IID FedProx | 0.727 | 0.597 | 0.685 | 0.638 | 0.811 |
| FL3 non-IID FedAvg | 0.695 | 0.552 | 0.685 | 0.612 | 0.792 |
| FL3 non-IID FedProx | 0.695 | 0.552 | 0.685 | 0.612 | 0.792 |
| FL5 IID FedAvg | 0.714 | 0.578 | 0.685 | 0.627 | 0.810 |
| FL5 IID FedProx | 0.708 | 0.571 | 0.667 | 0.615 | 0.809 |
| FL5 non-IID FedAvg | 0.675 | 0.532 | 0.611 | 0.569 | 0.763 |
| FL5 non-IID FedProx | 0.682 | 0.540 | 0.630 | 0.581 | 0.767 |
| FL5 IID FedAvg + DP | 0.727 | 0.583 | 0.778 | 0.667 | 0.782 |
| FL5 non-IID FedAvg + DP | 0.688 | 0.538 | 0.778 | 0.636 | 0.779 |

**Robustness:** the federated randomness (shard draws and SGD shuffles) was repeated over seeds
42–46 with the split held fixed. See `results/seed_repeats_summary.csv`. ROC-AUC mean ± SD:
FL5 IID FedAvg 0.810 ± 0.000, FL5 non-IID FedAvg 0.801 ± 0.022, FL5 non-IID FedProx 0.802 ± 0.020.
Seed 42 happens to give a harder-than-average non-IID split.

**DP noise sweep** (FL5 IID FedAvg, clip norm 1.0), from `results/dp_sweep.csv`. ROC-AUC by noise
multiplier σ: 0 → 0.810, 0.1 → 0.810, 0.25 → 0.804, 0.5 → 0.782, 1.0 → 0.744, 2.0 → 0.712.

**Key points**
* With IID clients, federated logistic regression is within 0.003 ROC-AUC of centralised logistic
  regression (0.810–0.812 against 0.813), and no raw row ever leaves its client.
* Strong label skew costs more: 0.792 with 3 clients and 0.763 with 5 (FedAvg). FedProx
  (μ = 0.1) gives at most a small gain (0.767).
* DP-style noise lowers ROC-AUC as σ grows. With σ = 0.5, recall goes up (0.778) because the
  noisy model predicts the positive class more often, while precision and ROC-AUC go down.
* No formal (ε, δ) privacy accounting is done. The DP runs show the utility cost of the
  mechanism, not a certified guarantee.

## Settings

All settings are in [`config.yaml`](config.yaml). Values stated in the report are marked
`(report)`; values the report does not give are marked `(chosen)`.

| Setting | Value | Source |
|---|---|---|
| Seed / split | 42 / stratified 80:20 (614 / 154) | report |
| Imputation | zeros in Glucose, BloodPressure, SkinThickness, Insulin, BMI → training median | report |
| Scaling | StandardScaler fitted on training rows | report |
| Central LR | `LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)` | report (class weight) / chosen |
| Central RF | 300 trees, `class_weight="balanced"` | chosen |
| FL model | logistic regression (8 weights + bias), class-balanced log-loss | chosen (report: "federated linear SGD") |
| Rounds / clients | 15 / 3 and 5 | report |
| Local epochs E / batch / learning rate / L2 | 5 / 16 / 0.05 / 1e-4 | chosen |
| Client participation | all clients every round | chosen |
| Non-IID | Dirichlet label skew, α = 0.5, at least 20 rows per client | chosen (report: "label-skew shards") |
| FedProx μ | 0.1 | chosen |
| DP-style | update clipping C = 1.0, Gaussian noise σ·C, headline σ = 0.5 | chosen |

## Project structure

```
nikhil-fl-diabetes/
├── README.md  DEMO.md  config.yaml  requirements.txt  requirements-dev.txt  .gitignore
├── .streamlit/            config.toml (theme), secrets.toml.example
├── app/streamlit_app.py   peer-evaluation demo
├── data/                  download_data.py, pima_diabetes.csv, SOURCE.txt
├── models/models.json     preprocessing + Central LR + FL5 IID FedAvg weights (for the app)
├── notebooks/             fl_diabetes_experiments.ipynb (executed)
├── report_figures/        cropped report images + captions.md
├── results/               metrics.csv/json, round_history.csv, dp_sweep.csv, seed_repeats*.csv,
│                          test_predictions.csv, flower_crosscheck.json, test_log.txt,
│                          figures/*.png, screenshots/*.png
├── scripts/               build_notebook.py, capture_screenshots.py, make_report_figures.py
├── src/                   data_prep, centralised, federated, flower_sim, metrics, plots
├── tests/                 pytest suite
└── run_experiments.py
```

## Ethics

Model training uses only the public, de-identified CC0 Pima data, with no NHS or patient data.
The human part is a voluntary usability evaluation by adult SHU Computing peers under the
approved UREC2 low-risk application. The Feedback tab requires every consent statement to be
ticked before the questionnaire appears, asks for no names, and stores answers anonymously.
