# Report figure captions

Each image in this folder has a caption below: the figure number and name, then three plain lines
on what it is and what it does. The numbers are placeholders; renumber them to fit the report order.
Every number quoted comes from the executed run (seed 42, n_test = 154).

## Code snippets (from `src/` and `app/`)

**Figure 1: Leakage-safe data preparation** (`code_01_prepare_data.png`)
The `prepare_data` function in `src/data_prep.py`.
It makes the stratified 80/20 split first (614 train and 154 test rows, seed 42), then fits the imputer and scaler.
Because both are fitted on training rows only, no test-set information leaks into training.

**Figure 2: Median imputation of implausible zeros** (`code_02_median_imputer.png`)
The `fit` and `transform` methods of `MedianZeroImputer`.
A value of 0 in Glucose, BloodPressure, SkinThickness, Insulin or BMI is treated as missing.
It is filled with the training-set median, for example Glucose 117 and BMI 32.4.

**Figure 3: Non-IID client shards by label skew** (`code_03_label_skew.png`)
The `partition_label_skew` function, which draws each client's share of every class from a Dirichlet(α = 0.5) distribution.
This creates realistic imbalance, so one client can hold mostly diabetes cases while another holds almost none.
Shards are disjoint, and every client gets at least 20 rows.

**Figure 4: Local SGD with optional FedProx term** (`code_04_local_sgd.png`)
The client-side training loop: E epochs of mini-batch SGD on class-weighted log-loss, run on the client's own rows only.
Setting mu > 0 adds the FedProx pull back towards the global model (lines 87–89).
This is the only place training happens in the federated pipeline.

**Figure 5: DP-style clipping, Gaussian noise and FedAvg aggregation** (`code_05_dp_and_fedavg.png`)
`clip_and_noise` limits each client update to an L2 norm of C and adds N(0, (σC)²) noise.
`fedavg_aggregate` then averages the updates, weighted by each client's number of rows.
Together they show exactly what the server receives and how it combines the updates.

**Figure 6: One federated communication round** (`code_06_fl_round_loop.png`)
The main loop of `run_federated`.
Each round, every client trains locally and sends back only the parameter difference, which can optionally be clipped and noised; the server then applies the weighted average.
The global model is scored on the held-out test set after every round to produce the convergence curves.

**Figure 7: Flower client used for the cross-check** (`code_07_flower_client.png`)
A Flower `NumPyClient` that wraps the same `local_sgd` routine and the same data shards.
Running it through Flower's own FedAvg and FedProx strategies checks the hand-written loop against a standard framework.
FL5 IID FedAvg gave ROC-AUC 0.810 in both versions.

**Figure 8: Consent gate in the Streamlit feedback tab** (`code_08_consent_gate.png`)
The questionnaire code in `app/streamlit_app.py` only runs once every consent statement is ticked.
This enforces the UREC2 consent step inside the software itself.
Answers are stored anonymously, with an optional participant code such as P01.

## Jupyter notebook cells and outputs (`notebooks/fl_diabetes_experiments.ipynb`)

**Figure 9: Dataset summary and implausible zeros** (`nb_01_dataset_summary.png`)
Notebook output showing the mean, standard deviation, minimum and maximum of each feature, plus how many rows hold a 0.
Insulin (374 zeros) and SkinThickness (227) have the most implausible zeros.
This is the evidence behind the median imputation step.

**Figure 10: Preprocessing cell** (`nb_02_preprocessing_cell.png`)
The notebook cell that calls `prepare_data` with the settings from `config.yaml`.
It prints the split sizes, the training medians used for imputation and the scaled training means.
It runs the pipeline exactly as `run_experiments.py` does.

**Figure 11: Preprocessing output** (`nb_03_preprocessing_output.png`)
The output of Figure 10: 614 training rows (34.9% diabetes) and 154 test rows (35.1% diabetes).
The near-identical class rates confirm the split is stratified.
Scaled training means of 0 confirm the scaler was fitted on the training data.

**Figure 12: Centralised baseline results** (`nb_04_centralised_results.png`)
Held-out metrics for the pooled models: logistic regression (ROC-AUC 0.813), random forest (0.831) and SGD logistic regression (0.812).
SGD-LR uses the same optimiser as the federated clients but trains on pooled data.
It isolates the effect of keeping data local from the effect of the optimiser.

**Figure 13: Client shard composition** (`nb_05_client_shards_table.png`)
Rows and diabetes share for each of the 5 clients under IID and non-IID partitioning.
The IID clients all sit at about 35% diabetes; the non-IID clients range from 4% to 96%.
This table shows that the non-IID setting really is skewed.

**Figure 14: Federated training cell** (`nb_06_fl_training_cell.png`)
The notebook cell that trains all eight federated configurations: 3 or 5 clients, IID or non-IID, and FedAvg or FedProx.
Every configuration runs 15 rounds with 5 local epochs per client, with all settings taken from `config.yaml`.
Each final global model is scored on the same 154 test rows.

**Figure 15: Federated ROC-AUC per round (notebook)** (`nb_07_fl_auc_curve.png`)
Held-out ROC-AUC after each of the 15 rounds, plotted for every federated configuration.
The dashed line marks centralised logistic regression (0.813); the IID runs climb steadily and level off at about 0.81 by round 9–11.
The non-IID runs level off lower (about 0.79 with 3 clients and 0.76–0.77 with 5).

**Figure 16: Results comparison table** (`nb_08_results_comparison.png`)
All centralised and federated models on one held-out table, with the ROC-AUC gap to centralised LR.
The IID federated runs are within 0.003 of the baseline; FL5 non-IID FedAvg is 0.050 lower.
This is the main evidence for the research question.

**Figure 17: Effect of DP-style noise** (`nb_09_dp_sweep.png`)
FL5 IID FedAvg retrained with update clipping (C = 1.0) and noise levels σ from 0 to 2.
ROC-AUC falls from 0.810 (σ = 0) to 0.782 (σ = 0.5) and 0.712 (σ = 2.0).
This shows the privacy–utility trade-off; no formal ε is claimed.

**Figure 18: Reproducibility cross-check** (`nb_10_crosscheck.png`)
The notebook results match `results/metrics.json` exactly (largest difference 0).
The Flower simulation gives ROC-AUC 0.810 for FL5 IID FedAvg and 0.771 for FL5 non-IID FedProx.
These are close to the hand-written loop's 0.810 and 0.767.

## Result plots (`results/figures/`, generated by `run_experiments.py`)

**Figure 19: Outcome class balance** (`plot_class_balance.png`)
Bar chart of Outcome 0 (500 rows) against Outcome 1 (268 rows) in the full dataset (n = 768).
The classes are imbalanced at roughly 65:35.
That is why precision, recall, F1 and ROC-AUC are reported alongside accuracy.

**Figure 20: ROC curves, centralised and federated** (`plot_roc_curves.png`)
ROC curves on the 154 test rows, with each model's AUC in the legend; the dashed diagonal is chance.
The IID federated curves almost overlap centralised LR, while the non-IID and DP curves sit slightly lower.
Made with scikit-learn `roc_curve`.

**Figure 21: Federated ROC-AUC by round** (`plot_fl_auc_per_round.png`)
Held-out ROC-AUC per round for the eight non-DP federated runs, with centralised LR as a dashed reference.
It shows when each run stabilises within the 15 rounds.
The final-round values are the ones reported in the results table.

**Figure 22: Metric comparison bars** (`plot_metric_comparison.png`)
Grouped bars of accuracy, precision, recall, F1 and ROC-AUC for all 13 runs, on a shared 0–1 scale.
They make it easy to compare models on more than one metric at once.
The values match `results/metrics.csv`.

**Figure 23: DP noise sweep** (`plot_dp_effect.png`)
ROC-AUC, F1 and accuracy of FL5 IID FedAvg as the noise multiplier σ rises (clip norm 1.0).
The dashed line marks the run with no clipping or noise (AUC 0.810).
Higher σ means more noise in each update and lower utility.

**Figure 24: Confusion matrix, centralised LR** (`plot_cm_central_lr.png`)
Counts of actual against predicted classes on the test set: TN 75, FP 25, FN 16, TP 38.
Rows are actual classes and columns are predicted classes, using a 0.5 threshold.
Recall for diabetes is 0.704.

**Figure 25: Confusion matrix, FL5 IID FedAvg** (`plot_cm_fl5_iid_fedavg.png`)
The same layout for the 5-client IID federated model: TN 73, FP 27, FN 17, TP 37.
This model was trained without pooling any raw rows.
Recall is 0.685, one fewer correctly detected diabetes case than centralised LR.

**Figure 26: Label mix on each simulated client** (`plot_client_shards.png`)
Stacked bars of Outcome 0 and Outcome 1 training rows for each client, in four settings: 3 or 5 clients, IID or non-IID.
The IID clients have identical mixes; the non-IID clients are heavily skewed.
These bars explain why non-IID performance is lower.

**Figure 27: Random forest feature importances** (`plot_rf_feature_importance.png`)
Mean decrease in impurity for each feature in the centralised random forest (n_train = 614).
Glucose is the most important feature, followed by BMI, age and the pedigree function.
This is consistent with clinical expectations for type 2 diabetes risk.

## Streamlit demo screenshots (headless Chromium, `results/screenshots/`)

**Figure 28: Streamlit home tab** (`app_01_home.png`)
The landing view, with a short project introduction, headline metrics and a 4-step guide.
It tells testers what to do in a 5-minute session.
It also shows the 'not a diagnostic device' notice.

**Figure 29: Streamlit results tab** (`app_02_results.png`)
An interactive metrics table, a per-round chart, ROC and DP plots, and three confusion matrices.
All of it is loaded from the committed results files, with no retraining.
Testers can filter models and switch the metric shown in the chart.

**Figure 30: Streamlit privacy explanation tab** (`app_03_privacy.png`)
A plain-language account of what stays on each client and what is shared, with a client–server diagram.
It states the limits honestly: updates can still leak information, and there is no formal ε.
The DP sweep table shows the cost of adding noise.

**Figure 31: Streamlit prediction tab** (`app_04_prediction.png`)
A form for eight feature values that returns a risk score from centralised LR and from FL5 IID FedAvg.
It applies the same imputation and scaling used in training.
A red banner states that it is not a diagnostic device.

**Figure 32: Streamlit feedback questionnaire** (`app_06_feedback_questionnaire.png`)
The UREC2 Annex D questionnaire: 10 Likert items and 2 open questions, unlocked only after consent.
Responses are saved anonymously, and a link to a matching Microsoft Form is provided.
This tab is the user-evaluation instrument.
