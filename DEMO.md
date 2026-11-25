# 5-minute demo script (for peer testers)

**Before you start:** the researcher gives you the Participant Information Sheet and a participant
code such as `P07`. Open the app at the link you were given, or run
`streamlit run app/streamlit_app.py` locally. This is a software usability test, **not** a health
check, and the tool is not a diagnostic device.

| Time | Tab | Task for the tester | What to notice |
|---|---|---|---|
| 0:00–0:45 | 🏠 **Home** | Read the short introduction and the four headline numbers. | The question being tested: can a model trained without pooling rows match a normal one? |
| 0:45–2:00 | 📊 **Results** | 1. Find the ROC-AUC of **Central LR** and **FL5 IID FedAvg** in the table. 2. In *Training curve per round*, add **FL5 non-IID FedAvg** and compare it with the IID line. 3. Switch the metric to **F1**. | IID federated ≈ centralised; skewed (non-IID) clients score lower. |
| 2:00–3:00 | 🛡️ **Privacy explained** | Read the four steps and the diagram. Then find what ROC-AUC drops to at noise σ = 1.0. | Only model updates travel, never raw rows. More noise means more privacy but less accuracy. |
| 3:00–3:45 | 🧪 **Try a prediction** | Click **Get risk score** with the default values. Then raise **Glucose** to 180 and score again. | Both models respond in the same direction. Read the "not a diagnostic device" banner. |
| 3:45–5:00+ | 📝 **Feedback** | Tick the consent statements, enter your participant code, and answer questions 1–10 (plus 11–12 if you wish). Click **Submit**. | You can stop at any time. Alternatively, use the **Microsoft Forms** button. |

## Task-completion checklist (researcher notes, one row per tester)

| Code | T1 found both AUCs | T2 added non-IID line | T3 read DP effect | T4 got a prediction | T5 submitted feedback | Time (min) | Notes |
|---|---|---|---|---|---|---|---|
| P01 | ☐ | ☐ | ☐ | ☐ | ☐ | | |

Success criteria (from the report's UAT plan): the tester can view the comparison, read the privacy
text and submit the form without help.
