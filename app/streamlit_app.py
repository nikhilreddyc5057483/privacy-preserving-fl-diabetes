"""Peer-evaluation demo for the Pima federated-learning prototype.

Runs entirely from committed files (results/*.csv|json, results/figures/*.png,
models/models.json). Nothing is retrained when the app starts.

Local:  streamlit run app/streamlit_app.py
Cloud:  Streamlit Community Cloud -> main file path: app/streamlit_app.py
"""
from __future__ import annotations

import csv
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]          # repo root, works locally and on Cloud
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
MODELS = ROOT / "models" / "models.json"
FEEDBACK_DIR = ROOT / "feedback"

# Live Microsoft Forms questionnaire (can be overridden with MS_FORM_URL in Streamlit secrets)
MS_FORM_URL_DEFAULT = "https://forms.cloud.microsoft/Pages/ResponsePage.aspx?id=ofZoiROsL0e4mfcxbkOfQw69TnFTThZMhj03lZluKwdUMjREUzNQM1NORjdSVTNEWk1PNFFEMDhJNi4u"

# Annex D (UREC2) evaluation questionnaire - wording kept exactly as approved.
LIKERT_ITEMS = [
    "The demonstration of the prototype was easy to follow.",
    "The purpose of the prototype (privacy-preserving diabetes-risk prediction using federated "
    "learning on public research data) was clearly explained.",
    "I understood the difference between the centralised baseline and the federated learning "
    "approach as presented.",
    "The privacy explanation (for example, keeping raw records local to simulated clients and "
    "sharing model updates rather than raw rows) was clear.",
    "The comparison of results (for example accuracy or other metrics) between approaches was "
    "understandable.",
    "The prototype interface / demonstration materials were usable for a short academic "
    "evaluation session.",
    "I found the prototype potentially useful as a learning or demonstration tool for "
    "privacy-preserving AI concepts.",
    "Overall, I am satisfied with the clarity and presentation of this student deliverable.",
    "I felt the session length (about 25–30 minutes including questionnaire) was reasonable.",
    "I did not feel pressured to give positive feedback.",
]
OPEN_ITEMS = [
    "What, if anything, was unclear about the privacy-preserving design or the results comparison?",
    "What one improvement would most help future users understand the prototype?",
]
SCALE = {1: "1 Strongly disagree", 2: "2 Disagree", 3: "3 Neutral", 4: "4 Agree",
         5: "5 Strongly agree"}
CONSENT_ITEMS = [
    "I am aged 18 or over.",
    "I have read the Participant Information Sheet and had the chance to ask questions.",
    "I understand my participation is voluntary and I can stop at any time without giving a reason.",
    "I understand this is a software usability evaluation, not a medical study, and the tool is "
    "not a diagnostic device.",
    "I agree that my anonymised answers can be used in this MSc dissertation.",
]
DISCLAIMER = ("**Not a diagnostic device.** This is a student research prototype trained on a "
              "public research dataset collected in the 1980s (768 rows). Its output is a statistical score for "
              "demonstration only and must not be used for any health decision. If you are "
              "worried about diabetes, speak to a GP or pharmacist.")


# ----------------------------------------------------------------- data loading
@st.cache_data
def load_metrics():
    return json.loads((RESULTS / "metrics.json").read_text())


@st.cache_data
def load_csv(name):
    return pd.read_csv(RESULTS / name)


@st.cache_data
def load_models():
    return json.loads(MODELS.read_text())


def ms_form_url() -> str:
    try:
        return st.secrets.get("MS_FORM_URL", os.environ.get("MS_FORM_URL", MS_FORM_URL_DEFAULT))
    except Exception:                                    # no secrets.toml present
        return os.environ.get("MS_FORM_URL", MS_FORM_URL_DEFAULT)


def fig(name, caption):
    path = FIGS / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")


# ----------------------------------------------------------------------- tabs
def tab_home(m):
    st.markdown(
        "This demo shows a **federated learning (FL)** prototype for diabetes-risk prediction "
        "built for an MSc Computing dissertation at Sheffield Hallam University. "
        "The question is simple: *can a model trained without pooling raw patient rows score "
        "about as well as a normal, centralised model?*")
    runs = {r["run_id"]: r for r in m["runs"]}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset rows", m["meta"]["dataset"]["rows"])
    c2.metric("Held-out test rows", m["meta"]["dataset"]["n_test"])
    c3.metric("Central LR ROC-AUC", f"{runs['central_lr']['roc_auc']:.3f}")
    c4.metric("FL5 IID FedAvg ROC-AUC", f"{runs['fl5_iid_fedavg']['roc_auc']:.3f}",
              delta=f"{runs['fl5_iid_fedavg']['roc_auc'] - runs['central_lr']['roc_auc']:+.3f}")
    st.markdown("""
**How to use this demo (about 5 minutes):**
1. **Results** – compare the centralised and federated models.
2. **Privacy explained** – see what stays on each client and what is shared.
3. **Try a prediction** – enter example values and see the score from both models.
4. **Feedback** – give consent, then answer the 12-question questionnaire.
""")
    st.info("All numbers come from `run_experiments.py` and are loaded from saved files. "
            "Nothing is retrained when this page opens.")
    st.caption(DISCLAIMER)


def tab_results(m):
    df = pd.DataFrame(m["runs"])
    cols = ["label", "family", "accuracy", "precision", "recall", "f1", "roc_auc"]
    st.subheader("Held-out test performance (n = 154)")
    fam = st.multiselect("Show model families", ["centralised", "federated"],
                         default=["centralised", "federated"])
    view = df[df.family.isin(fam)][cols].rename(columns={
        "label": "Model", "family": "Type", "accuracy": "Accuracy", "precision": "Precision",
        "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC"})
    st.dataframe(view.style.format({c: "{:.3f}" for c in
                                    ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]})
                 .highlight_max(subset=["ROC-AUC"], color="#d4edda"),
                 hide_index=True, width="stretch", height=(len(view) + 1) * 35 + 3)
    st.caption("Positive class = diabetes (Outcome 1). Threshold 0.5. Same seeded split "
               "(random_state = 42) for every model.")

    st.subheader("Training curve per round")
    hist = load_csv("round_history.csv")
    metric = st.radio("Metric", ["roc_auc", "f1", "accuracy", "recall", "precision"],
                      horizontal=True, format_func=lambda s: s.upper().replace("_", "-"))
    labels = sorted(hist.label.unique())
    chosen = st.multiselect("Federated runs", labels,
                            default=[l for l in labels if "FedAvg" in l and "DP" not in l])
    sub = hist[hist.label.isin(chosen)]
    chart = (alt.Chart(sub).mark_line(point=True)
             .encode(x=alt.X("round:Q", title="Communication round", scale=alt.Scale(domain=[1, 15])),
                     y=alt.Y(f"{metric}:Q", title=metric.upper().replace("_", "-"),
                             scale=alt.Scale(zero=False)),
                     color=alt.Color("label:N", title="Run"),
                     tooltip=["label", "round", alt.Tooltip(f"{metric}:Q", format=".3f")])
             .properties(height=320))
    st.altair_chart(chart, width="stretch")

    st.subheader("Figures")
    a, b = st.columns(2)
    with a:
        fig("roc_curves.png", "ROC curves, centralised vs federated")
    with b:
        fig("dp_effect.png", "Effect of DP-style clipping + Gaussian noise")
    st.markdown("**Confusion matrices** (rows = actual, columns = predicted)")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig("cm_central_lr.png", "Central LR")
    with c2:
        fig("cm_fl5_iid_fedavg.png", "FL5 IID FedAvg")
    with c3:
        fig("cm_fl5_non_iid_fedprox.png", "FL5 non-IID FedProx")
    with st.expander("How were clients' data split? (IID vs non-IID)"):
        fig("client_shards.png", "Label mix on each simulated client")


def tab_privacy(m):
    st.subheader("What federated learning keeps private")
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("""
**Centralised (normal) training:** every hospital sends its patient rows to one server,
which trains one model. Simple, but the raw data is copied and pooled.

**Federated training (this prototype):**
1. The server sends the current model (8 weights + 1 bias) to each client.
2. Each client trains it for a few passes (**local SGD**) on **its own rows only**.
3. Each client sends back **only the updated numbers**, never its rows.
4. The server averages the updates, weighted by client size (**FedAvg**), and repeats.
   This prototype runs 15 rounds.

**FedProx** adds a small pull back towards the shared model so clients with very
different patients (non-IID data) do not drift apart.

**DP-style noise:** each update is clipped to a maximum size and random Gaussian noise
is added before sending, which makes it harder to infer any one person from an update.
""")
    with right:
        st.graphviz_chart("""
digraph G { rankdir=TB; node [shape=box, style="rounded,filled", fontname=Helvetica];
  S [label="Server\\n(FedAvg / FedProx)", fillcolor="#c6e0b4"];
  C1 [label="Client 1\\nlocal rows", fillcolor="#ffe699"];
  C2 [label="Client 2\\nlocal rows", fillcolor="#ffe699"];
  C3 [label="Client 3\\nlocal rows", fillcolor="#ffe699"];
  S -> C1 [label="global model"]; S -> C2; S -> C3;
  C1 -> S [label="update only", color="#1f4e79"]; C2 -> S [color="#1f4e79"]; C3 -> S [color="#1f4e79"];
}""")
    st.warning("**Limits, stated honestly.** Model updates can still leak some information "
               "(e.g. membership inference). The noise here is a light demonstration with no "
               "formal (ε, δ) guarantee, and all clients are simulated on one computer.")
    sweep = pd.DataFrame(m["dp_sweep"])
    st.markdown("**Privacy–utility trade-off (5 clients, IID):** more noise → lower ROC-AUC.")
    st.dataframe(sweep[["noise_multiplier", "accuracy", "f1", "roc_auc"]].rename(columns={
        "noise_multiplier": "Noise σ", "accuracy": "Accuracy", "f1": "F1", "roc_auc": "ROC-AUC"})
        .style.format("{:.3f}"), hide_index=True, width="stretch")


def tab_predict():
    models = load_models()
    prep = models["preprocessing"]
    st.error(DISCLAIMER)
    defaults = {"Pregnancies": 2, "Glucose": 120, "BloodPressure": 72, "SkinThickness": 29,
                "Insulin": 125, "BMI": 32.0, "DiabetesPedigreeFunction": 0.37, "Age": 33}
    limits = {"Pregnancies": (0, 17, 1), "Glucose": (40, 200, 1), "BloodPressure": (30, 125, 1),
              "SkinThickness": (5, 100, 1), "Insulin": (10, 850, 5), "BMI": (15.0, 68.0, 0.1),
              "DiabetesPedigreeFunction": (0.05, 2.5, 0.01), "Age": (21, 81, 1)}
    with st.form("predict"):
        cols = st.columns(4)
        values = {}
        for i, feat in enumerate(prep["features"]):
            lo, hi, step = limits[feat]
            values[feat] = cols[i % 4].number_input(feat, min_value=lo, max_value=hi,
                                                    value=defaults[feat], step=step)
        go = st.form_submit_button("Get risk score", type="primary")
    if go:
        x = np.array([values[f] for f in prep["features"]], dtype=float)
        for j, f in enumerate(prep["features"]):          # same preprocessing as training
            if f in prep["zero_as_missing"] and x[j] == 0:
                x[j] = prep["medians"][f]
        z = (x - np.array(prep["scaler_mean"])) / np.array(prep["scaler_scale"])
        a, b = st.columns(2)
        for col, key in ((a, "central_lr"), (b, "fl5_iid_fedavg")):
            mdl = models[key]
            p = 1 / (1 + np.exp(-(z @ np.array(mdl["w"]) + mdl["b"])))
            col.metric(mdl["label"], f"{p:.0%}", help="Model score for Outcome = 1")
            col.progress(float(p))
        st.caption("Both are logistic-regression models; the federated one never saw pooled "
                   "rows. Scores above 50% are classed as 'diabetes' in the results tab.")


def tab_feedback():
    st.markdown("Please complete this **after** the demo. It evaluates the software, not your "
                "health, and takes about 10 minutes. Do not enter your name.")
    url = ms_form_url()
    st.link_button("Prefer Microsoft Forms? Open the same questionnaire there", url)
    if "REPLACE_WITH_FORM_ID" in url:
        st.caption("(Form link placeholder – the researcher will add the live link.)")

    st.subheader("1. Consent")
    ticks = [st.checkbox(t, key=f"consent_{i}") for i, t in enumerate(CONSENT_ITEMS)]
    if not all(ticks):
        st.info("Tick all consent statements to unlock the questionnaire.")
        return

    st.subheader("2. Questionnaire")
    with st.form("feedback", clear_on_submit=False):
        code = st.text_input("Participant code from the researcher (e.g. P01) – optional")
        answers = {}
        for i, item in enumerate(LIKERT_ITEMS, start=1):
            answers[f"Q{i}"] = st.radio(f"**{i}.** {item}", list(SCALE), index=None,
                                        format_func=SCALE.get, horizontal=True, key=f"q{i}")
        for i, item in enumerate(OPEN_ITEMS, start=11):
            answers[f"Q{i}"] = st.text_area(f"**{i}.** {item} (optional)", key=f"q{i}")
        sent = st.form_submit_button("Submit answers", type="primary")
    if sent:
        missing = [k for k in (f"Q{i}" for i in range(1, 11)) if answers[k] is None]
        if missing:
            st.error(f"Please answer: {', '.join(missing)}")
            return
        row = {"response_id": uuid.uuid4().hex[:8],
               "submitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "participant_code": code.strip(), **answers}
        FEEDBACK_DIR.mkdir(exist_ok=True)
        out = FEEDBACK_DIR / "responses.csv"
        new = not out.exists()
        with out.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        st.success("Thank you – your anonymous answers were saved.")
        st.download_button("Download a copy of my answers (CSV)",
                           pd.DataFrame([row]).to_csv(index=False), "my_feedback.csv")


def main():
    st.set_page_config(page_title="Privacy-preserving diabetes prediction (FL demo)",
                       page_icon="🔒", layout="wide")
    st.title("🔒 Privacy-Preserving Diabetes Prediction")
    st.caption("Federated learning prototype on the Pima Indians Diabetes Database · "
               "MSc Computing, Sheffield Hallam University · Nikhil Reddy Meesala")
    m = load_metrics()
    tabs = st.tabs(["🏠 Home", "📊 Results", "🛡️ Privacy explained", "🧪 Try a prediction",
                    "📝 Feedback"])
    with tabs[0]:
        tab_home(m)
    with tabs[1]:
        tab_results(m)
    with tabs[2]:
        tab_privacy(m)
    with tabs[3]:
        tab_predict()
    with tabs[4]:
        tab_feedback()


if __name__ == "__main__":
    main()
