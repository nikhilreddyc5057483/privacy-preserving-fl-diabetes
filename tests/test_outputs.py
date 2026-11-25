import importlib.util
import json

import pytest

from conftest import ROOT
from src.metrics import METRIC_KEYS, compute_metrics


def test_compute_metrics_keys():
    m = compute_metrics([0, 1, 1, 0], [0.1, 0.9, 0.4, 0.6])
    for k in METRIC_KEYS + ["confusion_matrix", "n_test"]:
        assert k in m
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_exported_metrics_have_all_keys():
    path = ROOT / "results" / "metrics.json"
    if not path.exists():
        pytest.skip("run python run_experiments.py first")
    d = json.loads(path.read_text())
    assert len(d["runs"]) >= 7
    for r in d["runs"]:
        for k in METRIC_KEYS:
            assert 0.0 <= r[k] <= 1.0
        assert r["n_test"] == 154
    for rid, h in d["history"].items():
        assert len(h["rounds"]) == 15


def test_streamlit_app_imports():
    pytest.importorskip("streamlit")
    spec = importlib.util.spec_from_file_location("streamlit_app",
                                                  ROOT / "app" / "streamlit_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                  # import only; main() is guarded
    assert len(mod.LIKERT_ITEMS) == 10 and len(mod.OPEN_ITEMS) == 2
    assert (ROOT / "models" / "models.json").exists()


def test_streamlit_app_runs_headless():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 5
