import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heart_core import ensure_bundle, predict_proba, contributions, adjust_for_prevalence  # noqa: E402

EX = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
      "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}
HIGH = dict(EX, age=67, cp=4, exang=1, oldpeak=3.0, thalach=110, ca=3, thal=7, slope=2)
LOW = dict(EX, age=40, sex=0, cp=2, exang=0, oldpeak=0.0, thalach=180, ca=0, thal=3, slope=1)


@pytest.fixture(scope="module")
def bundle():
    return ensure_bundle()


@pytest.mark.parametrize("key", ["portable", "full"])
def test_probability_valid_and_ordered(bundle, key):
    e = bundle[key]
    lo, hi = predict_proba(e, LOW), predict_proba(e, HIGH)
    assert 0 <= lo <= 1 and 0 <= hi <= 1
    assert hi > 0.9 > 0.2 > lo


@pytest.mark.parametrize("key", ["portable", "full"])
def test_contributions_reproduce_probability(bundle, key):
    e = bundle[key]
    c, intercept = contributions(e, EX)
    p = 1 / (1 + np.exp(-(c.log_odds_contribution.sum() + intercept)))
    assert p == pytest.approx(predict_proba(e, EX), abs=1e-9)


def test_prevalence_adjustment(bundle):
    e = bundle["portable"]; tp = e["prevalence"]
    assert adjust_for_prevalence(0.6, tp, tp) == pytest.approx(0.6)
    assert adjust_for_prevalence(0.6, tp, 0.9) > 0.6 > adjust_for_prevalence(0.6, tp, 0.2)
    # preset thresholds keep the same decision after adjustment
    p, t = predict_proba(e, EX), e["screening_threshold"]
    for lp in (0.2, 0.5, 0.9):
        assert (p >= t) == (adjust_for_prevalence(p, tp, lp) >= adjust_for_prevalence(t, tp, lp))


def test_bundle_quality(bundle):
    assert bundle["portable"]["cv_auc"] > 0.85 and bundle["full"]["cv_auc"] > 0.88
    assert 0.40 < bundle["full"]["prevalence"] < 0.50


def test_retrains_when_model_file_missing_or_stale(tmp_path):
    import joblib
    missing = tmp_path / "nope.joblib"
    b = ensure_bundle(missing)                      # missing -> retrain, writes file
    assert missing.exists() and "portable" in b
    stale = tmp_path / "stale.joblib"
    joblib.dump({"_meta": {"sklearn_version": "0.0.1"}, "portable": None}, stale)
    b2 = ensure_bundle(stale)                       # version mismatch -> retrain
    assert b2["portable"]["pipeline"] is not None


def test_streamlit_app_runs_without_exception():
    at_mod = pytest.importorskip("streamlit.testing.v1")
    app = Path(__file__).resolve().parents[1] / "app.py"
    at = at_mod.AppTest.from_file(str(app), default_timeout=120).run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Full (13 features)").run()
    assert not at.exception
