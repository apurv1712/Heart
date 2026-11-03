"""Prediction + per-feature explanation helpers (no Streamlit dependency, unit-testable)."""
from pathlib import Path
import joblib, numpy as np, pandas as pd, sklearn

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "heart_models.joblib"


def load_bundle(path=MODEL_PATH):
    return joblib.load(path)


def ensure_bundle(path=MODEL_PATH):
    """Load the saved models. If the file is missing, unreadable, or was built with a different
    scikit-learn version (pickles are not portable across versions), retrain from data/ instead."""
    try:
        b = load_bundle(path)
        if b.get("_meta", {}).get("sklearn_version") == sklearn.__version__:
            return b
    except Exception:
        pass
    from train_model import train
    try:
        return train(out_path=path, verbose=False)
    except OSError:  # read-only filesystem: keep the retrained models in memory only
        return train(out_path=None, verbose=False)

def predict_proba(entry, inputs: dict) -> float:
    X = pd.DataFrame([{f: inputs.get(f, np.nan) for f in entry["features"]}])
    return float(entry["pipeline"].predict_proba(X)[0, 1])

def contributions(entry, inputs: dict) -> pd.DataFrame:
    """Per-feature contribution to the log-odds (coefficient x transformed value), grouped by original
    feature. Positive = pushes toward 'heart disease'. Exact for logistic regression."""
    pipe = entry["pipeline"]; prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    X = pd.DataFrame([{f: inputs.get(f, np.nan) for f in entry["features"]}])
    z = np.asarray(prep.transform(X)).ravel() * clf.coef_.ravel()
    rows = {}
    for name, v in zip(prep.get_feature_names_out(), z):
        kind, rest = name.split("__", 1)
        orig = rest.rsplit("_", 1)[0] if kind == "cat" else rest
        rows[orig] = rows.get(orig, 0.0) + float(v)
    out = pd.DataFrame({"feature": list(rows), "log_odds_contribution": list(rows.values())})
    out["abs"] = out["log_odds_contribution"].abs()
    return out.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True), float(clf.intercept_[0])


def adjust_for_prevalence(prob: float, train_prev: float, local_prev: float) -> float:
    """Prior-shift correction: re-weights the baseline odds from the training prevalence to the local one.
    logit(p') = logit(p) + log(local_odds / train_odds). Fixes the average level of the probabilities only;
    it does not change how patients are ranked (AUC is unchanged)."""
    eps = 1e-6
    p = min(max(prob, eps), 1 - eps)
    shift = np.log((local_prev / (1 - local_prev)) / (train_prev / (1 - train_prev)))
    z = np.log(p / (1 - p)) + shift
    return float(1 / (1 + np.exp(-z)))
