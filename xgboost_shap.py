"""
XGBoost + SHAP on the Cleveland data.
Setup (inside your activated .venv):   pip install xgboost shap
Run:   python xgboost_shap.py [path/to/processed_cleveland.data]

What it does
 1. Fairly compares XGBoost with Logistic Regression (repeated 5x10 stratified CV, same splits)
 2. Fits XGBoost on a training split and explains it with SHAP (global + one patient)
 3. Explains Logistic Regression with SHAP's LinearExplainer for comparison
Figures are saved in ./figures
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
SEED = 42
DATA = sys.argv[1] if len(sys.argv) > 1 else "heart_app/data/processed_cleveland.data"
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"]
df = pd.read_csv(DATA, header=None, names=cols, na_values="?")
y = (df.pop("num") > 0).astype(int)
X = df

onehot_cols = ["cp", "restecg", "slope", "thal"]
numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
binary_cols = ["sex", "fbs", "exang"]


def make_prep():
    return ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), numeric_cols),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), onehot_cols),
        ("bin", SimpleImputer(strategy="most_frequent"), binary_cols)])


def make_xgb():
    return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, eval_metric="logloss", random_state=SEED)


def make_lr():
    return LogisticRegression(max_iter=2000, random_state=SEED)


# ---------------------------------------------------------------- 1. fair comparison
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
print("=== Repeated 5x10 CV, ROC-AUC ===")
for name, mk in [("LogReg", make_lr), ("XGBoost", make_xgb)]:
    s = cross_val_score(Pipeline([("prep", make_prep()), ("clf", mk())]), X, y, cv=cv, scoring="roc_auc")
    print(f"{name:8s} {s.mean():.3f} +/- {s.std():.3f}")
print("(On ~300 rows expect XGBoost to be about equal to, not better than, logistic regression.)\n")

# ---------------------------------------------------------------- 2. SHAP for XGBoost
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
prep = make_prep().fit(X_tr)


def transform(data):
    arr = prep.transform(data)
    arr = arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr)
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    return pd.DataFrame(arr, columns=names, index=data.index)


Xt_tr, Xt_te = transform(X_tr), transform(X_te)

xgb = make_xgb().fit(Xt_tr, y_tr)
print(f"XGBoost test ROC-AUC (single split, n={len(y_te)}): {roc_auc_score(y_te, xgb.predict_proba(Xt_te)[:, 1]):.3f}")

explainer = shap.TreeExplainer(xgb)
sv = explainer(Xt_te)                    # shap.Explanation, values are in log-odds units

plt.figure(); shap.summary_plot(sv, Xt_te, show=False)
plt.tight_layout(); plt.savefig(f"{OUT}/10_shap_xgb_summary.png", dpi=130, bbox_inches="tight"); plt.close()

plt.figure(); shap.plots.bar(sv, show=False)
plt.tight_layout(); plt.savefig(f"{OUT}/11_shap_xgb_bar.png", dpi=130, bbox_inches="tight"); plt.close()

# one patient: pick the test patient the model thinks is riskiest
i = int(np.argmax(xgb.predict_proba(Xt_te)[:, 1]))
plt.figure(); shap.plots.waterfall(sv[i], show=False)
plt.tight_layout(); plt.savefig(f"{OUT}/12_shap_xgb_patient.png", dpi=130, bbox_inches="tight"); plt.close()

mean_abs = pd.Series(np.abs(sv.values).mean(0), index=Xt_te.columns).sort_values(ascending=False)
print("\nTop features by mean |SHAP| (XGBoost):")
print(mean_abs.head(8).round(3).to_string())

# ---------------------------------------------------------------- 3. SHAP for logistic regression
lr = make_lr().fit(Xt_tr, y_tr)
lin = shap.LinearExplainer(lr, Xt_tr)
sv_lr = lin(Xt_te)
plt.figure(); shap.summary_plot(sv_lr, Xt_te, show=False)
plt.tight_layout(); plt.savefig(f"{OUT}/13_shap_logreg_summary.png", dpi=130, bbox_inches="tight"); plt.close()
print("\nSaved SHAP figures 10-13 to", OUT)