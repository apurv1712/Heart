"""Train and save the two deployable models.
 portable : 8 features available at every hospital
 full     : all 13 Cleveland features
CLI:  python train_model.py [data_path] [out_path]
"""
import sys
from pathlib import Path
import joblib, numpy as np, pandas as pd, sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import roc_curve

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data" / "processed_cleveland.data"
DEFAULT_OUT = HERE / "models" / "heart_models.joblib"
SEED = 42
COLS = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"]
ONEHOT, BINARY = {"cp","restecg","slope","thal"}, {"sex","fbs","exang"}
SETS = {"portable": ["age","sex","cp","trestbps","restecg","thalach","exang","oldpeak"],
        "full": ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]}


def make_pipe(feats):
    num = [c for c in feats if c not in ONEHOT | BINARY]
    oh = [c for c in feats if c in ONEHOT]
    bi = [c for c in feats if c in BINARY]
    prep = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore"))]), oh),
        ("bin", SimpleImputer(strategy="most_frequent"), bi)])
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000, random_state=SEED))])


def train(data_path=DEFAULT_DATA, out_path=DEFAULT_OUT, verbose=True):
    df = pd.read_csv(data_path, header=None, names=COLS, na_values="?")
    y = (df.pop("num") > 0).astype(int).values
    bundle = {"_meta": {"sklearn_version": sklearn.__version__}}
    for name, feats in SETS.items():
        X = df[feats]
        auc = cross_val_score(make_pipe(feats), X, y, scoring="roc_auc",
                              cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED))
        oof = cross_val_predict(make_pipe(feats), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                method="predict_proba")[:, 1]
        fpr, tpr, thr = roc_curve(y, oof)
        screen_thr = float(thr[np.where(tpr >= 0.90)[0][0]])
        bundle[name] = {"pipeline": make_pipe(feats).fit(X, y), "features": feats,
                        "cv_auc": float(auc.mean()), "cv_auc_std": float(auc.std()),
                        "screening_threshold": round(screen_thr, 2),
                        "ranges": {c: (float(df[c].min()), float(df[c].max())) for c in feats},
                        "n_train": int(len(df)), "prevalence": float(y.mean())}
        if verbose:
            print(f"{name}: CV AUC {auc.mean():.3f} +/- {auc.std():.3f}, screening threshold {screen_thr:.2f}")
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, out_path)
        if verbose:
            print("saved", out_path)
    return bundle


if __name__ == "__main__":
    train(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA,
          sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT)
