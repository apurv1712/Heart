# Heart Disease Risk Demo

Educational Streamlit app built on the UCI Heart Disease database. **Not a medical device.**

* Two logistic-regression models: **Portable** (8 features, usable at any hospital) and **Full** (13 features).
* Threshold presets (balanced / ~90% sensitivity screening), optional prevalence adjustment, per-feature explanation.
* Internal CV ROC-AUC: portable 0.877, full 0.911. External hospitals (portable): about 0.73-0.87.

## Run locally
```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Models are in `models/`. If the file is missing or was built with a different scikit-learn version, the app
retrains automatically from `data/processed_cleveland.data` (about 10 seconds, once per server start).
Retrain by hand with `python train_model.py`.

## Tests
```
pip install -r requirements-dev.txt
pytest -q
```

## Deploy on Streamlit Community Cloud (free)
1. Create a GitHub account and a **new public repository** (e.g. `heart-disease-demo`).
2. Upload everything in this folder to the repository root (so `app.py` sits at the top level):
   ```
   git init && git add . && git commit -m "Heart disease demo"
   git branch -M main
   git remote add origin https://github.com/<your-username>/heart-disease-demo.git
   git push -u origin main
   ```
3. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**, and choose
   *Deploy a public app from GitHub*.
4. Set **Repository** = your repo, **Branch** = `main`, **Main file path** = `app.py`.
5. Under **Advanced settings**, pick Python **3.12** (matches the CI workflow), then click **Deploy**.
6. First start takes a few minutes while packages install. You get a public `*.streamlit.app` link.
   Updates are redeployed automatically on every `git push`.

Alternatives: Hugging Face Spaces (choose the Streamlit SDK) or any host that runs `streamlit run app.py`.

## Before you share the link
* Keep the on-screen disclaimer and the dataset attribution in the footer.
* Free Streamlit apps go to sleep after inactivity, so the first visitor may wait about 30 seconds.
* Open the app on a phone and a laptop, and try both models, the threshold presets and the prevalence slider.
* If you change the model, re-run `python train_model.py` and commit the new `models/heart_models.joblib`.

## Data credit
UCI Heart Disease database. Please cite the investigators when publishing results: Robert Detrano M.D., Ph.D.
(V.A. Medical Center Long Beach and Cleveland Clinic Foundation), Andras Janosi M.D. (Hungarian Institute of
Cardiology), William Steinbrunn M.D. (University Hospital Zurich), Matthias Pfisterer M.D. (University Hospital Basel).
Donor: David W. Aha. Detrano et al., *Am J Cardiol* 1989;64:304-310.

## Layout
```
app.py              Streamlit UI
heart_core.py       prediction, explanation, prevalence adjustment, model loading/fallback retrain
train_model.py      trains and saves the models
data/               processed_cleveland.data
models/             heart_models.joblib
tests/test_core.py  unit tests (+ Streamlit AppTest when streamlit is installed)
.github/workflows/  CI (runs pytest on every push)
.streamlit/         server/theme config
```
