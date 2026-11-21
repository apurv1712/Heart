"""Heart disease risk screening demo (educational). Run:  streamlit run app.py"""
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from heart_core import ensure_bundle, predict_proba, contributions, adjust_for_prevalence

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"


def asset(name):
    """Find a figure or table in assets/, or next to app.py if it was uploaded there by mistake."""
    for folder in (ASSETS, HERE):
        p = folder / name
        if p.exists():
            return p
    return None

st.set_page_config(page_title="Heart Disease Risk Screening", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
#MainMenu, footer, .stDeployButton {visibility: hidden;}
.block-container {padding-top: 2.2rem; max-width: 1180px;}
h1, h2, h3, h4 {font-family: Georgia, "Times New Roman", serif !important; font-weight: 600 !important; letter-spacing: 0;}
.topbar {border-bottom: 2px solid #1F4E79; padding-bottom: 0.7rem; margin-bottom: 0.9rem;}
.topbar .t {font-family: Georgia, "Times New Roman", serif; font-size: 1.85rem; font-weight: 600; color: #1B2733;}
.topbar .s {font-size: 0.9rem; color: #5B6770; margin-top: 0.15rem;}
.notice {font-size: 0.86rem; color: #3D4852; background: #F4F6F8; border-left: 3px solid #8A949D; padding: 0.55rem 0.8rem; margin-bottom: 1.1rem;}
.sec {font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: #5B6770; font-weight: 600; margin: 0.2rem 0 0.4rem 0;}
.result {border: 1px solid #D5DAE0; border-left: 5px solid #8A949D; border-radius: 3px; padding: 1rem 1.1rem 0.9rem 1.1rem; background: #FFFFFF;}
.result.pos {border-left-color: #A63D40;}
.result.neg {border-left-color: #3C7A5A;}
.result .lab {font-size: 0.82rem; color: #5B6770;}
.result .val {font-family: Georgia, "Times New Roman", serif; font-size: 2.9rem; line-height: 1.15; color: #1B2733;}
.track {position: relative; height: 8px; background: #E6EAEE; border-radius: 2px; margin: 0.6rem 0 0.25rem 0;}
.track .fill {position: absolute; left: 0; top: 0; bottom: 0; background: #1F4E79; border-radius: 2px;}
.track .mark {position: absolute; top: -4px; width: 2px; height: 16px; background: #1B2733;}
.ticks {display: flex; justify-content: space-between; font-size: 0.74rem; color: #6B757E;}
.outcome {font-weight: 600; margin-top: 0.8rem; font-size: 0.98rem;}
.outcome.pos {color: #A63D40;}
.outcome.neg {color: #2F6B4D;}
.hint {font-size: 0.84rem; color: #5B6770; margin-top: 0.25rem;}
table.kv {width: 100%; border-collapse: collapse; margin-top: 0.8rem; font-size: 0.88rem;}
table.kv td {padding: 0.35rem 0; border-top: 1px solid #E6EAEE;}
table.kv td:last-child {text-align: right; font-weight: 600;}
.crow {display: flex; align-items: center; margin: 0.28rem 0; font-size: 0.88rem;}
.crow .n {width: 34%; color: #1B2733;}
.crow .tr {position: relative; flex: 1; height: 14px; background: #F4F6F8;}
.crow .tr:before {content: ""; position: absolute; left: 50%; top: -2px; bottom: -2px; width: 1px; background: #8A949D;}
.crow .b {position: absolute; top: 2px; bottom: 2px;}
.crow .b.up {background: #A63D40;}
.crow .b.dn {background: #2F6F8F;}
.crow .v {width: 64px; text-align: right; color: #3D4852; font-variant-numeric: tabular-nums;}
.legend {font-size: 0.8rem; color: #5B6770; margin-bottom: 0.5rem;}
.legend i {display: inline-block; width: 10px; height: 10px; margin: 0 4px 0 10px; vertical-align: middle;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models (retrains once if needed, about 10 s)...")
def get_bundle():
    return ensure_bundle()


bundle = get_bundle()

st.markdown(
    '<div class="topbar"><div class="t">Heart Disease Risk Screening</div>'
    '<div class="s">Logistic regression models trained on the UCI Heart Disease database '
    '(Cleveland Clinic, 303 patients)</div></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="notice"><b>Educational project, not a medical device.</b> '
    'Do not use it for real diagnosis or treatment decisions.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("Model settings")
    mode = st.radio("Feature set", ["Portable (8 features)", "Full (13 features)"],
                    help="Portable uses only measurements available at all four hospitals. Full also needs "
                         "fluoroscopy vessel count (ca), thallium scan (thal), slope, cholesterol and fasting sugar.")
    key = "portable" if mode.startswith("Portable") else "full"
    entry = bundle[key]
    st.caption(f"Cross-validated ROC-AUC: **{entry['cv_auc']:.3f}** ± {entry['cv_auc_std']:.3f} (Cleveland only)")
    if key == "portable":
        st.caption("External hospitals (trained on Cleveland): AUC about 0.73–0.87.")
    else:
        st.caption("Full model cannot be run at other hospitals: `ca` is 96–99% missing there.")

    st.divider()
    st.markdown("**Population**")
    train_prev = entry["prevalence"]
    adjust = st.checkbox("Adjust for local disease prevalence", value=False,
                         help="The model learned from a population where "
                              f"{train_prev:.0%} of patients had disease. In a population with a different "
                              "rate, raw probabilities are systematically too high or too low.")
    local_prev = train_prev
    if adjust:
        local_prev = st.slider("Disease prevalence in your population", 0.02, 0.98, float(round(train_prev, 2)), 0.01)
        st.caption("Shifts the baseline risk only (prior-shift correction). It cannot fix other site "
                   "differences such as different measurement practices or patient mix.")
    else:
        st.caption(f"Probabilities are for a Cleveland-like population ({train_prev:.0%} prevalence).")

    st.divider()
    st.markdown("**Decision threshold**")
    bal_thr = adjust_for_prevalence(0.5, train_prev, local_prev)
    scr_thr = adjust_for_prevalence(entry["screening_threshold"], train_prev, local_prev)
    preset = st.radio("Preset", [f"Balanced ({bal_thr:.2f})", f"Screening (≈90% sensitivity, {scr_thr:.2f})", "Custom"])
    if preset.startswith("Balanced"):
        threshold = bal_thr
    elif preset.startswith("Screening"):
        threshold = scr_thr
    else:
        threshold = st.slider("Threshold", 0.05, 0.95, 0.5, 0.01)
    st.caption("Lower threshold = fewer missed sick patients but more false alarms. "
               "Thresholds tuned on Cleveland do not transfer to other populations.")

# ------------------------------------------------------------------ inputs
CP = {"Typical angina": 1, "Atypical angina": 2, "Non-anginal pain": 3, "Asymptomatic": 4}
ECG = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
SLOPE = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
THAL = {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}
YN = {"No": 0, "Yes": 1}
LABELS = {"age": "Age", "sex": "Sex (male)", "cp": "Chest pain type", "trestbps": "Resting blood pressure",
          "chol": "Cholesterol", "fbs": "Fasting blood sugar", "restecg": "Resting ECG",
          "thalach": "Max heart rate", "exang": "Exercise-induced angina", "oldpeak": "ST depression",
          "slope": "ST slope", "ca": "Vessels by fluoroscopy", "thal": "Thallium scan"}

DEFAULTS = {"in_age": 55, "in_sex": "Male", "in_cp": "Asymptomatic", "in_trestbps": 130,
            "in_restecg": "Normal", "in_thalach": 150, "in_exang": "No", "in_oldpeak": 1.0,
            "in_chol": 240, "in_fbs": "No", "in_slope": "Flat", "in_ca": 0, "in_thal": "Normal"}
EXAMPLES = {
    "low": {"in_age": 40, "in_sex": "Female", "in_cp": "Atypical angina", "in_trestbps": 120,
            "in_restecg": "Normal", "in_thalach": 180, "in_exang": "No", "in_oldpeak": 0.0,
            "in_chol": 200, "in_fbs": "No", "in_slope": "Upsloping", "in_ca": 0, "in_thal": "Normal"},
    "high": {"in_age": 67, "in_sex": "Male", "in_cp": "Asymptomatic", "in_trestbps": 160,
             "in_restecg": "Left ventricular hypertrophy", "in_thalach": 110, "in_exang": "Yes",
             "in_oldpeak": 3.0, "in_chol": 286, "in_fbs": "No", "in_slope": "Flat", "in_ca": 3,
             "in_thal": "Reversible defect"},
    "reset": DEFAULTS,
}

# Remember every input so values survive switching between the Portable and Full model
# (Streamlit drops the state of widgets that are not drawn on a given run).
_keep = st.session_state.setdefault("_keep", dict(DEFAULTS))
for _k, _v in _keep.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def load_example(name):
    for k, v in EXAMPLES[name].items():
        st.session_state[k] = v
        st.session_state["_keep"][k] = v


tab_pred, tab_eval, tab_about = st.tabs(["Risk estimate", "Model evaluation", "About the data and limits"])

with tab_pred:
    left, right = st.columns([3, 2], gap="large")
    inputs = {}

    with left:
        st.markdown('<div class="sec">Quick fill (illustrative values, not real patients)</div>',
                    unsafe_allow_html=True)
        q1, q2, q3 = st.columns(3)
        q1.button("Low-risk example", on_click=load_example, args=("low",))
        q2.button("High-risk example", on_click=load_example, args=("high",))
        q3.button("Reset values", on_click=load_example, args=("reset",))

        st.markdown('<div class="sec" style="margin-top:1rem">Patient</div>', unsafe_allow_html=True)
        with st.container(border=True):
            a, b = st.columns(2)
            inputs["age"] = a.number_input("Age (years)", 18, 100, key="in_age")
            inputs["sex"] = 1 if b.selectbox("Sex", ["Female", "Male"], key="in_sex") == "Male" else 0
            inputs["cp"] = CP[a.selectbox("Chest pain type", list(CP), key="in_cp")]
            inputs["trestbps"] = b.number_input("Resting blood pressure (mm Hg)", 80, 220, key="in_trestbps")

        st.markdown('<div class="sec" style="margin-top:1rem">Resting ECG and exercise test</div>',
                    unsafe_allow_html=True)
        with st.container(border=True):
            a, b = st.columns(2)
            inputs["restecg"] = ECG[a.selectbox("Resting ECG", list(ECG), key="in_restecg")]
            inputs["thalach"] = b.number_input("Max heart rate achieved (bpm)", 60, 220, key="in_thalach")
            inputs["exang"] = YN[a.selectbox("Exercise-induced angina", list(YN), key="in_exang")]
            inputs["oldpeak"] = b.number_input("ST depression (oldpeak)", 0.0, 7.0, step=0.1, key="in_oldpeak")

        if key == "full":
            st.markdown('<div class="sec" style="margin-top:1rem">Laboratory and imaging</div>',
                        unsafe_allow_html=True)
            with st.container(border=True):
                a, b = st.columns(2)
                inputs["chol"] = a.number_input("Serum cholesterol (mg/dl)", 100, 600, key="in_chol")
                inputs["fbs"] = YN[b.selectbox("Fasting blood sugar > 120 mg/dl", list(YN), key="in_fbs")]
                inputs["slope"] = SLOPE[a.selectbox("Peak exercise ST slope", list(SLOPE), key="in_slope")]
                inputs["ca"] = b.selectbox("Major vessels coloured by fluoroscopy (0–3)", [0, 1, 2, 3], key="in_ca")
                inputs["thal"] = THAL[a.selectbox("Thallium scan result", list(THAL), key="in_thal")]
        else:
            st.caption("Cholesterol, fasting sugar, ST slope, vessel count and thallium result are used only by "
                       "the Full model. Choose it under Model settings in the sidebar.")

        for _k in DEFAULTS:
            if _k in st.session_state:
                st.session_state["_keep"][_k] = st.session_state[_k]

        out_of_range = [f"{LABELS.get(f, f)} = {inputs[f]}" for f, (lo, hi) in entry["ranges"].items()
                        if f in inputs and not (lo <= inputs[f] <= hi)]
        if out_of_range:
            st.warning("Outside the training data range (prediction is extrapolating): " + ", ".join(out_of_range))

    # ------------------------------------------------------------ prediction
    raw_prob = predict_proba(entry, inputs)
    prob = adjust_for_prevalence(raw_prob, train_prev, local_prev) if adjust else raw_prob
    positive = prob >= threshold
    cls = "pos" if positive else "neg"
    outcome = ("At or above threshold: refer for further testing" if positive
               else "Below threshold: no referral triggered")
    hint = ("In a screening setting this would trigger follow-up." if positive
            else "A negative result does not rule out heart disease.")
    pct = min(max(prob, 0.0), 1.0) * 100
    thr_pct = min(max(threshold, 0.0), 1.0) * 100

    with right:
        st.markdown('<div class="sec">Result</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="result {cls}">'
            f'<div class="lab">Estimated probability of heart disease</div>'
            f'<div class="val">{prob:.1%}</div>'
            f'<div class="track"><div class="fill" style="width:{pct:.1f}%"></div>'
            f'<div class="mark" style="left:{thr_pct:.1f}%"></div></div>'
            f'<div class="ticks"><span>0%</span><span>threshold {threshold:.2f}</span><span>100%</span></div>'
            f'<div class="outcome {cls}">{escape(outcome)}</div>'
            f'<div class="hint">{escape(hint)}</div>'
            f'</div>', unsafe_allow_html=True)

        rows = [("Model", "Portable, 8 features" if key == "portable" else "Full, 13 features"),
                ("Threshold used", f"{threshold:.2f}"),
                ("Cleveland-baseline probability", f"{raw_prob:.1%}")]
        if adjust:
            rows.append(("Local prevalence assumed", f"{local_prev:.0%}"))
        st.markdown('<table class="kv">' + "".join(
            f"<tr><td>{escape(k)}</td><td>{escape(v)}</td></tr>" for k, v in rows) + "</table>",
            unsafe_allow_html=True)

    # ---------------------------------------------------------- explanation
    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
    st.subheader("What drove this estimate")
    contrib, intercept = contributions(entry, inputs)
    st.markdown('<div class="legend">Contribution to the log-odds of disease'
                '<i style="background:#A63D40"></i>raises risk<i style="background:#2F6F8F"></i>lowers risk</div>',
                unsafe_allow_html=True)
    vmax = max(float(contrib["log_odds_contribution"].abs().max()), 1e-9)
    html_rows = []
    for _, r in contrib.iterrows():
        v = float(r["log_odds_contribution"])
        w = abs(v) / vmax * 50
        bar = (f'<div class="b up" style="left:50%;width:{w:.1f}%"></div>' if v >= 0
               else f'<div class="b dn" style="left:{50 - w:.1f}%;width:{w:.1f}%"></div>')
        html_rows.append(f'<div class="crow"><div class="n">{escape(LABELS.get(r["feature"], r["feature"]))}</div>'
                         f'<div class="tr">{bar}</div><div class="v">{v:+.2f}</div></div>')
    st.markdown("".join(html_rows), unsafe_allow_html=True)
    st.caption("Exact for logistic regression. Numeric inputs are measured against the training average; yes/no "
               "and category inputs count from zero, so 'Sex (male)' shows a large positive value because males "
               "had higher disease rates in this dataset.")
    with st.expander("Numeric details"):
        shown = contrib.copy()
        shown["feature"] = shown["feature"].map(lambda f: LABELS.get(f, f))
        st.dataframe(shown.round(3), hide_index=True)
        st.write(f"Intercept (baseline log-odds): {intercept:.3f}. Sum of contributions + intercept = "
                 f"{contrib['log_odds_contribution'].sum() + intercept:.3f}, which gives a probability of "
                 f"{raw_prob:.1%} before any prevalence adjustment.")

# --------------------------------------------------------------- evaluation
with tab_eval:
    st.subheader("Model comparison")
    cmp_path = asset("model_comparison.csv")
    if cmp_path:
        cmp = pd.read_csv(cmp_path).rename(columns={
            "model": "Model", "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
            "f1": "F1", "roc_auc": "ROC-AUC", "auc_std": "AUC std"})
        st.dataframe(cmp.style.format({c: "{:.3f}" for c in cmp.columns if c != "Model"}),
                     hide_index=True)
        st.caption("Cross-validated results on the Cleveland data. Logistic regression gave the highest "
                   "ROC-AUC and accuracy and is the easiest to explain, so it is the model used in this app.")
    figures = [
        ("04_test_eval.png", "Held-out test evaluation"),
        ("05_permutation_importance.png", "Permutation importance"),
        ("06_repeated_cv_boxplot.png", "Repeated cross-validation"),
        ("07_threshold_tradeoff.png", "Threshold trade-off"),
        ("08_cross_hospital_roc.png", "Cross-hospital ROC"),
        ("09_calibration.png", "Calibration"),
        ("14_fairness_subgroups.png", "Subgroup performance"),
    ]
    available = [(f, t) for f, t in figures if asset(f)]
    if not available:
        st.info("Evaluation figures were not found. Add the files from the assets folder "
                "(the .png files and model_comparison.csv) to the repository.")
    if available:
        st.subheader("Figures")
        cols = st.columns(2, gap="large")
        for i, (f, t) in enumerate(available):
            with cols[i % 2]:
                st.markdown(f'<div class="sec">{escape(t)}</div>', unsafe_allow_html=True)
                st.image(str(asset(f)))

    with st.expander("How reliable are these probabilities at other hospitals?"):
        st.markdown("""
The probabilities are well calibrated **on Cleveland** (calibration slope 0.94), but a Cleveland-trained
portable model was **miscalibrated elsewhere**:

| Hospital | Disease rate in data | Model's average prediction | Raw Brier | After prevalence shift |
|---|---|---|---|---|
| Hungarian | 36% | 41% | 0.133 | 0.128 |
| VA (Long Beach) | 74% | 70% | 0.178 | 0.176 (over-corrects to 85%) |
| Switzerland | 94% | 68% | 0.168 | 0.057 |

- The **ranking** of patients (AUC about 0.73–0.87) holds up better than the **probability values**.
- Adjusting for prevalence fixes Hungary and Switzerland but over-corrects at VA, whose error is not just
  prevalence. Real deployment would need recalibration on local data.
- Switzerland and VA are referral populations; Switzerland has only 8 healthy patients, so its numbers are noisy.
""")

# -------------------------------------------------------------------- about
with tab_about:
    st.subheader("Data used")
    data_used = pd.DataFrame({
        "Hospital": ["Cleveland Clinic", "Hungarian Institute of Cardiology", "VA Medical Center, Long Beach",
                     "University Hospital, Zurich/Basel", "Total"],
        "Role": ["Training and internal cross-validation", "External validation", "External validation",
                 "External validation", ""],
        "Patients": [303, 294, 200, 123, 920],
        "With disease": [139, 106, 149, 115, 509],
        "Disease rate": ["46%", "36%", "75%", "94%", "55%"],
        "Portable-model ROC-AUC": ["0.877 (cross-validated)", "0.873", "0.733", "0.753", ""],
    })
    st.dataframe(data_used, hide_index=True)
    st.caption("The models are trained on Cleveland only. The other three hospitals (617 patients) are used only "
               "to test how the portable 8-feature model behaves at sites it never saw. Disease rates differ a "
               "lot between sites, which is why probabilities need local recalibration.")

    st.subheader("About the model and its limits")
    st.markdown(f"""
- **Model:** logistic regression, {len(entry['features'])} inputs, trained on {entry['n_train']} patients.
- **Internal performance:** ROC-AUC {entry['cv_auc']:.2f} (repeated 5×10 cross-validation).
- **External validity:** on other hospitals the portable model reached AUC ≈ 0.73–0.87, and fixed thresholds
  did not transfer (specificity fell sharply at sites with high disease prevalence).
- **Data:** Cleveland Clinic data, collected in the 1980s, with predominantly male patients.
- **Target:** angiographic narrowing > 50% in any major vessel (binary version of the `num` field).
""")
    st.caption("Data: UCI Heart Disease database (Cleveland Clinic Foundation; Hungarian Institute of Cardiology, "
               "Budapest; V.A. Medical Center, Long Beach; University Hospital, Zurich). Principal investigators: "
               "Robert Detrano M.D., Ph.D. (Cleveland, Long Beach), Andras Janosi M.D. (Hungary), William "
               "Steinbrunn M.D. (Zurich), Matthias Pfisterer M.D. (Basel). Donor: David W. Aha. Reference: "
               "Detrano et al., Am J Cardiol 1989;64:304-310. Educational use only.")
