"""Heart disease risk screening demo (educational). Run:  streamlit run app.py"""
import numpy as np
import pandas as pd
import streamlit as st
from heart_core import ensure_bundle, predict_proba, contributions, adjust_for_prevalence

st.set_page_config(page_title="Heart Disease Risk Screening", page_icon="🫀", layout="wide")

@st.cache_resource(show_spinner="Loading models (retrains once if needed, about 10 s)...")
def get_bundle():
    return ensure_bundle()

bundle = get_bundle()

st.title("Heart Disease Risk Screening")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Model")
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

    st.header("Population")
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

    st.header("Decision threshold")
    bal_thr = adjust_for_prevalence(0.5, train_prev, local_prev)
    scr_thr = adjust_for_prevalence(entry["screening_threshold"], train_prev, local_prev)
    preset = st.radio("Preset", [f"Balanced ({bal_thr:.2f})", f"Screening (≈90% sensitivity, {scr_thr:.2f})", "Custom"])
    if preset.startswith("Balanced"):
        threshold = adjust_for_prevalence(0.5, train_prev, local_prev)
    elif preset.startswith("Screening"):
        threshold = adjust_for_prevalence(entry["screening_threshold"], train_prev, local_prev)
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

st.subheader("Patient measurements")
c1, c2, c3 = st.columns(3)
inputs = {}
with c1:
    inputs["age"] = st.number_input("Age (years)", 18, 100, 55)
    inputs["sex"] = 1 if st.selectbox("Sex", ["Female", "Male"], index=1) == "Male" else 0
    inputs["cp"] = CP[st.selectbox("Chest pain type", list(CP), index=3)]
    inputs["trestbps"] = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 130)
with c2:
    inputs["restecg"] = ECG[st.selectbox("Resting ECG", list(ECG))]
    inputs["thalach"] = st.number_input("Max heart rate achieved (bpm)", 60, 220, 150)
    inputs["exang"] = YN[st.selectbox("Exercise-induced angina", list(YN))]
    inputs["oldpeak"] = st.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, 0.1)
with c3:
    if key == "full":
        inputs["chol"] = st.number_input("Serum cholesterol (mg/dl)", 100, 600, 240)
        inputs["fbs"] = YN[st.selectbox("Fasting blood sugar > 120 mg/dl", list(YN))]
        inputs["slope"] = SLOPE[st.selectbox("Peak exercise ST slope", list(SLOPE), index=1)]
        inputs["ca"] = st.selectbox("Major vessels coloured by fluoroscopy (0–3)", [0, 1, 2, 3])
        inputs["thal"] = THAL[st.selectbox("Thallium scan result", list(THAL))]
    else:
        st.info("Switch to the **Full** model in the sidebar to also enter cholesterol, fasting sugar, "
                "ST slope, vessel count and thallium result.")

# range sanity warnings (values outside the training range are extrapolation)
out_of_range = [f"{f} = {inputs[f]}" for f, (lo, hi) in entry["ranges"].items()
                if f in inputs and not (lo <= inputs[f] <= hi)]
if out_of_range:
    st.warning("Outside the training data range (prediction is extrapolating): " + ", ".join(out_of_range))

# -------------------------------------------------------------- prediction
raw_prob = predict_proba(entry, inputs)
prob = adjust_for_prevalence(raw_prob, train_prev, local_prev) if adjust else raw_prob
positive = prob >= threshold

st.subheader("Result")
m1, m2, m3 = st.columns(3)
m1.metric("Estimated probability of heart disease", f"{prob:.1%}",
          delta=(f"{(prob - raw_prob) * 100:+.1f} pts vs Cleveland baseline" if adjust else None), delta_color="off")
m2.metric("Threshold used", f"{threshold:.2f}")
m3.metric("Screening outcome", "⚠️ Positive (refer for testing)" if positive else "✅ Negative")
st.progress(min(max(prob, 0.0), 1.0))
if positive:
    st.error("Probability is at or above the threshold. In a screening setting this would trigger follow-up.")
else:
    st.success("Probability is below the threshold. This does not rule out heart disease.")

# ------------------------------------------------------------ explanation
st.subheader("Why this prediction?")
contrib, intercept = contributions(entry, inputs)
st.caption("Each bar is how much a feature pushed the log-odds toward disease (positive) or away from it "
           "(negative). Exact for logistic regression. Numeric features are measured relative to the "
           "training average; yes/no and category features count from zero, so e.g. 'sex' shows a large "
           "positive value for males because males had higher disease rates in this dataset.")
chart = contrib.set_index("feature")["log_odds_contribution"]
st.bar_chart(chart, horizontal=True)
with st.expander("Details"):
    st.dataframe(contrib.round(3), use_container_width=True, hide_index=True)
    st.write(f"Intercept (baseline log-odds): {intercept:.3f}. Sum of contributions + intercept = "
             f"{contrib['log_odds_contribution'].sum() + intercept:.3f} → probability {prob:.1%}.")


with st.expander("⚠️ How reliable are these probabilities at other hospitals?", expanded=False):
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

with st.expander("About the model and its limits"):
    st.markdown(f"""
- **Model:** logistic regression, {len(entry['features'])} inputs, trained on {entry['n_train']} patients.
- **Internal performance:** ROC-AUC {entry['cv_auc']:.2f} (repeated 5×10 cross-validation).
- **External validity:** on other hospitals the portable model reached AUC ≈ 0.73–0.87, and fixed thresholds
  did not transfer (specificity fell sharply at sites with high disease prevalence).
- **Data:** Cleveland Clinic data, collected in the 1980s, with predominantly male patients.
- **Target:** angiographic narrowing > 50% in any major vessel (binary version of the `num` field).
""")

st.divider()
st.caption("Data: UCI Heart Disease database (Cleveland Clinic Foundation; Hungarian Institute of Cardiology, Budapest; "
           "V.A. Medical Center, Long Beach; University Hospital, Zurich). Principal investigators: Robert Detrano M.D., Ph.D. "
           "(Cleveland, Long Beach), Andras Janosi M.D. (Hungary), William Steinbrunn M.D. (Zurich), "
           "Matthias Pfisterer M.D. (Basel). Donor: David W. Aha. Reference: Detrano et al., Am J Cardiol 1989;64:304-310. "
           "Educational use only.")
