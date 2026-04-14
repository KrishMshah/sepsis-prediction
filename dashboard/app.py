import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Sepsis Prediction Dashboard",
    page_icon  = "🏥",
    layout     = "wide"
)

# ────────────────────────────────────────────────────────────────────────────
#  Load resources (cached so they only load once)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return joblib.load("models/xgboost.pkl")

@st.cache_data
def load_feature_cols():
    return pd.read_csv("models/feature_cols.csv").iloc[:, 0].tolist()

@st.cache_data
def load_metrics():
    return pd.read_csv("outputs/results/metrics.csv")

@st.cache_data
def load_metrics_top30():
    return pd.read_csv("outputs/results/metrics_top30.csv")

@st.cache_data
def load_shap_importance():
    return pd.read_csv("outputs/results/shap_feature_importance.csv")

@st.cache_data
def load_eri():
    return pd.read_csv("outputs/results/eri_paper_table.csv")

model        = load_model()
feature_cols = load_feature_cols()

# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Go to",
    ["Patient Risk Predictor", "Model Performance", "Global Explainability", "Stress Test Results"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** Explainable Sepsis Onset Prediction")
st.sidebar.markdown("**Dataset:** PhysioNet CinC 2019")
st.sidebar.markdown("**Model:** XGBoost + SHAP")
st.sidebar.markdown("**Team:** Krish | Yuvraj | Riaan")

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: generate SHAP waterfall for a single patient
# ─────────────────────────────────────────────────────────────────────────────

def shap_waterfall(input_array: np.ndarray, feature_cols: list):
    explainer   = shap.TreeExplainer(model)
    shap_vals   = explainer.shap_values(input_array)
    fig, ax     = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_vals[0],
            base_values   = explainer.expected_value,
            data          = input_array[0],
            feature_names = feature_cols
        ),
        max_display = 15,
        show        = False
    )
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 1 — Patient Risk Predictor
# ─────────────────────────────────────────────────────────────────────────────

if page == "Patient Risk Predictor":
    st.title("Patient Sepsis Risk Predictor")
    st.markdown("Enter the patient's current ICU readings to get a sepsis risk score and explanation.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Vitals")
        hr      = st.number_input("Heart Rate (HR)",            0.0, 300.0, 85.0)
        sbp     = st.number_input("Systolic BP (SBP)",          0.0, 300.0, 120.0)
        map_val = st.number_input("Mean Arterial Pressure (MAP)", 0.0, 200.0, 85.0)
        dbp     = st.number_input("Diastolic BP (DBP)",         0.0, 200.0, 70.0)
        resp    = st.number_input("Respiratory Rate (Resp)",    0.0, 60.0,  18.0)
        o2sat   = st.number_input("O2 Saturation (O2Sat)",      0.0, 100.0, 98.0)
        temp    = st.number_input("Temperature (Temp)",         30.0, 45.0, 37.0)

    with col2:
        st.subheader("Labs")
        lactate    = st.number_input("Lactate",     0.0, 30.0,  1.5)
        creatinine = st.number_input("Creatinine",  0.0, 20.0,  0.9)
        wbc        = st.number_input("WBC",         0.0, 100.0, 8.0)
        platelets  = st.number_input("Platelets",   0.0, 1000.0, 200.0)
        bun        = st.number_input("BUN",         0.0, 200.0, 15.0)
        glucose    = st.number_input("Glucose",     0.0, 500.0, 110.0)
        bilirubin  = st.number_input("Bilirubin Total", 0.0, 30.0, 0.8)

    with col3:
        st.subheader("Patient Info")
        age          = st.number_input("Age",              0.0,  100.0, 60.0)
        iculos       = st.number_input("ICU Hour (ICULOS)", 1.0, 300.0,  6.0)
        hospadmtime  = st.number_input("Hospital Admission Time", -300.0, 0.0, -10.0)
        gender       = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        unit1        = st.selectbox("Unit 1", [0, 1])
        unit2        = st.selectbox("Unit 2", [0, 1])
        fio2         = st.number_input("FiO2",  0.0, 1.0, 0.21)
    st.markdown("---")
    if st.button("Predict Sepsis Risk", type="primary"):
        with st.spinner("Computing risk and explanation..."):

            # Build a feature vector with defaults for engineered features
            # Real deployment would compute these from patient history
            feature_defaults = {f: 0.0 for f in feature_cols}

            # Fill known raw features
            known = {
                "HR": hr, "SBP": sbp, "MAP": map_val, "DBP": dbp,
                "Resp": resp, "O2Sat": o2sat, "Temp": temp,
                "Lactate": lactate, "Creatinine": creatinine, "WBC": wbc,
                "Platelets": platelets, "BUN": bun, "Glucose": glucose,
                "Bilirubin_total": bilirubin, "Age": age,
                "HospAdmTime": hospadmtime, "iculos_hour": iculos,
                "Gender": gender, "Unit1": unit1, "Unit2": unit2,
                "FiO2": fio2,
                "shock_index": hr / sbp if sbp > 0 else 0,
                "icu_stay_progress": min(iculos / max(iculos, 1), 1.0),
                "sofa_renal": 2 if creatinine > 2.0 else (1 if creatinine > 1.2 else 0),
                "sofa_liver": 2 if bilirubin > 2.0 else (1 if bilirubin > 1.2 else 0),
                "sofa_coag": 2 if platelets < 100 else (1 if platelets < 150 else 0),
                "sofa_resp": 1 if o2sat < 95 else 0,
                "sofa_proxy_total": 0,
                "HR_delta": 0.0, "MAP_delta": 0.0,
                "Lactate_delta": 0.0, "Creatinine_delta": 0.0,
                "Platelets_delta": 0.0,
            }

            # Compute sofa total
            known["sofa_proxy_total"] = (
                known["sofa_renal"] + known["sofa_liver"] +
                known["sofa_coag"] + known["sofa_resp"]
            )

            # Fill rolling features with raw values as approximations
            for col in feature_cols:
                if col in known:
                    feature_defaults[col] = known[col]
                elif "_roll" in col:
                    base = col.split("_roll")[0]
                    if base in known:
                        feature_defaults[col] = known[base]

            X_input = np.array([[feature_defaults[f] for f in feature_cols]], dtype=np.float32)
            risk    = model.predict_proba(X_input)[0, 1]

            # Display risk score
            st.markdown("### Sepsis Risk Score")
            col_r1, col_r2 = st.columns([1, 3])

            with col_r1:
                color = "#28a745" if risk < 0.3 else ("#ffc107" if risk < 0.6 else "#dc3545")
                st.markdown(
                    f"<div style='background:{color};padding:20px;border-radius:10px;"
                    f"text-align:center;color:white;font-size:36px;font-weight:bold'>"
                    f"{risk:.1%}</div>",
                    unsafe_allow_html=True
                )
                label = "LOW RISK" if risk < 0.3 else ("MODERATE RISK" if risk < 0.6 else "HIGH RISK")
                st.markdown(f"<div style='text-align:center;font-size:16px;font-weight:bold;margin-top:8px'>{label}</div>", unsafe_allow_html=True)

            with col_r2:
                st.progress(float(risk))
                st.markdown(f"The model predicts a **{risk:.1%}** probability of sepsis onset for this patient at this ICU hour.")
                if risk >= 0.6:
                    st.error("High risk detected. Immediate clinical review recommended.")
                elif risk >= 0.3:
                    st.warning("Moderate risk. Monitor closely and reassess within 2 hours.")
                else:
                    st.success("Low risk. Continue standard monitoring protocol.")

            # SHAP explanation
            st.markdown("### Why This Score - SHAP Explanation")
            st.markdown("Red bars = features pushing risk UP. Blue bars = features pushing risk DOWN.")
            fig = shap_waterfall(X_input, feature_cols)
            st.pyplot(fig)
            plt.close()

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 2 — Model Performance
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown("Evaluation results on the held-out test set (310,997 real patient records never seen during training).")
    st.markdown("---")

    tab1, tab2 = st.tabs(["104 Features (Primary Model)", "Top 30 Features (Ablation Study)"])

    with tab1:
        st.subheader("All Models - 104 Features")
        metrics = load_metrics()
        st.dataframe(metrics.set_index("model").style.highlight_max(axis=0, color="#d4edda"))
        st.markdown("---")
        st.subheader("Performance Plots")
        col1, col2 = st.columns(2)
        with col1:
            roc_path = Path("outputs/figures/evaluation/roc_curve.png")
            if roc_path.exists():
                st.image(str(roc_path), caption="ROC Curve - All Models")
        with col2:
            prc_path = Path("outputs/figures/evaluation/prc_curve.png")
            if prc_path.exists():
                st.image(str(prc_path), caption="Precision-Recall Curve - All Models")
        cal_path = Path("outputs/figures/evaluation/calibration_curve.png")
        if cal_path.exists():
            st.image(str(cal_path), caption="Calibration Curve - All Models")

    with tab2:
        st.subheader("All Models - Top 30 Features")
        metrics30 = load_metrics_top30()
        st.dataframe(metrics30.set_index("model").style.highlight_max(axis=0, color="#d4edda"))
        st.markdown("---")
        abl_path = Path("outputs/figures/evaluation_top30/ablation_comparison.png")
        if abl_path.exists():
            st.image(str(abl_path), caption="XGBoost - 104 Features vs Top 30 Features")
        col1, col2 = st.columns(2)
        with col1:
            roc30 = Path("outputs/figures/evaluation_top30/roc_curve.png")
            if roc30.exists():
                st.image(str(roc30), caption="ROC Curve - Top 30 Features")
        with col2:
            prc30 = Path("outputs/figures/evaluation_top30/prc_curve.png")
            if prc30.exists():
                st.image(str(prc30), caption="PRC Curve - Top 30 Features")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 3 — Global Explainability
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Global Explainability":
    st.title("Global Explainability")
    st.markdown("SHAP analysis showing which features drive sepsis predictions across the entire patient cohort.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        bar_path = Path("outputs/figures/shap/shap_global_bar.png")
        if bar_path.exists():
            st.image(str(bar_path), caption="Global SHAP - Mean Absolute Importance")
    with col2:
        sum_path = Path("outputs/figures/shap/shap_global_summary.png")
        if sum_path.exists():
            st.image(str(sum_path), caption="Global SHAP - Summary Dot Plot")

    st.markdown("---")
    st.subheader("Top 20 Features by SHAP Importance")
    shap_df = load_shap_importance()
    st.dataframe(shap_df.head(20))

    st.markdown("---")
    st.subheader("Local SHAP - Individual Patient Explanations")
    col1, col2, col3 = st.columns(3)
    with col1:
        low = Path("outputs/figures/shap/shap_local_low_risk.png")
        if low.exists():
            st.image(str(low), caption="Low Risk Patient")
    with col2:
        med = Path("outputs/figures/shap/shap_local_medium_risk.png")
        if med.exists():
            st.image(str(med), caption="Medium Risk Patient")
    with col3:
        high = Path("outputs/figures/shap/shap_local_high_risk.png")
        if high.exists():
            st.image(str(high), caption="High Risk Patient")

    st.markdown("---")
    st.subheader("Temporal SHAP - Feature Importance Across ICU Hours")
    temp_path = Path("outputs/figures/shap/shap_temporal.png")
    if temp_path.exists():
        st.image(str(temp_path), caption="How Feature Importance Shifts Across ICU Hours 1-24")
    st.info("How SHAP feature importance shifts dynamically across ICU hours.")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 4 — Stress Test Results
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Stress Test Results":
    st.title("Infection Escalation Stress Test")
    st.markdown("Tests whether the model behaves correctly when a patient is actively deteriorating.")
    st.markdown("---")

    st.subheader("What is the Stress Test")
    st.markdown("""
    200 patients were sampled and their clinical readings were artificially worsened at three levels.
    We measure two things:
    - **ERI (Escalation Robustness Index):** Does the risk score go up when the patient gets worse?
    - **SHAP Rank Correlation:** Do the model's explanations stay consistent under stress?
    """)

    st.markdown("---")
    st.subheader("ERI Results Table")
    eri_df = load_eri()
    st.dataframe(eri_df)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        eri_path = Path("outputs/figures/stress_test/eri_final_summary.png")
        if eri_path.exists():
            st.image(str(eri_path), caption="ERI and SHAP Rank Correlation Across Perturbation Levels")
    with col2:
        risk_path = Path("outputs/figures/stress_test/risk_shift.png")
        if risk_path.exists():
            st.image(str(risk_path), caption="Risk Score Distribution - Baseline vs Stress Levels")

    shap_cmp = Path("outputs/figures/stress_test/shap_comparison.png")
    if shap_cmp.exists():
        st.image(str(shap_cmp), caption="SHAP Feature Importance - Baseline vs Severe Stress")

    st.markdown("---")
    st.subheader("Key Findings")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean ERI", "0.0038", help="Higher = model responds more strongly to deterioration")
    with col2:
        st.metric("Mean SHAP Rank Correlation", "0.9827", help="Closer to 1.0 = explanations more stable")
    with col3:
        st.metric("Risk Increased (avg)", "72%", help="% of patients where risk correctly increased under stress")