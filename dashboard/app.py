import streamlit as st
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sepsis Prediction Dashboard",
    page_icon="🏥",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Load CSVs (cached)
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Go to",
    ["Model Performance", "Global Explainability", "Stress Test Results"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** Explainable Sepsis Onset Prediction")
st.sidebar.markdown("**Dataset:** PhysioNet CinC 2019")
st.sidebar.markdown("**Model:** XGBoost + SHAP")
st.sidebar.markdown("**Team:** Krish | Yuvraj | Riaan")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 1 — Model Performance
# ─────────────────────────────────────────────────────────────────────────────

if page == "Model Performance":
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
#  PAGE 2 — Global Explainability
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
#  PAGE 3 — Stress Test Results
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