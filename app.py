import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# === Page Setup ===
st.set_page_config(page_title="Heart Diagnosis Model", layout="wide")
st.title("🫀 Diagnosis of Cardiometabolic Conditions")
st.markdown("---")

# === Load Metrics ===
@st.cache_data
def load_metrics():
    try:
        report = pd.read_csv("results/classification_report.csv", index_col=0)
        metrics = {
            "Accuracy": round(report.loc["accuracy"]["f1-score"], 4),
            "ROC AUC": round(report.loc["weighted avg"]["f1-score"], 4)
        }
        return metrics
    except Exception:
        return None

# === Load Base Model Accuracies ===
@st.cache_data
def load_model_comparison():
    try:
        return pd.read_csv("results/base_model_accuracies.csv")
    except FileNotFoundError:
        return pd.DataFrame()

# === Load SHAP Summary ===
@st.cache_data
def load_shap_summary():
    try:
        return pd.read_csv("results/shap/shap_top_features_summary.csv")
    except FileNotFoundError:
        return pd.DataFrame()

# === Save Bar Chart as PNG ===
def save_global_bar_chart(df, output_path="results/shap/global_feature_importance.png", top_n=15):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    df.head(top_n).set_index("Feature").plot(kind="barh", legend=False)
    plt.gca().invert_yaxis()
    plt.title("Global SHAP Feature Importance")
    plt.xlabel("Mean Absolute SHAP Value")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# === Main App ===
def main():
    # --- Overall Metrics ---
    metrics = load_metrics()
    if metrics:
        col1, col2 = st.columns(2)
        col1.metric("✅ Accuracy", metrics["Accuracy"])
        col2.metric("📈 ROC AUC", metrics["ROC AUC"])
    else:
        st.warning("❌ Metrics not found. Please run the model first.")

    st.markdown("---")

    # --- Base Model Accuracies ---
    st.subheader("📊 Base Model Accuracies")
    model_df = load_model_comparison()
    if not model_df.empty:
        st.dataframe(model_df.set_index("Model"), use_container_width=True)
    else:
        st.warning("No base model accuracy file found.")

    st.markdown("---")

    # --- SHAP Visualizations per Diagnosis ---
    st.subheader("🔍 SHAP Explanations by Diagnosis")
    shap_summary = load_shap_summary()
    if not shap_summary.empty:
        diagnoses = shap_summary["Class"].unique().tolist()
        selected_diagnosis = st.selectbox("Select Diagnosis Class", diagnoses)

        # SHAP Table - Full feature display with height
        st.markdown(f"#### 🔬 SHAP Feature Ranking - {selected_diagnosis}")
        class_df = shap_summary[shap_summary["Class"] == selected_diagnosis].copy()
        st.dataframe(
            class_df[["Feature", "MeanAbsSHAP"]].set_index("Feature"),
            use_container_width=True,
            height=650
        )

        # SHAP Plots
        bar_path = f"results/shap/shap_bar_{selected_diagnosis}.png"
        swarm_path = f"results/shap/shap_beeswarm_{selected_diagnosis}.png"
        cols = st.columns(2)

        if os.path.exists(bar_path):
            with cols[0]:
