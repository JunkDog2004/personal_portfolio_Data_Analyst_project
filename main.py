import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import google.generativeai as genai
import io

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="Analytics Platform", layout="wide")

# Soft Dark Theme CSS
st.markdown("""
    <style>
    .main {
        background-color: #1e1e26;
        color: #e0e0e0;
    }
    .stButton>button {
        background-color: #3d3d4d;
        color: white;
        border-radius: 5px;
        border: none;
    }
    .stDataFrame {
        background-color: #2b2b36;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Gemini (Replace with your actual Secret Key management)
# genai.configure(api_key="YOUR_GEMINI_API_KEY")

def get_gemini_insight(df_summary, task):
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"As a data expert, provide 3 brief insights for a {task} task based on this data summary: {df_summary}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return "Insight generation currently unavailable."

# --- APP HEADER ---
st.title("📊 AutoML Deployment Agent")
st.subheader("Automated Pipeline: Bronze to Gold Layer Insights")
st.write("Upload your dataset to begin the automated cleaning, training, and explanation process.")

# --- SIDEBAR: DATA INGESTION ---
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    # 1. BRONZE LAYER: Raw Data Load
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### 🥉 Bronze Layer: Raw Data Preview")
    st.dataframe(df.head(10))

    # 2. SILVER LAYER: Automated Cleaning
    st.sidebar.header("Settings")
    target_col = st.sidebar.selectbox("Select Target Column", df.columns)
    task_type = st.sidebar.selectbox("Task Type", ["classification", "regression"])

    # Simple Cleaning (Handling Nulls)
    df_clean = df.dropna()
    X = df_clean.drop(columns=[target_col])
    # Convert categorical to numeric for FLAML
    X = pd.get_dummies(X, drop_first=True)
    y = df_clean[target_col]

    st.write("### 🥈 Silver Layer: Cleaned Data Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Total Samples:** {df_clean.shape[0]}")
        st.write(f"**Features Count:** {df_clean.shape[1] - 1}")
    with col2:
        if st.button("Generate AI Insights"):
            try:
                summary = df_clean.describe().to_markdown()
                insights = get_gemini_insight(summary, task_type)
                st.info(insights)
            except Exception as e:
                st.warning(f"AI Insights unavailable: {e}")

    # 3. GOLD LAYER: AutoML Training
    if st.sidebar.button("Run AutoML Pipeline"):
        st.write("---")
        st.write("### 🥇 Gold Layer: Model Training & Evaluation")

        with st.spinner("Optimizing model with FLAML..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            automl = AutoML()
            settings = {
                "time_budget": 30,  # 30 seconds for demo
                "metric": "auto",
                "task": task_type,
                "log_file_name": "automl.log",
            }

            automl.fit(X_train=X_train, y_train=y_train, **settings)

            st.success(f"✅ Best Model Found: **{automl.best_estimator}**")

            # FIX: Convert predictions to clean numpy arrays
            y_pred = np.array(automl.predict(X_test)).flatten()
            y_test_arr = np.array(y_test).flatten()

            if task_type == "classification":
                score = accuracy_score(y_test_arr, y_pred)
                st.metric("Model Accuracy", f"{score:.2%}")
            else:
                score = np.sqrt(mean_squared_error(y_test_arr, y_pred))
                st.metric("RMSE", f"{score:.4f}")

        # 4. EXPLAINABILITY (SHAP)
        st.write("### 🔍 Interpretability (SHAP Analysis)")
        try:
            explainer = shap.Explainer(automl.model.estimator, X_train)
            shap_values = explainer(X_test)

            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_test, show=False)
            plt.gcf().set_facecolor('#1e1e26')  # Match theme
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"⚠️ SHAP explainability unavailable for this model type: {e}")

else:
    st.info("📂 Please upload a dataset in the sidebar to start the agent.")

# --- FOOTER ---
st.markdown("---")
st.caption("AutoML Deployment Agent | Built with Streamlit & FLAML")
