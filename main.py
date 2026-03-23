import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
from groq import Groq  # Updated to Groq
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

warnings.filterwarnings("ignore")

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="AutoML Analytics Platform", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #1e1e26; color: #e0e0e0; }
    .stButton>button { background-color: #3d3d4d; color: white; border-radius: 5px; border: none; }
    .stDataFrame { background-color: #2b2b36; }
    </style>
    """, unsafe_allow_html=True)


# --- GROQ INSIGHT (Updated) ---
def get_groq_insight(df_summary, task):
    try:
        api_key = st.secrets.get("GROQ_API_KEY", None)
        if not api_key:
            return "⚠️ No Groq API key found in Streamlit Secrets."
        
        client = Groq(api_key=api_key)
        prompt = (
            f"As a data expert, provide exactly 3 concise insights for a {task} task "
            f"based on this data summary:\n{df_summary}\n"
            f"Format as: 1. ... 2. ... 3. ..."
        )
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Insight generation failed: {e}"


# --- SAFE CV SPLIT CALCULATOR ---
def get_safe_cv_splits(y, task_type, max_splits=5):
    if task_type == "classification":
        counts = pd.Series(y).value_counts()
        if len(counts) < 2: return None
        min_class_count = int(counts.min())
        n_splits = min(max_splits, min_class_count)
    else:
        n_splits = min(max_splits, len(y) // 2)
    return n_splits if n_splits >= 2 else None


# --- MODEL EVALUATOR ---
def evaluate_model(model, X_train, y_train, X_test, y_test, task_type, n_splits):
    scoring = "accuracy" if task_type == "classification" else "r2"
    if n_splits is not None:
        cv = (
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            if task_type == "classification"
            else KFold(n_splits=n_splits, shuffle=True, random_state=42)
        )
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        return scores.mean()
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        if task_type == "classification":
            return accuracy_score(y_test, preds)
        else:
            ss_res = np.sum((np.array(y_test) - preds) ** 2)
            ss_tot = np.sum((np.array(y_test) - np.mean(y_test)) ** 2)
            return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0


# --- AUTOML RUNNER ---
def run_automl(X_train, X_test, y_train, y_test, task_type):
    if task_type == "classification":
        candidates = {
            "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        }
    else:
        candidates = {
            "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Ridge Regression":  Ridge(),
        }

    n_splits = get_safe_cv_splits(y_train, task_type)
    cv_label = f"{n_splits}-fold CV" if n_splits else "holdout"

    if n_splits is None:
        st.info("ℹ️ Dataset too small for cross-validation — using holdout evaluation.")
    st.write(f"📋 Evaluation method: **{cv_label}**")

    best_name, best_score, best_model = None, -np.inf, None
    results = {}
    progress = st.progress(0)
    status = st.empty()

    for i, (name, model) in enumerate(candidates.items()):
        status.write(f"🔍 Trying **{name}**...")
        try:
            score = evaluate_model(model, X_train, y_train, X_test, y_test, task_type, n_splits)
            results[name] = score
            if score > best_score:
                best_score, best_name, best_model = score, name, model
        except Exception as e:
            results[name] = None
            st.warning(f"⚠️ {name} failed: {e}")
        progress.progress((i + 1) / len(candidates))

    status.empty()
    progress.empty()

    if best_model is not None:
        best_model.fit(X_train, y_train) 
    return best_name, best_model, results

# --- SHAP ---
def show_shap(model, X_train, X_test):
    st.write("### 🔍 Interpretability (SHAP Feature Importance)")
    try:
        with st.spinner("Computing SHAP..."):
            explainer = shap.TreeExplainer(model)
            X_sample = X_test.iloc[:min(100, len(X_test))]
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, show=False, plot_type="bar")
            plt.gcf().set_facecolor("#1e1e26")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    except:
        st.warning("⚠️ SHAP not available for this model.")

# --- APP LAYOUT ---
st.title("📊 AutoML Deployment Agent")
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    st.write("### 🥉 Bronze Layer: Raw Data Preview")
    st.dataframe(df.head(10))

    st.sidebar.header("Settings")
    # Hint: Avoid LocationID for classification!
    target_col = st.sidebar.selectbox("Select Target Column", df.columns)
    task_type  = st.sidebar.selectbox("Task Type", ["classification", "regression"])

    df_clean = df.dropna()
    X = pd.get_dummies(df_clean.drop(columns=[target_col]), drop_first=True)
    y = df_clean[target_col]

    if task_type == "classification" and y.dtype == object:
        y = y.astype("category").cat.codes

    st.write("### 🥈 Silver Layer: Cleaned Data Information")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Total Samples:** {df_clean.shape[0]}")
        st.write(f"**Features Count:** {df_clean.shape[1] - 1}")
    with c2:
        if st.button("💡 Generate AI Insights"):
            with st.spinner("Asking Groq..."):
                summary = df_clean.describe().to_markdown()
                insights = get_groq_insight(summary, task_type)
                st.info(insights)

    if st.sidebar.button("🚀 Run AutoML Pipeline"):
        st.write("---")
        st.write("### 🥇 Gold Layer")
        try:
            strat = y if task_type == "classification" else None
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
            best_name, best_model, all_results = run_automl(X_train, X_test, y_train, y_test, task_type)
            st.success(f"✅ Best Model: **{best_name}**")
            show_shap(best_model, X_train, X_test)
        except Exception as e:
            st.error(f"❌ Training failed: {e}")
else:
    st.info("📂 Please upload a dataset in the sidebar.")
