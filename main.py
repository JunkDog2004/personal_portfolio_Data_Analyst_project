import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
import google.generativeai as genai
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


# --- GEMINI INSIGHT (Streamlit Secrets only — no .env) ---
def get_gemini_insight(df_summary, task):
    try:
        api_key = st.secrets.get("GROQ_API_KEY", None)
        if not api_key:
            return "⚠️ No API key found. Go to **Manage App → Settings → Secrets** and add:\n```\nGEMINI_API_KEY = 'your-key-here'\n```"
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"As a data expert, provide exactly 3 concise insights for a {task} task "
            f"based on this data summary:\n{df_summary}\n"
            f"Format as: 1. ... 2. ... 3. ..."
        )
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Insight generation failed: {e}"


# --- SAFE CV SPLIT CALCULATOR ---
def get_safe_cv_splits(y, task_type, max_splits=5):
    if task_type == "classification":
        min_class_count = int(pd.Series(y).value_counts().min())
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
        if task_type == "classification":
            return accuracy_score(y_test, model.predict(X_test))
        else:
            preds = model.predict(X_test)
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
        best_model.fit(X_train, y_train)  # Final fit on full training data

    return best_name, best_model, results


# --- FAST SHAP (TreeExplainer only — no slow KernelExplainer) ---
def show_shap(model, X_train, X_test):
    st.write("### 🔍 Interpretability (SHAP Feature Importance)")
    try:
        with st.spinner("Computing SHAP values..."):
            explainer = shap.TreeExplainer(model)
            X_sample = X_test.iloc[:min(100, len(X_test))]
            shap_values = explainer.shap_values(X_sample)

            # For classifiers, shap_values is a list (one per class) — take class 1
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, show=False, plot_type="bar")
            plt.gcf().set_facecolor("#1e1e26")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    except Exception:
        # Fallback: plain feature importances for non-tree models
        st.warning("⚠️ SHAP not available for this model — showing feature importances instead.")
        try:
            if hasattr(model, "feature_importances_"):
                importances = pd.Series(model.feature_importances_, index=X_train.columns)
            elif hasattr(model, "coef_"):
                coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                importances = pd.Series(np.abs(coef), index=X_train.columns)
            else:
                st.info("Feature importance not available for this model.")
                return

            importances = importances.sort_values(ascending=True).tail(20)
            fig, ax = plt.subplots(figsize=(10, 6))
            importances.plot(kind="barh", ax=ax, color="#7c7cff")
            ax.set_facecolor("#2b2b36")
            fig.patch.set_facecolor("#1e1e26")
            ax.tick_params(colors="white")
            ax.set_title("Feature Importances", color="white")
            st.pyplot(fig)
            plt.close()
        except Exception as e2:
            st.warning(f"Could not generate feature importance chart: {e2}")


# =====================================================================
# APP LAYOUT
# =====================================================================

st.title("📊 AutoML Deployment Agent")
st.subheader("Automated Pipeline: Bronze → Silver → Gold Layer Insights")
st.write("Upload your dataset to begin the automated cleaning, training, and explanation process.")

st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    # ── 1. BRONZE LAYER ──────────────────────────────────────────────
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    st.write("### 🥉 Bronze Layer: Raw Data Preview")
    st.dataframe(df.head(10))

    # ── 2. SILVER LAYER ──────────────────────────────────────────────
    st.sidebar.header("Settings")
    target_col = st.sidebar.selectbox("Select Target Column", df.columns)
    task_type  = st.sidebar.selectbox("Task Type", ["classification", "regression"])

    df_clean = df.dropna()
    X = pd.get_dummies(df_clean.drop(columns=[target_col]), drop_first=True)
    y = df_clean[target_col]

    # Encode string targets for classification
    if task_type == "classification" and y.dtype == object:
        y = y.astype("category").cat.codes

    # Warn if any class has very few samples
    if task_type == "classification":
        tiny = pd.Series(y).value_counts()
        tiny = tiny[tiny < 5]
        if not tiny.empty:
            st.warning(f"⚠️ Some classes have very few samples {tiny.to_dict()}. CV folds will be adjusted automatically.")

    st.write("### 🥈 Silver Layer: Cleaned Data Information")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Total Samples:** {df_clean.shape[0]}")
        st.write(f"**Features Count:** {df_clean.shape[1] - 1}")
        st.write(f"**Missing Values Dropped:** {df.shape[0] - df_clean.shape[0]}")
    with c2:
        if st.button("💡 Generate AI Insights"):
            with st.spinner("Generating insights..."):
                try:
                    summary = df_clean.describe().to_markdown()
                    insights = get_gemini_insight(summary, task_type)
                    st.info(insights)
                except Exception as e:
                    st.warning(f"AI Insights unavailable: {e}")

    # ── 3. GOLD LAYER ────────────────────────────────────────────────
    if st.sidebar.button("🚀 Run AutoML Pipeline"):
        st.write("---")
        st.write("### 🥇 Gold Layer: Model Training & Evaluation")

        try:
            # Stratified split for classification, plain for regression
            try:
                stratify = y if task_type == "classification" else None
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=stratify
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

            best_name, best_model, all_results = run_automl(
                X_train, X_test, y_train, y_test, task_type
            )

            if best_model is None:
                st.error("❌ All models failed. Please check your dataset for sufficient samples per class.")
                st.stop()

            st.success(f"✅ Best Model: **{best_name}**")

            # Show all model scores
            score_label = "Accuracy (CV)" if task_type == "classification" else "R² (CV)"
            with st.expander("📊 All Model Scores"):
                for name, score in all_results.items():
                    if score is not None:
                        st.write(f"- **{name}**: {score:.4f} ({score_label})")
                    else:
                        st.write(f"- **{name}**: ❌ Failed")

            # Final test set evaluation
            y_pred     = np.array(best_model.predict(X_test)).flatten()
            y_test_arr = np.array(y_test).flatten()

            if task_type == "classification":
                st.metric("Test Accuracy", f"{accuracy_score(y_test_arr, y_pred):.2%}")
            else:
                st.metric("Test RMSE", f"{np.sqrt(mean_squared_error(y_test_arr, y_pred)):.4f}")

        except Exception as e:
            st.error(f"❌ Training failed: {e}")
            st.stop()

        # ── 4. SHAP ──────────────────────────────────────────────────
        show_shap(best_model, X_train, X_test)

else:
    st.info("📂 Please upload a dataset in the sidebar to start the agent.")

# --- FOOTER ---
st.markdown("---")
st.caption("AutoML Deployment Agent | Built with Streamlit & Scikit-learn")
