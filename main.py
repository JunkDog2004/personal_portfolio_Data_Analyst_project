import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
import warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="Analytics Platform", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #1e1e26; color: #e0e0e0; }
    .stButton>button { background-color: #3d3d4d; color: white; border-radius: 5px; border: none; }
    .stDataFrame { background-color: #2b2b36; }
    </style>
    """, unsafe_allow_html=True)

def get_gemini_insight(df_summary, task):
    try:
        import google.generativeai as genai
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"As a data expert, provide 3 brief insights for a {task} task based on this data summary: {df_summary}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Insight generation currently unavailable: {e}"

def get_safe_cv_splits(y, task_type, max_splits=5):
    """
    Calculate the maximum safe number of CV folds.
    For classification: limited by the smallest class size.
    For regression: limited by total sample count.
    Minimum is 2. If even 2 splits is unsafe, returns None (use holdout only).
    """
    if task_type == "classification":
        min_class_count = int(pd.Series(y).value_counts().min())
        n_splits = min(max_splits, min_class_count)
    else:
        n_splits = min(max_splits, len(y) // 2)

    return n_splits if n_splits >= 2 else None

def evaluate_model(model, X_train, y_train, X_test, y_test, task_type, n_splits):
    """
    Evaluate a model. Uses CV if n_splits is valid, otherwise simple holdout scoring.
    """
    scoring = "accuracy" if task_type == "classification" else "r2"

    if n_splits is not None:
        if task_type == "classification":
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        return scores.mean()
    else:
        # Fallback: fit on train, score on test directly
        model.fit(X_train, y_train)
        if task_type == "classification":
            return accuracy_score(y_test, model.predict(X_test))
        else:
            preds = model.predict(X_test)
            ss_res = np.sum((np.array(y_test) - preds) ** 2)
            ss_tot = np.sum((np.array(y_test) - np.mean(y_test)) ** 2)
            return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

def run_automl(X_train, X_test, y_train, y_test, task_type):
    """
    Try multiple sklearn models, pick the best one.
    Safely handles small datasets and imbalanced classes.
    """
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

    # Calculate safe CV splits ONCE based on training data
    n_splits = get_safe_cv_splits(y_train, task_type)

    if n_splits is None:
        st.info("ℹ️ Dataset too small for cross-validation. Using holdout evaluation instead.")
        cv_method = "holdout"
    else:
        cv_method = f"{n_splits}-fold CV"

    st.write(f"📋 Evaluation method: **{cv_method}**")

    best_name = None
    best_score = -np.inf
    best_model = None
    results = {}

    progress = st.progress(0)
    status = st.empty()

    for i, (name, model) in enumerate(candidates.items()):
        status.write(f"🔍 Trying **{name}**...")
        try:
            score = evaluate_model(model, X_train, y_train, X_test, y_test, task_type, n_splits)
            results[name] = score
            if score > best_score:
                best_score = score
                best_name = name
                best_model = model
        except Exception as e:
            results[name] = None
            st.warning(f"⚠️ {name} failed: {e}")
        progress.progress((i + 1) / len(candidates))

    status.empty()
    progress.empty()

    if best_model is None:
        return None, None, results

    # Final fit on full training data (needed if CV was used)
    best_model.fit(X_train, y_train)
    return best_name, best_model, results

# --- APP HEADER ---
st.title("📊 AutoML Deployment Agent")
st.subheader("Automated Pipeline: Bronze → Silver → Gold Layer Insights")
st.write("Upload your dataset to begin the automated cleaning, training, and explanation process.")

# --- SIDEBAR ---
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    # 1. BRONZE LAYER
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### 🥉 Bronze Layer: Raw Data Preview")
    st.dataframe(df.head(10))

    # 2. SILVER LAYER
    st.sidebar.header("Settings")
    target_col = st.sidebar.selectbox("Select Target Column", df.columns)
    task_type = st.sidebar.selectbox("Task Type", ["classification", "regression"])

    df_clean = df.dropna()
    X = df_clean.drop(columns=[target_col])
    X = pd.get_dummies(X, drop_first=True)
    y = df_clean[target_col]

    # Encode string targets for classification
    if task_type == "classification" and y.dtype == object:
        y = y.astype("category").cat.codes

    # ⚠️ Warn user if any class is very small
    if task_type == "classification":
        class_counts = pd.Series(y).value_counts()
        tiny_classes = class_counts[class_counts < 5]
        if not tiny_classes.empty:
            st.warning(
                f"⚠️ Some classes have very few samples ({tiny_classes.to_dict()}). "
                "Cross-validation will be adjusted automatically."
            )

    st.write("### 🥈 Silver Layer: Cleaned Data Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Total Samples:** {df_clean.shape[0]}")
        st.write(f"**Features Count:** {df_clean.shape[1] - 1}")
        st.write(f"**Missing Values Dropped:** {df.shape[0] - df_clean.shape[0]}")
    with col2:
        if st.button("Generate AI Insights"):
            with st.spinner("Generating insights..."):
                try:
                    summary = df_clean.describe().to_markdown()
                    insights = get_gemini_insight(summary, task_type)
                    st.info(insights)
                except Exception as e:
                    st.warning(f"AI Insights unavailable: {e}")

    # 3. GOLD LAYER
    if st.sidebar.button("Run AutoML Pipeline"):
        st.write("---")
        st.write("### 🥇 Gold Layer: Model Training & Evaluation")

        try:
            # Safe split: use stratify only for classification
            split_kwargs = {"stratify": y} if task_type == "classification" else {}
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, **split_kwargs
                )
            except ValueError:
                # If stratified split fails (too few samples per class), do plain split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

            best_name, best_model, all_results = run_automl(
                X_train, X_test, y_train, y_test, task_type
            )

            if best_model is None:
                st.error("❌ All models failed to train. Please check your dataset (ensure enough samples per class).")
                st.stop()

            st.success(f"✅ Best Model: **{best_name}**")

            # Show all model scores
            score_label = "Accuracy" if task_type == "classification" else "R²"
            with st.expander("📊 All Model Scores (CV)"):
                for name, score in all_results.items():
                    if score is not None:
                        st.write(f"- **{name}**: {score:.4f} ({score_label})")
                    else:
                        st.write(f"- **{name}**: ❌ Failed")

            # Final test set evaluation
            y_pred = np.array(best_model.predict(X_test)).flatten()
            y_test_arr = np.array(y_test).flatten()

            if task_type == "classification":
                score = accuracy_score(y_test_arr, y_pred)
                st.metric("Test Accuracy", f"{score:.2%}")
            else:
                score = np.sqrt(mean_squared_error(y_test_arr, y_pred))
                st.metric("Test RMSE", f"{score:.4f}")

        except Exception as e:
            st.error(f"❌ Training failed: {e}")
            st.stop()

        # 4. SHAP EXPLAINABILITY
        st.write("### 🔍 Interpretability (SHAP Analysis)")
        try:
            with st.spinner("Computing SHAP values..."):
                try:
                    explainer = shap.TreeExplainer(best_model)
                    shap_values = explainer.shap_values(X_test)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                except Exception:
                    sample = shap.sample(X_train, min(50, len(X_train)))
                    explainer = shap.KernelExplainer(best_model.predict, sample)
                    shap_values = explainer.shap_values(X_test.iloc[:min(50, len(X_test))])

                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test, show=False, plot_type="bar")
                plt.gcf().set_facecolor('#1e1e26')
                plt.tight_layout()
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"⚠️ SHAP unavailable for this model: {e}")

else:
    st.info("📂 Please upload a dataset in the sidebar to start the agent.")

# --- FOOTER ---
st.markdown("---")
st.caption("AutoML Deployment Agent | Built with Streamlit & Scikit-learn")
