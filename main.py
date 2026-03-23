import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
from groq import Groq
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

warnings.filterwarnings("ignore")

# --- 🎨 UI CONFIGURATION (Black, Yellow, White) ---
st.set_page_config(page_title="AutoML Analytics Platform", layout="wide")

st.markdown("""
    <style>
    /* Main Background - Black 50% */
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    /* Sidebar - Dark Contrast */
    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #ffe300;
    }

    /* Buttons - Yellow 30% */
    .stButton>button {
        background-color: #ffe300;
        color: #000000;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ffffff;
        color: #000000;
        border: 2px solid #ffe300;
    }

    /* Data Containers - White 20% / Accents */
    .stDataFrame, div[data-testid="stExpander"] {
        background-color: #ffffff;
        color: #000000;
        border-radius: 5px;
    }

    /* Text & Headers */
    h1, h2, h3, p, span, label {
        color: #ffffff !important;
    }
    
    /* Metrics Accent */
    [data-testid="stMetricValue"] {
        color: #ffe300 !important;
    }
    
    /* Success/Info Box styling to match theme */
    .stAlert {
        background-color: #1a1a1a;
        color: #ffe300;
        border: 1px solid #ffe300;
    }
    </style>
    """, unsafe_allow_html=True)


# --- 🤖 GROQ AI INSIGHTS ---
def get_groq_insight(df_summary, task):
    try:
        api_key = st.secrets.get("GROQ_API_KEY", None)
        if not api_key:
            return "⚠️ No **GROQ_API_KEY** found in Streamlit Secrets."
        
        client = Groq(api_key=api_key)
        prompt = (
            f"As a data expert, provide 3 short, high-impact insights for a {task} task "
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


# --- ⚙️ UTILITIES ---
def get_safe_cv_splits(y, task_type, max_splits=5):
    if task_type == "classification":
        counts = pd.Series(y).value_counts()
        if len(counts) < 2: return None
        n_splits = min(max_splits, int(counts.min()))
    else:
        n_splits = min(max_splits, len(y) // 2)
    return n_splits if n_splits >= 2 else None

def evaluate_model(model, X_train, y_train, X_test, y_test, task_type, n_splits):
    scoring = "accuracy" if task_type == "classification" else "r2"
    if n_splits is not None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) if task_type == "classification" else KFold(n_splits=n_splits, shuffle=True, random_state=42)
        return cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring).mean()
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return accuracy_score(y_test, preds) if task_type == "classification" else model.score(X_test, y_test)

# --- 🔍 SHAP VISUALIZATION ---
def show_shap(model, X_train, X_test):
    st.write("### 🔍 Interpretability (SHAP)")
    try:
        with st.spinner("Calculating feature impact..."):
            explainer = shap.TreeExplainer(model)
            X_sample = X_test.iloc[:min(100, len(X_test))]
            shap_values = explainer.shap_values(X_sample)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#000000')
            ax.set_facecolor('#000000')
            
            shap.summary_plot(shap_values, X_sample, show=False, plot_type="bar")
            
            # Match UI Colors
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.tick_params(colors='white')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    except:
        st.warning("⚠️ SHAP visual failed or model not compatible. Showing raw coefficients.")

# --- 🚀 MAIN APP ---
st.title("📊 AutoML Deployment Agent")
st.subheader("Bronze → Silver → Gold Pipeline")

st.sidebar.header("📁 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    # 🥉 BRONZE
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    st.write("### 🥉 Bronze Layer: Raw Data")
    st.dataframe(df.head(10), use_container_width=True)

    # 🥈 SILVER
    st.sidebar.header("⚙️ Settings")
    target_col = st.sidebar.selectbox("Target Column (Avoid IDs!)", df.columns)
    task_type = st.sidebar.selectbox("Task Type", ["classification", "regression"])

    df_clean = df.dropna()
    X = pd.get_dummies(df_clean.drop(columns=[target_col]), drop_first=True)
    y = df_clean[target_col]

    if task_type == "classification" and y.dtype == object:
        y = y.astype("category").cat.codes

    st.write("### 🥈 Silver Layer: Cleaned Data")
    col1, col2 = st.columns(2)
    col1.metric("Total Samples", len(df_clean))
    col2.metric("Features", len(X.columns))

    if st.button("💡 Generate AI Insights"):
        summary = df_clean.describe().to_markdown()
        st.info(get_groq_insight(summary, task_type))

    # 🥇 GOLD
    if st.sidebar.button("🚀 Run AutoML Pipeline"):
        st.write("---")
        st.write("### 🥇 Gold Layer: Model Training")
        
        try:
            # Handle class imbalance for splitting
            strat = y if (task_type == "classification" and pd.Series(y).value_counts().min() > 1) else None
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
            
            if task_type == "classification":
                models = {"Random Forest": RandomForestClassifier(), "Gradient Boosting": GradientBoostingClassifier(), "Logistic": LogisticRegression()}
            else:
                models = {"Random Forest": RandomForestRegressor(), "Gradient Boosting": GradientBoostingRegressor(), "Ridge": Ridge()}

            n_splits = get_safe_cv_splits(y_train, task_type)
            best_name, best_score, best_model = None, -np.inf, None

            for name, m in models.items():
                score = evaluate_model(m, X_train, y_train, X_test, y_test, task_type, n_splits)
                if score > best_score:
                    best_score, best_name, best_model = score, name, m

            st.success(f"✅ Best Model: **{best_name}** (Score: {best_score:.4f})")
            best_model.fit(X_train, y_train)
            show_shap(best_model, X_train, X_test)
            
        except Exception as e:
            st.error(f"❌ Training Error: {e}")
            st.info("💡 Try selecting a different Target Column (like 'Borough').")

else:
    st.info("📂 Please upload a dataset in the sidebar to begin.")

st.markdown("---")
st.caption("AutoML Agent | Cyber-Theme Edition")
