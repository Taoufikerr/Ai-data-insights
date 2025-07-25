import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import base64
import importlib

# Try to import XGBoost if available
xgb_available = importlib.util.find_spec("xgboost") is not None
if xgb_available:
    from xgboost import XGBRegressor

st.set_page_config(page_title="AI Salary Predictor", layout="wide")

# Sidebar: Theme toggle
mode = st.sidebar.radio("Select Theme", ("Light", "Dark"))
if mode == "Dark":
    st.markdown("""
        <style>
            html, body, [class*="css"]  {
                background-color: #0e1117 !important;
                color: #fafafa !important;
            }
            .stSelectbox > div > div {
                background-color: #1e222a !important;
                color: #fafafa !important;
            }
            .stTextInput > div > div > input {
                background-color: #1e222a !important;
                color: #fafafa !important;
            }
            .stButton > button {
                background-color: #1e222a !important;
                color: #fafafa !important;
            }
            .stDataFrame, .stMarkdown, .stTable {
                background-color: #1e222a !important;
                color: #fafafa !important;
            }
            .css-1d391kg {
                background-color: #1e222a !important;
                color: #fafafa !important;
            }
        </style>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Sidebar Filters
    st.sidebar.subheader("🔍 Filter Data")
    for col in df.select_dtypes(include='object').columns:
        options = df[col].unique().tolist()
        selection = st.sidebar.multiselect(f"Filter by {col}", options, default=options)
        df = df[df[col].isin(selection)]

    # Summary
    st.subheader("📈 Summary Statistics")
    st.dataframe(df.describe())

    # Visualization
    st.subheader("📊 Data Visualizations")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    st.subheader("🎯 Define Target and Features")
    target = st.selectbox("Select Target Column", numeric_cols)
    features = st.multiselect("Select Feature Columns", df.columns.drop(target), default=list(df.columns.drop(target)))

    X = df[features].copy()
    y = df[target]

    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.subheader("🤖 Model Selection")
    available_models = ["Random Forest", "SVM"]
    if xgb_available:
        available_models.append("XGBoost")

    model_choice = st.radio("Choose Model", available_models)
    if model_choice == "Random Forest":
        model = RandomForestRegressor()
    elif model_choice == "SVM":
        model = SVR()
    elif model_choice == "XGBoost" and xgb_available:
        model = XGBRegressor(verbosity=0)

    if st.button("🚀 Train Model"):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = r2_score(y_test, predictions)
        st.success(f"Model trained. R² Score: {score:.4f}")

        df_preds = X_test.copy()
        df_preds['Actual'] = y_test.values
        df_preds['Predicted'] = predictions

        # AI Summary (Simple Stats Summary)
        st.subheader("🧠 AI Summary")
        try:
            summary = f"Target Mean: {y.mean():.2f}, Std: {y.std():.2f}, Min: {y.min()}, Max: {y.max()}. R² Score: {score:.4f}"
            st.info(summary)
        except:
            st.warning("Summary not available.")

        csv = df_preds.to_csv(index=False).encode()
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download Predictions as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

        st.subheader("📄 Predictions Sample")
        st.dataframe(df_preds.head())
