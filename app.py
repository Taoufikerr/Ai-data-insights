import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import base64

st.set_page_config(page_title="AI Salary Predictor", layout="wide")

# Sidebar: Theme toggle
mode = st.sidebar.radio("Select Theme", ("Light", "Dark"))
if mode == "Dark":
    st.markdown("""
        <style>
            body {
                background-color: #0e1117;
                color: #fafafa;
            }
            .stSelectbox, .stSlider, .stButton, .stFileUploader {
                background-color: #1e222a;
            }
        </style>
    """, unsafe_allow_html=True)

st.title("💼 AI Salary Predictor")
st.markdown("Upload a dataset, choose filters, train a model, and predict salary.")

# Upload data
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Sidebar Filters
    st.sidebar.subheader("🔍 Filter Data")
    filters = {}
    for col in df.select_dtypes(include='object').columns:
        options = df[col].unique().tolist()
        selection = st.sidebar.multiselect(f"Filter by {col}", options, default=options)
        filters[col] = selection
        df = df[df[col].isin(selection)]

    # Summary Statistics
    st.subheader("📈 Summary Statistics")
    st.dataframe(df.describe())

    # Visualizations
    st.subheader("📊 Data Visualizations")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    # Select Target and Features
    st.subheader("🎯 Define Target and Features")
    target = st.selectbox("Select Target Column", numeric_cols)
    features = st.multiselect("Select Feature Columns", df.columns.drop(target), default=list(df.columns.drop(target)))

    # Encode and prepare data
    X = df[features].copy()
    y = df[target]

    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Selection
    st.subheader("🤖 Model Selection")
    model_choice = st.radio("Choose Model", ["Random Forest", "SVM", "XGBoost"])
    if model_choice == "Random Forest":
        model = RandomForestRegressor()
    elif model_choice == "SVM":
        model = SVR()
    elif model_choice == "XGBoost":
        model = XGBRegressor(verbosity=0)

    if st.button("🚀 Train Model"):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = r2_score(y_test, predictions)
        st.success(f"Model trained. R² Score: {score:.4f}")

        # Export
        df_preds = X_test.copy()
        df_preds['Actual'] = y_test.values
        df_preds['Predicted'] = predictions

        csv = df_preds.to_csv(index=False).encode()
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download Predictions as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

        st.subheader("📄 Predictions Sample")
        st.dataframe(df_preds.head())
