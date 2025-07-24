import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, r2_score

# AI Summary
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

st.set_page_config(layout="wide")
st.title("AI Data Insights")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df)

    st.subheader("Summary Statistics")
    st.text(df.describe(include='all').to_string())

    target_column = st.selectbox("Select the target column (what you want to predict)", df.columns)

    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # One-hot encode features
        X = pd.get_dummies(X)

        # Classification or regression
        is_classification = False
        if y.dtype == 'object' or y.nunique() < 20:
            is_classification = True
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Check for nulls
        if X.isnull().any().any() or pd.isnull(y).any():
            st.warning("Missing values detected. Please clean or impute your data.")
        else:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestClassifier() if is_classification else RandomForestRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Show score
            st.subheader("Model Performance")
            if is_classification:
                score = accuracy_score(y_test, y_pred)
                st.write(f"Classification Accuracy: **{score:.2%}**")
            else:
                score = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                st.write(f"R² Score: **{score:.4f}**")
                st.write(f"Mean Squared Error: **{mse:,.2f}**")

            # Plot actual vs predicted
            st.subheader("Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(8, 6))
            if is_classification:
                cm = confusion_matrix(y_test, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(ax=ax, xticks_rotation=90)
                plt.title("Confusion Matrix")
            else:
                sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax)
                sns.lineplot(x=y_test, y=y_test, color='red', linestyle='--', ax=ax)
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Actual vs Predicted")
            st.pyplot(fig)

            # AI Summary
            st.subheader("AI Summary of Dataset and Results")
            try:
                if pipeline is None:
                    raise ImportError("transformers not installed")

                data_description = df.describe().to_string()
                prompt = f"Summarize the following dataset statistics and model performance:\n{data_description}\nScore: {score}"
                token_length = len(prompt.split())

                if token_length < 400:
                    model_name = "google/flan-t5-small"
                elif token_length < 800:
                    model_name = "google/flan-t5-base"
                else:
                    model_name = "google/flan-t5-large"

                summarizer = pipeline("summarization", model=model_name, tokenizer=model_name)
                prompt = prompt[:1500]
                summary = summarizer(prompt, max_length=100, min_length=30, do_sample=False)[0]['summary_text']

                st.text_area("AI Summary", value=summary, height=150)
            except Exception as e:
                st.warning("Could not generate AI summary. Please ensure GPU or internet is available.")
                st.text(str(e))
