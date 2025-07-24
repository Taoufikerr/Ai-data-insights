import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from transformers import pipeline

st.set_page_config(page_title="AI Data Insights App", layout="wide")
st.title("AI Data Insights")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.dropna()  # Drop rows with any missing values
    st.info(f"Rows with missing values have been removed. Remaining rows: {len(df)}")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Summary")
    st.write(df.describe())

    target_column = st.selectbox("Select the target column for prediction", df.columns)

    if target_column:
        y = df[target_column]
        X = df.drop(columns=[target_column])
        X = pd.get_dummies(X)
        X = X.select_dtypes(include=["int64", "float64"])

        if X.shape[1] > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = (
                RandomForestClassifier() if y.dtype == "object" else RandomForestRegressor()
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            score = (
                accuracy_score(y_test, y_pred)
                if y.dtype == "object"
                else r2_score(y_test, y_pred)
            )

            st.subheader("Model Performance")
            st.write(f"Score: {round(score, 3)}")

            st.subheader("Actual vs Predicted")
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.6)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)

            st.subheader("AI Summary (FLAN-T5)")
           try:
    # Prepare a cleaner, readable version of the summary
    summary_data = df.describe().T  # Transpose for easy access
    main_cols = summary_data.index[:3]  # Limit to first 3 numerical columns

    formatted_lines = []
    for col in main_cols:
        stats = summary_data.loc[col]
        formatted_lines.append(
            f"{col} — Mean: {round(stats['mean'], 2)}, Min: {round(stats['min'], 2)}, Max: {round(stats['max'], 2)}"
        )

    clean_description = "\\n".join(formatted_lines)

    # Compose prompt
    prompt = (
        f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns. "
        f"Here are selected statistics:\n{clean_description}\n"
        f"The model performance score is {round(score, 3)}. "
        f"Generate a concise, professional summary."
    )

    # Token-based model selection
    token_length = len(prompt.split())
    if token_length < 400:
        model_name = "google/flan-t5-small"
    elif token_length < 800:
        model_name = "google/flan-t5-base"
    else:
        model_name = "google/flan-t5-large"

    # Generate summary
    summarizer = pipeline("summarization", model=model_name, tokenizer=model_name)
    prompt = prompt[:1500]
    summary = summarizer(prompt, max_length=120, min_length=50, do_sample=False)[0]['summary_text']

    st.subheader("AI Summary (FLAN-T5)")
    st.text_area("Summary", value=summary, height=160)

except Exception as e:
    st.warning("Could not generate AI summary. Please ensure GPU or internet is available.")
    st.text(str(e))
