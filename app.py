import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, r2_score

# Optional Hugging Face summarization
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

st.set_page_config(page_title="AI Data Insights", layout="wide")
st.title("📊 AI Data Insights App")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    tabs = st.tabs(["🔍 Preview", "📈 Analysis", "📊 Model", "🧠 AI Summary"])

    with tabs[0]:
        st.subheader("Dataset Overview")
        st.dataframe(df, use_container_width=True)
        st.markdown(f"**Rows:** {df.shape[0]} &nbsp;&nbsp; **Columns:** {df.shape[1]}")

    with tabs[1]:
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe(include='all').transpose(), use_container_width=True)

    with tabs[2]:
        st.subheader("Model Training and Evaluation")

        target_column = st.selectbox("Select the target column", df.columns)

        if target_column:
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Encode categorical features
            X = pd.get_dummies(X)

            # Determine task type
            is_classification = False
            if y.dtype == 'object' or y.nunique() < 20:
                is_classification = True
                le = LabelEncoder()
                y = le.fit_transform(y)

            # Drop rows with nulls
            X = X.dropna()
            y = pd.Series(y).dropna()

            # Ensure alignment
            X, y = X.loc[y.index], y.loc[X.index]

            if X.empty or y.empty:
                st.warning("Data is empty after cleaning. Please check your file.")
            else:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train model
                model = RandomForestClassifier() if is_classification else RandomForestRegressor()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                st.success("Model trained successfully.")

                # Metrics
                if is_classification:
                    score = accuracy_score(y_test, y_pred)
                    st.write(f"**Classification Accuracy:** {score:.2%}")
                else:
                    score = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    st.write(f"**R² Score:** {score:.4f}")
                    st.write(f"**Mean Squared Error:** {mse:,.2f}")

                # Visualization
                st.subheader("Performance Visualization")
                fig, ax = plt.subplots(figsize=(8, 5))

                if is_classification:
                    cm = confusion_matrix(y_test, y_pred)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    disp.plot(ax=ax)
                    ax.set_title("Confusion Matrix")
                else:
                    sns.scatterplot(x=y_test, y=y_pred, ax=ax, alpha=0.7)
                    sns.lineplot(x=y_test, y=y_test, color='red', ax=ax, linestyle='--')
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")
                    ax.set_title("Actual vs Predicted")

                st.pyplot(fig)

    with tabs[3]:
        st.subheader("AI-Generated Summary")

        try:
            if pipeline is None:
                raise ImportError("Transformers not installed.")

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

            st.text_area("AI Summary", value=summary, height=160)

        except Exception as e:
            st.warning("Could not generate summary. Ensure you have internet access and the required packages installed.")
            st.text(str(e))
