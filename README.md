# AI Data Insights App

This Streamlit application allows users to upload a CSV dataset, select a target column for prediction, and receive an automated analysis. The app uses machine learning to generate predictions and a transformer model (FLAN-T5) from Hugging Face to summarize the results in natural language.

## Features

- Upload and preview any CSV file
- Automatically detect and prepare features using pandas and one-hot encoding
- Train a machine learning model (Random Forest Classifier or Regressor)
- Display model performance with R² or accuracy score
- Visualize actual vs predicted values
- Generate AI-written summaries of the dataset and results using FLAN-T5

## Setup Instructions

### Run Locally

1. Clone or download the repository.
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

### Deploy to Streamlit Cloud

1. Push the files to a public GitHub repository.
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in.
3. Click “New App”, connect your GitHub repo, and select `app.py`.
4. Streamlit will build and host your app online.

## Requirements

- Python 3.7+
- Internet connection for Hugging Face model inference

## License

This project is open-source and free to use for educational or demonstration purposes.
