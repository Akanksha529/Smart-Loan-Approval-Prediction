# 🏦 Smart Loan Prediction

A machine learning web application that predicts whether a loan application is likely to be approved based on applicant details. The project includes data preprocessing, feature engineering, model training, hyperparameter tuning, and an interactive Streamlit dashboard for real-time predictions.

---

## 📌 Features

- Predicts loan approval using a trained machine learning model.
- Interactive Streamlit web application.
- Automated data preprocessing pipeline.
- Feature engineering and encoding.
- Hyperparameter tuning using GridSearchCV.
- Probability score for each prediction.
- Log-odds explainability visualization.
- Clean and production-ready project structure.

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib

---

## 📂 Project Structure

```
smart-loan-prediction/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── src/
│   ├── preprocessing.py
│   └── train.py
│
├── model/
│   └── best_pipeline.joblib
│
├── notebook/
│   └── eda.py
│
└── plots/
```

---

## 📊 Workflow

1. Load dataset
2. Perform data preprocessing
3. Handle missing values
4. Feature engineering
5. Train multiple machine learning models
6. Hyperparameter tuning using GridSearchCV
7. Save the best model
8. Deploy using Streamlit

---

## 📈 Model Performance

| Metric | Value |
|---------|--------|
| Accuracy | XX% |
| Precision | XX |
| Recall | XX |
| F1 Score | XX |
| ROC-AUC | XX |

*(Replace the values with your actual results.)*

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/smart-loan-prediction.git
```

Move into the project directory

```bash
cd smart-loan-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📸 Application Preview

Add screenshots of your Streamlit dashboard here.

Example:

- Home Page
- Prediction Form
- Prediction Result
- Probability Gauge
- Explainability Plot

---

## 📁 Dataset

The project uses the Loan Prediction dataset containing applicant information such as:

- Gender
- Married
- Education
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Term
- Credit History
- Property Area

Target Variable:

- Loan Status (Approved / Rejected)

---

## 🔍 Machine Learning Pipeline

- Missing Value Imputation
- Categorical Encoding
- Feature Scaling
- Model Training
- Hyperparameter Optimization
- Model Serialization using Joblib

---

## 📌 Future Improvements

- Deploy on Streamlit Cloud
- Support batch predictions
- User authentication
- Database integration

---

