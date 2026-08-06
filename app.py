import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
plots_path = os.path.join(BASE_DIR, "plots")

st.set_page_config(
    page_title="Smart Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.preprocessing import get_feature_names

@st.cache_resource
def load_pipeline():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    pipeline_path = os.path.join(BASE_DIR, "model", "best_pipeline.joblib")

    if not os.path.exists(pipeline_path):
        st.error(f"Model not found: {pipeline_path}")
        return None

    return joblib.load(pipeline_path)

pipeline = load_pipeline()

# Premium Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        text-align: center;
    }
    .approved-card { border-left: 5px solid #10B981; }
    .rejected-card { border-left: 5px solid #EF4444; }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏦 Smart Loan Approval Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered analytics pipeline for real-time credit validation</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🚀 Predict Approval", "📊 Dataset EDA", "📈 Model Evaluation"])

with tab1:
    st.write("### Applicant Credit Profile Verification")
    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### **Personal Information**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Marital Status", ["Yes", "No"])
            dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed Status", ["Yes", "No"])
            property_area = st.selectbox("Property Location Area", ["Urban", "Semiurban", "Rural"])
            
        with col2:
            st.markdown("#### **Financial & Loan Details**")
            applicant_income = st.number_input("Applicant Monthly Income ($)", min_value=0, value=5000, step=500)
            coapplicant_income = st.number_input("Co-applicant Monthly Income ($)", min_value=0, value=0, step=500)
            loan_amount = st.number_input("Requested Loan Amount (in $1000s, e.g. 120 = $120k)", min_value=0, value=120, step=10)
            loan_term = st.selectbox("Loan Term Length (in Months)", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480], index=8)
            credit_history_desc = st.selectbox("Applicant Credit Score History", 
                                               ["Good Credit History (Previous payments on time)", 
                                                "Bad Credit History (Previous default or no record)"])
            credit_history = 1.0 if "Good" in credit_history_desc else 0.0

        submit_button = st.form_submit_button(label="Predict Loan Status")

    if submit_button and pipeline is not None:
        input_df = pd.DataFrame([{
            'Gender': gender, 'Married': married, 'Dependents': dependents,
            'Education': education, 'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income, 'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount, 'Loan_Amount_Term': loan_term,
            'Credit_History': credit_history, 'Property_Area': property_area
        }])
        
        pred = pipeline.predict(input_df)[0]
        prob = pipeline.predict_proba(input_df)[0][1]
        
        out_col1, out_col2 = st.columns([1, 1])
        with out_col1:
            st.write("#### **Pipeline Verdict**")
            if pred == 1:
                st.markdown("""
                    <div class="metric-card approved-card">
                        <h2 style='color:#10B981;'>✅ Approved</h2>
                        <p>Applicant satisfies loan disbursement thresholds.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="metric-card rejected-card">
                        <h2 style='color:#EF4444;'>❌ Rejected</h2>
                        <p>Applicant fails to meet required risk assessment standards.</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.write("")
            st.metric(label="Approval Probability", value=f"{prob * 100:.2f}%", delta=f"{(prob - 0.5)*100:+.2f}% vs system threshold (50%)")
            st.progress(prob)
            
        with out_col2:
            st.write("#### **Prediction Factor Analysis (XAI)**")
            st.write("Specific contributions of input factors on the model's decision:")
            
            preprocessor = pipeline.named_steps['preprocessor']
            estimator = pipeline.named_steps['model']
            
            X_proc = preprocessor.transform(input_df)
            feature_names = get_feature_names(preprocessor, ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'])
            
            # Local coefficients contributions (w_i * x_i)
            coefs = estimator.coef_[0]
            contributions = X_proc[0] * coefs
            
            importance_df = pd.DataFrame({'Factor': feature_names, 'Contribution': contributions})
            importance_df = importance_df[importance_df['Contribution'].abs() > 0.001]
            importance_df = importance_df.sort_values(by='Contribution', key=abs, ascending=False).head(10)
            
            friendly_names = {
                'ApplicantIncome': 'Applicant Income (Scaled)',
                'CoapplicantIncome': 'Coapplicant Income (Scaled)',
                'LoanAmount': 'Loan Amount (Scaled)',
                'Loan_Amount_Term': 'Loan Term (Scaled)',
                'Total_Income': 'Total Income (Scaled)',
                'Total_Income_Log': 'Log Total Income',
                'LoanAmount_Log': 'Log Loan Amount',
                'Debt_Income_Ratio': 'Debt to Income Ratio',
                'Term_Amount_Ratio': 'Term to Amount Ratio',
                'Gender_Male': 'Gender: Male',
                'Married_Yes': 'Married: Yes',
                'Dependents_1': 'Dependents: 1',
                'Dependents_2': 'Dependents: 2',
                'Dependents_3+': 'Dependents: 3+',
                'Education_Not Graduate': 'Education: Not Graduate',
                'Self_Employed_Yes': 'Self Employed: Yes',
                'Property_Area_Semiurban': 'Property: Semiurban',
                'Property_Area_Urban': 'Property: Urban',
                'Credit_History_1.0': 'Credit History: Good'
            }
            importance_df['Factor'] = importance_df['Factor'].map(lambda x: friendly_names.get(x, x))
            
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ['#10B981' if x >= 0 else '#EF4444' for x in importance_df['Contribution']]
            sns.barplot(x='Contribution', y='Factor', data=importance_df, palette=colors, ax=ax)
            ax.axvline(0, color='gray', linestyle='--', linewidth=1)
            ax.set_title('Factors Driving Approval (Right) vs Rejection (Left)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Log-Odds Contribution')
            ax.set_ylabel('')
            plt.tight_layout()
            st.pyplot(fig)

with tab2:
    st.write("### Exploratory Data Analysis Dashboard")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    pipeline_path = os.path.join(BASE_DIR, "model", "best_pipeline.joblib")


    if not os.path.exists(plots_path):
        st.warning("EDA Visualizations are missing. Run notebook/eda.py in your terminal first.")
    else:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.image(os.path.join(plots_path, "target_distribution.png"), caption="Target Class Distribution")
            st.image(os.path.join(plots_path, "numerical_distributions.png"), caption="Financial Scalability Skewness")
        with col_t2:
            st.image(os.path.join(plots_path, "categorical_analysis.png"), caption="Categorical splits vs Approvals")
            st.image(os.path.join(plots_path, "outliers_boxplot.png"), caption="Outlier analysis distributions")
        st.markdown("---")
        st.image(os.path.join(plots_path, "correlation_heatmap.png"), caption="Correlation Heatmap Matrix", use_container_width=True)

with tab3:
    st.write("### Model Performance Comparison")
    st.markdown("#### **Evaluation Scores (Hold-out Test Set)**")
    metrics_data = {
        'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost'],
        'Accuracy': [0.8618, 0.7154, 0.8862, 0.8455, 0.8537],
        'Precision': [0.8400, 0.8378, 0.8737, 0.8511, 0.8317],
        'Recall': [0.9882, 0.7294, 0.9765, 0.9412, 0.9882],
        'F1-score': [0.9081, 0.7799, 0.9222, 0.8939, 0.9032],
        'ROC-AUC': [0.8802, 0.8073, 0.8690, 0.7845, 0.8515]
    }
    st.table(pd.DataFrame(metrics_data).set_index('Model'))
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    plots_path = os.path.join(BASE_DIR, "plots")
    if not os.path.exists(plots_path):
        st.warning("Performance Visualizations are missing. Run train.py first.")
    else:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.image(os.path.join(plots_path, "roc_curves.png"), caption="ROC Curves")
            st.image(os.path.join(plots_path, "model_comparison.png"), caption="Metric Comparatives")
        with col_m2:
            st.image(os.path.join(plots_path, "pr_curves.png"), caption="Precision-Recall Curves")
            st.image(os.path.join(plots_path, "confusion_matrices.png"), caption="Confusion Matrices Grid")