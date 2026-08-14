import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

def perform_eda():
    base_dir = r"C:\Users\akank\.gemini\antigravity\scratch\smart-loan-prediction"
    data_path = os.path.join(base_dir, "data", "train.csv")
    plots_dir = os.path.join(base_dir, "notebook", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    # 1. Target Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Loan_Status', data=df, palette='viridis')
    plt.title('Loan Approval Status Distribution (Target)', fontsize=14, fontweight='bold')
    plt.xlabel('Loan Approved (Y/N)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'target_distribution.png'), dpi=300)
    plt.close()

    # 2. Numerical Distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.histplot(df['ApplicantIncome'].dropna(), kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Applicant Income Distribution', fontsize=12, fontweight='bold')
    sns.histplot(df['CoapplicantIncome'].dropna(), kde=True, ax=axes[1], color='salmon')
    axes[1].set_title('Coapplicant Income Distribution', fontsize=12, fontweight='bold')
    sns.histplot(df['LoanAmount'].dropna(), kde=True, ax=axes[2], color='lightgreen')
    axes[2].set_title('Loan Amount Distribution', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'numerical_distributions.png'), dpi=300)
    plt.close()

    # 3. Outlier Analysis Boxplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(y='ApplicantIncome', x='Education', data=df, ax=axes[0], palette='pastel')
    axes[0].set_title('Applicant Income by Education Level (Outliers)', fontsize=12, fontweight='bold')
    sns.boxplot(y='LoanAmount', x='Self_Employed', data=df, ax=axes[1], palette='pastel')
    axes[1].set_title('Loan Amount by Self Employment Status', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'outliers_boxplot.png'), dpi=300)
    plt.close()

    # 4. Categorical vs Target Analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.countplot(x='Credit_History', hue='Loan_Status', data=df, ax=axes[0, 0], palette='coolwarm')
    axes[0, 0].set_title('Loan Status by Credit History', fontsize=12, fontweight='bold')
    sns.countplot(x='Education', hue='Loan_Status', data=df, ax=axes[0, 1], palette='coolwarm')
    axes[0, 1].set_title('Loan Status by Education Level', fontsize=12, fontweight='bold')
    sns.countplot(x='Property_Area', hue='Loan_Status', data=df, ax=axes[1, 0], palette='coolwarm')
    axes[1, 0].set_title('Loan Status by Property Area', fontsize=12, fontweight='bold')
    sns.countplot(x='Married', hue='Loan_Status', data=df, ax=axes[1, 1], palette='coolwarm')
    axes[1, 1].set_title('Loan Status by Marital Status', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'categorical_analysis.png'), dpi=300)
    plt.close()

    # 5. Correlation Heatmap
    encoded_df = df.copy()
    encoded_df['Gender'] = encoded_df['Gender'].map({'Male': 1, 'Female': 0})
    encoded_df['Married'] = encoded_df['Married'].map({'Yes': 1, 'No': 0})
    encoded_df['Education'] = encoded_df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    encoded_df['Self_Employed'] = encoded_df['Self_Employed'].map({'Yes': 1, 'No': 0})
    encoded_df['Loan_Status'] = encoded_df['Loan_Status'].map({'Y': 1, 'N': 0})
    
    numerical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 
                      'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                      'Loan_Amount_Term', 'Credit_History', 'Loan_Status']
    
    corr_matrix = encoded_df[numerical_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Heatmap including Encoded Categoricals', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    
    print("EDA completed. Visualizations generated.")

if __name__ == "__main__":
    perform_eda()