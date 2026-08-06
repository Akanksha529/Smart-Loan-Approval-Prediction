import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn transformer to perform feature engineering
    on the Loan Approval Prediction dataset.
    """
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # If it's a numpy array, convert to DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'])
        else:
            X = X.copy()
        
        # 1. Total Income
        X['Total_Income'] = X['ApplicantIncome'] + X['CoapplicantIncome']
        
        # 2. Log conversion for right-skewed numerical values
        X['Total_Income_Log'] = np.log1p(X['Total_Income'])
        X['LoanAmount_Log'] = np.log1p(X['LoanAmount'])
        
        # 3. Ratios 
        X['Debt_Income_Ratio'] = X['LoanAmount'] / (X['Total_Income'] + 1)
        
        # 4. Loan term per amount ratio
        X['Term_Amount_Ratio'] = X['Loan_Amount_Term'] / (X['LoanAmount'] + 1)
        
        return X
def get_preprocessing_pipeline():
    """
    Constructs and returns the complete scikit-learn preprocessing ColumnTransformer pipeline.
    """
    num_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    cat_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']
    # Numerical pipeline: Impute -> Feature Engineer -> Scale
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('engineer', LoanFeatureEngineer()),
        ('scaler', StandardScaler())
    ])
    # Categorical pipeline: Impute -> One-Hot Encode
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    # Combine numerical and categorical processors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_features),
            ('cat', cat_pipeline, cat_features)
        ],
        remainder='drop'
    )
    return preprocessor
def get_feature_names(preprocessor, num_base_cols):
    """
    Utility helper to retrieve feature names out of the fitted ColumnTransformer.
    """
    # 4 base columns + 5 engineered columns = 9 numerical features
    num_cols = num_base_cols + ['Total_Income', 'Total_Income_Log', 'LoanAmount_Log', 'Debt_Income_Ratio', 'Term_Amount_Ratio']
    
    cat_transformer = preprocessor.named_transformers_['cat']
    onehot_encoder = cat_transformer.named_steps['onehot']
    
    cat_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']
    cat_cols = onehot_encoder.get_feature_names_out(cat_features).tolist()
    
    return num_cols + cat_cols