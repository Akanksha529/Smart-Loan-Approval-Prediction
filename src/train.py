import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from src.preprocessing import get_preprocessing_pipeline

sns.set_theme(style="whitegrid")

def train_and_evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "train.csv")
    plots_dir = os.path.join(base_dir, "plots")
    model_dir = os.path.join(base_dir, "model")
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print("Loading raw training data...")
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['Loan_ID', 'Loan_Status'])
    y = df['Loan_Status'].map({'Y': 1, 'N': 0})
    
    # Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Split size: Train={X_train.shape}, Test={X_test.shape}")
    
    preprocessor = get_preprocessing_pipeline()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Model Configurations
    models_config = {
        'Logistic Regression': {
            'estimator': LogisticRegression(random_state=42, max_iter=500),
            'params': {'C': [0.01, 0.1, 1, 10, 100]}
        },
        'Decision Tree': {
            'estimator': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [3, 5, 8, 10, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'Random Forest': {
            'estimator': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 8, 10],
                'min_samples_split': [2, 5, 10]
            }
        },
        'Gradient Boosting': {
            'estimator': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 4, 5]
            }
        },
        'XGBoost': {
            'estimator': XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 4, 5]
            }
        }
    }
    
    results = {}
    best_estimators = {}
    
    for name, c in models_config.items():
        print(f"\nTuning hyperparameters for {name}...")
        grid = GridSearchCV(c['estimator'], c['params'], cv=5, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train_proc, y_train)
        
        best_est = grid.best_estimator_
        best_estimators[name] = best_est
        
        y_pred = best_est.predict(X_test_proc)
        y_prob = best_est.predict_proba(X_test_proc)[:, 1]
        
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred),
            'F1-score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob),
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
    performance_df = pd.DataFrame({
        name: {k: v for k, v in metrics.items() if k not in ['y_pred', 'y_prob']}
        for name, metrics in results.items()
    }).T
    
    print("\n" + "="*50)
    print("                 MODEL COMPARISON TABLE")
    print("="*50)
    print(performance_df.to_string())
    
    best_model_name = performance_df['ROC-AUC'].idxmax()
    best_model = best_estimators[best_model_name]
    
    # Save components individually
    joblib.dump(best_model, os.path.join(model_dir, "best_model.joblib"))
    joblib.dump(preprocessor, os.path.join(model_dir, "preprocessor.joblib"))
    
    # Save modular production Pipeline
    entire_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', best_model)
    ])
    joblib.dump(entire_pipeline, os.path.join(model_dir, "best_pipeline.joblib"))
    
    # --- Generate Validation Charts ---
    # 1. Metric Comparison Plot
    plt.figure(figsize=(10, 6))
    comparison_melted = performance_df.reset_index().rename(columns={'index': 'Model'}).melt(
        id_vars='Model', value_vars=['Accuracy', 'F1-score', 'ROC-AUC'], 
        var_name='Metric', value_name='Score'
    )
    sns.barplot(x='Score', y='Model', hue='Metric', data=comparison_melted, palette='muted')
    plt.title('Classifier Evaluation Metrics Comparison', fontsize=14, fontweight='bold')
    plt.xlim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_comparison.png'), dpi=300)
    plt.close()
    
    # 2. ROC Curves Plot
    plt.figure(figsize=(10, 8))
    for name, metrics in results.items():
        fpr, tpr, _ = roc_curve(y_test, metrics['y_prob'])
        plt.plot(fpr, tpr, label=f"{name} (AUC={metrics['ROC-AUC']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'roc_curves.png'), dpi=300)
    plt.close()
    
    # 3. Precision-Recall Curves Plot
    plt.figure(figsize=(10, 8))
    for name, metrics in results.items():
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, metrics['y_prob'])
        plt.plot(rec_curve, prec_curve, label=f"{name}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'pr_curves.png'), dpi=300)
    plt.close()
    
    # 4. Confusion Matrices Grid
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, (name, metrics) in enumerate(results.items()):
        cm = confusion_matrix(y_test, metrics['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                    xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
        axes[i].set_title(name, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    axes[5].axis('off')
    plt.suptitle('Confusion Matrices Grid', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confusion_matrices.png'), dpi=300)
    plt.close()
    
    print("\nVisualizations saved. Training completed successfully.")

if __name__ == "__main__":
    train_and_evaluate()