import os
import joblib
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Create a directory for saving models
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

# Load dataset
df = pd.read_csv("data/heart_disease_1000.csv")

# Define selected features
selected_features = [ 'cp','thal', 'ca','slope', 'exang', 'oldpeak',  'sex', 'thalach', 'age', 'chol']

# Define features and target
X = df[selected_features]
y = df['target']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- XGBoost Tuning with Optuna ---
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5)
    }
    model = XGBClassifier(eval_metric='logloss', random_state=42, **params)
    return cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=False)

best_params = study.best_params
best_xgb_model = XGBClassifier(eval_metric='logloss', random_state=42, **best_params)
best_xgb_model.fit(X_train, y_train)

# Save the best XGBoost model
xgb_model_path = os.path.join(save_dir, "best_xgb_model_selected.pkl")
joblib.dump(best_xgb_model, xgb_model_path)
print(f"Best XGBoost Model saved at: {xgb_model_path}")

# --- Model Training and Evaluation ---
models = {
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42, **best_params),
    "CatBoost": CatBoostClassifier(verbose=False, random_state=42)
}

results = {}

for name, model in models.items():
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_pred_prob)
    }
    print(f"Model: {name}, Evaluation:\n{classification_report(y_test, y_pred)}")

results_df = pd.DataFrame(results).T
print("\nModel Performance Comparison:")
print(results_df)

# Save the best CatBoost model
catboost_model = models["CatBoost"]
catboost_model.fit(X_train, y_train)

catboost_pkl_path = os.path.join(save_dir, "best_catboost_model_selected.pkl")
joblib.dump(catboost_model, catboost_pkl_path)

print(f"Best CatBoost Model saved at: {catboost_pkl_path}")