import shap
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# Load saved models
xgb_model = joblib.load('saved_models/best_xgb_model.pkl')  # Adjust path if needed
catboost_model = joblib.load('saved_models/best_catboost_model.pkl')

selected_features = ['thal', 'cp', 'ca', 'exang', 'oldpeak', 'slope', 'sex', 'trestbps', 'age', 'thalach', 'chol']

# Ensure XGBoost recognizes the correct feature names
xgb_model.get_booster().feature_names = selected_features

# Load dataset (use the same dataset as training)
df = pd.read_csv("data/heart_disease_1000.csv")
X = df[selected_features]

# Select a sample input for testing
sample_input = X.iloc[0:1]  # Change index to test different samples

# Ensure data types match training
sample_input = sample_input.astype(float)

# --- XGBoost SHAP ---
explainer_xgb = shap.TreeExplainer(xgb_model)  # Use TreeExplainer for tree-based models
shap_values_xgb = explainer_xgb.shap_values(sample_input)

# --- CatBoost SHAP ---
explainer_cat = shap.TreeExplainer(catboost_model)
shap_values_cat = explainer_cat.shap_values(sample_input)

# --- SHAP Force Plot ---
shap.force_plot(explainer_xgb.expected_value, shap_values_xgb, sample_input, matplotlib=True)
shap.force_plot(explainer_cat.expected_value, shap_values_cat, sample_input, matplotlib=True)

# --- SHAP Summary Plot ---
plt.figure(figsize=(12, 6))
plt.suptitle("Feature Importance Comparison (XGBoost vs. CatBoost)")

plt.subplot(1, 2, 1)
shap.summary_plot(shap_values_xgb, sample_input, show=False)
plt.title("XGBoost Feature Impact")

plt.subplot(1, 2, 2)
shap.summary_plot(shap_values_cat, sample_input, show=False)
plt.title("CatBoost Feature Impact")

plt.tight_layout()
plt.show()