import shap
import joblib
import pandas as pd
import xgboost as xgb

# Load trained model
model = joblib.load("models/xgboost_model.pkl")

# Load processed test data
processed_data = pd.read_csv("data/heart_disease_cleaned_updated.csv")
X_test = processed_data[["age", "sex", "cp", "trestbps", "chol", "fbs", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"]]  # Selected features

# Compute SHAP values
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Plot feature importance
shap.summary_plot(shap_values, X_test)
