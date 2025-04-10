import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/xgboost_model.pkl")

# Define the selected features used in training
selected_features = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# Example input for prediction (adjust values for testing)
user_data = {
    "age": 35,  # Younger age
    "sex": 0,  # Female
    "cp": 0,  # Typical angina
    "trestbps": 120,  # Normal blood pressure
    "chol": 180,  # Healthy cholesterol level
    "fbs": 0,  # Normal fasting blood sugar
    "thalach": 170,  # Higher max heart rate
    "exang": 0,  # No exercise-induced angina
    "oldpeak": 0.1,  # Minimal ST depression
    "slope": 2,  # Upsloping ST segment
    "ca": 0,  # No major vessels blocked
    "thal": 1  # Normal thalassemia result
}


# Convert input into a DataFrame
user_df = pd.DataFrame([user_data])

# Ensure the features are in the correct order
user_df = user_df[selected_features]

# Make a direct prediction
prediction = model.predict(user_df)

# Get prediction probability instead of direct prediction
proba = model.predict_proba(user_df)[:, 1]  # Probability of being high risk

# Define a threshold (adjust based on model performance)
threshold = 0.6 # Example: increase sensitivity to detect high risk

# Apply threshold to classify risk
prediction = 1 if proba[0] >= threshold else 0

# Output the risk assessment
if prediction == 1:
    print("Prediction: The person is at high risk of heart disease.")
else:
    print("Prediction: The person is not at risk of heart disease.")
