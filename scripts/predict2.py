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
    "age": 55,  # Example age
    "sex": 1,  # Male
    "cp": 2,  # Non-anginal pain
    "trestbps": 140,  # Blood pressure
    "chol": 230,  # Cholesterol level
    "fbs": 0,  # Fasting blood sugar <= 120 mg/dl
    "thalach": 150,  # Max heart rate
    "exang": 1,  # Exercise-induced angina
    "oldpeak": 2.3,  # ST depression
    "slope": 1,  # Slope of ST segment
    "ca": 0,  # Number of major vessels
    "thal": 2  # Fixed defect
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
