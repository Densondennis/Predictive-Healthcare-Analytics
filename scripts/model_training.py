# Import necessary libraries
import os
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
print("Loading dataset...")
data = pd.read_csv("data/heart_disease_cleaned_updated.csv")

# Display dataset information
print("\nDataset Overview:")
print(data.info())
print("\nFirst 5 rows of the dataset:")
print(data.head())

# Define the 12 best features for training
selected_features = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
     "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# Define feature matrix (X) and target variable (y)
X = data[selected_features]
y = data["num"]  # Target variable (Presence of heart disease)
      
# Split the dataset into training (80%) and testing (20%) sets
print("\nSplitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Print dataset shapes
print(f"Training set size: {X_train.shape}, Testing set size: {X_test.shape}")

# Initialize and train the XGBoost classifier with hyperparameters
print("\nTraining XGBoost model with optimized hyperparameters...")
xgb_model = xgb.XGBClassifier(
    use_label_encoder=False, 
    eval_metric="mlogloss", 
    n_estimators=300,        # Number of boosting rounds (trees)
    max_depth=6,             # Maximum depth of a tree
    learning_rate=0.05,      # Learning rate (step size)
    subsample=0.8,           # Fraction of data used per tree
    colsample_bytree=0.8,    # Fraction of features used per tree
    gamma=0.2,               # Minimum loss reduction for split
    reg_lambda=1,            # L2 regularization term
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Make predictions on the test set
print("\nMaking predictions...")
y_pred = xgb_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display evaluation metrics
print(f"\nOptimized Model Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)

# Save the trained XGBoost model
print("\nSaving trained model...")
os.makedirs("models", exist_ok=True)  # Ensure the "models" directory exists
joblib.dump(xgb_model, "models/xgboost_model.pkl")
print("Optimized model successfully saved at: models/xgboost_model.pkl")

print("\nProcess completed successfully!")
