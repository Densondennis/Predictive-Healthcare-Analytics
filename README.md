# ❤️ Heart Disease Prediction using Machine Learning

## Overview
Heart disease remains a leading cause of death globally. Early prediction can significantly reduce risks by enabling timely interventions. This project aims to develop a predictive model using machine learning, specifically XGBoost, to accurately identify the presence of heart disease based on patient data.

## Objectives
- Predict heart disease presence using clinical and demographic data.
- Improve early diagnosis to assist healthcare providers in decision-making.
- Evaluate and optimize model performance through various metrics.

## Dataset
- Source: [Kaggle - Heart Disease UCI Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- Records: 921
- Features: 16 attributes including age, sex, cholesterol, resting blood pressure, chest pain type, and others.

## Methodology

### 1. Data Preprocessing
- Removed duplicates and unnecessary columns.
- Encoded categorical variables (e.g., sex, chest pain type, thal).
- Handled missing values using median imputation.
- Balanced the dataset using SMOTE to address class imbalance.
- Saved the cleaned dataset as `heart_disease_cleaned_updated.csv`.

### 2. Model Selection & Training
- Selected XGBoost for its high accuracy and efficiency.
- Tuned hyperparameters (learning rate, max depth, n_estimators) to improve performance.
- Used stratified K-Fold cross-validation to ensure robust training.

### 3. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## Results
The XGBoost model showed strong predictive performance across all evaluation metrics, demonstrating its potential for use in clinical decision support.

## Future Work
- Develop a mobile app for real-time prediction and integration into clinical workflows.
- Incorporate additional datasets to improve model generalization.
- Implement Explainable AI (XAI) tools like SHAP to interpret model predictions.

## Deliverables
- Source code for preprocessing, modeling, and evaluation.
- Cleaned dataset and results summary.
- Visual dashboard (in development) for healthcare insights.

## Team
Developed by Denis Kipkemboi  — Computer Science student at Maasai Mara University, with contributions from healthcare domain experts and data scientists.

## License
This project is open-source under the MIT License.
