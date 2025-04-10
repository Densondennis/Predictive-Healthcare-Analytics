import pandas as pd

# Load the raw data
data = pd.read_csv("data/heart_disease_uci.csv")

# Display initial data information
print("Initial Data Preview:")
print(data.head())

print("\nData Types:")
print(data.dtypes)

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Count duplicate rows, print, and remove duplicates
duplicate_count = data.duplicated().sum()
print(f"\nNumber of Duplicate Rows: {duplicate_count}")
if duplicate_count > 0:
    data = data.drop_duplicates()
    print("Duplicates removed.")

# Drop unnecessary columns (Ensure 'dateset' exists before dropping)
columns_to_drop = ['id', 'dataset']
existing_columns = [col for col in columns_to_drop if col in data.columns]
data.drop(existing_columns, axis=1, inplace=True)

# Rename columns for consistency
data.rename(columns={'thalch': 'thalach'}, inplace=True)

# Convert categorical columns safely
if 'sex' in data.columns:
    data['sex'] = data['sex'].map({'Male': 1, 'Female': 0}).astype('Int64', errors='ignore')
if 'fbs' in data.columns:
    data['fbs'] = data['fbs'].map({1: 1, 0: 0}).astype('Int64', errors='ignore')
if 'exang' in data.columns:
    data['exang'] = data['exang'].map({1: 1, 0: 0}).astype('Int64', errors='ignore')
if 'cp' in data.columns:
    data['cp'] = data['cp'].map({'typical angina': 0, 'atypical angina': 1, 'non-anginal': 2, 'asymptomatic': 3}).astype('Int64', errors='ignore')
if 'slope' in data.columns:
    data['slope'] = data['slope'].map({'upsloping': 0, 'flat': 1, 'downsloping': 2}).astype('Int64', errors='ignore')
if 'thal' in data.columns:
    data['thal'] = data['thal'].map({'normal': 1, 'fixed defect': 2, 'reversible defect': 3}).astype('Int64', errors='ignore')
if 'restecg' in data.columns:
    data['restecg'] = data['restecg'].map({'normal': 0, 'lv hypertrophy': 1}).astype('Int64', errors='ignore')
if 'num' in data.columns:
    data['num'] = data['num'].map(lambda x: 0 if x == 0 else 1).astype('Int64', errors='ignore')

# Fill missing values with median or mode
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
numerical_cols = [col for col in numerical_cols if col in data.columns]  # Ensure they exist

for col in numerical_cols:
    if data[col].isna().sum() > 0:
        data[col].fillna(data[col].median(), inplace=True)

# Detect outliers using the IQR method
if numerical_cols:
    Q1 = data[numerical_cols].quantile(0.25)
    Q3 = data[numerical_cols].quantile(0.75)
    IQR = Q3 - Q1

    # Outlier condition
    outliers = ((data[numerical_cols] < (Q1 - 1.5 * IQR)) | (data[numerical_cols] > (Q3 + 1.5 * IQR))).sum()
    print("\nOutlier Counts (IQR Method):")
    print(outliers)

# Save the cleaned dataset
data.to_csv("data/heart_disease_cleaned_updated.csv", index=False)
print("\nProcessed data saved to data/heart_disease_cleaned_updated.csv")
