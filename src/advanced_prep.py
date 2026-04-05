import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

# ------------------------------------------------------------------
# Load Data
# ------------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv('data/raw/dataset.csv')

# ------------------------------------------------------------------
# 1. Handle missing values
# ------------------------------------------------------------------
# The UCI Heart Failure dataset is pre-cleaned and has no missing values,
# but in a real-world scenario, you would drop them or impute them.
# We apply dropna() as a standard safeguard.
df = df.dropna()

# ------------------------------------------------------------------
# 2. Feature Selection
# ------------------------------------------------------------------
print("\n--- 2a. Correlation Matrix ---")
# Calculate Pearson correlation to find linear relationships with the target
corr = df.corr()
print(corr['DEATH_EVENT'].sort_values(ascending=False))

print("\n--- 2b. Random Forest Feature Importance ---")
# Use a Random Forest to find non-linear feature importances
X_temp = df.drop(columns=['DEATH_EVENT'])
y_temp = df['DEATH_EVENT']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_temp, y_temp)

importances = pd.Series(rf.feature_importances_, index=X_temp.columns)
print(importances.sort_values(ascending=False))

# Clinical Insight: `time` is highly predictive but it represents the follow-up period.
# A doctor cannot know the "follow-up time" at the moment of prediction. 
# We select the top 3 actionable clinical metrics for our web application model.
selected_features = ['age', 'ejection_fraction', 'serum_creatinine']
print(f"\nSelected Actionable Features for Web App: {selected_features}")

# ------------------------------------------------------------------
# 3. Split data into X (features) and y (target)
# ------------------------------------------------------------------
X = df[selected_features]
y = df['DEATH_EVENT']

# ------------------------------------------------------------------
# 4. Train-test split (80/20, Stratified)
# ------------------------------------------------------------------
# Stratification ensures the 80/20 train-test split maintains the
# exact same ratio of 0s and 1s as the original dataset.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------------------------------------------------
# 5. Scale features using StandardScaler
# ------------------------------------------------------------------
# CRITICAL: We fit the scaler on the TRAINING set only to prevent data leakage.
# Medical data (like age vs creatinine) operates on vastly different numerical scales.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test set using the parameters learned from the training set
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------
# 6. Handle class imbalance using SMOTE
# ------------------------------------------------------------------
# Synthetic Minority Over-sampling Technique generates synthetic patient records
# for the minority class (Deaths) so the model doesn't become biased toward guessing "Survived".
# Note: SMOTE must ONLY be applied to the training set!
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"\n--- Class Imbalance Resolution (SMOTE) ---")
print(f"Original Training Target Distribution:\n{y_train.value_counts()}")
print(f"Resampled Training Target Distribution:\n{pd.Series(y_train_resampled).value_counts()}")

# ------------------------------------------------------------------
# 7. Save the scaler using joblib
# ------------------------------------------------------------------
os.makedirs('models', exist_ok=True)
scaler_path = 'models/scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"\n✅ Scaler successfully saved to: {scaler_path}")
