import pandas as pd
import numpy as np
import os
import joblib
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

def train_advanced_model():
    data_path = "data/raw/dataset.csv"
    if not os.path.exists(data_path):
        print("Error: Dataset missing. Please download it first.")
        sys.exit(1)
        
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    
    # 1. Handle missing values
    df = df.dropna()
    
    # 2. Feature selection
    selected_features = ['age', 'ejection_fraction', 'serum_creatinine']
    X = df[selected_features]
    y = df['DEATH_EVENT']
    
    # 3. Train-test split (80/20, Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 4. Scale features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved to models/scaler.pkl")
    
    # 5. Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # 6. Train the Model
    print("Training Random Forest on SMOTE-resampled data...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    
    # 7. Evaluate the Model (on un-SMOTEd test data)
    preds = model.predict(X_test_scaled)
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))
    
    # Save the model
    joblib.dump(model, 'models/model.pkl')
    print("Model saved to models/model.pkl")

if __name__ == "__main__":
    train_advanced_model()
