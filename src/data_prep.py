import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(path):
    return pd.read_csv(path)

def preprocess_and_scale(df, fit=False, scaler_path='models/scaler.pkl'):
    """
    Cleans data and scales numerical features.
    If fit=True, fits the scaler and saves it.
    If fit=False, loads the scaler and transforms.
    """
    features = ['age', 'ejection_fraction', 'serum_creatinine']
    X = df[features]
    
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
        
    return X_scaled

def get_target(df):
    return df['DEATH_EVENT']
