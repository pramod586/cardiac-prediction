import joblib
import pandas as pd
import os
import sys

# Add src to path so relative imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_prep import preprocess_and_scale

def predict_mortality(age, ef, sc):
    """
    Uses the saved scaler and model to make a prediction.
    """
    df = pd.DataFrame({
        'age': [age],
        'ejection_fraction': [ef],
        'serum_creatinine': [sc]
    })
    
    X_scaled = preprocess_and_scale(df, fit=False, scaler_path='models/scaler.pkl')
    
    model = joblib.load('models/best_model.pkl')
    prediction = model.predict(X_scaled)[0]
    
    probability = model.predict_proba(X_scaled)[0][1]
    
    return {
        'risk_class': int(prediction), # 1 for high risk, 0 for low risk
        'probability': float(probability)
    }
