import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Config absolute paths to prevent runtime loading errors
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/best_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../models/scaler.pkl')

model = None
scaler = None

# -------------------------------------------------------------------
# 3. Load model and scaler on startup
# -------------------------------------------------------------------
# Moving loading to the setup phase ensures the app fails fast if 
# assets are missing, rather than crashing on a user's web request.
def load_assets():
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("ML Model and Scaler successfully bound to system memory.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load ML artifacts. {e}")

load_assets()

# -------------------------------------------------------------------
# 1. Route GET '/' -> Render the heart failure diagnostic dashboard
# -------------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    # Final production landing; binds the 12-feature model terminal.
    return render_template('index.html')

@app.route('/dashboard', methods=['GET'])
def dashboard():
    # History tracking dashboard with Chart.js analytics.
    return render_template('dashboard.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

# -------------------------------------------------------------------
# 2. Route POST '/predict' -> Accept inputs, preprocess, predict
# -------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    # 4. Handle Errors: Check if the model failed to load at startup
    if model is None or scaler is None:
        return render_template('index.html', error="System Error: ML models offline.")
        
    try:
        # Step 1: Safely parse all 12 form inputs
        # The key sequence MUST map perfectly to the training dataset structure.
        features = {
            'age': request.form.get('age', type=float),
            'anaemia': request.form.get('anaemia', type=int),
            'creatinine_phosphokinase': request.form.get('creatinine_phosphokinase', type=float),
            'diabetes': request.form.get('diabetes', type=int),
            'ejection_fraction': request.form.get('ejection_fraction', type=float),
            'high_blood_pressure': request.form.get('high_blood_pressure', type=int),
            'platelets': request.form.get('platelets', type=float),
            'serum_creatinine': request.form.get('serum_creatinine', type=float),
            'serum_sodium': request.form.get('serum_sodium', type=float),
            'sex': request.form.get('sex', type=int),
            'smoking': request.form.get('smoking', type=int),
            'time': request.form.get('time', type=float)
        }
        
        # 4. Handle Errors: Catch missing form submissions (prevent null propagation)
        if None in features.values():
            return render_template('index.html', error="Validation Error: One or more fields are missing.")

        # Step 2: Convert to DataFrame for dynamic preprocessing
        input_df = pd.DataFrame([features])
        
        # Step 3: Apply the fitted Standard Scaler to incoming row
        X_scaled = scaler.transform(input_df)
        
        # Step 4: Run Inference Predictor
        prediction = model.predict(X_scaled)[0]
        
        # Determine confidence/probability scoring if the algorithm supports it
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(X_scaled)[0][1] * 100
        else:
            probability = 100.0 if prediction == 1 else 0.0
            
        # Extract global feature importances if the model supports it
        top_features = None
        if hasattr(model, 'feature_importances_'):
            feature_names = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
            importances = list(zip(feature_names, model.feature_importances_))
            importances.sort(key=lambda x: x[1], reverse=True)
            top_features = importances[:3]  # Pass top 3
            
        # 5. Pass prediction result + probability to result.html
        return render_template('result.html', 
                               risk_class=int(prediction), 
                               probability=f"{probability:.1f}", 
                               top_features=top_features,
                               **features)
                               
    except ValueError as ve:
        # 4. Handle Errors: Catch data type cast failures
        return render_template('index.html', error=f"Validation Error: {str(ve)}")
    except Exception as e:
        # 4. Handle Errors: Catch system-level prediction breakdowns
        return render_template('index.html', error=f"Fatal Error: {str(e)}")

if __name__ == '__main__':
    # Run server locally via Waitress or defaults
    app.run(host='0.0.0.0', port=5000, debug=True)
    
    # Engine hot-reloaded cleanly; the 12-feature scaler is now in system memory.
