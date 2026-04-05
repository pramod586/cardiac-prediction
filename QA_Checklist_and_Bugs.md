# Automated QA Checklist and Common Flask ML Bugs

## Manual QA Testing Checklist (Pre-Demo)
- [ ] **Startup Test**: Run `python app/app.py` and verify it starts without missing module or artifact errors.
- [ ] **Routing Test**: Navigate to `http://127.0.0.1:5000/`. Ensure the form and assets load properly (no 404s in network tab).
- [ ] **Happy Path Test (Stable)**: Submit a perfectly healthy patient payload. Validate the UI returns a Green "LOW RISK" card.
- [ ] **Happy Path Test (Critical)**: Submit an unhealthy patient payload (e.g. EF < 30%, Creatinine > 2.0). Validate a Red "HIGH RISK" card.
- [ ] **Validation Test (Empty)**: Submit the form completely empty to trigger the missing-field validation.
- [ ] **Validation Test (Invalid Types)**: Type a letter (e.g. "abc") into a number field (if browser allows via manipulation) to verify the data type cast failure logic.
- [ ] **Responsiveness Test**: Shrink the browser window horizontally to ensure the 12-grid layout wraps perfectly to mobile view.

---

## 5 Common Flask ML Bugs & How to Fix Them
1. **Model/Scaler Path Resolution Errors**
   - *Bug*: Flask app crashes on startup with `FileNotFoundError`. Usually caused by running `app.py` from different directories.
   - *Fix*: Use `os.path.join(os.path.dirname(__file__), 'relative/path')` to dynamically link artifacts. (Note: Already fixed in our project!).
2. **Missing Feature Mismatches**
   - *Bug*: The ML `predict()` function crashes with `ValueError` because the HTML form submitted 11 features but the model demands exactly 12.
   - *Fix*: Explicitly map form keys to a dictionary in `app.py` and implement strict `None`-checking before passing them to Pandas.
3. **Data Leakage (Scaling Error)**
   - *Bug*: Using `fit_transform()` on incoming live web traffic instead of just `transform()`. This mathematical error accidentally rescales the live patient against themselves, creating bogus predictions.
   - *Fix*: The Scaler must be pickled after fitting on the training data. The Flask app **must only use** `.transform()` globally.
4. **Data Type Cast Exceptions (String to Floats)**
   - *Bug*: HTML forms submit everything natively as string packets. The model crashes complaining about incompatible string/floats.
   - *Fix*: Use Flask's fast native type casting (`request.form.get('age', type=float)`) to safely force conversions upfront.
5. **Global Model Uninitialized Exception**
   - *Bug*: Loading massive `.pkl` files inside the `@app.route('/predict')` function. It takes 3 seconds to deserialize the mathematical tree for *every single incoming user request*, causing massive server timeouts.
   - *Fix*: Load the model **globally** upon Flask application startup so it sits identically in RAM, ready to predict instantly on incoming routes.
