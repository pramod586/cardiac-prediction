import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Ensure output directories exist
os.makedirs('eda_plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ------------------------------------------------------------------
# Data Loading and Preparation
# ------------------------------------------------------------------
print("Loading and preparing data...")
df = pd.read_csv('data/raw/dataset.csv').dropna()
X = df.drop(columns=['DEATH_EVENT'])
y = df['DEATH_EVENT']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'models/scaler.pkl')

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# ------------------------------------------------------------------
# 1. Train 4 Models
# ------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "SVM": SVC(probability=True, random_state=42) # probability=True is required for ROC-AUC scoring
}

# ------------------------------------------------------------------
# 2. Evaluate Models
# ------------------------------------------------------------------
results = []
best_model = None
best_roc_auc = 0
best_model_name = ""

plt.figure(figsize=(10, 8)) # Setup for ROC Curve

fig_cm, axes_cm = plt.subplots(2, 2, figsize=(12, 10))
axes_cm = axes_cm.flatten()

for idx, (name, model) in enumerate(models.items()):
    # Fit model on the SMOTE-resampled training data
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predict on untouched test data
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate performance metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec, # CRITICAL METRIC (We want to catch all true deaths)
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })
    
    # Subplot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm[idx], cbar=False)
    axes_cm[idx].set_title(f'{name} Confusion Matrix')
    axes_cm[idx].set_xlabel('Predicted Death')
    axes_cm[idx].set_ylabel('Actual Death')
    
    # Plot ROC Curve Line
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    
    # 3. Select BEST model based on ROC-AUC
    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        best_model = model
        best_model_name = name

# Save Confusion Matrices Chart
fig_cm.tight_layout()
fig_cm.savefig('eda_plots/06_confusion_matrices.png', bbox_inches='tight')
plt.close(fig_cm)

# Finalize ROC Curve Plot
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate (Missing a death)')
plt.ylabel('True Positive Rate (Catching a death)')
plt.title('ROC Curve Comparison Across 4 Models')
plt.legend(loc='lower right')
plt.savefig('eda_plots/07_roc_curves.png', bbox_inches='tight')
plt.close()

# ------------------------------------------------------------------
# 4. Save the best model using joblib
# ------------------------------------------------------------------
joblib.dump(best_model, 'models/best_model.pkl')

# ------------------------------------------------------------------
# 5. Print a final comparison table
# ------------------------------------------------------------------
results_df = pd.DataFrame(results).set_index("Model")

# As per requirements, we will sort primarily by Recall, then ROC-AUC
results_df = results_df.sort_values(by=['Recall', 'ROC-AUC'], ascending=False)

print("\n=======================================================")
print("FINAL MODEL COMPARISON (Sorted by Recall -> ROC-AUC)")
print("=======================================================")
print(results_df.round(4))
print("=======================================================")
print(f"\n[+] Best Model Selected: {best_model_name}")
print(f"[+] Saved successfully as 'models/best_model.pkl'")
