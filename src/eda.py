import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure the output directory for plots exists
os.makedirs('eda_plots', exist_ok=True)

# --- Define the DataFrame ---
# Load the real clinical dataset downloaded previously:
df = pd.read_csv('data/raw/dataset.csv') 

# Set a cohesive aesthetic style for medical plots
sns.set_theme(style="whitegrid", palette="muted")

# =====================================================================
# 1. Count Plot of Target Variable (Class Imbalance Check)
# =====================================================================
# MEDICAL INSIGHT: We need to see how many patients survived (0) vs died (1). 
# A massive imbalance could trick a model into always guessing "Survived"
# and failing to identify high-risk patients.
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='DEATH_EVENT', data=df, palette=['#2ecc71', '#e74c3c'])
plt.title('Distribution of Mortality (DEATH_EVENT)')
plt.xlabel('0 = Survived, 1 = Deceased')
plt.ylabel('Patient Count')
plt.savefig('eda_plots/01_target_distribution.png', bbox_inches='tight')
plt.close()

# =====================================================================
# 2. Distribution Plots for Key Numerical Features
# =====================================================================
# MEDICAL INSIGHT: Age, Ejection Fraction, and Serum Creatinine are clinically
# proven to be the strongest univariate predictors of heart failure outcomes.
# We plot their distributions to understand normal ranges vs dangerous outliers.
numerical_features = ['age', 'ejection_fraction', 'serum_creatinine', 'creatinine_phosphokinase']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distributions of Key Clinical Features', fontsize=16)

for idx, feature in enumerate(numerical_features):
    row, col = idx // 2, idx % 2
    sns.histplot(df[feature], kde=True, ax=axes[row, col], color='#3498db')
    axes[row, col].set_title(f'Distribution of {feature}')

plt.tight_layout()
plt.savefig('eda_plots/02_numerical_distributions.png', bbox_inches='tight')
plt.close()

# =====================================================================
# 3. Correlation Heatmap
# =====================================================================
# MEDICAL INSIGHT: The heatmap shows linear relationships between all variables.
# We are looking for high correlations (positive or negative) with 'DEATH_EVENT'.
# We also check for multi-collinearity (if two features measure the exact same thing).
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Clinical Features Correlation Heatmap')
plt.savefig('eda_plots/03_correlation_heatmap.png', bbox_inches='tight')
plt.close()

# Note: In the heatmap, you will notice 'time', 'ejection_fraction', 'age', 
# and 'serum_creatinine' have the strongest correlation coefficients with DEATH_EVENT.

# =====================================================================
# 4. Box Plots for Key Features vs DEATH_EVENT
# =====================================================================
# MEDICAL INSIGHT: Box plots clearly show clinical thresholds. 
# For example, we should see that deceased patients typically have a much LOWER
# ejection fraction (weaker heart pump) and HIGHER serum creatinine (failing kidneys).
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Age vs Death
sns.boxplot(x='DEATH_EVENT', y='age', data=df, ax=axes[0], palette=['#2ecc71', '#e74c3c'])
axes[0].set_title('Age vs Mortality')

# Ejection Fraction vs Death
sns.boxplot(x='DEATH_EVENT', y='ejection_fraction', data=df, ax=axes[1], palette=['#2ecc71', '#e74c3c'])
axes[1].set_title('Ejection Fraction (%) vs Mortality')

# Serum Creatinine vs Death
sns.boxplot(x='DEATH_EVENT', y='serum_creatinine', data=df, ax=axes[2], palette=['#2ecc71', '#e74c3c'])
axes[2].set_title('Serum Creatinine (mg/dL) vs Mortality')

plt.tight_layout()
plt.savefig('eda_plots/04_boxplots_key_features.png', bbox_inches='tight')
plt.close()

# =====================================================================
# 5. Pairplot for Top Correlated Features
# =====================================================================
# MEDICAL INSIGHT: A pairplot graphs every major feature against one another,
# colored by the patient's survival status. If we see clear clusters (e.g., 
# red dots grouping in the high creatinine + low ejection fraction corner), 
# it proves our machine learning model will easily be able to draw decision boundaries.
top_features = ['age', 'ejection_fraction', 'serum_creatinine', 'time', 'DEATH_EVENT']

# Using corner=True so we don't display redundant mirrored plots
pairplot_fig = sns.pairplot(df[top_features], hue='DEATH_EVENT', palette=['#2ecc71', '#e74c3c'], corner=True)
pairplot_fig.fig.suptitle('Pairplot of Top Predictive Features', y=1.02)
pairplot_fig.savefig('eda_plots/05_pairplot_top_features.png', bbox_inches='tight')
plt.close('all')

print("EDA generation complete! All plots saved to the 'eda_plots/' directory.")
