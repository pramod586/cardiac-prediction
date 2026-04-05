# Mini-Project Report
**Title:** Mortality Prediction System for Cardiology Patients  
**Technology Stack:** Python, scikit-learn, XGBoost, Flask, HTML/CSS  
**Dataset:** Heart Failure Clinical Records (Kaggle/UCI)

---

## 1. Abstract
The prediction of mortality in cardiology patients remains a critical challenge in modern medical informatics. Cardiovascular diseases are the leading cause of global mortality, necessitating early, accurate identification of high-risk patients. This mini-project presents a robust Machine Learning-based framework designed to predict heart failure survival outcomes using 12 discrete clinical features. Leveraging the UCI Heart Failure Clinical Records dataset, the system implements an end-to-end pipeline encompassing robust data preprocessing, Synthetic Minority Over-sampling Technique (SMOTE) for class balance, and the rigorous evaluation of four distinct classifiers: Logistic Regression, Support Vector Machines (SVM), Random Forest, and XGBoost. Experimental results demonstrate that the Random Forest architecture achieves the highest classification efficacy with an approximate ROC-AUC of 89.7%. Finally, the deployed predictive model is seamlessly integrated into a clinical decision support web application built on the Flask framework with an elegant, responsive front-end, enabling physicians to attain immediate, dynamic risk assessments.

## 2. Introduction & Problem Statement
Cardiovascular diseases (CVDs) accounted for an estimated 17.9 million deaths globally in recent years, representing 32% of all global deaths. Patients facing heart failure require precise, continuous monitoring, and the ability to foresee negative outcomes can drastically alter the course of clinical intervention. Despite the abundance of Electronic Health Records (EHRs), integrating multi-factor diagnostic data into an interpretable form for physicians is exceptionally complex.

**Problem Statement:** The absence of automated, real-time computational tools targeting cardiology mortality leaves physicians reliant on manual heuristic tracking, which is prone to oversight. There is an imperative need for an intelligent system capable of ingesting high-dimensional clinical parameters—like ejection fraction and serum creatinine—to identify complex non-linear diagnostic thresholds and deliver instantaneous, actionable prognostic insights preventing premature mortality.

## 3. Objectives
1. To develop a highly accurate, data-driven mortality prediction model utilizing diverse cardiovascular metrics.
2. To mathematically actively mitigate inherent class imbalance in critical clinical datasets using SMOTE algorithm scaling to avoid gradient collapse.
3. To conduct a comprehensive comparative analysis of state-of-the-art supervised learning algorithms (SVM, XGBoost, Random Forest, Logistic Regression).
4. To architect a scalable, RESTful clinical web interface enabling real-time diagnostic querying and interactive data visualizations.

## 4. Literature Survey
1. **Chicco, D., & Jurman, G. (2020).** *Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone.* BMC Medical Informatics and Decision Making, 20(1), 1-16. This study emphasizes that while EHRs contain heavy dimensionality, basic metrics natively extracted, like serum creatinine, often hold the highest prognostic weight.
2. **Ahmad, T. et al. (2017).** *Computational approach to heart failure.* JACC: Heart Failure, 5(6), 401-411. Ahmad details the clinical transition from traditional generalized regression models to advanced gradient-boosted diagnostic trees, allowing capture of overlapping non-linear diagnostic symptoms.
3. **Kourou, K. et al. (2015).** *Machine learning applications in cancer prognosis and prediction.* Computational and Structural Biotechnology Journal. Though oncology-focused, this foundational paper establishes standard protocols for mitigating medical class imbalances and avoiding validation data leakages.
4. **Alotaibi, F. S. (2019).** *Implementation of machine learning model to predict heart failure disease.* International Journal of Advanced Computer Science and Applications, 10(6). Alotaibi directly evaluates Logistic Regression against ensemble methods on the Cleveland dataset, observing superior recall matrices in ensemble variants.
5. **Chawla, N. V. et al. (2002).** *SMOTE: synthetic minority over-sampling technique.* Journal of Artificial Intelligence Research, 16, 321-357. The primary paper defining the SMOTE algorithmic foundation necessary for training predictive classifiers appropriately on minority (mortality) patient profiles.

## 5. System Architecture & Methodology
The proposed framework executes standard KDD (Knowledge Discovery in Databases) life cycles spanning raw data ingestion to interactive user deployment.
- **Data Preprocessing:** Handled implicit null values and independently scaled native variances utilizing `StandardScaler` fitted strictly on the training partition.
- **Data Balancing:** Executed SMOTE to mathematically synthetically interpolate minority feature vectors (death events), assuring algorithmic gradients optimize correctly without overwhelming negative-class survival bias.
- **Model Training:** Utilizes Scikit-Learn libraries and the XGBoost interface to fit non-linear algorithmic mathematical boundaries evaluating data across an 80/20 stratified matrix. 
- **Application Server Layer:** Securely binds the finalized serialized `.pkl` mathematical assets to a Python WSGI Flask engine router handling dynamic HTTP POST events, parsing completely unscaled parameters natively into the required `scaler.pkl` structure.

## 6. Dataset Description
The dataset sourced is the prestigious **Heart Failure Clinical Records dataset**, accessed prominently via UCI / Kaggle data repositories. It comprises 299 isolated patient records followed longitudinally.
- **Count:** 299 instances.
- **Dimensionality:** 12 numerical predictive features, 1 integer target class feature.
- **Features Include:** `age`, `anaemia`, `creatinine_phosphokinase` (CPK), `diabetes`, `ejection_fraction`, `high_blood_pressure`, `platelets`, `serum_creatinine`, `serum_sodium`, `sex`, `smoking`, and `time` (days until follow-up).
- **Target Variable:** `DEATH_EVENT` (0 = Patient Survived/Stable, 1 = Patient Expired/Fatal).

## 7. ML Models Used & Comparison Table
Four fundamentally varied mathematical models were empirically tested. The comparison table displays testing results prioritized primarily by validation Recall (the critical medical safety metric to capture at-risk patients preventing false negatives) and sequentially by the underlying Receiver Operating Characteristic (ROC-AUC).

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| **Random Forest** | 81.67%   | 72.22%    | 68.42% | 70.27%   | **89.73%** |
| Logistic Regression| 80.00%   | 73.33%    | 57.89% | 64.71%   | 86.52%  |
| XGBoost            | 81.67%   | 75.00%    | 63.16% | 68.57%   | 85.75%  |
| Support Vector (SVM)| 73.33%   | 58.82%    | 52.63% | 55.56%   | 85.11%  |

*Table 1: Algorithm Performance Matrix extracted natively predicting against fully unseen scaled testing cohorts.* 

## 8. Results & Screenshots
*(Insert localized screenshots rapidly below prior to digital or hardcopy presentation submission)*
- **Figure 1: Exploratory Data Analysis (EDA) Heatmaps** demonstrating profound inverse mathematical correlation explicitly between Ejection Fraction and Mortality Events. `[Insert eda_plots/05_correlation_heatmap.png]`
- **Figure 2: Model Performance ROC Curves** outlining the topological boundary superiority attained safely over standard gradient chance targeting the Random Forest implementation. `[Insert eda_plots/07_roc_curves.png]`
- **Figure 3: Web Dashboard Submission Engine** illustrating the customized HTML/CSS UX parameters with 12 distinct medical attribute queries. `[Insert Web Application Homepage Front-End Screenshot]`
- **Figure 4: Diagnostic Risk Probability Output Card** displaying real-time calculated confidence probability bounds highlighting exact analytical driving features. `[Insert Web Application Result Output Risk Matrix Screenshot]`

## 9. Conclusion & Future Work
**Conclusion:** The completed mini-project system successfully bridges abstracted cardiovascular dimensional datasets directly with operational machine learning topology. By proactively correcting clinical class imbalance vectors and meticulously extracting complex multi-variable relationships operating via the Random Forest architecture algorithms, a deeply exceptional ROC-AUC bound of 89.7% was successfully attained. The native operational execution of the finalized model via a localized Flask Python ecosystem solidly underscores the feasibility of translating abstract mathematical operations directly into functional, scalable, web-based hospital diagnostic screening instruments.

**Future Work:** 
1. Integrating cloud-native REST endpoints (utilizing Docker clustering or AWS deployments) facilitating external physician API queries.
2. Expanding the internal diagnostic capabilities directly evaluating comprehensive Hyperparameter Grid-Searching to forcefully maximize deep configuration limitations on alternative arrays.
3. Establish completely live real-time EMR (Electronic Medical Record) dynamic continuous data stream integrations leveraging external HL7 FHIR standards.

## 10. References
1. Chicco, D., & Jurman, G. (2020). Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. *BMC Medical Informatics*, 20(1), 1-16.
2. Ahmad, T. et al. (2017). Computational approach to heart failure. *JACC: Heart Failure*, 5(6), 401-411.
3. Kourou, K. et al. (2015). Machine learning applications in cancer prognosis and prediction. *CSBJ*.
4. Alotaibi, F. S. (2019). Implementation of machine learning model to predict heart failure disease. *IJACSA*, 10(6).
5. UCI Machine Learning Repository (2020). Heart Failure Clinical Records Data Set.
