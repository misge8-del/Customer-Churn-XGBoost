# 📉 Enterprise Customer Churn Prediction Engine with XGBoost

**Author:** Misgina Gebregergs  
**Frameworks & Tools:** Python, XGBoost, Scikit-Learn (Pipelines, GridSearchCV), Pandas, NumPy, Matplotlib, Seaborn, Joblib  

---

## 📌 Project Architecture Overview
This repository contains an end-to-end production-grade machine learning pipeline designed to predict customer churn using telecom account profiles. 

Rather than training a detached mathematical model, this project wraps data preprocessing, feature engineering, and classification steps into an airtight `scikit-learn Pipeline` architecture. This design guarantees clean data containment, mitigates data leakage during cross-validation, and exports a unified `.joblib` binary artifact ready for immediate containerized API deployment.

### 📁 Repository Structure
```text
├── models/
│   └── churn_xgboost_model.joblib      # Self-contained optimized model pipeline
├── src/
│   └── customer_churn_XGBoost.ipynb   # Complete verified development notebook
├── requirements.txt                    # Explicit version pinned dependencies
└── README.md                           # Professional portfolio documentation
```

---

## 📊 Dataset & Feature Engineering Details
The data footprint mirrors a 7043-row schema with 22 core customer behavioral and demographic columns. 

### 🛠️ Advanced Data Wrangling & Feature Injections
- **Type Coercion & Imputation:** Identified that the critical numerical property `TotalCharges` was contaminated with blank spaces (imported as strings) due to new accounts with `tenure == 0`. Leveraged `pd.to_numeric` forcing errors to `NaN`, followed by a fallback `median` imputation strategy inside the numerical pipeline wrapper.
- **ValueRatio (Engineered Metric):** Injected a custom business metric mapping capital footprint: 
  \[\text{ValueRatio} = \frac{\text{TotalCharges}}{\text{tenure} + 1}\]
- **TenureGroup (Categorical Binning):** Discretized raw tenure ranges into intuitive account lifestyle stages (`New`, `Short`, `Medium`, `Long`) via `pd.cut`.

### 🔗 Airtight Preprocessing Pipelines
Features are split by data type and processed dynamically using a `ColumnTransformer` block to ensure completely clean model scoring matrices:
- **Numerical Track:** `SimpleImputer(strategy='median')` → `StandardScaler()`
- **Categorical Track:** `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore')`

---

## 🏋️ Model Architecture & Hyperparameter Tuning Sweep
To establish a definitive performance benchmark, a baseline **Logistic Regression** model was evaluated against an ensemble gradient boosted tree architecture using **XGBoost**.

### 🔍 3-Fold Grid Search Optimization
To prevent overfitting on tree leaf distributions, a systematic grid sweep was executed across **108 hyperparameter weight combinations** via `GridSearchCV` scoring on `roc_auc`:

```python
param_grid = {
    "model__n_estimators":,
    "model__max_depth":,
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0]
}
```

**Optimal Parameters Discovered:**
- `learning_rate`: 0.03 | `max_depth`: 3 | `n_estimators`: 200 | `subsample`: 0.8 | `colsample_bytree`: 0.8  
- **Best Cross-Validation ROC-AUC:** `0.8511`

---

## 📊 Final Evaluation & Model Performance Analysis

The final optimized model weights were tested against an untouched 20% validation split, proving stable generalization across both classifiers:

| Metric Evaluation Split | Logistic Regression Baseline | Optimized Tuned XGBoost Engine |
| :--- | :---: | :---: |
| **Accuracy** | 80.41% | **80.70%** |
| **Precision (Positive Accuracy)** | 66.67% | **68.09%** |
| **Recall (Sensitivity)** | **52.41%** | 51.34% |
| **F1-Score (Harmonic Mean)** | **58.68%** | 58.54% |
| **ROC-AUC (Discriminatory Power)** | 84.66% | **84.81%** |

### 🧠 Strategic Trade-off Insights for Hiring Managers
1. **Model Choices:** While XGBoost yielded the strongest overall discriminatory power (`ROC-AUC: 0.8481`) and Precision (`68.09%`), the baseline Logistic Regression captured a slightly higher proportion of actual churners (`Recall: 52.41%`).
2. **Imbalance Mitigation:** By executing both Classification Threshold Tuning and testing a `scale_pos_weight` ratio of `2.76` to counter the dataset imbalance, the pipeline can be adjusted downstream depending on business intent: maximizing recall to capture all retention risks, or maximizing precision to limit false alarms.

---

## 🚀 Deployment & Production Inference API Simulation
The finalized pipeline is persisted to a unified binary storage file using `joblib`. The framework includes a clean prediction function capable of dynamically vectorizing dictionary inputs into schema-compliant DataFrames:

```python
import joblib
import pandas as pd

# Load the production artifact
model = joblib.load('churn_xgboost_model.joblib')

# Score custom customer profile dictionary in real-time
new_customer_record = {
    "gender": "Male", "SeniorCitizen": 1, "Contract": "Month-to-month",
    "MonthlyCharges": 95.50, "TotalCharges": 286.50, "tenure": 3, ...
}

# Returns classification arrays and true probability metrics instantly
prob = model.predict_proba(pd.DataFrame([new_customer_record]))[0][1]
print(f"Calculated Account Churn Probability: {prob:.2%}")
```

---

## 🛠️ Step-by-Step Local Setup Guide

1. **Clone the Repository:**
   ```bash
   git clone https://github.com
   cd customer-churn-xgboost
   ```

2. **Initialize Environment and Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Explore Development Notebook:**
   Launch Jupyter or upload `src/customer_churn_pipeline.ipynb` directly to Google Colab to run the verified data pipeline.
