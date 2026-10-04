# 📉 Customer Churn Prediction — Telecom

An end-to-end machine learning project for predicting customer churn in the telecommunications sector. The project follows a leakage-safe workflow from data inspection and feature engineering through model benchmarking, cross-validation, hyperparameter tuning, probability calibration, threshold optimization, explainability, final evaluation, and production-style inference.

**Author:** Misgina Gebregergs
**Project Type:** End-to-End Machine Learning Classification
**Problem Type:** Binary Classification
**Target:** `Churn` (`No` / `Yes`)
**Primary Advanced Candidate:** XGBoost
**Random State:** `42`

---

## Table of Contents

* [Overview](#overview)
* [Business Problem](#business-problem)
* [Project Objectives](#project-objectives)
* [Dataset](#dataset)
* [Machine Learning Workflow](#machine-learning-workflow)
* [Data Cleaning](#data-cleaning)
* [Exploratory Data Analysis](#exploratory-data-analysis)
* [Feature Engineering](#feature-engineering)
* [Data Splitting](#data-splitting)
* [Preprocessing Pipeline](#preprocessing-pipeline)
* [Models](#models)
* [Model Evaluation](#model-evaluation)
* [Cross-Validation](#cross-validation)
* [Hyperparameter Tuning](#hyperparameter-tuning)
* [Class Imbalance](#class-imbalance)
* [Model Selection and Calibration](#model-selection-and-calibration)
* [Threshold Optimization](#threshold-optimization)
* [Explainability](#explainability)
* [Error Analysis](#error-analysis)
* [Final Test Evaluation](#final-test-evaluation)
* [Prediction Function](#prediction-function)
* [Model Artifact](#model-artifact)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Technologies](#technologies)
* [Limitations and Future Improvements](#limitations-and-future-improvements)
* [Reproducibility](#reproducibility)
* [Author](#author)

---

## Overview

Customer churn is an important business problem for telecommunications companies because losing customers can reduce revenue and increase customer-acquisition costs.

This project develops a machine learning system that estimates the probability that a customer will churn. The predicted risk can then be used to prioritize customer-retention activities.

The notebook implements a complete machine learning pipeline:

> **Data Understanding → Cleaning → EDA → Feature Engineering → Preprocessing → Model Benchmarking → Cross-Validation → Hyperparameter Tuning → Imbalance Handling → Model Selection → Probability Calibration → Threshold Optimization → Explainability → Error Analysis → Final Test Evaluation → Production Prediction**

The workflow is designed to reduce data leakage and keep the final test set untouched until the final evaluation.

---

## Business Problem

The business goal is to identify customers who are at elevated risk of leaving the telecommunications service.

A useful churn model can help a company:

* Identify high-risk customers.
* Prioritize retention campaigns.
* Allocate customer-support resources.
* Develop targeted offers.
* Understand patterns associated with customer churn.
* Balance false positives against missed churners.

### Machine Learning Formulation

This is a **binary classification** problem:

| Class | Meaning  |
| ----- | -------- |
| `0`   | No Churn |
| `1`   | Churn    |

The target variable is:

```text
Churn
```

with the mapping:

```text
No  → 0
Yes → 1
```

---

## Project Objectives

The project aims to:

1. Inspect and understand the real dataset.
2. Identify data-quality issues.
3. Remove non-predictive identifiers.
4. Handle hidden missing values.
5. Perform useful exploratory data analysis.
6. Create domain-informed features.
7. Build a leakage-safe preprocessing pipeline.
8. Establish a Logistic Regression baseline.
9. Compare multiple machine learning algorithms.
10. Apply stratified cross-validation.
11. Optimize model hyperparameters using PR-AUC.
12. Handle class imbalance.
13. Select the strongest candidate using validation evidence.
14. Calibrate predicted probabilities.
15. Optimize the classification threshold.
16. Explain tree-based predictions with SHAP.
17. Analyze false positives and false negatives.
18. Evaluate the selected model on an untouched test set.
19. Save the complete production model artifact.
20. Provide a reusable prediction function.

---

## Dataset

The project uses the **Telco Customer Churn** dataset.

The notebook identifies:

* **7,043 rows**
* **21 columns**
* `customerID` as a unique customer identifier
* `Churn` as the target variable
* Customer demographic information
* Account information
* Contract information
* Internet and telephone services
* Payment information
* Monthly charges
* Total charges
* Customer tenure
* Additional subscribed services

### Target Variable

```text
Churn
```

The positive class is:

```text
Yes
```

and the negative class is:

```text
No
```

### Identifier

`customerID` is unique and is removed before model training because it is an identifier rather than a predictive feature.

---

## Machine Learning Workflow

The project follows these major stages:

```text
1. Dataset Inspection
2. Problem Formulation
3. Data Cleaning
4. Exploratory Data Analysis
5. Train / Validation / Test Split
6. Feature Engineering
7. Preprocessing
8. Baseline Modeling
9. Model Benchmarking
10. Stratified Cross-Validation
11. Hyperparameter Optimization
12. Class-Imbalance Handling
13. Candidate Selection
14. Probability Calibration
15. Threshold Optimization
16. Feature Ablation
17. SHAP Explainability
18. Error Analysis
19. Final Test Evaluation
20. Production Prediction
21. Model Artifact Saving
```

---

## Data Cleaning

The notebook performs professional data-quality handling before modeling.

### Missing Values

Whitespace-only strings are converted to missing values:

```python
data[col] = data[col].replace(r"^\s*$", np.nan, regex=True)
```

`TotalCharges` is converted to numeric:

```python
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)
```

Zero-tenure customers are retained rather than being silently removed. Their resulting missing `TotalCharges` values are handled by the preprocessing pipeline.

### Duplicate Checks

The notebook checks:

* Duplicate rows
* Duplicate customer IDs

---

## Exploratory Data Analysis

The project investigates customer and churn relationships using:

* Churn distribution
* Numerical feature distributions
* Churn rates by contract type
* Churn rates by internet service
* Churn rates by payment method
* Churn rates by online security
* Churn rates by technical support
* Churn rates by paperless billing
* Churn rates across tenure bands

The goal is to understand the data and identify useful patterns before modeling rather than generating visualizations without a modeling purpose.

---

## Feature Engineering

Feature engineering is implemented as a custom scikit-learn transformer so that the same transformations are applied during both training and inference.

The engineered features include:

| Feature                         | Description                                                             |
| ------------------------------- | ----------------------------------------------------------------------- |
| `avg_monthly_charge_from_total` | Average monthly charge derived from total charges and tenure            |
| `total_to_monthly_ratio`        | Ratio between total charges and monthly charges                         |
| `tenure_months_squared`         | Squared tenure to capture nonlinear tenure effects                      |
| `is_new_customer`               | Indicates customers with tenure of 1 month or less                      |
| `service_count`                 | Number of subscribed services                                           |
| `has_security_support`          | Indicates whether the customer has online security or technical support |

The engineered features are also tested through an ablation experiment to determine whether they provide measurable value.

---

## Data Splitting

The dataset is divided using a **stratified 70/15/15 split**:

| Dataset    | Purpose                                                    |
| ---------- | ---------------------------------------------------------- |
| Training   | Model fitting, cross-validation, and hyperparameter tuning |
| Validation | Model comparison, calibration, and threshold selection     |
| Test       | Final one-time generalization evaluation                   |

The final test set is not used for:

* Hyperparameter tuning
* Feature selection
* Threshold selection
* Model selection
* Ensemble selection

This helps provide a more reliable estimate of generalization performance.

---

## Preprocessing Pipeline

The project uses scikit-learn pipelines to keep preprocessing consistent and leakage-safe.

### Numerical Features

The numerical pipeline uses:

1. Median imputation
2. Standard scaling

```text
Numerical Data
     ↓
Median Imputation
     ↓
StandardScaler
```

### Categorical Features

The categorical pipeline uses:

1. Most-frequent imputation
2. One-hot encoding

```text
Categorical Data
       ↓
Most-Frequent Imputation
       ↓
One-Hot Encoding
```

The complete preprocessing process is integrated into the model pipeline.

---

## Models

Four model families are benchmarked:

### 1. Logistic Regression

Used as an interpretable baseline.

```text
Logistic Regression
```

Class weighting is used to help account for the imbalanced target.

### 2. Random Forest

A nonlinear ensemble model based on multiple decision trees.

```text
Random Forest
```

### 3. HistGradientBoosting

A gradient-boosting model designed to capture nonlinear relationships.

```text
HistGradientBoosting
```

### 4. XGBoost

An advanced gradient-boosting algorithm and major candidate in the project.

```text
XGBoost
```

The notebook does **not** assume that XGBoost is automatically the best model. The final candidate is selected using validation evidence.

---

## Model Evaluation

The project evaluates models using several complementary metrics.

### Accuracy

The proportion of all predictions that are correct.

### Precision

Among customers predicted as churners, precision measures how many actually churn.

### Recall

Among actual churners, recall measures how many the model successfully identifies.

### F1 Score

The harmonic mean of precision and recall.

### ROC-AUC

Measures ranking performance across different classification thresholds.

### PR-AUC

Measures precision-recall performance and is especially useful when the positive class is the minority class.

### Brier Score

Measures the quality of predicted probabilities. Lower values indicate better probability calibration.

### Confusion Matrix

The final evaluation also reports:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

## Cross-Validation

The project uses **5-fold Stratified K-Fold Cross-Validation**.

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

The following metrics are tracked:

* ROC-AUC mean and standard deviation
* PR-AUC mean and standard deviation
* F1 mean and standard deviation

PR-AUC is used as an important selection criterion because churn is the minority class.

---

## Hyperparameter Tuning

Hyperparameter optimization is performed using:

```python
RandomizedSearchCV
```

The optimization objective is:

```text
average_precision
```

which corresponds to PR-AUC.

The following models are tuned independently:

* Random Forest
* HistGradientBoosting
* XGBoost

The searches are performed using the training data and stratified cross-validation rather than the final test set.

---

## Class Imbalance

Because churn is the minority class, the project does not rely on accuracy alone.

### Logistic Regression

Uses:

```python
class_weight="balanced"
```

### Random Forest

Uses:

```python
class_weight="balanced"
```

### XGBoost

Uses a training-data-derived:

```python
scale_pos_weight
```

### Threshold Optimization

The decision threshold is also optimized using validation data, providing another way to control the precision-recall trade-off.

---

## Model Selection and Calibration

After hyperparameter tuning, the tuned candidates are compared on validation data:

* Tuned Random Forest
* Tuned HistGradientBoosting
* Tuned XGBoost

The candidate with the highest validation **PR-AUC** is selected.

The selected candidate is then calibrated using:

```python
CalibratedClassifierCV
```

with sigmoid calibration.

Calibration quality is assessed using:

* Brier score
* Calibration curve

This separates two important questions:

1. How well does the model rank customers by risk?
2. How trustworthy are the predicted probabilities?

---

## Threshold Optimization

The project does not automatically assume that:

```text
threshold = 0.50
```

is optimal.

Instead, validation thresholds from **0.05 to 0.95** are evaluated.

The current notebook selects the threshold that maximizes validation:

```text
F1 Score
```

The selected threshold is then frozen before the final test evaluation.

### Business Extension

If reliable business costs become available, the threshold can instead be optimized using expected cost:

```text
Expected Cost =
(False Positives × FP Cost)
+
(False Negatives × FN Cost)
```

This would allow the model's operating point to reflect the actual cost of retention actions and missed churners.

---

## Explainability

The notebook includes an optional SHAP explainability stage.

SHAP can be used to understand:

* Which features influence predictions most strongly.
* Which features increase predicted churn risk.
* Which features decrease predicted churn risk.
* Individual customer prediction explanations.

The notebook attempts to generate a SHAP summary plot for the selected tree-based candidate.

> **Important:** SHAP describes the model's association with a prediction. It does not prove that a feature causes customer churn.

---

## Error Analysis

The validation predictions are categorized into:

* Correct predictions
* False positives
* False negatives

The notebook also examines false positives and false negatives by contract group.

This helps answer:

> **Where does the model make mistakes, and what types of customers are difficult to classify?**

False negatives are especially important because they represent churners that the model failed to identify.

---

## Final Test Evaluation

The final test set is evaluated **once**, after model selection, calibration, and threshold selection.

The final evaluation reports:

* Accuracy
* Precision
* Recall
* F1
* ROC-AUC
* PR-AUC
* Brier score
* Classification report
* Confusion matrix
* ROC curve
* Precision-recall curve

The notebook explicitly avoids further tuning after the final test result is viewed.

This protects the test set from becoming part of the model-development process.

---

## Prediction Function

The notebook provides a reusable production-style function:

```python
predict_customer(customer_data)
```

It accepts either:

* A Python dictionary
* A pandas DataFrame

and returns:

| Output               | Description                    |
| -------------------- | ------------------------------ |
| `churn_probability`  | Predicted probability of churn |
| `prediction`         | Binary churn prediction        |
| `risk_level`         | Low, Medium, or High           |
| `recommended_action` | Suggested retention action     |

### Risk Levels

The current implementation uses:

```text
Probability >= 0.70 → High
Probability >= 0.40 → Medium
Otherwise            → Low
```

The recommended actions are:

```text
High   → Prioritize retention outreach
Medium → Monitor and consider targeted retention
Low    → Standard customer engagement
```

---

## Model Artifact

The notebook saves the complete model artifact using Joblib:

```text
artifacts/customer_churn_model.joblib
```

It also saves metadata:

```text
artifacts/model_metadata.json
```

The artifact contains:

* Calibrated model
* Selected classification threshold
* Target mapping
* Selected model name
* Random state

Saving the complete pipeline rather than only the classifier helps preserve the preprocessing and inference workflow required by the model.

---

## Project Structure

A typical project structure is:

```text
Customer-Churn-Prediction/
│
├── Customer_Churn_Prediction.ipynb
├── Telco-Customer-Churn.csv
├── README.md
│
├── artifacts/
│   ├── customer_churn_model.joblib
│   └── model_metadata.json
│
└── figures/
    ├── target_distribution.png
    ├── numeric_distributions.png
    ├── churn_by_tenure.png
    ├── calibration_curve.png
    ├── threshold_analysis.png
    ├── shap_summary.png
    ├── final_roc_curve.png
    └── final_pr_curve.png
```

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Customer-Churn-Prediction
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn xgboost shap joblib jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Then open:

```text
Customer_Churn_Prediction.ipynb
```

### Google Colab

The notebook can also be executed in Google Colab after uploading the dataset and updating:

```python
DATA_PATH
```

to the location of the CSV file.

---

## Usage

### Train the model

Run the notebook from top to bottom.

The notebook will:

1. Load the dataset.
2. Inspect and clean the data.
3. Perform EDA.
4. Engineer features.
5. Split the data.
6. Build preprocessing pipelines.
7. Benchmark models.
8. Run cross-validation.
9. Tune Random Forest, HistGradientBoosting, and XGBoost.
10. Select the best candidate using validation PR-AUC.
11. Calibrate probabilities.
12. Optimize the classification threshold.
13. Perform SHAP analysis.
14. Analyze prediction errors.
15. Evaluate the final model on the untouched test set.
16. Save the model artifact and metadata.

### Example Prediction

```python
example_customer = X_val.iloc[[0]].copy()

prediction = predict_customer(example_customer)

display(prediction)
```

---

## Technologies

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib

### Machine Learning

* Scikit-learn
* XGBoost

### Explainability

* SHAP

### Model Persistence

* Joblib

### Development Environment

* Jupyter Notebook
* Google Colab

---

## Limitations and Future Improvements

Although the project implements a comprehensive machine learning workflow, several improvements could be considered.

### 1. Business Cost Optimization

The current threshold objective is maximum validation F1.

A production system should ideally use real business costs for:

* False positives
* False negatives
* Retention campaigns

### 2. External Validation

Testing the model on another telecommunications dataset or future customer cohort would provide stronger evidence of generalization.

### 3. Model Monitoring

A production deployment should monitor:

* Prediction drift
* Feature drift
* Churn-rate changes
* Calibration
* Performance degradation

### 4. Fairness Analysis

If sensitive demographic information is used operationally, model performance should be evaluated across relevant customer groups.

### 5. Deployment

The saved Joblib artifact could be integrated into:

* FastAPI
* Flask
* Streamlit
* A cloud inference service

### 6. Further Model Experiments

Future experiments could investigate:

* LightGBM
* CatBoost
* Ensemble methods
* Cost-sensitive learning
* Advanced calibration methods
* More sophisticated threshold strategies

Any additional experiment should be evaluated using a leakage-safe methodology.

---

## Reproducibility

The project uses:

```python
RANDOM_STATE = 42
```

for reproducibility across:

* Train/validation/test splitting
* Cross-validation
* Randomized hyperparameter search
* Random forest
* XGBoost
* Feature sampling

The final artifact also stores the random state and selected threshold in metadata.

---

## Important Modeling Practices

This project follows several important machine learning practices:

* Do not use the final test set for tuning.
* Fit preprocessing only within the training workflow.
* Use stratification for an imbalanced classification target.
* Compare multiple models rather than assuming one algorithm is best.
* Use PR-AUC in addition to accuracy.
* Calibrate probabilities when probability quality matters.
* Optimize the classification threshold using validation data.
* Perform feature ablation before assuming engineered features help.
* Use explainability tools to understand model behavior.
* Save preprocessing together with the trained model.
* Do not retune the model after viewing final test performance.

---

## Author

**Misgina Gebregergs**

Bachelor's student in Mathematics Science
Addis Ababa University

Interested in:

* Machine Learning
* Data Science
* Artificial Intelligence
* Mathematical Modeling
* Optimization
* Python

---

## Project Status

**Status:** Completed end-to-end ML project

The notebook contains the full workflow from raw customer data through model development, evaluation, explainability, and model artifact generation.
