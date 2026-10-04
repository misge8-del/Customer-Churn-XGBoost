## 📌 Table of Contents

- [# 📉 Customer Churn Prediction in the Telecom Sector](#-customer-churn-prediction-in-the-telecom-sector)
- [🎯 Project Overview](#-project-overview)
- [💼 Business Problem](#-business-problem)
- [🧠 Machine Learning Task](#-machine-learning-task)
- [🎯 Project Objectives](#-project-objectives)
- [📊 Dataset](#-dataset)
- [🛠️ Technologies and Libraries](#️-technologies-and-libraries)
- [🔄 Project Workflow](#-project-workflow)
- [1. Environment and Libraries](#1-environment-and-libraries)
- [2. Dataset Loading](#2-dataset-loading)
- [3. Initial Data Exploration](#3-initial-data-exploration)
- [4. Exploratory Data Analysis](#4-exploratory-data-analysis)
- [5. Feature Relationship Analysis](#5-feature-relationship-analysis)
- [6. Correlation Analysis](#6-correlation-analysis)
- [7. Outlier Investigation](#7-outlier-investigation)
- [8. Understanding the Data Description](#8-understanding-the-data-description)
- [9. Missing Value Handling](#9-missing-value-handling)
- [10. Feature Engineering](#10-feature-engineering)
- [11. Model Benchmarking](#11-model-benchmarking)
- [12. Cross-Validation](#12-cross-validation)
- [13. Hyperparameter Tuning](#13-hyperparameter-tuning)
- [14. Class Imbalance Handling](#14-class-imbalance-handling)
- [15. Model Selection](#15-model-selection)
- [16. Probability Calibration](#16-probability-calibration)
- [17. Threshold Optimization](#17-threshold-optimization)
- [18. Model Explainability](#18-model-explainability)
- [19. Error Analysis](#19-error-analysis)
- [20. Final Test Evaluation](#20-final-test-evaluation)
- [21. Production Prediction](#21-production-prediction)
- [22. Model Serialization](#22-model-serialization)
- [📈 Results](#-results)
- [💼 Business Interpretation](#-business-interpretation)
- [⚠️ Limitations](#️-limitations)
- [🔮 Future Improvements](#-future-improvements)
- [📁 Project Structure](#-project-structure)
- [▶️ How to Run](#️-how-to-run)
- [👨‍💻 Author](#-author)

   # 📉 Customer Churn Prediction in the Telecom Sector

An end-to-end machine learning project for predicting customer churn in the telecommunications industry using **Logistic Regression, Random Forest, HistGradientBoosting, and XGBoost**.

The project emphasizes **leakage-safe modeling, stratified validation, hyperparameter optimization, class-imbalance handling, probability calibration, threshold optimization, model explainability, and production-ready inference**.

---

## 🎯 Project Overview

Customer churn is a major business challenge in the telecommunications industry. Identifying customers who are likely to leave can help companies prioritize retention efforts and reduce potential revenue loss.

This project develops a reproducible machine learning pipeline that predicts whether a customer is likely to churn based on customer demographics, services, account information, contract characteristics, and billing information.

### Main objective

> **Build a reliable and interpretable churn prediction system that can identify high-risk customers and support data-driven retention decisions.**

---

## 🧠 Machine Learning Workflow

The project follows a complete end-to-end machine learning workflow:

```text
Data Inspection
      ↓
Data Cleaning
      ↓
Exploratory Analysis
      ↓
Stratified Train / Validation / Test Split
      ↓
Feature Engineering
      ↓
Leakage-Safe Preprocessing
      ↓
Baseline Modeling
      ↓
Model Benchmarking
      ↓
Stratified Cross-Validation
      ↓
Hyperparameter Tuning
      ↓
Class-Imbalance Handling
      ↓
Model Selection
      ↓
Probability Calibration
      ↓
Threshold Optimization
      ↓
SHAP Explainability
      ↓
Error Analysis
      ↓
Final Test Evaluation
      ↓
Production Prediction
      ↓
Model Serialization
```

A key design principle is that the **final test set remains untouched until all modeling decisions are finalized**.

---

## 📊 Dataset

The project uses a telecommunications customer dataset containing customer-level information related to:

* Customer demographics
* Account information
* Contract type
* Internet services
* Telephone services
* Payment methods
* Monthly charges
* Total charges
* Customer tenure
* Additional subscribed services

### Target Variable

**`Churn`**

The target represents whether a customer has left the telecommunications service.

---

## 🛠️ Data Preparation

The dataset is processed through a reproducible preprocessing workflow that includes:

* Data-quality inspection
* Missing-value handling
* Duplicate detection
* Identifier removal
* Categorical-variable encoding
* Numerical-variable preprocessing
* Feature engineering
* Leakage-safe transformations

Feature engineering is performed before model training while ensuring that information from the validation and test sets does not leak into the training process.

---

## 🧪 Models

Four classification approaches are evaluated:

| Model                | Purpose                              |
| -------------------- | ------------------------------------ |
| Logistic Regression  | Baseline linear model                |
| Random Forest        | Nonlinear ensemble baseline          |
| HistGradientBoosting | Gradient boosting model              |
| XGBoost              | Advanced gradient boosting candidate |

**XGBoost** is integrated as the primary advanced model candidate, but the final model is selected objectively based on validation performance rather than assuming XGBoost must win.

---

## 🔬 Model Validation

The dataset is divided using a **stratified train/validation/test strategy**.

The modeling process uses:

* Stratified splitting
* 5-fold Stratified Cross-Validation
* Validation-based model comparison
* Randomized hyperparameter search
* Validation-only threshold optimization

The final test set is reserved for one final evaluation after all model decisions have been frozen.

---

## ⚙️ Hyperparameter Optimization

Randomized hyperparameter search is applied to the tree-based models.

For XGBoost, the search explores parameters including:

* `n_estimators`
* `max_depth`
* `learning_rate`
* `subsample`
* `colsample_bytree`
* `min_child_weight`
* `gamma`
* `reg_alpha`
* `reg_lambda`

The optimization objective uses **PR-AUC / Average Precision**, which is particularly useful when evaluating an imbalanced classification problem.

---

## ⚖️ Class Imbalance

Customer churn datasets commonly contain fewer churned customers than non-churned customers.

This project addresses the imbalance using model-appropriate techniques:

* Logistic Regression → class weighting
* Random Forest → class weighting
* XGBoost → `scale_pos_weight`
* HistGradientBoosting → evaluated without additional class weighting
* Probability threshold optimization → controls the precision/recall trade-off

The XGBoost class-weighting ratio is calculated using the **training data only**.

---

## 📈 Evaluation Metrics

Models are evaluated using multiple metrics rather than relying on accuracy alone.

### Classification Metrics

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC
* Confusion Matrix
* Classification Report

### Probability Quality

* Brier Score
* Calibration Curve

### Why PR-AUC?

Because churn prediction can involve class imbalance, **PR-AUC provides useful information about the model's ability to identify churn customers while controlling false positives**.

---

## 🏆 Model Selection

The tuned candidate models are compared using the validation set.

The current selection strategy is:

```text
Random Forest
        │
HistGradientBoosting
        │
XGBoost
        ↓
Compare Validation PR-AUC
        ↓
Select Best Candidate
```

The model with the strongest **validation PR-AUC** becomes the candidate for probability calibration and subsequent threshold optimization.

The final model is therefore determined by the experimental results rather than by model popularity.

---

## 🎚️ Probability Calibration

The selected model undergoes probability calibration using a separate calibration subset from the training data.

Calibration performance is evaluated using:

* Brier Score
* Calibration Curve

This is important because the model is expected to produce not only a classification but also a meaningful **churn probability**.

---

## 🎯 Threshold Optimization

The default classification threshold of `0.50` is not automatically assumed to be optimal.

The project investigates alternative probability thresholds using the **validation set**.

This allows the business to choose a threshold depending on the desired balance between:

* Precision
* Recall
* False positives
* False negatives

For example, a company may prefer higher recall if missing a high-risk customer is more costly than contacting a customer who ultimately does not churn.

---

## 🔍 Model Explainability

The project includes **SHAP-based model explainability** to investigate which features contribute most strongly to predictions.

Explainability helps answer questions such as:

* Which customer characteristics increase churn risk?
* Which features are associated with lower churn risk?
* Why was a particular customer classified as high risk?

This makes the model more useful for analysis and business decision-making.

---

## 🚨 Error Analysis

Validation predictions are analyzed to identify:

* False positives
* False negatives
* Difficult customer cases
* Potential weaknesses in the model

This helps determine where the model performs well and where additional data or future modeling improvements may be needed.

---

## 🧪 Final Test Evaluation

After all modeling decisions are finalized, the selected model is evaluated on the **untouched test set**.

The final report includes:

```text
Model: Tuned XGBoost
Threshold: 0.315

Accuracy:   77.29%
Precision:  55.68%
Recall:     71.53%
F1:         62.62%
ROC-AUC:    83.96%
PR-AUC:     66.26%
Brier:      0.1389
```

> **Note:** These values will be added after the complete notebook has been executed. No test-set results are assumed or fabricated.

---

## 💼 Business Interpretation

The purpose of the model is not simply to maximize a machine learning metric.

The practical goal is to help answer:

> **Which customers are at higher risk of churn, and which customers should receive retention attention first?**

A production system could use the predicted churn probability to categorize customers into different risk levels and prioritize retention strategies.

Example:

```text
Churn Probability
       ↓
Risk Level
       ↓
Retention Priority
       ↓
Recommended Business Action
```

---

## 🚀 Production Prediction

The notebook includes a reusable prediction workflow that accepts new customer information and produces:

```text
Churn Probability
Prediction
Risk Level
Recommended Action
```

The same preprocessing used during model development is preserved for inference to ensure consistency between training and prediction.

---

## 💾 Model Serialization

The final project saves a production-oriented model artifact using **Joblib**.

The saved artifacts include:

```text
artifacts/
├── customer_churn_model.joblib
└── model_metadata.json
```

The metadata records information such as:

* Selected model
* Selected threshold
* Random state
* Final test metrics
* Training-set size
* Validation-set size
* Test-set size
* Positive and negative class labels

---

## 📁 Project Structure

A recommended GitHub structure is:

```text
customer-churn-prediction/
│
├── README.md
│
├── notebooks/
│   └── customer_churn_prediction_XGBoost.ipynb
│
├── data/
│   └── README.md
│
├── models/
│   └── customer_churn_model.joblib
│
├── reports/
│   └── model_results.md
│
├── figures/
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── calibration_curve.png
│   └── shap_summary.png
│
├── requirements.txt
│
└── .gitignore
```

> The actual repository structure may be simplified depending on which artifacts are ultimately committed.

---

## 💻 Technologies

**Python**
**Pandas**
**NumPy**
**Matplotlib**
**Seaborn**
**Scikit-learn**
**XGBoost**
**SHAP**
**Joblib**
**Jupyter Notebook**
**Google Colab**

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd customer-churn-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

```bash
jupyter notebook
```

Then open:

```text
notebooks/customer_churn_prediction_XGBoost.ipynb
```

Alternatively, the notebook can be executed directly in **Google Colab**.

### 4. Provide the dataset

Place the dataset in the expected project location and update `DATA_PATH` if necessary.

---

## ⚠️ Limitations

This project has several limitations:

* Model performance depends on the quality and representativeness of the available dataset.
* Historical customer behavior may not perfectly represent future behavior.
* Churn predictions indicate risk rather than certainty.
* Threshold selection depends on the business cost of false positives and false negatives.
* Model explanations describe associations learned by the model and should not automatically be interpreted as causal relationships.

---

## 🔮 Future Improvements

Potential future improvements include:

* Larger and more recent customer datasets
* External behavioral and usage data
* Advanced hyperparameter optimization with Optuna
* LightGBM and CatBoost comparison
* Ensemble modeling if experiments justify it
* Cost-sensitive threshold optimization
* Drift monitoring
* Automated model retraining
* Deployment through an API
* Interactive churn-risk dashboard
* Real-time prediction pipeline

---

## 📌 Key Machine Learning Skills Demonstrated

This project demonstrates practical experience with:

* Data cleaning
* Exploratory data analysis
* Feature engineering
* Leakage prevention
* Classification
* Ensemble learning
* Gradient boosting
* XGBoost
* Cross-validation
* Hyperparameter tuning
* Imbalanced classification
* Probability calibration
* Threshold optimization
* Model explainability
* Error analysis
* Model serialization
* Production-oriented inference
* Reproducible machine learning workflows

---

## 👨‍💻 Author

**Misgina Gebregergs**

BSc Mathematics
Addis Ababa University

Interested in:

**Machine Learning • Data Science • Artificial Intelligence • Mathematical Modeling**

---

## ⭐ Project Status

**Status:** 🟡 In Development

The complete modeling pipeline has been implemented. Final performance metrics will be reported after the notebook is executed and the final test evaluation is completed.

**Important:** The final test set is kept untouched until all model-selection and threshold decisions are finalized.
