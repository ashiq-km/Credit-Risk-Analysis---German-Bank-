# 🏦 Credit Risk Analysis – German Credit Dataset

A complete end-to-end Credit Risk Analysis project using the German Credit Dataset.
This repository covers everything from EDA → preprocessing → feature engineering → modeling → evaluation → deployment.

## 📘 Overview

This project analyzes credit applicant data to understand patterns that lead to good or bad credit outcomes and builds a predictive model to assess credit risk.

It includes:

Clean and documented datasets

Notebooks for each stage

Final model pipeline

Streamlit deployment code

## 🎯 Project Objectives
## ✅ Primary Goals

Understand customer-level credit factors

Clean and preprocess raw credit data

Engineer meaningful and interpretable features

Build and evaluate ML models

Implement the best model in a deployable format

### 🧠 Key Questions Answered

Which customer attributes influence creditworthiness?

What patterns separate defaulters from non-defaulters?

Which model performs best for predicting loan default?

## 📊 Dataset Description
📂 Dataset: German Credit Risk Dataset

Contains 1,000 applicants with categorical + numeric attributes:

Personal information

Credit history

Loan purpose & amount

Payment behavior

Financial stability

Many features come with coded values (e.g., A41, A93), which were decoded during preprocessing.

## 🛠️ Project Workflow
### 1. 🔍 Exploratory Data Analysis (EDA)

Distribution checks

Correlation visualization

Categorical decoding

Outlier identification

### 2. 🧹 Data Preprocessing

Handling missing values

Feature type correction

Ordinal & One-Hot Encoding

Scaling numeric variables

Outlier treatment

### 3. ⚙️ Feature Engineering

Creation of ratio-based variables

Credit utilisation features

Binning & transformations

SMOTE for class imbalance

### 4. 🤖 Modeling

Models evaluated:

Logistic Regression

Random Forest

XGBoost

LightGBM

Grid Search & cross-validation used for tuning

Performance evaluation on Recall, Precision, F1, ROC-AUC

### 5. 🚀 Deployment

Streamlit app created for model prediction

User-friendly UI with input legends/explanations

Final model pipeline saved via joblib

📈 Results Summary

Best model achieved strong Recall for identifying risky applicants

Proper feature engineering significantly improved performance

Model generalized well on unseen test data

(You can add exact scores if needed.)
```
📂 Project Structure
Credit-Risk-Analysis/
│
├── data/
│   ├── gd.csv
│   ├── german.data
│   ├── german.data-numeric
│   ├── german.doc
│   └── Index
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── feature_engineering.ipynb
│   ├── modeling.ipynb
│   └── evaluation.ipynb
│
├── app/
│   ├── streamlit_app.py
│   └── best_model/
│       └── xgb_pipeline.joblib
│
└── README.md```

💻 Technologies Used

Python 🐍

Pandas, NumPy

Scikit-Learn

XGBoost / LightGBM

Imbalanced-Learn

Matplotlib & Seaborn

Streamlit

Joblib

🚧 Future Enhancements

Add SHAP-based interpretability

Add API endpoints for production use

Add monitoring & drift detection

🙌 Acknowledgements

Dataset source: UCI Machine Learning Repository – German Credit Dataset.
