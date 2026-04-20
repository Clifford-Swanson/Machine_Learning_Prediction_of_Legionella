# Machine_Learning_Prediction_of_Legionella
This is the machine learning script and input data for machine learning model development used for the Environmental Engineering Science Journal article Schulz et al. 2026
Machine Learning Pipeline for Environmental Prediction and Interpretation
Overview

This repository contains a publication-ready machine learning pipeline for classification and regression tasks, designed for small-to-moderate sample size environmental datasets.

The framework emphasizes:

Robust model comparison
Statistical rigor
Interpretability
Reproducibility

The script (run_ml_models.py) performs:

Nested cross-validation with hyperparameter tuning
Model comparison across multiple algorithms
AUC / RMSE performance evaluation
Confidence intervals via bootstrapping
Statistical comparison using DeLong test
Decision Curve Analysis (DCA)
Feature importance (Permutation + SHAP)
Feature stability and breakpoint analysis
Models Included
Classification
Logistic Regression (Elastic Net)
Support Vector Machine (RBF)
Random Forest
XGBoost
Gaussian Process
Artificial Neural Network (MLP)
Regression
Elastic Net
Support Vector Regression (SVR)
Random Forest
XGBoost
Gaussian Process
Artificial Neural Network (MLP)
Repository Structure
project/
│
├── data/
│   ├── example_X.csv
│   └── example_y.csv
│
├── scripts/
│   └── run_ml_models.py
│
├── config/
│   └── config.yaml
│
├── results/                # auto-generated
│
├── requirements.txt
└── README.md



How to Run
Basic command
python scripts/run_ml_models.py \
    --data_X data/example_X.csv \
    --data_y data/example_y.csv \
    --task classification
Using a config file (recommended for reproducibility)
python scripts/run_ml_models.py --config config/config.yaml
Input Data Format
X (features)
Rows = samples
Columns = features
First column = index
y (target)
Single column CSV
Binary (classification) or continuous (regression)
Outputs

All outputs are saved to /results/

Performance
*_performance.csv
ALL_MODEL_PERFORMANCE_RANKED.csv
Predictions
*_train_predictions.csv
*_test_predictions.csv
ROC Analysis
*_ROC.csv
ALL_MODELS_ROC_CURVES.csv
Statistical Testing
ALL_MODELS_DeLong_Test.csv
*_Test_AUC_CI.txt
Decision Curve Analysis
*_DECISION_CURVE.csv
ALL_MODELS_DECISION_CURVE.csv
Feature Analysis
*_permutation_importance.csv
*_shap_values.csv
*_shap_importance.csv
FEATURE_STABILITY.csv
Breakpoint Analysis
*_Break_Point_Analysis.csv


Key Methodological Features
Nested Cross-Validation → prevents overfitting
Bootstrap Confidence Intervals → quantifies uncertainty
DeLong Test → statistical comparison of AUC
Decision Curve Analysis → evaluates practical utility
Correlation Filtering + VIF → reduces multicollinearity
SHAP Values → interpretable feature contributions
Reproducibility
Fixed random seed (random_state)
Config-driven execution
Environment dependencies specified
All intermediate outputs saved
Notes
Designed for small datasets (n < 100) where overfitting risk is high
Supports both prediction and mechanistic interpretation
Particularly suited for environmental microbiology, water quality, and public health datasets
Citation

If used in a publication, please cite the associated manuscript and repository.

