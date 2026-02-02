# Credit Card Fraud Detection using Machine Learning

## Overview
This project implements an end-to-end Machine Learning pipeline to detect fraudulent credit card transactions using a highly imbalanced real-world dataset. The focus is on handling class imbalance correctly, choosing appropriate evaluation metrics, and building a robust fraud detection model.

## Dataset
Source: ULB Credit Card Fraud Dataset (Kaggle)

Total transactions: 284,807  
Fraud cases: 492 (~0.17%)

Features:
- V1–V28: PCA-transformed, anonymized features
- Amount: Transaction amount
- Time: Seconds elapsed since the first transaction
- Class: Target variable (0 = Normal, 1 = Fraud)

The dataset is highly imbalanced, reflecting real-world fraud detection scenarios.

## Project Workflow
1. Environment setup and dependency management
2. Dataset loading and validation
3. Exploratory Data Analysis (EDA)
4. Data preprocessing and scaling
5. Baseline Logistic Regression model
6. Handling class imbalance using class weights
7. Threshold tuning and ROC-AUC evaluation
8. Random Forest model training
9. Model comparison and selection
10. Saving the final model and scaler

## Models and Results

Logistic Regression  
- Precision (Fraud): 0.81  
- Recall (Fraud): 0.69  
- ROC-AUC: ~0.97  

Logistic Regression (Balanced)  
- Precision (Fraud): 0.06  
- Recall (Fraud): 0.92  
- ROC-AUC: ~0.97  

Random Forest  
- Precision (Fraud): 0.94  
- Recall (Fraud): 0.82  
- ROC-AUC: ~0.96  

Final Model: Random Forest  
Chosen for the best precision–recall trade-off.

## Project Structure
credit-card-fraud-detection/
├── data/
│   └── raw/
│       └── creditcard.csv
├── models/
│   ├── random_forest_fraud.pkl
│   └── amount_scaler.pkl
├── notebooks/
│   └── credit_card_fraud_detection.ipynb
├── requirements.txt
└── README.md

## Using the Saved Model
import joblib

model = joblib.load("models/random_forest_fraud.pkl")
scaler = joblib.load("models/amount_scaler.pkl")

Ensure the Amount feature is scaled before prediction.

## Key Learnings
- Fraud detection is an inherently imbalanced classification problem
- Accuracy is misleading for rare-event prediction
- Precision–recall trade-offs are driven by business requirements
- Tree-based models perform well on anonymized, PCA-transformed data

