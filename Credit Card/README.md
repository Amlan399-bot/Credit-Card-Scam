<<<<<<< HEAD
# Credit Card Fraud Detection

Complete anomaly detection pipeline for imbalanced credit card fraud dataset.

## Features
- EDA with visualizations (`eda_plots.png`)
- Anomaly detection (IsolationForest, LOF, OneClassSVM, EllipticEnvelope)
- Imbalanced handling (SMOTE, SMOTETomek, class weights)
- Model comparison table with ROC-AUC, PR-AUC, F1
- Preprocessing pipeline (RobustScaler, Time engineering)

## Usage
```bash
python FraudDetection.py
```

## Results Summary
See console output for model rankings. Best anomaly model typically achieves ROC-AUC > 0.95.

## Files
- `FraudDetection.py` - Complete pipeline
- `TODO.md` - Progress tracking
- `creditcard.csv` - Dataset
- `scaler.pkl` - Trained scaler
- `eda_plots.png` - Visualizations
=======
# Credit-Card-Scam
Fraud detection system using Isolation Forest, One-Class SVM, Random Forest, and XGBoost, combined with SMOTE-based resampling to handle highly imbalanced data.
>>>>>>> 3cfbcf6d628317165fc627c19fcc9cf31d99842c
