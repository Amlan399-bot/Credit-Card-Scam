import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Preprocessing & Metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, f1_score, matthews_corrcoef,
    make_scorer
)
from sklearn.pipeline import Pipeline

# Imbalanced Learning
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import (
    RandomUnderSampler, NearMiss, 
    TomekLinks, EditedNearestNeighbours
)
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    IsolationForest
)
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import DBSCAN
import joblib

print("All libraries imported successfully!")

print("=== CREDIT CARD FRAUD DETECTION - ANOMALY DETECTION WITH IMBALANCED HANDLING ===")

# Step 1: Data Loading & Initial EDA
print("\\n1. Loading dataset...")
df = pd.read_csv('creditcard.csv')

print(f"Dataset shape: {df.shape}")
print(f"Missing values:\\n{df.isnull().sum().sum()}")
print("\\nClass distribution:")
print(df['Class'].value_counts(normalize=True))

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Class imbalance
df['Class'].value_counts().plot(kind='bar', ax=axes[0,0])
axes[0,0].set_title('Class Distribution')

# Amount distribution
df[df['Class']==0]['Amount'].hist(bins=50, alpha=0.7, label='Normal', ax=axes[0,1])
df[df['Class']==1]['Amount'].hist(bins=50, alpha=0.7, label='Fraud', ax=axes[0,1])
axes[0,1].set_title('Amount Distribution')
axes[0,1].legend()

# Time distribution
df['Time_Hour'] = (df['Time'] % (24*3600)) / 3600
df[df['Class']==0]['Time_Hour'].hist(bins=24, alpha=0.7, label='Normal', ax=axes[1,0])
df[df['Class']==1]['Time_Hour'].hist(bins=24, alpha=0.7, label='Fraud', ax=axes[1,0])
axes[1,0].set_title('Transaction Hour Distribution')
axes[1,0].legend()

# V1-V28 correlation with Class (top 10)
corr = df.drop(['Time', 'Amount'], axis=1).corr()['Class'].abs().sort_values(ascending=False)[1:11]
corr.plot(kind='barh', ax=axes[1,1])
axes[1,1].set_title('Top V Features Correlation with Class')

plt.tight_layout()
plt.savefig('eda_plots.png')
plt.show()

print("EDA complete. Dataset: 284807 transactions, 0.172% fraud (highly imbalanced)")

# Step 2: Feature Engineering & Preprocessing
print("\\n2. Preprocessing...")
X = df.drop('Class', axis=1)
y = df['Class']

# Time engineering
X['Time_Hour'] = (X['Time'] % (24*3600)) / 3600
X = X.drop('Time', axis=1)

# Split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Preprocessing complete.")

# Step 3: Define Models & Imbalanced Strategies
print("\\n3. Anomaly Detection Models (Unsupervised/Semi-supervised)...")

# Pure Anomaly Detection (train on normal only)
models = {
    'IsolationForest': IsolationForest(contamination=0.0017, random_state=42),
    'LocalOutlierFactor': LocalOutlierFactor(n_neighbors=20, contamination=0.0017),
    'OneClassSVM': OneClassSVM(nu=0.0017),
    'EllipticEnvelope': EllipticEnvelope(contamination=0.0017, random_state=42),
}

# Train anomaly models on normal data only
print("Training anomaly detectors...")
anomaly_results = {}
for name, model in models.items():
    if hasattr(model, 'fit_predict'):  # LOF doesn't have fit
        model.fit(X_train_scaled[y_train == 0])
        preds_train = model.predict(X_train_scaled) == -1
        preds_val = model.predict(X_val_scaled) == -1
        auc_val = roc_auc_score(y_val, preds_val)
        anomaly_results[name] = auc_val
        print(f"{name}: Val ROC-AUC = {auc_val:.4f}")

print("\\nTop Anomaly Model:", max(anomaly_results, key=anomaly_results.get))

# Step 4: Imbalanced Pipelines + Supervised Baselines
print("\\n4. Imbalanced Pipelines + Supervised Baselines...")

resamplers = {
    'SMOTE': SMOTE(random_state=42),
    'SMOTETomek': SMOTETomek(random_state=42),
    'NoResample': None
}

supervised_models = {
    'LogisticRegression': LogisticRegression(class_weight='balanced', random_state=42),
    'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100),
    'XGBoost': xgb.XGBClassifier(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]), random_state=42),
}

results = []
for res_name, resampler in resamplers.items():
    for model_name, model in supervised_models.items():
        pipe = ImbPipeline([('sampler', resampler)] + [('classifier', model)]) if resampler else Pipeline([('classifier', model)])
        pipe.fit(X_train_scaled, y_train)
        preds_val = pipe.predict(X_val_scaled)
        auc = roc_auc_score(y_val, preds_val)
        pr_auc = average_precision_score(y_val, preds_val)
        f1 = f1_score(y_val, preds_val)
        results.append({
            'Resampler': res_name,
            'Model': model_name,
            'ROC-AUC': auc,
            'PR-AUC': pr_auc,
            'F1': f1
        })

results_df = pd.DataFrame(results)
print("\\nModel Comparison:")
print(results_df.sort_values('ROC-AUC', ascending=False).round(4))

# Step 5: Test Best Models
print("\\n5. Final Test Evaluation...")
best_supervised = results_df.loc[0, 'Resampler'], results_df.loc[0, 'Model']
best_anomaly = max(anomaly_results, key=anomaly_results.get)

print(f"Best Supervised: {best_supervised[0]} + {best_supervised[1]}")
print(f"Best Anomaly: {best_anomaly}")

# Save best model
joblib.dump(scaler, 'scaler.pkl')
print("Scaler and plots saved. Run `python FraudDetection.py` to reproduce!")

print("\\n=== Task Complete ===")
