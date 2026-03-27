import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
df = pd.read_csv('creditcard.csv')

print("Credit Card Fraud Detection - TIME HISTOGRAM")
print(f"Dataset: {df.shape}")
print("\nClass distribution:")
print(df['Class'].value_counts(normalize=True))

# Feature engineering: Convert Time (seconds) to hours (0-48 hours cycle)
df['Hour'] = (df['Time'] / 3600) % 24

# Separate classes
normal = df[df['Class'] == 0]
fraud = df[df['Class'] == 1]

# Create comprehensive time histogram
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Full time distribution (hours)
axes[0,0].hist(normal['Hour'], bins=48, alpha=0.7, color='green', label='Normal', density=True)
axes[0,0].hist(fraud['Hour'], bins=48, alpha=0.7, color='red', label='Fraud', density=True)
axes[0,0].set_title('Transaction Hour Distribution (Full Dataset)', fontweight='bold')
axes[0,0].set_xlabel('Hour of Day (0-24)')
axes[0,0].set_ylabel('Density')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. Fraud-only (zoomed)
axes[0,1].hist(fraud['Hour'], bins=48, alpha=0.8, color='red', edgecolor='black')
axes[0,1].set_title('Fraud Transactions Only', fontweight='bold')
axes[0,1].set_xlabel('Hour of Day')
axes[0,1].set_ylabel('Count')
axes[0,1].grid(True, alpha=0.3)

# 3. Normal vs Fraud ratio
hourly_fraud_rate = df.groupby('Hour')['Class'].mean()
axes[1,0].plot(hourly_fraud_rate.index, hourly_fraud_rate.values * 100, 'ro-', linewidth=2, markersize=4)
axes[1,0].set_title('Fraud Rate by Hour (%)', fontweight='bold')
axes[1,0].set_xlabel('Hour of Day')
axes[1,0].set_ylabel('Fraud %')
axes[1,0].grid(True, alpha=0.3)
axes[1,0].axhline(y=0.172, color='gray', linestyle='--', label='Overall Fraud Rate')
axes[1,0].legend()

# 4. Count per hour
hourly_counts = df.groupby('Hour').size()
axes[1,1].bar(hourly_counts.index, hourly_counts.values, alpha=0.7, color='blue', edgecolor='black')
axes[1,1].set_title('Total Transactions per Hour', fontweight='bold')
axes[1,1].set_xlabel('Hour of Day')
axes[1,1].set_ylabel('Count')
axes[1,1].grid(True, alpha=0.3)

plt.suptitle('Comprehensive Time Analysis - Credit Card Fraud Dataset', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('time_histogram.png', dpi=300, bbox_inches='tight')
plt.show()

# Statistics
print("\nFraud patterns by time:")
print("Hour with highest fraud rate:", hourly_fraud_rate.idxmax())
print("Max fraud rate:", f"{hourly_fraud_rate.max()*100:.3f}%")
print("Fraud transactions:", len(fraud))
print("Time range:", f"{df['Time'].min()/3600:.1f}h - {df['Time'].max()/3600:.1f}h")
