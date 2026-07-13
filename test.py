import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib
import json 

df = pd.read_csv('creditcard.csv')

print(df.shape)
print(df['Class'].value_counts())
print(df['Class'].value_counts(normalize=True))
print(df.isnull().sum().sum())
print(df[['Time', 'Amount']].describe())

FEATURES = [c for c in df.columns if c not in ['Class', 'Time']]

normal = df[df['Class'] == 0]
fraud = df[df['Class'] == 1]

normal_train, normal_test = train_test_split(normal, test_size=0.2, random_state=42)

test_set = pd.concat([normal_test, fraud])

X_train = normal_train[FEATURES]
X_test = test_set[FEATURES]
y_test = test_set['Class']

print(X_train.shape, X_test.shape)
print(y_test.value_counts())

iso_forest = IsolationForest(
    n_estimators=100,
    max_samples='auto',
    contamination = 'auto',
    random_state=42,
    n_jobs=-1
)

iso_forest.fit(X_train)
test_scores = -iso_forest.score_samples(X_test)

fraud_scores = test_scores[y_test == 1]
normal_scores = test_scores[y_test == 0]

print("Fraud   — mean: {:.4f}, median: {:.4f}, min: {:.4f}, max: {:.4f}".format(
    fraud_scores.mean(), np.median(fraud_scores), fraud_scores.min(), fraud_scores.max()))
print("Normal  — mean: {:.4f}, median: {:.4f}, min: {:.4f}, max: {:.4f}".format(
    normal_scores.mean(), np.median(normal_scores), normal_scores.min(), normal_scores.max()))

roc_auc = roc_auc_score(y_test, test_scores)
pr_auc = average_precision_score(y_test, test_scores)

print("ROC AUC: {:.4f}".format(roc_auc))
print("PR AUC: {:.4f}".format(pr_auc))

for pct in [0.1, 0.5, 1, 2, 5]:
    threshold = np.percentile(test_scores, 100 - pct)
    predicted_fraud = test_scores >= threshold
    
    true_positives = np.sum((predicted_fraud) & (y_test.values == 1))
    false_positives = np.sum((predicted_fraud) & (y_test.values == 0))
    total_frauds = np.sum(y_test.values == 1)
    
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / total_frauds
    
    print(f"Top {pct}% flagged | threshold={threshold:.4f} | "
          f"precision={precision:.3f} | recall={recall:.3f} | "
          f"caught {true_positives}/{total_frauds} fraud, {false_positives} false alarms")
    
sample_fraud = X_test[y_test.values == 1].iloc[0][FEATURES].to_dict()
sample_normal = X_test[y_test.values == 0].iloc[0][FEATURES].to_dict()

print(json.dumps(sample_fraud))
print()
print(json.dumps(sample_normal))

THRESHOLDS = {
    "critical": 0.7039,
    "high" : 0.6402,
    "medium" : 0.5583
}

def risk_tier(score):
    if score >= THRESHOLDS["critical"]: return "critical"
    elif score >= THRESHOLDS["high"]: return "high"
    elif score >= THRESHOLDS["medium"]: return "medium"
    return "low"

fraud_tiers = pd.Series([risk_tier(s) for s in fraud_scores])
print(fraud_tiers.value_counts())
print(fraud_tiers.value_counts(normalize=True))

