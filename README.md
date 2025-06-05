# CodeAlpha_MachineLearning
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls'
df = pd.read_excel(url, header=1)
df.rename(columns={'default payment next month': 'default'}, inplace=True)

# Drop ID column
df.drop(columns=['ID'], inplace=True)

# Feature Engineering
df['LIMIT_BAL_log'] = np.log1p(df['LIMIT_BAL'])  # Log transformation
df['PAY_SUM'] = df[[f'PAY_{i}' for i in range(1, 7)]].sum(axis=1)  # Total payment status
df['BILL_SUM'] = df[[f'BILL_AMT{i}' for i in range(1, 7)]].sum(axis=1)
df['PAY_AMT_SUM'] = df[[f'PAY_AMT{i}' for i in range(1, 7)]].sum(axis=1)
df['UTIL_RATIO'] = df['BILL_SUM'] / (df['LIMIT_BAL'] + 1)

# Drop redundant features
drop_cols = [f'BILL_AMT{i}' for i in range(1, 7)] + [f'PAY_AMT{i}' for i in range(1, 7)]
df.drop(columns=drop_cols, inplace=True)

# Define features and target
X = df.drop(columns='default')
y = df['default']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# Optional: scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importances
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nTop 10 Features:")
print(importances.sort_values(ascending=False).head(10))


 Output Example
Precision, Recall, F1

ROC-AUC ~ 0.75–0.80 (typical)

Top features like PAY_SUM, UTIL_RATIO, and LIMIT_BAL_log
