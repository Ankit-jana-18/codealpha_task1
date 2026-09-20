import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

# Load dataset
df = pd.read_csv("cs_training.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

# Remove unnecessary index column
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Target distribution
print("\nTarget Distribution:")
print(df["SeriousDlqin2yrs"].value_counts())

# Correlation Heatmap
plt.figure(figsize=(12, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Credit Scoring Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Feature Engineering
df["TotalPastDue"] = (
    df["NumberOfTime30-59DaysPastDueNotWorse"]
    + df["NumberOfTime60-89DaysPastDueNotWorse"]
    + df["NumberOfTimes90DaysLate"]
)

df["CreditLinesPerDependent"] = (
    df["NumberOfOpenCreditLinesAndLoans"] /
    (df["NumberOfDependents"] + 1)
)

# Separate features and target
X = df.drop("SeriousDlqin2yrs", axis=1)
y = df["SeriousDlqin2yrs"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:", y.name)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# Handle missing values
for column in X_train.columns:
    if X_train[column].isnull().sum() > 0:
        median_value = X_train[column].median()
        X_train[column] = X_train[column].fillna(median_value)
        X_test[column] = X_test[column].fillna(median_value)

print("\nMissing Values After Cleaning:")
print("Training:", X_train.isnull().sum().sum())
print("Testing:", X_test.isnull().sum().sum())

# Feature Scaling for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

logistic_model.fit(X_train_scaled, y_train)

logistic_pred = logistic_model.predict(X_test_scaled)
logistic_prob = logistic_model.predict_proba(X_test_scaled)[:, 1]

logistic_metrics = [
    accuracy_score(y_test, logistic_pred),
    precision_score(y_test, logistic_pred),
    recall_score(y_test, logistic_pred),
    f1_score(y_test, logistic_pred),
    roc_auc_score(y_test, logistic_prob)
]

# Decision Tree
decision_tree = DecisionTreeClassifier(
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

decision_tree.fit(X_train, y_train)

dt_pred = decision_tree.predict(X_test)
dt_prob = decision_tree.predict_proba(X_test)[:, 1]

dt_metrics = [
    accuracy_score(y_test, dt_pred),
    precision_score(y_test, dt_pred),
    recall_score(y_test, dt_pred),
    f1_score(y_test, dt_pred),
    roc_auc_score(y_test, dt_prob)
]

# Random Forest
random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

random_forest.fit(X_train, y_train)

rf_pred = random_forest.predict(X_test)
rf_prob = random_forest.predict_proba(X_test)[:, 1]

rf_metrics = [
    accuracy_score(y_test, rf_pred),
    precision_score(y_test, rf_pred),
    recall_score(y_test, rf_pred),
    f1_score(y_test, rf_pred),
    roc_auc_score(y_test, rf_prob)
]

# Model Comparison
results = pd.DataFrame(
    [logistic_metrics, dt_metrics, rf_metrics],
    columns=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
    index=["Logistic Regression", "Decision Tree", "Random Forest"]
)

print("\nModel Performance:")
print(results.round(4))

# Best Model
best_model = results["ROC-AUC"].idxmax()

print("\nBest Model:", best_model)
print("Best ROC-AUC:", round(results.loc[best_model, "ROC-AUC"], 4))

# Random Forest Classification Report
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=1
)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ROC Curve
lr_fpr, lr_tpr, _ = roc_curve(y_test, logistic_prob)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_prob)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)

plt.figure(figsize=(8, 6))
plt.plot(
    lr_fpr, lr_tpr,
    label=f"Logistic Regression (AUC={logistic_metrics[4]:.3f})"
)
plt.plot(
    dt_fpr, dt_tpr,
    label=f"Decision Tree (AUC={dt_metrics[4]:.3f})"
)
plt.plot(
    rf_fpr, rf_tpr,
    label=f"Random Forest (AUC={rf_metrics[4]:.3f})"
)
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Model Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Feature Importance
importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": random_forest.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nFeature Importance:")
print(importance)

# Feature Importance Heatmap
importance_matrix = importance.set_index("Feature").T

plt.figure(figsize=(14, 2.5))
sns.heatmap(
    importance_matrix,
    annot=True,
    fmt=".3f",
    cmap="YlGnBu",
    cbar=True
)
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

print("\nCredit Scoring Model Completed Successfully!")