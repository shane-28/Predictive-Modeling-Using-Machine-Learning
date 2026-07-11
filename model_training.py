"""
Predictive Modeling Using Machine Learning
--------------------------------------------
Predicts customer churn (Yes/No) from account and usage features.

Models compared:
  - Logistic Regression  (the linear/baseline model — used here instead of
                           plain Linear Regression, since the target is
                           categorical and we need class probabilities for
                           the confusion matrix / ROC curve)
  - Decision Tree
  - Random Forest

Run: python model_training.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

DATA_PATH = "data/customer_churn.csv"
VIS_DIR = "visuals"
os.makedirs(VIS_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. LOAD & PREPROCESS DATA
# ----------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("=" * 60)
print(f"Dataset shape: {df.shape}")

# Handle the few missing values (median impute)
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Drop ID column (not predictive)
df = df.drop(columns=["CustomerID"])

# Target distribution plot
plt.figure(figsize=(5, 4))
sns.countplot(x="Churn", data=df, hue="Churn", palette="Set2", legend=False)
plt.title("Target Distribution: Churn")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/01_target_distribution.png")
plt.close()

# Encode categorical features
categorical_cols = ["ContractType", "InternetService", "PaymentMethod"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

le = LabelEncoder()
df_encoded["Churn"] = le.fit_transform(df_encoded["Churn"])  # Yes=1, No=0

# Correlation heatmap (numeric features only)
plt.figure(figsize=(8, 6))
numeric_cols = ["Age", "TenureMonths", "MonthlyCharges", "TotalCharges",
                 "SupportCallsLastYear", "Churn"]
sns.heatmap(df_encoded[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/02_correlation_heatmap.png")
plt.close()

# ----------------------------------------------------------------------
# 2. TRAIN / TEST SPLIT
# ----------------------------------------------------------------------
X = df_encoded.drop(columns=["Churn"])
y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# Scale features (helps Logistic Regression converge cleanly)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------------------
# 3. TRAIN MODELS
# ----------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
}

results = {}
roc_data = {}

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": auc}
    roc_data[name] = roc_curve(y_test, y_proba)

    print(f"\n{name}")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall   : {rec:.3f}")
    print(f"  F1 Score : {f1:.3f}")
    print(f"  ROC AUC  : {auc:.3f}")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"{VIS_DIR}/03_confusion_matrix_{safe_name}.png")
    plt.close()

# ----------------------------------------------------------------------
# 4. ROC CURVE COMPARISON
# ----------------------------------------------------------------------
plt.figure(figsize=(7, 6))
for name, (fpr, tpr, _) in roc_data.items():
    auc = results[name]["AUC"]
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/04_roc_curves_comparison.png")
plt.close()

# ----------------------------------------------------------------------
# 5. MODEL ACCURACY / METRIC COMPARISON
# ----------------------------------------------------------------------
results_df = pd.DataFrame(results).T
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print(results_df.round(3))

plt.figure(figsize=(9, 5))
results_df[["Accuracy", "Precision", "Recall", "F1", "AUC"]].plot(
    kind="bar", figsize=(9, 5), colormap="viridis"
)
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/05_model_comparison.png")
plt.close()

# ----------------------------------------------------------------------
# 6. FEATURE IMPORTANCE (Random Forest)
# ----------------------------------------------------------------------
rf_model = models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
            palette="mako", legend=False)
plt.title("Feature Importance — Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/06_feature_importance_rf.png")
plt.close()

results_df.round(3).to_csv(f"{VIS_DIR}/model_comparison_metrics.csv")

print(f"\nAll visualizations saved to '{VIS_DIR}/'")
print("=" * 60)
print("PREDICTIVE MODELING COMPLETE")
print("=" * 60)
