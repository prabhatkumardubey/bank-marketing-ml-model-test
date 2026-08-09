"""
This script trains 5 classification models on the UCI Bank Marketing dataset
(bank-additional-full.csv). It checks each model with 6 metrics (Accuracy, AUC,
Precision, Recall, F1, MCC), saves the trained models and the fitted scaler,
and also saves the test split as test_data.csv.

Dataset details:
- Name: Bank Marketing (bank-additional-full.csv)
- Source: UCI Machine Learning Repository (Moro, Rita & Cortez, 2014)
- Rows: 41,188
- Input features used: 19 out of 20 (we removed the 'duration' column)
- Task: Binary classification - y = 1 means client subscribed, y = 0 means not
- Dataset ID: 222 at UCI ML Repository

Important note: We are NOT using the 'duration' column. Duration is the length
of the last call, and this is known only after the call is done. So it is like
a leakage feature. If we use it, the model will look very good but it will not
be useful in real life because we will not know the call duration before
calling the customer. We keep 'duration' only in the raw CSV for reference, but
no model uses it.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

SEED = 42

# ---------------------------------------------------------------------------
# Step 1: Load the raw CSV file
# UCI gives this file with semicolons (;) instead of commas
# ---------------------------------------------------------------------------
raw = pd.read_csv("bank_marketing_raw.csv", sep=";")
print(f"Raw dataset shape: {raw.shape[0]} instances, {raw.shape[1] - 1} raw columns (incl. duration)")

DROP_COL = "duration"
TARGET_COL = "y"

cat_cols = raw.select_dtypes(include=["object", "string"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != TARGET_COL]
num_cols = [c for c in raw.columns if c not in cat_cols and c not in (TARGET_COL, DROP_COL)]
FEATURE_COLS = cat_cols + num_cols  # 19 columns total, duration removed

print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
print(f"Numeric columns, excl. duration ({len(num_cols)}): {num_cols}")

# Convert target to numbers: yes = 1, no = 0
# 1 means the client subscribed, which is the class we care about
y = raw[TARGET_COL].map({"yes": 1, "no": 0})
print(f"Class balance: {dict(y.value_counts())}  (1=yes/subscribed, 0=no)")
print(f"'Always predict no' accuracy baseline: {(y == 0).mean():.4f}")

# ---------------------------------------------------------------------------
# Step 2: One-hot encode the categorical columns
# We keep 'unknown' as its own category because it is a real answer, not missing data
# ---------------------------------------------------------------------------
X = pd.get_dummies(raw[FEATURE_COLS], columns=cat_cols, drop_first=True)
encoded_features = list(X.columns)
print(f"Feature count after one-hot encoding: {len(encoded_features)}")

# ---------------------------------------------------------------------------
# Step 3: Split the data into train and test sets (80% train, 20% test)
# stratify=y keeps the same yes/no ratio in both sets
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# ---------------------------------------------------------------------------
# Step 4: Create the 5 machine learning models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, random_state=SEED),
    "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=SEED),
    "kNN":                 KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes":         GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=SEED, n_jobs=-1
    ),
}
scale_models = {"Logistic Regression", "kNN"}

results = []
conf_mats = {}
reports = {}

for name, model in models.items():
    x_train = X_train_scaled.values if name in scale_models else X_train.values
    x_test = X_test_scaled.values if name in scale_models else X_test.values

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]  # probability that the client said yes (class 1)

    result = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    results.append(result)
    conf_mats[name] = confusion_matrix(y_test, y_pred).tolist()
    reports[name] = classification_report(
        y_test, y_pred, target_names=["no(0)", "yes(1)"]
    )
    print(f"trained: {name}")

    joblib.dump(model, f"model/{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib")

joblib.dump(scaler, "model/scaler.joblib")

# ---------------------------------------------------------------------------
# Step 5: Show and save the results
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results).set_index("Model").round(4)
print("\n=== Comparison Table (pos_label=1='yes'/subscribed) ===")
print(results_df.to_string())

results_df.to_csv("model/results_summary.csv")

with open("model/confusion_matrices.json", "w") as f:
    json.dump(conf_mats, f, indent=2)

for name, report in reports.items():
    print(f"\n--- Classification report: {name} ---")
    print(report)

# ---------------------------------------------------------------------------
# Step 6: Save the test split as test_data.csv
# We save the raw columns only, with duration removed and target as yes/no
# ---------------------------------------------------------------------------
test_raw = raw.loc[X_test.index, FEATURE_COLS + [TARGET_COL]].copy()
test_raw.to_csv("test_data.csv", index=False)
print(f"\nSaved test_data.csv with {test_raw.shape[0]} rows and {test_raw.shape[1]} columns (raw, unencoded, no duration)")

with open("model/meta.json", "w") as f:
    json.dump({
        "raw_columns": FEATURE_COLS,
        "categorical_columns": cat_cols,
        "numeric_columns": num_cols,
        "encoded_feature_names": encoded_features,
        "target_names": ["no(0)", "yes(1)"],
        "needs_scaling": list(scale_models),
        "target_col": TARGET_COL,
        "excluded_leakage_col": DROP_COL,
    }, f, indent=2)

print("\nDone. Models saved in model/, test split saved as test_data.csv")
