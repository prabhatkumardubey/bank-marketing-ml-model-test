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

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load raw dataset (semicolon-delimited, as distributed by UCI)
# ---------------------------------------------------------------------------
raw = pd.read_csv("bank_marketing_raw.csv", sep=";")
print(f"Raw dataset shape: {raw.shape[0]} instances, {raw.shape[1] - 1} raw columns (incl. duration)")

LEAKAGE_COL = "duration"
TARGET = "y"

cat_cols = raw.select_dtypes(include=["object", "string"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != TARGET]
num_cols = [c for c in raw.columns if c not in cat_cols and c not in (TARGET, LEAKAGE_COL)]
RAW_FEATURE_COLS = cat_cols + num_cols  # 19 columns, duration excluded

print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
print(f"Numeric columns, excl. duration ({len(num_cols)}): {num_cols}")

y = raw[TARGET].map({"yes": 1, "no": 0})
print(f"Class balance: {dict(y.value_counts())}  (1=yes/subscribed, 0=no)")
print(f"'Always predict no' accuracy baseline: {(y == 0).mean():.4f}")

# ---------------------------------------------------------------------------
# 2. One-hot encode categorical columns ('unknown' kept as its own category -
#    it is a real response, not a missing value, so it is not imputed away)
# ---------------------------------------------------------------------------
X = pd.get_dummies(raw[RAW_FEATURE_COLS], columns=cat_cols, drop_first=True)
feature_names = list(X.columns)
print(f"Feature count after one-hot encoding: {len(feature_names)}")

# ---------------------------------------------------------------------------
# 3. Train / test split (stratified, 80/20)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# ---------------------------------------------------------------------------
# 4. Define the 5 required models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
    "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    "kNN":                 KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes":         GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
    ),
}
needs_scaling = {"Logistic Regression", "kNN"}

results = []
confusion_matrices = {}
class_reports = {}

for name, model in models.items():
    Xtr = X_train_scaled.values if name in needs_scaling else X_train.values
    Xte = X_test_scaled.values if name in needs_scaling else X_test.values

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]  # P(class = 1 = subscribed)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    results.append(metrics)
    confusion_matrices[name] = confusion_matrix(y_test, y_pred).tolist()
    class_reports[name] = classification_report(
        y_test, y_pred, target_names=["no(0)", "yes(1)"]
    )
    print(f"trained: {name}")

    joblib.dump(model, f"model/{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib")

joblib.dump(scaler, "model/scaler.joblib")

# ---------------------------------------------------------------------------
# 5. Results table
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results).set_index("Model").round(4)
print("\n=== Comparison Table (pos_label=1='yes'/subscribed) ===")
print(results_df.to_string())

results_df.to_csv("model/results_summary.csv")

with open("model/confusion_matrices.json", "w") as f:
    json.dump(confusion_matrices, f, indent=2)

for name, report in class_reports.items():
    print(f"\n--- Classification report: {name} ---")
    print(report)

# ---------------------------------------------------------------------------
# 6. Save held-out test split as test_data.csv (RAW columns, duration
#    excluded, + target y as text 'yes'/'no' as originally distributed)
# ---------------------------------------------------------------------------
test_raw = raw.loc[X_test.index, RAW_FEATURE_COLS + [TARGET]].copy()
test_raw.to_csv("test_data.csv", index=False)
print(f"\nSaved test_data.csv with {test_raw.shape[0]} rows and {test_raw.shape[1]} columns (raw, unencoded, no duration)")

with open("model/meta.json", "w") as f:
    json.dump({
        "raw_columns": RAW_FEATURE_COLS,
        "categorical_columns": cat_cols,
        "numeric_columns": num_cols,
        "encoded_feature_names": feature_names,
        "target_names": ["no(0)", "yes(1)"],
        "needs_scaling": list(needs_scaling),
        "target_col": TARGET,
        "excluded_leakage_col": LEAKAGE_COL,
    }, f, indent=2)

print("\nDone. Models saved in model/, test split saved as test_data.csv")
