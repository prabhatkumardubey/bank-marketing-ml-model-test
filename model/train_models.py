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
# 1. Load raw dataset (semicolon seperated as distributed by UCI)
# ---------------------------------------------------------------------------
raw = pd.read_csv("bank_marketing_raw.csv", sep=";")
print(f"Raw dataset shape: {raw.shape[0]} instances, {raw.shape[1] - 1} raw columns (incl. duration)")


# ---------------------------------------------------------------------------
# 2. One-hot encode categorical columns ('unknown' keep as its own category -
#    it is a real response, not a missing value, so it does not get ommited)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3. Train/test split (80/20)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 4. Define the required models
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
print("Done")
