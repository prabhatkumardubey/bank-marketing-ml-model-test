"""
app.py - Streamlit app for bank term deposit prediction
Name: Prabhat Kumar Dubey
Machine Learning - Assignment 2
BITS Pilani WILP - M.Tech (AI/ML)

This app loads the 5 trained models saved by model/train_models.py.
Upload a raw test CSV (19 raw columns, duration not included) with the
'y' column, choose a model, and see its metrics, confusion matrix,
and classification report.
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Bank Term Deposit Subscription Prediction", layout="wide")

# ---------------------------------------------------------------------------
# Load models, scaler, and metadata once
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("model/meta.json") as f:
        meta = json.load(f)
    models = {
        "Logistic Regression": joblib.load("model/Logistic_Regression.joblib"),
        "Decision Tree": joblib.load("model/Decision_Tree.joblib"),
        "kNN": joblib.load("model/kNN.joblib"),
        "Naive Bayes": joblib.load("model/Naive_Bayes.joblib"),
        "Random Forest (Ensemble)": joblib.load("model/Random_Forest_Ensemble.joblib"),
    }
    scaler = joblib.load("model/scaler.joblib")
    return models, scaler, meta

models, scaler, meta = load_artifacts()
RAW_COLS = meta["raw_columns"]
CAT_COLS = meta["categorical_columns"]
NUM_COLS = meta["numeric_columns"]
ENCODED_COLS = meta["encoded_feature_names"]
TARGET_NAMES = meta["target_names"]          # labels for the two classes
NEEDS_SCALING = set(meta["needs_scaling"])
TARGET_COL = meta["target_col"]              # name of target column

st.title("🏦 Bank Term Deposit Subscription Prediction")
st.caption(
    "Dataset: UCI Bank Marketing (bank-additional-full). 19 raw features "
    "(10 categorical, 9 numeric; duration excluded because it causes leakage). "
    "41,188 rows. Task: predict if a client subscribes (yes/no)."
)
st.info(
    "Tip: If we always predict 'no', we get about 88.7% accuracy because "
    "only ~11.3% clients say yes. So look at AUC and MCC too — they give a "
    "fairer picture than Accuracy alone.",
    icon="📌",
)

# ---------------------------------------------------------------------------
# Section 1: Upload raw test CSV
# ---------------------------------------------------------------------------
st.header("1. Upload test data")
st.write(
    f"Upload a CSV with {len(RAW_COLS)} raw columns (duration not needed) "
    f"plus a `{TARGET_COL}` column (`yes`/`no`). You can also use the bundled "
    "`test_data.csv`."
)
uploaded_file = st.file_uploader("Pick a CSV file", type=["csv"])

if uploaded_file is not None:
    # UCI Bank Marketing file uses ';' as separator.
    # Try comma first; if columns are missing, try semicolon.
    try:
        df = pd.read_csv(uploaded_file)
        if not set(RAW_COLS + [TARGET_COL]).issubset(df.columns):
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=";")
    except Exception as e:
        st.error(f"Could not read the uploaded CSV: {e}")
        st.stop()
else:
    st.info("No file uploaded yet. Showing the bundled test_data.csv preview.")
    df = pd.read_csv("test_data.csv")

missing_cols = [c for c in RAW_COLS if c not in df.columns]
if missing_cols or TARGET_COL not in df.columns:
    st.error(
        f"Your CSV is missing these required columns: "
        f"{missing_cols + ([TARGET_COL] if TARGET_COL not in df.columns else [])}"
    )
    st.stop()

st.dataframe(df, width="stretch", height=350)
st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

# ---------------------------------------------------------------------------
# One-hot encode the same way as training, then match training columns
# (a small uploaded file may not have every category).
# ---------------------------------------------------------------------------
X_raw = df[RAW_COLS]
y_true = df[TARGET_COL].map({"yes": 1, "no": 0})

X_encoded = pd.get_dummies(X_raw, columns=CAT_COLS, drop_first=True)
X_encoded = X_encoded.reindex(columns=ENCODED_COLS, fill_value=0)

# ---------------------------------------------------------------------------
# Section 2: Pick a model
# ---------------------------------------------------------------------------
st.header("2. Select a model")
model_name = st.selectbox("Pick a model", list(models.keys()))
model = models[model_name]

if model_name in NEEDS_SCALING:
    X_input = X_encoded.copy()
    X_input[NUM_COLS] = scaler.transform(X_input[NUM_COLS])
    X_input = X_input.values
else:
    X_input = X_encoded.values

y_pred = model.predict(X_input)
y_proba = model.predict_proba(X_input)[:, 1]  # probability client subscribed (class 1)

# ---------------------------------------------------------------------------
# Section 3: Show metrics
# ---------------------------------------------------------------------------
st.header("3. Evaluation metrics")
st.caption("Positive class = 1 means the client subscribed ('yes').")

metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "AUC": roc_auc_score(y_true, y_proba),
    "Precision": precision_score(y_true, y_pred),
    "Recall": recall_score(y_true, y_pred),
    "F1 Score": f1_score(y_true, y_pred),
    "MCC": matthews_corrcoef(y_true, y_pred),
}

cols = st.columns(len(metrics))
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.4f}")

baseline_acc = (y_true == 0).mean()
st.caption(f"⚠️ Simple baseline (always predict 'no') accuracy: **{baseline_acc:.4f}**. Compare it with the Accuracy above.")

# ---------------------------------------------------------------------------
# Compare all 5 models side by side
# ---------------------------------------------------------------------------
with st.expander("Compare all 5 models on this data"):
    rows = []
    for name, m in models.items():
        if name in NEEDS_SCALING:
            Xi = X_encoded.copy()
            Xi[NUM_COLS] = scaler.transform(Xi[NUM_COLS])
            Xi = Xi.values
        else:
            Xi = X_encoded.values
        yp = m.predict(Xi)
        ypr = m.predict_proba(Xi)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_true, yp),
            "AUC": roc_auc_score(y_true, ypr),
            "Precision": precision_score(y_true, yp),
            "Recall": recall_score(y_true, yp),
            "F1": f1_score(y_true, yp),
            "MCC": matthews_corrcoef(y_true, yp),
        })
    comparison_df = pd.DataFrame(rows).set_index("Model").round(4)
    st.dataframe(comparison_df, width="stretch")

# ---------------------------------------------------------------------------
# Section 4: Confusion matrix & classification report
# ---------------------------------------------------------------------------
st.header("4. Confusion matrix & classification report")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES, ax=ax
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    st.pyplot(fig)

with col2:
    st.subheader("Classification report")
    report = classification_report(
        y_true, y_pred, target_names=TARGET_NAMES, output_dict=True
    )
    st.dataframe(pd.DataFrame(report).transpose().round(3), width="stretch")

st.divider()
st.caption(
    "Machine Learning - Assignment 2 | Prabhat Kumar Dubey"
)
