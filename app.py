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
# Load artifacts (models, scaler, metadata) once
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
TARGET_NAMES = meta["target_names"]          # ["no(0)", "yes(1)"]
NEEDS_SCALING = set(meta["needs_scaling"])
TARGET_COL = meta["target_col"]              # "y"

st.title("🏦 Bank Term Deposit Subscription Prediction")
st.caption(
    "Dataset: UCI Bank Marketing (used bank-additional-full) — 19 raw features "
    "(10 categorical, 9 numeric; excluded duration as there is a known leakage "
    "column), 41,188 instances, binary classification (subscribed vs. not subscribed)."
)
st.info(
    "📌 Reference point: always predicting **'no'** already scores **88.7% accuracy** "
    "on this dataset (only ~11.3% of clients subscribe). Keep that in mind when "
    "reading the Accuracy numbers below — AUC and MCC tell a more honest story.",
    icon="📌",
)

# ---------------------------------------------------------------------------
# Feature 1: Dataset upload option (CSV file) - upload RAW test data
# ---------------------------------------------------------------------------
st.header("1. Upload test data (CSV file for prediction)")
st.write(
    f"Upload a CSV with the {len(RAW_COLS)} original raw columns (no `duration`) "
    f"plus a `{TARGET_COL}` column (`yes`/`no`). A ready-made `test_data.csv` is "
    "included in this repository."
)
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # UCI distributes Bank Marketing as a semicolon-delimited file.
    # using comma first and then if fails will use semicolon
    # if the expected columns are missing.
    try:
        df = pd.read_csv(uploaded_file)
        if not set(RAW_COLS + [TARGET_COL]).issubset(df.columns):
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=";")
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")
        st.stop()
else:
    st.info("No file uploaded yet - showing a preview using the bundled test_data.csv.")
    df = pd.read_csv("test_data.csv")

missing_cols = [c for c in RAW_COLS if c not in df.columns]
if missing_cols or TARGET_COL not in df.columns:
    st.error(
        f"Uploaded CSV is missing required columns: "
        f"{missing_cols + ([TARGET_COL] if TARGET_COL not in df.columns else [])}"
    )
    st.stop()

st.dataframe(df, width="stretch", height=350)
st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

# ---------------------------------------------------------------------------
# Re-create the SAME one-hot encoding used at training time, then reindex
# to the exact training-time columns (a small uploaded batch may not contain
# every category the model was trained on).
# ---------------------------------------------------------------------------
X_raw = df[RAW_COLS]
y_true = df[TARGET_COL].map({"yes": 1, "no": 0})

X_encoded = pd.get_dummies(X_raw, columns=CAT_COLS, drop_first=True)
X_encoded = X_encoded.reindex(columns=ENCODED_COLS, fill_value=0)

# ---------------------------------------------------------------------------
# Feature 2: Model selection dropdown
# ---------------------------------------------------------------------------
st.header("2. Select a model")
model_name = st.selectbox("Choose a classification model", list(models.keys()))
model = models[model_name]

if model_name in NEEDS_SCALING:
    X_input = X_encoded.copy()
    X_input[NUM_COLS] = scaler.transform(X_input[NUM_COLS])
    X_input = X_input.values
else:
    X_input = X_encoded.values

y_pred = model.predict(X_input)
y_proba = model.predict_proba(X_input)[:, 1]  # P(class = 1 = subscribed)

# ---------------------------------------------------------------------------
# Feature 3: Display of evaluation metrics
# ---------------------------------------------------------------------------
st.header("3. Evaluation metrics")
st.caption("Positive class = 1 ('yes' / subscribed) — scikit-learn's default, and the business-relevant class here.")

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
st.caption(f"⚠️ 'Always predict no' baseline accuracy on this uploaded data: **{baseline_acc:.4f}** — compare against the Accuracy above.")

# ---------------------------------------------------------------------------
# Compare against all 5 models side-by-side
# ---------------------------------------------------------------------------
with st.expander("Compare all 5 models on this uploaded data"):
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
# Feature 4: Confusion matrix or classification report
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
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with col2:
    st.subheader("Classification report")
    report = classification_report(
        y_true, y_pred, target_names=TARGET_NAMES, output_dict=True
    )
    st.dataframe(pd.DataFrame(report).transpose().round(3), width="stretch")

st.divider()
st.caption(
    "Machine Learning - Assignment 2 by Prabhat Kumar Dubey"
)
