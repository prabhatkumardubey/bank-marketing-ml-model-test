import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(page_title="Bank Marketing - Term Deposit Subscription Prediction", layout="wide")

# ---------------------------------------------------------------------------
# Load artifacts (models, scaler, metadata) once
# ---------------------------------------------------------------------------
st.title("🏦 Bank Marketing - Term Deposit Subscription Prediction")

# ---------------------------------------------------------------------------
# Feature-1: Dataset upload option (CSV file) - upload RAW test data
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Re-create the SAME one-hot encoding used at training time, then reindex
# to the exact training-time columns. a small uploaded batch may not contain
# every category the model was trained on
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Feature-2: Model selection dropdown
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
# Feature-3: Display of evaluation metrics
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Compare against all models side-by-side
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Feature-4: Confusion matrix or classification report
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Machine Learning - Assignment 2 by Prabhat Kumar Dubey"
)
