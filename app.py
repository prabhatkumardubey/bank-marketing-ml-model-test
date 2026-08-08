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

st.title("🏦 Bank Term Deposit Subscription Prediction")
st.divider()
st.caption(
    "Machine Learning - Assignment 2 by Prabhat Kumar Dubey"
)
