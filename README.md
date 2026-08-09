# Bank Term Deposit Subscription Prediction

This is my second assignment for the Machine Learning course in BITS Pilani WILP (M.Tech AI/ML). In this project, I built and compared five classification models to predict whether a bank customer will subscribe to a term deposit. I also made a Streamlit app where you can upload test data and check how each model performs.

## Problem Statement

A Portuguese bank runs phone marketing campaigns to sell term deposits. The goal is to predict whether a client will subscribe to a term deposit **before** the call is made, using only information that is available before contacting the client. The final best model is also exposed through an interactive Streamlit app.

## Dataset Description

- **Original UCI file:** `bank-additional-full.csv`
- **Local file in repo:** `bank_marketing_raw.csv` (same data, just renamed for clarity)
- **Source:** UCI Machine Learning Repository, dataset id 222 (Moro, Rita & Cortez, 2014)
- **Instances:** 41,188
- **Input features used:** 19 in total — 10 categorical (job, marital, education, default, housing, loan, contact, month, day of week, poutcome) and 9 numeric (age, campaign, pdays, previous, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed).
- **Excluded column:** `duration` is not used because it is only known after the call ends; using it would cause data leakage.
- **Target:** `y` — `yes` means subscribed (mapped to 1), `no` means not subscribed (mapped to 0).
- **Class balance:** 36,548 no / 4,640 yes (about 88.7% / 11.3%).

## Preprocessing

1. **Remove leakage column:** `duration` is dropped because it is only known after the call ends.
2. **Target encoding:** `y` is mapped to `yes = 1` and `no = 0`.
3. **One-hot encoding:** The 10 categorical columns are encoded with `drop_first=True`, giving 52 encoded features.
4. **Scaling:** Numeric columns are standardized for **Logistic Regression** and **kNN** using `StandardScaler`. The other models use the original numeric values.
5. **Train/test split:** 80% of the data is used for training and 20% for testing, with stratification to keep the same yes/no ratio in both sets.

## Evaluation Metrics

Each model is checked with 6 metrics:

- **Accuracy**
- **AUC (Area Under ROC Curve)**
- **Precision**
- **Recall**
- **F1 Score**
- **MCC (Matthews Correlation Coefficient)**

Because the data is imbalanced (88.7% "no"), **Accuracy** alone can be misleading. A model that always predicts "no" already gets 88.7% accuracy, so **AUC** and **MCC** give a fairer picture of how well the model really performs.

## GitHub Repository Link

`https://github.com/prabhatkumardubey/bank-marketing-ml-model-test`

## Models Used

Five classification models were trained and compared on the same 80/20 stratified split:

1. **Logistic Regression**
2. **Decision Tree**
3. **kNN** (k-Nearest Neighbors, `k=15`)
4. **Naive Bayes**
5. **Random Forest (Ensemble)** (`200` trees, `max_depth=10`)

Logistic Regression and kNN used standardized numeric features. The other models used the original numeric features.

### Comparison Table

*(Precision, Recall and F1 are calculated for the positive class `1 = yes/subscribed`.)*

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Logistic Regression | 0.9014 | 0.8007 | 0.6933 | 0.2241 | 0.3388 | 0.3571 |
| Decision Tree | 0.8995 | 0.7864 | 0.6263 | 0.2672 | 0.3746 | 0.3651 |
| kNN | 0.8991 | 0.7743 | 0.6293 | 0.2543 | 0.3622 | 0.3569 |
| Naive Bayes | 0.8694 | 0.7730 | 0.4237 | 0.4429 | 0.4331 | 0.3594 |
| Random Forest (Ensemble) | 0.9022 | 0.8089 | 0.7075 | 0.2241 | 0.3404 | 0.3619 |

*Baseline: always predicting "no" gives 0.8873 accuracy and 0.0 for all other metrics. Every model beats it on AUC and MCC, but only slightly on accuracy.*

### Observations

- **Logistic Regression:** Accuracy is only a little better than the always-"no" baseline. MCC and AUC show it has learned something useful, but recall is low (0.2241), so it misses about 3 out of 4 real subscribers.
- **Decision Tree:** Best MCC (0.3651) among all models. This shows that MCC and AUC can rank models differently because they measure different things.
- **kNN:** Average results. With 52 one-hot encoded columns, Euclidean distance does not work very well because the data becomes sparse.
- **Naive Bayes:** Lowest accuracy and precision, but best recall (0.4429). It catches more real subscribers but also produces more false alarms.
- **Random Forest (Ensemble):** Best accuracy, best AUC and best precision. It is the strongest overall, but its recall is tied for lowest, so it is conservative — when it says "yes" it is usually right, but it misses many real subscribers.

**Overall winner:** **Random Forest (Ensemble)** is the best overall because it wins on 3 out of 6 metrics, including AUC, which is less affected by class imbalance. If the bank wants to catch as many real subscribers as possible (even if it means more wasted calls), then **Naive Bayes** is a better choice.

---

## Repository Structure

```
project-folder/
├── .gitattributes             # normalizes line endings and marks binary files
├── .gitignore                 # excludes local/generated files from commits
├── app.py                     # Streamlit app for predictions
├── requirements.txt           # Python packages needed
├── README.md                  # this file
├── bank_marketing_raw.csv     # original UCI dataset (renamed from bank-additional-full.csv)
├── test_data.csv              # held-out 20% test split (raw, no duration)
└── model/
    ├── train_models.py        # trains all 5 models and computes metrics
    ├── Logistic_Regression.joblib
    ├── Decision_Tree.joblib
    ├── kNN.joblib
    ├── Naive_Bayes.joblib
    ├── Random_Forest_Ensemble.joblib
    ├── scaler.joblib
    ├── meta.json              # feature, encoding, and scaling info used by app.py
    ├── results_summary.csv
    └── confusion_matrices.json
```

The `.joblib`, `.json`, and `.csv` files inside `model/` and `test_data.csv` are generated when `train_models.py` runs.

## How to Run Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the models and create the artifacts:
   ```bash
   python model/train_models.py
   ```
3. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## How to Deploy on Streamlit Cloud

1. Run `python model/train_models.py` locally so the `model/` artifacts and `test_data.csv` are created.
2. Push all files to GitHub, including the generated `model/` and `test_data.csv` files.
3. Connect the GitHub repo to Streamlit Cloud and deploy.

Without the `model/` artifacts, `app.py` will fail with `FileNotFoundError`.

## Tools and Technologies

- Python 3
- pandas, NumPy
- scikit-learn
- joblib
- Streamlit
- Matplotlib, Seaborn

## Author

**Prabhat Kumar Dubey**  
BITS Pilani WILP — M.Tech (AI/ML)  
Machine Learning — Assignment 2
