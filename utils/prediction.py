"""
prediction.py
--------------
Core ML pipeline: preprocessing, training, evaluation, and inference utilities
for the placement prediction model. Designed to be imported by the Streamlit
app and the training notebook alike.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

FEATURES = [
    "CGPA", "Tenth_Percentage", "Twelfth_Percentage", "Internships",
    "Projects", "Coding_Skills", "Communication_Skills", "Aptitude_Score",
    "Technical_Skills", "Backlogs", "Certifications", "Hackathon_Participation",
]
TARGET = "Placement_Status"


def preprocess(df: pd.DataFrame):
    """
    Cleans and prepares the dataframe for modeling.
    Steps:
      1. Impute missing numeric values with column median (robust to outliers).
      2. Clip extreme outliers using the IQR method (cap, don't drop, to avoid
         losing data).
      3. Scale features with StandardScaler (mean=0, std=1) so models like
         Logistic Regression / SVM converge well and aren't biased by scale.
    No categorical encoding is needed here since all features are numeric,
    but the function is structured to make adding categorical columns easy.
    """
    df = df.copy()

    # 1. Missing value imputation (median is robust to skew/outliers)
    for col in FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # 2. Outlier handling via IQR clipping
    for col in FEATURES:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = df[col].clip(lower, upper)

    return df


def split_and_scale(df: pd.DataFrame, test_size=0.2, random_state=42):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200),
        "SVM": SVC(probability=True, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            random_state=42, eval_metric="logloss", use_label_encoder=False
        )
    return models


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def train_all_models(X_train, X_test, y_train, y_test):
    results = {}
    fitted = {}
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test)
        fitted[name] = model
    return fitted, results


def tune_best_model(best_name, X_train, y_train):
    """
    Hyperparameter tuning via GridSearchCV for the top-performing model family.
    Add/adjust grids as needed for the model that wins on your data.
    """
    param_grids = {
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10, 15],
                "min_samples_split": [2, 5, 10],
            },
        ),
        "XGBoost": (
            XGBClassifier(random_state=42, eval_metric="logloss") if HAS_XGB else None,
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
            },
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"C": [0.01, 0.1, 1, 10, 100]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]},
        ),
        "SVM": (
            SVC(probability=True, random_state=42),
            {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
        ),
    }
    base_model, grid = param_grids[best_name]
    search = GridSearchCV(base_model, grid, cv=5, scoring="f1", n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def predict_placement(model, scaler, student_dict):
    """
    student_dict: dict with keys matching FEATURES.
    Returns (prediction_label, probability_percentage, confidence_text)
    """
    x = pd.DataFrame([student_dict])[FEATURES]
    x_scaled = scaler.transform(x)
    prob = model.predict_proba(x_scaled)[0, 1]
    pred = "Likely to be Placed" if prob >= 0.5 else "Unlikely to be Placed (Needs Improvement)"
    if prob >= 0.8 or prob <= 0.2:
        confidence = "High"
    elif prob >= 0.65 or prob <= 0.35:
        confidence = "Moderate"
    else:
        confidence = "Low"
    return pred, round(prob * 100, 1), confidence


def save_artifacts(model, scaler, path="placement_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "features": FEATURES}, f)


def load_artifacts(path="placement_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
