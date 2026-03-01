from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
MODEL_FILE = "churn_model.joblib"

FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [c for c in FEATURE_COLUMNS if c not in NUMERIC_FEATURES]

model_pipeline = None
metrics_data = None


def safe_int(value, default=0):
    try:
        return int(value)
    except:
        return default


def safe_float(value, default=0.0):
    try:
        return float(value)
    except:
        return default


def load_and_clean_data():
    df = pd.read_csv(DATA_URL)

    # Drop ID
    df = df.drop(columns=["customerID"])

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Impute TotalCharges
    mask = df["TotalCharges"].isna()
    df.loc[mask, "TotalCharges"] = df.loc[mask, "tenure"] * df.loc[mask, "MonthlyCharges"]
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Replace service texts
    replace_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    for col in replace_cols:
        df[col] = df[col].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def train_and_evaluate(save_model=True):
    df = load_and_clean_data()

    X = df[FEATURE_COLUMNS].copy()
    y = df["Churn"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", clf)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test))
    }

    if save_model:
        joblib.dump({"pipeline": pipeline, "metrics": metrics}, MODEL_FILE)

    return pipeline, metrics


def load_model_or_train():
    if os.path.exists(MODEL_FILE):
        bundle = joblib.load(MODEL_FILE)
        return bundle["pipeline"], bundle["metrics"]
    return train_and_evaluate(save_model=True)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", metrics=metrics_data)


@app.route("/train", methods=["POST"])
def retrain():
    global model_pipeline, metrics_data
    model_pipeline, metrics_data = train_and_evaluate(save_model=True)
    return render_template("train.html", metrics=metrics_data)


@app.route("/predict", methods=["POST"])
def predict():
    global model_pipeline

    gender = request.form.get("gender", "Male")
    senior = safe_int(request.form.get("SeniorCitizen", "0"), 0)
    partner = request.form.get("Partner", "No")
    dependents = request.form.get("Dependents", "No")
    tenure = safe_int(request.form.get("tenure", "0"), 0)
    phone = request.form.get("PhoneService", "Yes")
    multiple = request.form.get("MultipleLines", "No")
    internet = request.form.get("InternetService", "DSL")
    online_sec = request.form.get("OnlineSecurity", "No")
    online_backup = request.form.get("OnlineBackup", "No")
    device = request.form.get("DeviceProtection", "No")
    tech = request.form.get("TechSupport", "No")
    tv = request.form.get("StreamingTV", "No")
    movies = request.form.get("StreamingMovies", "No")
    contract = request.form.get("Contract", "Month-to-month")
    paperless = request.form.get("PaperlessBilling", "Yes")
    payment = request.form.get("PaymentMethod", "Electronic check")
    monthly = safe_float(request.form.get("MonthlyCharges", "0"), 0.0)

    total_raw = request.form.get("TotalCharges", "").strip()
    if total_raw == "":
        total = tenure * monthly
    else:
        total = safe_float(total_raw, tenure * monthly)

    row = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    input_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    pred = int(model_pipeline.predict(input_df)[0])
    prob = float(model_pipeline.predict_proba(input_df)[0][1])

    prediction_text = "Churn: YES" if pred == 1 else "Churn: NO"

    return render_template(
        "result.html",
        prediction=prediction_text,
        probability=round(prob, 4),
        data=row
    )


if __name__ == "__main__":
    model_pipeline, metrics_data = load_model_or_train()
    app.run(debug=True)