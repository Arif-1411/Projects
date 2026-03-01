

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)

app = Flask(__name__)
MODEL_FILE = "kmeans_model.joblib"
model_bundle = None


# ============================================================
# EMBEDDED DATASET (200 rows — Mall Customers)
# ============================================================

def get_mall_data():
    """Mall Customers dataset embedded directly."""
    np.random.seed(42)

    # 5 realistic customer segments
    data = []

    # Segment 1: Young, Low Income, High Spending (40 customers)
    for _ in range(40):
        data.append({
            "CustomerID": len(data)+1,
            "Gender": np.random.choice(["Male", "Female"]),
            "Age": np.random.randint(18, 35),
            "Annual Income (k$)": np.random.randint(15, 40),
            "Spending Score (1-100)": np.random.randint(60, 99)
        })

    # Segment 2: Young, High Income, High Spending (40 customers)
    for _ in range(40):
        data.append({
            "CustomerID": len(data)+1,
            "Gender": np.random.choice(["Male", "Female"]),
            "Age": np.random.randint(25, 40),
            "Annual Income (k$)": np.random.randint(70, 137),
            "Spending Score (1-100)": np.random.randint(65, 99)
        })

    # Segment 3: Middle Age, Average Income, Average Spending (40 customers)
    for _ in range(40):
        data.append({
            "CustomerID": len(data)+1,
            "Gender": np.random.choice(["Male", "Female"]),
            "Age": np.random.randint(30, 55),
            "Annual Income (k$)": np.random.randint(40, 75),
            "Spending Score (1-100)": np.random.randint(35, 65)
        })

    # Segment 4: Older, High Income, Low Spending (40 customers)
    for _ in range(40):
        data.append({
            "CustomerID": len(data)+1,
            "Gender": np.random.choice(["Male", "Female"]),
            "Age": np.random.randint(40, 68),
            "Annual Income (k$)": np.random.randint(70, 137),
            "Spending Score (1-100)": np.random.randint(1, 35)
        })

    # Segment 5: Older, Low Income, Low Spending (40 customers)
    for _ in range(40):
        data.append({
            "CustomerID": len(data)+1,
            "Gender": np.random.choice(["Male", "Female"]),
            "Age": np.random.randint(45, 70),
            "Annual Income (k$)": np.random.randint(15, 40),
            "Spending Score (1-100)": np.random.randint(1, 40)
        })

    df = pd.DataFrame(data)
    print(f" Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):
    df_clean = df.copy()

    if "CustomerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["CustomerID"])

    le = LabelEncoder()
    df_clean["Gender"] = le.fit_transform(df_clean["Gender"])

    df_clean = df_clean.fillna(df_clean.median())

    before = len(df_clean)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    print(f"   Duplicates removed: {before - len(df_clean)}")

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_clean),
        columns=df_clean.columns
    )

    print(f"    Shape: {df_scaled.shape}")
    return df_clean, df_scaled, scaler, le


# ============================================================
# FIND BEST K
# ============================================================

def find_best_k(df_scaled):
    k_results = {}

    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(df_scaled)

        k_results[k] = {
            "inertia": round(float(km.inertia_), 2),
            "silhouette": round(float(silhouette_score(df_scaled, labels)), 4),
            "calinski_harabasz": round(float(calinski_harabasz_score(df_scaled, labels)), 2),
            "davies_bouldin": round(float(davies_bouldin_score(df_scaled, labels)), 4)
        }
        print(f"   K={k}: Sil={k_results[k]['silhouette']}, "
              f"Inertia={k_results[k]['inertia']}")

    best_k = max(k_results, key=lambda k: k_results[k]["silhouette"])
    print(f"    Best K = {best_k}")
    return best_k, k_results


# ============================================================
# BUILD PROFILES
# ============================================================

def build_profiles(df_clean, labels):
    df_temp = df_clean.copy()
    df_temp["Cluster"] = labels
    overall_mean = df_clean.mean()
    profiles = []

    for cl in sorted(df_temp["Cluster"].unique()):
        cl_data = df_temp[df_temp["Cluster"] == cl]
        cl_mean = cl_data.drop(columns=["Cluster"]).mean()

        high = [c for c in df_clean.columns if c != "Gender"
                and cl_mean[c] > overall_mean[c] * 1.15]
        low = [c for c in df_clean.columns if c != "Gender"
               and cl_mean[c] < overall_mean[c] * 0.85]

        gender_val = cl_data["Gender"].mode().values[0]
        gender_text = "Male majority" if gender_val == 1 else "Female majority"

        # Auto naming
        name = f"Cluster {cl}"
        inc = "Annual Income (k$)"
        sps = "Spending Score (1-100)"

        if sps in high and inc in high:
            name = "High Income High Spenders"
        elif sps in high and inc in low:
            name = "Low Income High Spenders"
        elif sps in low and inc in high:
            name = "High Income Low Spenders"
        elif sps in low and inc in low:
            name = "Low Income Low Spenders"
        elif "Age" in high:
            name = "Older Customers"
        elif "Age" in low:
            name = "Younger Customers"
        else:
            name = "Average Customers"

        profiles.append({
            "cluster": int(cl),
            "name": name,
            "count": int(len(cl_data)),
            "pct": round(len(cl_data) / len(df_temp) * 100, 1),
            "gender": gender_text,
            "means": {
                "Gender": round(float(cl_mean["Gender"]), 2),
                "Age": round(float(cl_mean["Age"]), 1),
                "Annual Income (k$)": round(float(cl_mean[inc]), 1),
                "Spending Score (1-100)": round(float(cl_mean[sps]), 1)
            },
            "high": high,
            "low": low
        })

    return profiles


# ============================================================
# FULL TRAINING PIPELINE
# ============================================================

def train_full_pipeline():
   

    df_raw = get_mall_data()
    df_clean, df_scaled, scaler, le = clean_data(df_raw)
    best_k, k_results = find_best_k(df_scaled)

    print(f"\n Training final KMeans (K={best_k})...")
    model = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = model.fit_predict(df_scaled)

    sil = silhouette_score(df_scaled, labels)
    ch = calinski_harabasz_score(df_scaled, labels)
    db = davies_bouldin_score(df_scaled, labels)

    profiles = build_profiles(df_clean, labels)

    centers_original = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=df_clean.columns
    )

    bundle = {
        "model": model,
        "scaler": scaler,
        "label_encoder": le,
        "best_k": best_k,
        "columns": list(df_scaled.columns),
        "centers_original": centers_original.round(1).to_dict("index"),
        "metrics": {
            "best_k": best_k,
            "n_samples": len(df_scaled),
            "silhouette": round(float(sil), 4),
            "calinski_harabasz": round(float(ch), 2),
            "davies_bouldin": round(float(db), 4),
            "inertia": round(float(model.inertia_), 2)
        },
        "k_results": {str(k): v for k, v in k_results.items()},
        "profiles": profiles
    }

    joblib.dump(bundle, MODEL_FILE)
    print(f"\n Saved: {MODEL_FILE}")
    return bundle


def load_or_train():
    if os.path.exists(MODEL_FILE):
        print("📂 Loading saved model...")
        return joblib.load(MODEL_FILE)
    return train_full_pipeline()


# ============================================================
# PREDICT
# ============================================================

def predict_customer(bundle, gender, age, income, spending):
    gender_encoded = 0 if gender.lower() == "female" else 1

    new_data = pd.DataFrame([{
        "Gender": gender_encoded,
        "Age": age,
        "Annual Income (k$)": income,
        "Spending Score (1-100)": spending
    }])

    new_scaled = pd.DataFrame(
        bundle["scaler"].transform(new_data),
        columns=new_data.columns
    )

    predicted = int(bundle["model"].predict(new_scaled)[0])

    distances = {}
    point = new_scaled.values[0]
    for i, center in enumerate(bundle["model"].cluster_centers_):
        dist = round(float(np.sqrt(np.sum((point - center) ** 2))), 4)
        distances[i] = dist

    profile = None
    for p in bundle["profiles"]:
        if p["cluster"] == predicted:
            profile = p
            break

    return predicted, distances, profile


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("index.html", bundle=model_bundle)


@app.route("/train", methods=["POST"])
def retrain():
    global model_bundle
    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)
    model_bundle = train_full_pipeline()
    return render_template("train.html", bundle=model_bundle)


@app.route("/predict", methods=["POST"])
def predict():
    gender = request.form.get("Gender", "Female")
    age = float(request.form.get("Age", 30))
    income = float(request.form.get("Income", 50))
    spending = float(request.form.get("Spending", 50))

    predicted, distances, profile = predict_customer(
        model_bundle, gender, age, income, spending
    )

    input_data = {
        "Gender": gender,
        "Age": int(age),
        "Annual Income (k$)": int(income),
        "Spending Score (1-100)": int(spending)
    }

    return render_template(
        "result.html",
        predicted=predicted,
        distances=distances,
        profile=profile,
        input_data=input_data
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    model_bundle = load_or_train()
    print(f"\n🌐 Open: http://127.0.0.1:5000")
    app.run(debug=True)