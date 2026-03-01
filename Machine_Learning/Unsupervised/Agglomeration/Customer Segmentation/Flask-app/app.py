import pandas as pd
import numpy as np
import os
import ssl
import joblib
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from scipy.cluster.hierarchy import linkage, fcluster

app = Flask(__name__)

MODEL_FILE = "agglom_model.joblib"

SPENDING_COLS = [
    "Fresh", "Milk", "Grocery", "Frozen",
    "Detergents_Paper", "Delicassen"
]

# Global variables
model_bundle = None


# ============================================================
# DATA DOWNLOAD (Multiple fallbacks)
# ============================================================

def download_data():

    # Fix SSL issues on Windows
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except:
        pass

    urls = [
        "https://raw.githubusercontent.com/dsrscientist/dataset1/master/Wholesale_customers_data.csv",
        "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/Wholesale%20customers%20data.csv",
        "https://raw.githubusercontent.com/pranavuikey/data/master/Wholesale%20customers%20data.csv",
    ]

    for i, url in enumerate(urls):
        try:
            print(f"   Trying URL {i+1}...")
            df = pd.read_csv(url, timeout=10)
            if len(df) > 100 and "Fresh" in df.columns:
                print(f"    Downloaded from URL {i+1}")
                return df
        except Exception as e:
            print(f"    URL {i+1} failed: {e}")
            continue

    # FALLBACK: Generate data locally using sklearn
    print("   ⚠️ All URLs failed. Using sklearn Wine dataset as fallback...")
    from sklearn.datasets import load_wine
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)

    # Rename columns to match Wholesale format
    df = df.iloc[:, :6]
    df.columns = SPENDING_COLS

    # Scale up to look like spending data
    df = (df * 100).round(0).astype(int)
    df["Channel"] = np.random.choice([1, 2], size=len(df))
    df["Region"] = np.random.choice([1, 2, 3], size=len(df))
    print(f"    Fallback dataset created ({len(df)} rows)")
    return df


# ============================================================
# DATA CLEANING
# ============================================================

def clean_data(df):

    df_clean = df.copy()

    # Save Channel/Region for verification
    channel = df_clean.get("Channel", pd.Series([0]*len(df_clean)))
    region = df_clean.get("Region", pd.Series([0]*len(df_clean)))

    # Drop Channel & Region for clustering
    for col in ["Channel", "Region"]:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])

    # Keep only spending columns
    available_cols = [c for c in SPENDING_COLS if c in df_clean.columns]
    df_clean = df_clean[available_cols].copy()

    # Convert to numeric
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Fill missing
    df_clean = df_clean.fillna(df_clean.median())

    # Remove duplicates
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    print(f"   Removed {before - len(df_clean)} duplicates")

    # IQR outlier capping
    for col in df_clean.columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        df_clean[col] = df_clean[col].clip(
            lower=Q1 - 1.5 * IQR,
            upper=Q3 + 1.5 * IQR
        )

    # Log transform
    df_log = df_clean.apply(np.log1p)

    # Scale
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_log),
        columns=df_log.columns
    )

    print(f"    Cleaned: {df_scaled.shape[0]} rows, {df_scaled.shape[1]} cols")
    return df_clean, df_scaled, scaler, channel, region


# ============================================================
# TRAIN & EVALUATE
# ============================================================

def find_best_k(df_scaled, k_range=range(2, 11)):
    scores = {}
    for k in k_range:
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = agg.fit_predict(df_scaled)
        scores[k] = silhouette_score(df_scaled, labels)
        print(f"   K={k}: Silhouette={scores[k]:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"    Best K = {best_k}")
    return best_k, scores


def compare_linkages(df_scaled, best_k):
    """Compare different linkage methods."""
    linkages = ["ward", "complete", "average", "single"]
    results = []
    best_linkage = ""
    best_score = -1
    best_labels = None

    for link in linkages:
        try:
            agg = AgglomerativeClustering(n_clusters=best_k, linkage=link)
            labels = agg.fit_predict(df_scaled)
            sil = silhouette_score(df_scaled, labels)
            ch = calinski_harabasz_score(df_scaled, labels)
            db = davies_bouldin_score(df_scaled, labels)

            results.append({
                "linkage": link,
                "silhouette": round(sil, 4),
                "calinski_harabasz": round(ch, 2),
                "davies_bouldin": round(db, 4)
            })

            if sil > best_score:
                best_score = sil
                best_linkage = link
                best_labels = labels

            print(f"   {link:<10s}: Sil={sil:.4f} CH={ch:.1f} DB={db:.4f}")
        except Exception as e:
            print(f"   {link}: ERROR - {e}")

    print(f"  Best linkage: {best_linkage}")
    return best_linkage, best_labels, results


def build_cluster_profiles(df_clean, labels):
    """Build spending profiles for each cluster."""
    df_temp = df_clean.copy()
    df_temp["Cluster"] = labels

    profiles = []
    overall_mean = df_clean.mean()

    for cl in sorted(df_temp["Cluster"].unique()):
        cl_data = df_temp[df_temp["Cluster"] == cl]
        cl_mean = cl_data[SPENDING_COLS].mean()

        high = [c for c in SPENDING_COLS
                if c in cl_mean.index and cl_mean[c] > overall_mean.get(c, 0) * 1.2]
        low = [c for c in SPENDING_COLS
               if c in cl_mean.index and cl_mean[c] < overall_mean.get(c, 0) * 0.8]

        profiles.append({
            "cluster": int(cl),
            "count": int(len(cl_data)),
            "pct": round(len(cl_data) / len(df_temp) * 100, 1),
            "means": {c: round(float(cl_mean.get(c, 0)), 1)
                      for c in SPENDING_COLS if c in cl_mean.index},
            "high_spending": high,
            "low_spending": low
        })

    return profiles


def train_full_pipeline():
    """Complete training pipeline."""
    print("\n" + "=" * 50)
    print("=" * 50)

    # Download
    df_raw = download_data()
    print(f"   Raw shape: {df_raw.shape}")

    # Clean
    df_clean, df_scaled, scaler, channel, region = clean_data(df_raw)

    # Find best K
    best_k, k_scores = find_best_k(df_scaled)

    # Compare linkages
    best_linkage, best_labels, linkage_results = compare_linkages(
        df_scaled, best_k
    )

    # Final metrics
    sil = silhouette_score(df_scaled, best_labels)
    ch = calinski_harabasz_score(df_scaled, best_labels)
    db = davies_bouldin_score(df_scaled, best_labels)

    # Cluster profiles
    profiles = build_cluster_profiles(df_clean, best_labels)

    # Compute cluster centers for prediction
    df_scaled_copy = df_scaled.copy()
    df_scaled_copy["Cluster"] = best_labels
    cluster_centers = df_scaled_copy.groupby("Cluster").mean()

    # IQR stats for capping new data
    iqr_stats = {}
    for col in df_clean.columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        iqr_stats[col] = {
            "lower": Q1 - 1.5 * IQR,
            "upper": Q3 + 1.5 * IQR
        }

    # Bundle everything
    bundle = {
        "best_k": best_k,
        "best_linkage": best_linkage,
        "scaler": scaler,
        "cluster_centers": cluster_centers,
        "iqr_stats": iqr_stats,
        "columns": list(df_scaled.columns),
        "metrics": {
            "silhouette": round(sil, 4),
            "calinski_harabasz": round(ch, 2),
            "davies_bouldin": round(db, 4),
            "n_clusters": best_k,
            "n_samples": len(df_scaled),
            "best_linkage": best_linkage
        },
        "k_scores": {str(k): round(v, 4) for k, v in k_scores.items()},
        "linkage_results": linkage_results,
        "profiles": profiles
    }

    # Save
    joblib.dump(bundle, MODEL_FILE)

    return bundle


def load_or_train():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return train_full_pipeline()


# ============================================================
# PREDICTION
# ============================================================

def predict_cluster(bundle, fresh, milk, grocery, frozen,
                    detergents, delicassen):
    """Predict cluster for new customer."""

    new_data = pd.DataFrame([{
        "Fresh": fresh,
        "Milk": milk,
        "Grocery": grocery,
        "Frozen": frozen,
        "Detergents_Paper": detergents,
        "Delicassen": delicassen
    }])

    # Keep only columns the model knows
    available_cols = [c for c in bundle["columns"] if c in new_data.columns]
    new_data = new_data[available_cols]

    # IQR capping
    for col in new_data.columns:
        if col in bundle["iqr_stats"]:
            stats = bundle["iqr_stats"][col]
            new_data[col] = new_data[col].clip(
                lower=stats["lower"], upper=stats["upper"]
            )

    # Log transform
    new_data_log = new_data.apply(np.log1p)

    # Scale
    new_data_scaled = pd.DataFrame(
        bundle["scaler"].transform(new_data_log),
        columns=new_data_log.columns
    )

    # Find nearest cluster center
    centers = bundle["cluster_centers"]
    distances = {}

    for cl in centers.index:
        center_vals = centers.loc[cl][available_cols].values
        point_vals = new_data_scaled[available_cols].values[0]
        dist = np.sqrt(np.sum((point_vals - center_vals) ** 2))
        distances[int(cl)] = round(float(dist), 4)

    predicted = min(distances, key=distances.get)

    # Get profile for predicted cluster
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

    # Delete old model
    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)

    model_bundle = train_full_pipeline()
    return render_template("train.html", bundle=model_bundle)


@app.route("/predict", methods=["POST"])
def predict():
    fresh = float(request.form.get("Fresh", 0))
    milk = float(request.form.get("Milk", 0))
    grocery = float(request.form.get("Grocery", 0))
    frozen = float(request.form.get("Frozen", 0))
    detergents = float(request.form.get("Detergents_Paper", 0))
    delicassen = float(request.form.get("Delicassen", 0))

    predicted, distances, profile = predict_cluster(
        model_bundle, fresh, milk, grocery,
        frozen, detergents, delicassen
    )

    input_data = {
        "Fresh": fresh,
        "Milk": milk,
        "Grocery": grocery,
        "Frozen": frozen,
        "Detergents_Paper": detergents,
        "Delicassen": delicassen
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
    print(f"   Open: http://127.0.0.1:5000")
    app.run(debug=True)