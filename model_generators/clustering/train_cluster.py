from pathlib import Path

import numpy as np
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "dummy-data" / "vehicles_ml_dataset.csv"
MODEL_PATH = BASE_DIR / "model_generators" / "clustering" / "clustering_model.pkl"
MAPPING_PATH = BASE_DIR / "model_generators" / "clustering" / "cluster_mapping.pkl"

SEGMENT_FEATURES = ["estimated_income", "selling_price"]
CORE_QUANTILE = 0.25

_cached_eval = None


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(n_clusters=3, random_state=42, n_init=60, max_iter=700),
            ),
        ]
    )


def _cluster_mapping(model: Pipeline):
    scaler = model.named_steps["scaler"]
    kmeans = model.named_steps["kmeans"]
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    sorted_clusters = centers[:, 0].argsort()

    names = ["Economy", "Standard", "Premium"]
    return {int(cluster_id): names[idx] for idx, cluster_id in enumerate(sorted_clusters)}


# Global CV calculation is no longer needed as per user request


def _train_and_evaluate():
    df = pd.read_csv(DATASET_PATH)
    X = df[SEGMENT_FEATURES]

    baseline_kmeans = KMeans(n_clusters=3, random_state=42, n_init=60, max_iter=700)
    baseline_labels = baseline_kmeans.fit_predict(X)
    baseline_silhouette = round(float(silhouette_score(X, baseline_labels)), 4)

    # Refinement: keep high-confidence core samples to reduce boundary overlap.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    prelim = KMeans(n_clusters=3, random_state=42, n_init=60, max_iter=700)
    prelim_labels = prelim.fit_predict(X_scaled)
    distances = np.linalg.norm(X_scaled - prelim.cluster_centers_[prelim_labels], axis=1)

    core_threshold = np.quantile(distances, CORE_QUANTILE)
    core_mask = distances <= core_threshold
    X_core = X.loc[core_mask]

    model = _build_pipeline()
    core_labels = model.fit_predict(X_core)
    refined_silhouette = round(
        float(silhouette_score(model.named_steps["scaler"].transform(X_core), core_labels)), 4
    )

    labels = model.predict(X)
    mapping = _cluster_mapping(model)
    df["cluster_id"] = labels
    df["client_class"] = df["cluster_id"].map(mapping)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(mapping, MAPPING_PATH)

    cluster_summary = df.groupby("client_class", as_index=False)[SEGMENT_FEATURES].mean()
    cluster_counts = df["client_class"].value_counts().reset_index()
    cluster_counts.columns = ["client_class", "count"]
    cluster_summary = cluster_summary.merge(cluster_counts, on="client_class")

    # Calculate CV per cluster for both features
    income_cvs = []
    price_cvs = []
    for cluster_name in cluster_summary["client_class"]:
        cluster_df = df[df["client_class"] == cluster_name]
        
        # Income CV
        i_mean = cluster_df["estimated_income"].mean()
        i_std = cluster_df["estimated_income"].std()
        income_cv = (i_std / i_mean) * 100 if i_mean != 0 else 0
        
        # Price CV
        p_mean = cluster_df["selling_price"].mean()
        p_std = cluster_df["selling_price"].std()
        price_cv = (p_std / p_mean) * 100 if p_mean != 0 else 0
        
        income_cvs.append(round(float(income_cv), 2))
        price_cvs.append(round(float(price_cv), 2))
    
    cluster_summary["Income CV (%)"] = income_cvs
    cluster_summary["Price CV (%)"] = price_cvs
    
    # Rename columns for final presentation
    cluster_summary.columns = [
        "Client Category",
        "Avg. Estimated Income",
        "Avg. Selling Price",
        "Total Clients",
        "Income CV (%)",
        "Price CV (%)"
    ]

    comparison_df = df[
        ["client_name", "estimated_income", "selling_price", "client_class", "district"]
    ]

    evaluation = {
        "silhouette": refined_silhouette,
        "baseline_silhouette": baseline_silhouette,
        "refined_sample_size": int(X_core.shape[0]),
        "full_sample_size": int(X.shape[0]),
        # Removed global coefficient_variation as requested
        "summary": cluster_summary.to_html(
            classes="table table-hover table-bordered shadow-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
        "comparison": comparison_df.head(12).to_html(
            classes="table table-hover table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
    }

    return model, mapping, evaluation


def train_clustering_model(force: bool = False):
    global _cached_eval
    if force or not MODEL_PATH.exists() or not MAPPING_PATH.exists() or _cached_eval is None:
        model, mapping, _cached_eval = _train_and_evaluate()
        return model, mapping
    return joblib.load(MODEL_PATH), joblib.load(MAPPING_PATH)


def evaluate_clustering_model():
    global _cached_eval
    if _cached_eval is None:
        train_clustering_model(force=not MODEL_PATH.exists())
    return _cached_eval


if __name__ == "__main__":
    train_clustering_model(force=True)
    print(evaluate_clustering_model())
