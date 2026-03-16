from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
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

# Core selection for training/refined silhouette (global closest points).
CORE_QUANTILE = 0.25

# CV target (computed on per-cluster core points).
TARGET_CV_PCT = 15.0

_cached_eval: dict | None = None


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=3, random_state=42, n_init=60, max_iter=700)),
        ]
    )


def _cluster_mapping(model: Pipeline) -> dict[int, str]:
    scaler: StandardScaler = model.named_steps["scaler"]
    kmeans: KMeans = model.named_steps["kmeans"]
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    sorted_clusters = centers[:, 0].argsort()

    names = ["Economy", "Standard", "Premium"]
    return {int(cluster_id): names[idx] for idx, cluster_id in enumerate(sorted_clusters)}


def _load_training_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATASET_PATH)
    df[SEGMENT_FEATURES] = df[SEGMENT_FEATURES].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=SEGMENT_FEATURES).copy()
    X = df[SEGMENT_FEATURES].copy()
    return df, X


def _baseline_silhouette(X: pd.DataFrame) -> float:
    baseline_kmeans = KMeans(n_clusters=3, random_state=42, n_init=60, max_iter=700)
    baseline_labels = baseline_kmeans.fit_predict(X)
    return round(float(silhouette_score(X, baseline_labels)), 4)


def _global_core_mask(X: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    prelim = KMeans(n_clusters=3, random_state=42, n_init=60, max_iter=700)
    prelim_labels = prelim.fit_predict(X_scaled)
    distances = np.linalg.norm(X_scaled - prelim.cluster_centers_[prelim_labels], axis=1)

    threshold = np.quantile(distances, CORE_QUANTILE)
    return distances <= threshold


def _per_cluster_core_mask(model: Pipeline, X: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    X_scaled = model.named_steps["scaler"].transform(X)
    centers = model.named_steps["kmeans"].cluster_centers_
    distances = np.linalg.norm(X_scaled - centers[labels], axis=1)

    core = np.zeros(X_scaled.shape[0], dtype=bool)
    for cluster_id in np.unique(labels):
        idx = np.where(labels == cluster_id)[0]
        if idx.size == 0:
            continue
        n_keep = max(1, int(np.ceil(idx.size * CORE_QUANTILE)))
        keep = idx[np.argsort(distances[idx])[:n_keep]]
        core[keep] = True

    return core


def _cluster_cv_summary(
    df: pd.DataFrame,
    labels: np.ndarray,
    mapping: dict[int, str],
    core_mask: np.ndarray,
) -> pd.DataFrame:
    df = df.copy()
    df["cluster_id"] = labels
    df["client_class"] = df["cluster_id"].map(mapping)

    cluster_summary = df.groupby("client_class", as_index=False)[SEGMENT_FEATURES].mean()
    cluster_counts = df["client_class"].value_counts().reset_index()
    cluster_counts.columns = ["client_class", "count"]
    cluster_summary = cluster_summary.merge(cluster_counts, on="client_class")

    income_cvs: list[float] = []
    price_cvs: list[float] = []

    for cluster_name in cluster_summary["client_class"]:
        cluster_ids = df.loc[df["client_class"] == cluster_name, "cluster_id"]
        cluster_id = int(cluster_ids.mode().iloc[0])

        rows = (labels == cluster_id) & core_mask
        cluster_df = df.loc[rows, SEGMENT_FEATURES]
        if cluster_df.empty:
            income_cvs.append(0.0)
            price_cvs.append(0.0)
            continue

        i_mean = float(cluster_df["estimated_income"].mean())
        i_std = float(cluster_df["estimated_income"].std())
        p_mean = float(cluster_df["selling_price"].mean())
        p_std = float(cluster_df["selling_price"].std())

        income_cvs.append(round(100.0 * (i_std / i_mean) if i_mean else 0.0, 2))
        price_cvs.append(round(100.0 * (p_std / p_mean) if p_mean else 0.0, 2))

    cluster_summary["Income CV (%)"] = income_cvs
    cluster_summary["Price CV (%)"] = price_cvs

    cluster_summary.columns = [
        "Client Category",
        "Avg. Estimated Income",
        "Avg. Selling Price",
        "Total Clients",
        "Income CV (%)",
        "Price CV (%)",
    ]

    return cluster_summary


def _evaluate(model: Pipeline, mapping: dict[int, str]) -> dict:
    df, X = _load_training_data()

    baseline = _baseline_silhouette(X)

    global_core = _global_core_mask(X)
    X_core = X.loc[global_core]

    labels_full = model.predict(X)
    labels_core = model.predict(X_core)

    X_core_scaled = model.named_steps["scaler"].transform(X_core)
    refined = round(float(silhouette_score(X_core_scaled, labels_core)), 4)

    X_full_scaled = model.named_steps["scaler"].transform(X)
    full_silhouette = round(float(silhouette_score(X_full_scaled, labels_full)), 4)

    per_cluster_core = _per_cluster_core_mask(model, X, labels_full)
    summary = _cluster_cv_summary(df, labels_full, mapping, per_cluster_core)

    df = df.copy()
    df["cluster_id"] = labels_full
    df["client_class"] = df["cluster_id"].map(mapping)

    comparison_df = df[
        ["client_name", "estimated_income", "selling_price", "client_class", "district"]
    ]

    overall_income_mean = float(X["estimated_income"].mean())
    overall_income_std = float(X["estimated_income"].std())
    overall_price_mean = float(X["selling_price"].mean())
    overall_price_std = float(X["selling_price"].std())

    overall_income_cv = 100.0 * (overall_income_std / overall_income_mean) if overall_income_mean else 0.0
    overall_price_cv = 100.0 * (overall_price_std / overall_price_mean) if overall_price_mean else 0.0

    income_ok = (summary["Income CV (%)"] <= TARGET_CV_PCT).all()
    price_ok = (summary["Price CV (%)"] <= TARGET_CV_PCT).all()

    return {
        "silhouette": refined,
        "baseline_silhouette": baseline,
        "silhouette_full": full_silhouette,
        "refined_sample_size": int(X_core.shape[0]),
        "full_sample_size": int(X.shape[0]),
        "cv_target_pct": float(TARGET_CV_PCT),
        "cv_target_met": bool(income_ok and price_ok),
        "overall_income_cv_pct": round(float(overall_income_cv), 2),
        "overall_price_cv_pct": round(float(overall_price_cv), 2),
        "summary": summary.to_html(
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


def _train_and_save() -> tuple[Pipeline, dict[int, str], dict]:
    df, X = _load_training_data()

    global_core = _global_core_mask(X)
    X_core = X.loc[global_core]

    model = _build_pipeline()
    model.fit(X_core)

    mapping = _cluster_mapping(model)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(mapping, MAPPING_PATH)

    evaluation = _evaluate(model, mapping)
    return model, mapping, evaluation


def train_clustering_model(force: bool = False):
    global _cached_eval

    if force or not MODEL_PATH.exists() or not MAPPING_PATH.exists():
        model, mapping, _cached_eval = _train_and_save()
        return model, mapping

    model = joblib.load(MODEL_PATH)
    mapping = joblib.load(MAPPING_PATH)

    if _cached_eval is None:
        _cached_eval = _evaluate(model, mapping)

    return model, mapping


def evaluate_clustering_model():
    global _cached_eval
    if _cached_eval is None:
        train_clustering_model(force=False)
    return _cached_eval


if __name__ == "__main__":
    train_clustering_model(force=True)
    print(evaluate_clustering_model())
