from pathlib import Path

import joblib
from django.shortcuts import render

from model_generators.classification.train_classifier import (evaluate_classification_model, train_classification_model,)
from model_generators.clustering.train_cluster import (evaluate_clustering_model, train_clustering_model,)
from model_generators.regression.train_regression import ( evaluate_regression_model, train_regression_model,)
from predictor.data_exploration import (data_exploration, dataset_exploration, district_map_html, world_map_html, load_dataset)

BASE_DIR = Path(__file__).resolve().parent.parent
REGRESSION_MODEL_PATH = BASE_DIR / "model_generators" / "regression" / "regression_model.pkl"
CLASSIFICATION_MODEL_PATH = (
    BASE_DIR / "model_generators" / "classification" / "classification_model.pkl"
)
CLUSTERING_MODEL_PATH = BASE_DIR / "model_generators" / "clustering" / "clustering_model.pkl"
CLUSTER_MAPPING_PATH = BASE_DIR / "model_generators" / "clustering" / "cluster_mapping.pkl"


def _load_models():
    try:
        if not REGRESSION_MODEL_PATH.exists():
            train_regression_model(force=True)
        if not CLASSIFICATION_MODEL_PATH.exists():
            train_classification_model(force=True)
        if not CLUSTERING_MODEL_PATH.exists() or not CLUSTER_MAPPING_PATH.exists():
            train_clustering_model(force=True)

        regression_model = joblib.load(REGRESSION_MODEL_PATH)
        classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
        clustering_model = joblib.load(CLUSTERING_MODEL_PATH)
        cluster_mapping = joblib.load(CLUSTER_MAPPING_PATH)
    except Exception:
        # Recover from missing/corrupted model artifacts by retraining once.
        train_regression_model(force=True)
        train_classification_model(force=True)
        train_clustering_model(force=True)
        regression_model = joblib.load(REGRESSION_MODEL_PATH)
        classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
        clustering_model = joblib.load(CLUSTERING_MODEL_PATH)
        cluster_mapping = joblib.load(CLUSTER_MAPPING_PATH)

    return regression_model, classification_model, clustering_model, cluster_mapping


regression_model, classification_model, clustering_model, cluster_mapping = _load_models()


def data_exploration_view(request):
    df = load_dataset()
    context = {
        "data_exploration": data_exploration(df),
        "dataset_exploration": dataset_exploration(df),
        "district_map": district_map_html(df),
    }
    return render(request, "predictor/index.html", context)


def world_map_view(request):
    df = load_dataset()
    context = {
        "world_map": world_map_html(df),
    }
    return render(request, "predictor/world_map.html", context)


def regression_analysis(request):
    context = {"evaluations": evaluate_regression_model()}

    if request.method == "POST":
        year = int(request.POST["year"])
        km = float(request.POST["km"])
        seats = int(request.POST["seats"])
        income = float(request.POST["income"])

        prediction = regression_model.predict([[year, km, seats, income]])[0]
        context["price"] = prediction

    return render(request, "predictor/regression_analysis.html", context)


def classification_analysis(request):
    context = {"evaluations": evaluate_classification_model()}

    if request.method == "POST":
        year = int(request.POST["year"])
        km = float(request.POST["km"])
        seats = int(request.POST["seats"])
        income = float(request.POST["income"])
        prediction = classification_model.predict([[year, km, seats, income]])[0]
        context["prediction"] = prediction

    return render(request, "predictor/classification_analysis.html", context)


def clustering_analysis(request):
    context = {"evaluations": evaluate_clustering_model()}

    if request.method == "POST":
        try:
            year = int(request.POST["year"])
            km = float(request.POST["km"])
            seats = int(request.POST["seats"])
            income = float(request.POST["income"])

            predicted_price = regression_model.predict([[year, km, seats, income]])[0]
            cluster_id = int(clustering_model.predict([[income, predicted_price]])[0])

            context.update(
                {
                    "prediction": cluster_mapping.get(cluster_id, "Unknown"),
                    "price": predicted_price,
                }
            )
        except Exception as exc:
            context["error"] = str(exc)

    return render(request, "predictor/clustering_analysis.html", context)
