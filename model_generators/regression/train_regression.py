from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "dummy-data" / "vehicles_ml_dataset.csv"
MODEL_PATH = BASE_DIR / "model_generators" / "regression" / "regression_model.pkl"

FEATURES = ["year", "kilometers_driven", "seating_capacity", "estimated_income"]
TARGET = "selling_price"

_cached_eval = None


def _train_and_evaluate():
    df = pd.read_csv(DATASET_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=220, random_state=42)
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    predictions = model.predict(X_test)
    r2 = round(r2_score(y_test, predictions) * 100, 2)

    comparison_df = pd.DataFrame(
        {
            "Actual": y_test.values,
            "Predicted": predictions.round(2),
            "Difference": (y_test.values - predictions).round(2),
        }
    )

    evaluation = {
        "r2": r2,
        "comparison": comparison_df.head(10).to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
    }

    return model, evaluation


def train_regression_model(force: bool = False):
    global _cached_eval
    if force or not MODEL_PATH.exists() or _cached_eval is None:
        model, _cached_eval = _train_and_evaluate()
        return model
    return joblib.load(MODEL_PATH)


def evaluate_regression_model():
    global _cached_eval
    if _cached_eval is None:
        train_regression_model(force=not MODEL_PATH.exists())
    return _cached_eval


if __name__ == "__main__":
    train_regression_model(force=True)
    print(evaluate_regression_model())
