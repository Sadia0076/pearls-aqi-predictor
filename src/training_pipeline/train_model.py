# src/training_pipeline/train_model.py

import os
import joblib
import shap
import numpy as np
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# STEP 1 — Load Features
# -----------------------------
def load_features():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("MONGO_URI is not set")

    client = MongoClient(mongo_uri)
    db = client["Pearls_aqi_feature_store"]
    col = db["karachi_air_qualityIndex"]

    df = pd.DataFrame(list(col.find({}, {"_id": 0})))

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df = df.dropna()
    df = df.sort_values("timestamp")

    return df


# -----------------------------
# STEP 2 — Prepare Training Data
# -----------------------------
def prepare_training_data():
    df = load_features()

    target = "pm25_next_hour"

    X = df.drop(columns=[
        "timestamp",
        "location",
        target
    ])

    y = df[target]

    return X, y


# -----------------------------
# STEP 3 — Model Registry
# -----------------------------
def save_model_to_registry(model, model_name, metrics):
    os.makedirs("models", exist_ok=True)

    version = datetime.utcnow().strftime("%Y%m%d_%H%M")
    model_path = f"models/aqi_model_{model_name}_{version}.pkl"

    joblib.dump(model, model_path)

    metadata = {
        "model_name": model_name,
        "version": version,
        "rmse": metrics["RMSE"],
        "mae": metrics["MAE"],
        "r2": metrics["R2"],
        "saved_at": version
    }

    registry_path = "models/model_registry.csv"

    if os.path.exists(registry_path):
        registry_df = pd.read_csv(registry_path)
        registry_df = pd.concat(
            [registry_df, pd.DataFrame([metadata])],
            ignore_index=True
        )
    else:
        registry_df = pd.DataFrame([metadata])

    registry_df.to_csv(registry_path, index=False)

    print(f"✅ Model saved: {model_path}")


# -----------------------------
# MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":
    X, y = prepare_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05
        )
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results[name] = {
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "MAE": mean_absolute_error(y_test, preds),
            "R2": r2_score(y_test, preds)
        }

    results_df = pd.DataFrame(results).T
    print(results_df)

    best_model_name = results_df["RMSE"].idxmin()
    best_model = models[best_model_name]

    save_model_to_registry(
        model=best_model,
        model_name=best_model_name,
        metrics=results[best_model_name]
    )

    # SHAP (Tree models only)
    if best_model_name in ["RandomForest", "GradientBoosting"]:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test)
