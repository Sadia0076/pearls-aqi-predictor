# src/training_pipeline/train_model.py

import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import os
from datetime import datetime
import shap

# -----------------------------
# STEP 1 — Fetch Historical Data
# -----------------------------
def load_features():
     client = MongoClient(os.getenv("MONGO_URI"))
    db = client["Pearls_aqi_feature_store"]
    col = db["karachi_air_qualityIndex"]

    df = pd.DataFrame(list(col.find({}, {"_id": 0})))

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )

    df = df.dropna()
    df = df.sort_values("timestamp")

    return df

# -----------------------------
# STEP 2 — Separate Features & Target
# -----------------------------
def prepare_training_data():
    df = load_features()

    TARGET = "pm25_next_hour"

    X = df.drop(columns=[
        "timestamp",
        "location",
        TARGET
    ])

    y = df[TARGET]

    return X, y

# -----------------------------
# STEP 4 — Model Registry
# -----------------------------
def save_model_to_registry(model, model_name, metrics):
    """
    Saves model with versioning and logs metadata
    """
    os.makedirs("models", exist_ok=True)

    version = datetime.now().strftime("%Y%m%d_%H%M")
    model_filename = f"aqi_model_{model_name}_{version}.pkl"
    model_path = os.path.join("models", model_filename)

    # Save model artifact
    joblib.dump(model, model_path)

    # Create metadata entry
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
    print("📘 Registry updated")

# -----------------------------
# MAIN TRAINING PIPELINE
# -----------------------------
if __name__ == "__main__":
    # STEP 2 — Prepare data
    X, y = prepare_training_data()

    # STEP 3 — Train/Test Split (time-series safe)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # STEP 4 — Define models
    models = {
        "LinearRegression": LinearRegression(),
        "RidgeRegression": Ridge(alpha=1.0),
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

    # STEP 5 — Train & Evaluate
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
    print("\n📊 Model Evaluation Results:")
    print(results_df)

    # STEP 6 — Select Best Model
    best_model_name = results_df["RMSE"].idxmin()
    best_model = models[best_model_name]
    best_model_metrics = results[best_model_name]

    print(f"\n Best model selected: {best_model_name}")

    # STEP 7 — Save Best Model to Registry
    save_model_to_registry(
        model=best_model,
        model_name=best_model_name,
        metrics=best_model_metrics
    )
     # -----------------------------
# STEP 8 — SHAP Explainability
# -----------------------------
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)



