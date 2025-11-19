from src.data.load_data import split_data, load_data
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import optuna
from catboost import CatBoostRegressor, Pool, cv
import os
import json
from datetime import datetime
import numpy as np


# -----------------------------
# Daten laden
# -----------------------------
data = load_data()
X_train, X_test, y_train, y_test = split_data(data, test_size=0.2, random_state=42)

# CatBoost: automatische Kategorieerkennung mit Strings
cat_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

train_pool = Pool(X_train, y_train, cat_features=cat_features)


# -----------------------------
# Optuna-Objective (optimiert)
# -----------------------------
def objective(trial):
    params = {
        "loss_function": "RMSE",
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "iterations": trial.suggest_int("iterations", 300, 1200),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 20),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.1, 5.0),
        "border_count": trial.suggest_int("border_count", 128, 254),
        "random_seed": 42,
        "early_stopping_rounds": 100,
        "task_type": "CPU",
        "verbose": False,
    }

    cv_results = cv(
        pool=train_pool,
        params=params,
        fold_count=5,           # stabiler als 8, schneller
        shuffle=True,
        partition_random_seed=42,
        verbose=False
    )

    best_rmse = np.min(cv_results["test-RMSE-mean"])
    return best_rmse


# -----------------------------
# Optuna-Tuning starten
# -----------------------------
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("\nBeste Hyperparameter:", study.best_params)


# -----------------------------
# Bestes Modell trainieren
# -----------------------------
best_model = CatBoostRegressor(
    **study.best_params,
    loss_function="RMSE",
    random_seed=42,
    verbose=False
)

best_model.fit(X_train, y_train, cat_features=cat_features)


# -----------------------------
# Evaluation
# -----------------------------
preds = best_model.predict(X_test)

mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\n📊 PERFORMANCE")
print("MAE:", mae)
print("RMSE:", np.sqrt(mse))
print("R2:", r2)

print("\n📌 Top Feature Importance:")
print(best_model.get_feature_importance(prettified=True))


# -----------------------------
# Speichern
# -----------------------------
output_dir = "models/tree/boosting"
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(output_dir, "catboost_model.cbm")
best_model.save_model(model_path)

# Metriken speichern
results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_path": model_path,
    "metrics": {"MAE": mae, "RMSE": np.sqrt(mse), "R2": r2},
    "best_params": study.best_params    
}

metrics_path = os.path.join(output_dir, "catboost_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(results, f, indent=4)

print("\n✅ Modell gespeichert:", model_path)