from src.data.load_data import load_data, split_data
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import joblib

# -----------------------------
# Daten laden
# -----------------------------
data = load_data()

print("Spalten im DataFrame:")
print(list(data.columns))

TARGET = "Auftragsende_IST"

# Target in Datetime umwandeln
data[TARGET] = pd.to_datetime(data[TARGET], errors="coerce")

# Zeilen ohne gültiges Ziel entfernen
data = data.dropna(subset=[TARGET]).copy()

# Nur Datum (Zeit auf 00:00:00)
data[TARGET] = data[TARGET].dt.normalize()

# -----------------------------
# Features / Target
# -----------------------------
X_full = data.drop(columns=[TARGET])
y_full = data[TARGET]

# bool-Spalte als numerisch casten (falls vorhanden)
if "is_transport_ruesten" in X_full.columns:
    X_full["is_transport_ruesten"] = X_full["is_transport_ruesten"].astype(int)

categorical = X_full.select_dtypes(include=["object"]).columns.tolist()
numeric = X_full.select_dtypes(include=[np.number]).columns.tolist()

print("Kategoriale Spalten:", categorical)
print("Numerische Spalten:", numeric)

# Train/Test Split – wie bei den linearen Modellen
X_train, X_test, y_train_dt, y_test_dt = split_data(
    data, test_size=0.2, random_state=42
)

# Targets als Datetime
y_train_dt = pd.to_datetime(y_train_dt, errors="coerce")
y_test_dt = pd.to_datetime(y_test_dt, errors="coerce")

# Nur Zeilen mit gültigem Target behalten
valid_idx_train = ~y_train_dt.isna()
valid_idx_test = ~y_test_dt.isna()

X_train = X_train[valid_idx_train]
y_train_dt = y_train_dt[valid_idx_train]

X_test = X_test[valid_idx_test]
y_test_dt = y_test_dt[valid_idx_test]

# Target: Datetime → int64 (ns seit 1970-01-01)
y_train = y_train_dt.astype("int64")
y_test = y_test_dt.astype("int64")

# -----------------------------
# Preprocessing (gleich wie oben)
# -----------------------------
cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
])

num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_pipeline, categorical),
        ("num", num_pipeline, numeric),
    ]
)

# -----------------------------
# Entscheidungsbäume
# -----------------------------
criterions = ["absolute_error", "squared_error"]

output_dir = "models/tree/pipeline"
os.makedirs(output_dir, exist_ok=True)

NS_PER_DAY = 24 * 60 * 60 * 1e9  # Nanosekunden pro Tag

for crit in criterions:
    tree = DecisionTreeRegressor(
        criterion=crit,
        max_depth=6,
        random_state=42,
        min_samples_split=10,
        min_samples_leaf=5,
    )

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", tree),
        ]
    )

    # Training
    pipe.fit(X_train, y_train)

    # Vorhersage (numeric ns)
    preds_numeric = pipe.predict(X_test)

    # Metriken in ns
    mse = mean_squared_error(y_test, preds_numeric)
    mae = mean_absolute_error(y_test, preds_numeric)
    r2 = r2_score(y_test, preds_numeric)

    # MAE in Tagen
    mae_days = mae / NS_PER_DAY

    print(f"\n✅ Ergebnisse Decision Tree ({crit}) Pipeline-Modell:")
    print(f"MAE (ns):   {mae:.2f}")
    print(f"MAE (Tage): {mae_days:.4f}")
    print(f"MSE:        {mse:.2e}")
    print(f"R2:         {r2:.4f}")

    # Modell speichern
    model_path = os.path.join(output_dir, f"decision_tree_{crit}_pipeline.pkl")
    joblib.dump(pipe, model_path)

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": model_path,
        "metrics": {
            "MAE_ns": float(mae),
            "MAE_days": float(mae_days),
            "MSE": float(mse),
            "R2": float(r2),
        },
        "model_type": f"DecisionTree_{crit}",
    }

    metrics_path = os.path.join(output_dir, f"decision_tree_{crit}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"📦 Modell & Metriken gespeichert unter: {metrics_path}")

    
