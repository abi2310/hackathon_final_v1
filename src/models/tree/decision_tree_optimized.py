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
# Feature Engineering
# -----------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Sicherstellen, dass relevante Spalten datetime sind
    dt_cols = [
        "Auftragseingang",
        "Auftragsende_SOLL",
        "AFO_Start_SOLL",
        "AFO_Ende_SOLL",
        "AFO_Start_IST",
        "AFO_Ende_IST",
    ]
    for c in dt_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Dauer SOLL: Auftrag in Tagen
    if "Auftragseingang" in df.columns and "Auftragsende_SOLL" in df.columns:
        delta = df["Auftragsende_SOLL"] - df["Auftragseingang"]
        df["soll_dauer_auftrag_tag"] = delta.dt.total_seconds() / 86400

        # Kalendereigenschaften
        df["auftragseingang_dow"] = df["Auftragseingang"].dt.weekday
        df["auftragseingang_monat"] = df["Auftragseingang"].dt.month

    # Dauer SOLL: AFO in Stunden
    if "AFO_Start_SOLL" in df.columns and "AFO_Ende_SOLL" in df.columns:
        delta_afo_soll = df["AFO_Ende_SOLL"] - df["AFO_Start_SOLL"]
        df["soll_dauer_afo_stunden"] = delta_afo_soll.dt.total_seconds() / 3600

    # Dauer IST: AFO in Stunden
    if "AFO_Start_IST" in df.columns and "AFO_Ende_IST" in df.columns:
        delta_afo_ist = df["AFO_Ende_IST"] - df["AFO_Start_IST"]
        df["ist_dauer_afo_stunden"] = delta_afo_ist.dt.total_seconds() / 3600

    return df

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

# Feature Engineering anwenden
data_fe = add_time_features(data)

# -----------------------------
# Features / Target
# -----------------------------
X_full = data_fe.drop(columns=[TARGET])
y_full = data_fe[TARGET]

# AuftragsID NICHT als Feature verwenden
if "AuftragsID" in X_full.columns:
    X_full = X_full.drop(columns=["AuftragsID"])

# MaschinenID eher als Kategorie behandeln
if "MaschinenID" in X_full.columns:
    X_full["MaschinenID"] = X_full["MaschinenID"].astype("Int64").astype("string")

# bool-Spalte als numerisch casten (falls vorhanden)
if "is_transport_ruesten" in X_full.columns:
    X_full["is_transport_ruesten"] = X_full["is_transport_ruesten"].astype(int)

# Spaltentypen bestimmen (Objekt + string = kategorial)
categorical = X_full.select_dtypes(include=["object", "string"]).columns.tolist()
numeric = X_full.select_dtypes(include=[np.number]).columns.tolist()

print("Kategoriale Spalten:", categorical)
print("Numerische Spalten:", numeric)

# DataFrame für split_data vorbereiten (X + Target)
data_fe_for_split = X_full.copy()
data_fe_for_split[TARGET] = y_full

# Train/Test Split – gleiche Logik wie bei deinen anderen Modellen
X_train, X_test, y_train_dt, y_test_dt = split_data(
    data_fe_for_split, test_size=0.2, random_state=42
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
# Preprocessing
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
# Entscheidungsbäume (etwas regularisiert)
# -----------------------------
criterions = ["squared_error", "absolute_error"]

output_dir = "models/tree/pipeline"
os.makedirs(output_dir, exist_ok=True)

NS_PER_DAY = 24 * 60 * 60 * 1e9  # Nanosekunden pro Tag

for crit in criterions:
    tree = DecisionTreeRegressor(
        criterion=crit,
        max_depth=10,          # etwas tiefer, aber nicht zu tief
        min_samples_split=200, # verhindert Mini-Blätter
        min_samples_leaf=100,
        random_state=42,
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
