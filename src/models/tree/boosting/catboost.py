from src.data.load_data import load_data
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import joblib

from lightgbm import LGBMRegressor

# =============================
# Konstanten
# =============================
TARGET_END = "Auftragsende_IST"
START_COL = "Auftragseingang"

DATE_COLS = [
    "Auftragseingang",
    "Auftragsende_SOLL",
    "AFO_Start_SOLL",
    "AFO_Ende_SOLL",
    "AFO_Start_IST",
    "AFO_Ende_IST",
]

# =============================
# Daten laden
# =============================
data = load_data()

print("Original Spalten:", list(data.columns))

# -----------------------------
# Datums-Spalten parsen
# -----------------------------
for col in DATE_COLS:
    data[col] = pd.to_datetime(data[col], errors="coerce")

# Target
data[TARGET_END] = pd.to_datetime(data[TARGET_END], errors="coerce")

# Valid rows
mask_valid = (~data[TARGET_END].isna()) & (~data[START_COL].isna())
data = data[mask_valid].copy()

start_dt = data[START_COL]

# =============================
# Dauer in Tagen berechnen
# =============================
duration_days = (data[TARGET_END] - start_dt).dt.total_seconds() / 86400.0
duration_days = duration_days.astype("float32")
y = duration_days

# =============================
# Date-features extrahieren
# =============================
for col in DATE_COLS:
    data[f"{col}_dow"] = data[col].dt.dayofweek
    data[f"{col}_hour"] = data[col].dt.hour
    data[f"{col}_day"] = data[col].dt.day
    data[f"{col}_month"] = data[col].dt.month
    data[f"{col}_week"] = data[col].dt.isocalendar().week.astype(int)

# Original datetime-Spalten entfernen
data = data.drop(columns=DATE_COLS + [TARGET_END])

# =============================
# ID-Spalten entfernen (vermeidet Overfitting)
# =============================
DROP_IDS = ["AuftragsID", "BauteilID", "MaschinenID"]
for col in DROP_IDS:
    if col in data.columns:
        data = data.drop(columns=[col])

# =============================
# Feature-Typen bestimmen
# =============================
categorical = data.select_dtypes(include=["object"]).columns.tolist()
numeric = data.select_dtypes(include=[np.number]).columns.tolist()

print("KATEGORIEN:", categorical)
print("NUMERISCH:", numeric)

# =============================
# Train/Test Split
# =============================
X_train, X_test, y_train, y_test, start_train_dt, start_test_dt = train_test_split(
    data, y, start_dt, test_size=0.2, random_state=42
)

# =============================
# Preprocessing
# =============================
cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
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

# =============================
# LightGBM Modell
# =============================
lgbm_model = LGBMRegressor(
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=64,
    min_data_in_leaf=100,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", lgbm_model),
])

# =============================
# Train
# =============================
pipe.fit(X_train, y_train)

preds_days = pipe.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, preds_days)
mse = mean_squared_error(y_test, preds_days)
r2 = r2_score(y_test, preds_days)

print("\n===== MODEL PERFORMANCE =====")
print("MAE Tage:", mae)
print("MSE Tage^2:", mse)
print("R²:", r2)

# =============================
# Speichern
# =============================
out_dir = "models/lightgbm/pipeline"
os.makedirs(out_dir, exist_ok=True)

model_path = os.path.join(out_dir, "lightgbm_pipeline.pkl")
joblib.dump(pipe, model_path)

print("📦 Saved:", model_path)
