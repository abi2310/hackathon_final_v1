import pandas as pd
import numpy as np
import joblib
import os

MODEL_PATH = "models/lightgbm/pipeline/best_lgbm_pipeline.pkl"

PUBLIC_PATH = "data/processed/df_eval_public_v2.csv"
PRIVATE_PATH = "data/processed/df_eval_privae_v2.csv"
IDS_PATH = "data/raw/df_IDs_for_eval_2025-11-03.csv"

OUTPUT_PATH = "submissions/XXX_Lightgbm_submission_v7.csv"
os.makedirs("submissions", exist_ok=True)

# Muss zu deinem Trainingsskript passen!
DATE_COLS = [
    "Auftragseingang",
    "Auftragsende_SOLL",
    "AFO_Start_SOLL",
    "AFO_Ende_SOLL",
    "AFO_Start_IST",
    "AFO_Ende_IST",
]

DROP_IDS = ["AuftragsID", "BauteilID", "MaschinenID"]

print("📥 Lade Modell...")
model = joblib.load(MODEL_PATH)

# -----------------------------
# Preprocessor & Spalten aus Pipeline
# -----------------------------
prep = model.named_steps["prep"]

expected_cols = []
cat_cols = []
num_cols = []

for name, transformer, cols in prep.transformers_:
    if cols is None or cols == "drop":
        continue
    cols = list(cols)
    expected_cols.extend(cols)
    if name == "cat":
        cat_cols = cols
    elif name == "num":
        num_cols = cols

expected_cols = list(dict.fromkeys(expected_cols))

print("Erwartete Spalten im Modell:", expected_cols)

# -----------------------------
# Eval-Daten laden
# -----------------------------
print("📥 Lade Testdaten...")
df_public = pd.read_csv(PUBLIC_PATH)
df_private = pd.read_csv(PRIVATE_PATH)
df_ids = pd.read_csv(IDS_PATH)

df_eval = pd.concat([df_public, df_private], ignore_index=True)
print("Eval shape:", df_eval.shape)

# IDs-Merge zur richtigen Reihenfolge
df_eval_sorted = df_ids.merge(df_eval, on="AuftragsID", how="left")
print("Shape nach Merge:", df_eval_sorted.shape)

# -----------------------------
# Auftragseingang (als Datetime) für Enddatum-Berechnung
# -----------------------------
start_dt_eval = pd.to_datetime(df_eval_sorted["Auftragseingang"], errors="coerce")

# -----------------------------
# Datums-Spalten als datetime parsen
# -----------------------------
for col in DATE_COLS:
    if col in df_eval_sorted.columns:
        df_eval_sorted[col] = pd.to_datetime(df_eval_sorted[col], errors="coerce")

# -----------------------------
# Date-Features wie im Training erzeugen
# -----------------------------
for col in DATE_COLS:
    if col in df_eval_sorted.columns:
        df_eval_sorted[f"{col}_dow"] = df_eval_sorted[col].dt.dayofweek
        df_eval_sorted[f"{col}_hour"] = df_eval_sorted[col].dt.hour
        df_eval_sorted[f"{col}_day"] = df_eval_sorted[col].dt.day
        df_eval_sorted[f"{col}_month"] = df_eval_sorted[col].dt.month
        df_eval_sorted[f"{col}_week"] = df_eval_sorted[col].dt.isocalendar().week.astype("Int64")

# Original datetime-Spalten entfernen (wie im Training)
for col in DATE_COLS:
    if col in df_eval_sorted.columns:
        df_eval_sorted = df_eval_sorted.drop(columns=[col])

# ID-Spalten droppen (wie im Training)
for col in DROP_IDS:
    if col in df_eval_sorted.columns:
        df_eval_sorted = df_eval_sorted.drop(columns=[col])

# -----------------------------
# Sicherstellen, dass alle erwarteten Spalten existieren
# -----------------------------
for col in expected_cols:
    if col not in df_eval_sorted.columns:
        df_eval_sorted[col] = np.nan
        print(f"⚠️ Fehlende Spalte ergänzt: {col}")

# -----------------------------
# Numerische Spalten castaen
# -----------------------------
df_eval_sorted = df_eval_sorted.replace({pd.NA: np.nan})

for col in num_cols:
    if col in df_eval_sorted.columns:
        df_eval_sorted[col] = pd.to_numeric(df_eval_sorted[col], errors="coerce")

# -----------------------------
# Features für Modell
# -----------------------------
X = df_eval_sorted[expected_cols]

print("🔮 Mache Predictions (Dauer in Tagen)...")
pred_duration_days = model.predict(X)

# -----------------------------
# Enddatum berechnen
# -----------------------------
pred_end_dt = start_dt_eval + pd.to_timedelta(np.round(pred_duration_days), unit="D")
preds_dates = pred_end_dt.dt.strftime("%Y-%m-%d")

# -----------------------------
# Submission erstellen
# -----------------------------
submission = pd.DataFrame({
    "AuftragsID": df_ids["AuftragsID"],  # Reihenfolge aus df_ids
    "Auftragsende_PREDICTED": preds_dates,
})

# Kaggle-required ID column
submission["ID"] = np.arange(1, len(submission) + 1)
submission = submission[["ID", "AuftragsID", "Auftragsende_PREDICTED"]]

submission.to_csv(OUTPUT_PATH, index=False)

print("✅ Fertig! Datei ist ready für Kaggle-Upload:", OUTPUT_PATH)
