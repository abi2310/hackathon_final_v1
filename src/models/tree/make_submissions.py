import pandas as pd
import numpy as np
import joblib
import os

MODEL_PATH = "models/tree/pipeline/decision_tree_squared_error_pipeline.pkl"

PUBLIC_PATH = "data/raw/df_eval_public_2025-11-03.csv"
PRIVATE_PATH = "data/raw/df_eval_private_2025-11-03.csv"
IDS_PATH = "data/raw/df_IDs_for_eval_2025-11-03.csv"

OUTPUT_PATH = "submissions/tree_sqaured_error_submission.csv"
os.makedirs("submissions", exist_ok=True)

print("📥 Lade Modell...")
model = joblib.load(MODEL_PATH)

# -----------------------------
# Preprocessor & Spalten aus dem Modell
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
print("Numerische Spalten laut Modell:", num_cols)
print("Kategoriale Spalten laut Modell:", cat_cols)

# -----------------------------
# Eval-Daten laden
# -----------------------------
print("📥 Lade Testdaten...")
df_public = pd.read_csv(PUBLIC_PATH)
df_private = pd.read_csv(PRIVATE_PATH)
df_ids = pd.read_csv(IDS_PATH)

df_eval = pd.concat([df_public, df_private], axis=0, ignore_index=True)

print("Eval shape:", df_eval.shape)
print("IDs shape:", df_ids.shape)

# Reihenfolge über IDs sicherstellen
df_eval_sorted = df_ids.merge(df_eval, on="AuftragsID", how="left")
print("Shape nach Merge:", df_eval_sorted.shape)

# -----------------------------
# Fehlende Spalten ergänzen
# -----------------------------
for col in expected_cols:
    if col not in df_eval_sorted.columns:
        print(f"⚠️ Spalte {col} fehlt in Eval-Daten – wird mit NaN ergänzt.")
        df_eval_sorted[col] = np.nan

# -----------------------------
# pd.NA -> np.nan GLOBAL fixen
# -----------------------------
df_eval_sorted = df_eval_sorted.replace({pd.NA: np.nan})

# -----------------------------
# Numerische Spalten sicher in numerisch casten
# -----------------------------
for col in num_cols:
    if col in df_eval_sorted.columns:
        df_eval_sorted[col] = pd.to_numeric(df_eval_sorted[col], errors="coerce")

# IDs für Submission merken
submission_ids = df_eval_sorted["AuftragsID"].copy()

# Für das Modell: genau die Spalten, die das Modell kennt
X = df_eval_sorted[expected_cols]

print("🔮 Mache Predictions...")
preds_numeric = model.predict(X)

print("Konvertiere Predictions in Datum...")
preds_dates = pd.to_datetime(
    np.round(preds_numeric).astype("int64")
).strftime("%Y-%m-%d")

submission = pd.DataFrame({
    "AuftragsID": submission_ids,
    "Auftragsende_PREDICTED": preds_dates
})


# Kaggle-required line
submission["ID"] = np.arange(1, len(submission) + 1)

# Optional: Spalten sortieren (schön & klar)
submission = submission[["ID", "AuftragsID", "Auftragsende_PREDICTED"]]


submission.to_csv(OUTPUT_PATH, index=False)

print("✅ Fertig! Datei ist ready für Kaggle-Upload.")
