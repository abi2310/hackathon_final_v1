import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path='data/processed/data_cleaned_maschinen_2.csv'):
   df = pd.read_csv(file_path)
   print("Spalten im DataFrame:")
   print(df.columns.tolist())
   return df


def split_data(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
):
    target_col = "Auftragsende_IST"
    if target_col not in data.columns:
        raise ValueError(f"Die Spalte '{target_col}' fehlt im DataFrame!")

    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Daten gesplittet: Train={len(X_train)}, Test={len(X_test)} (Testgröße={test_size})")
    return X_train, X_test, y_train, y_test